"""
Author: Gabe Grand

Tools for running inference of a pretrained ControlNet model.
Adapted from gradio_scribble2image.py from the original authors.

"""


import sys

sys.path.append("..")
from share import *

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from PIL import Image
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

# A_PROMPT_DEFAULT = "photography,4k, HDR High dynamic range, vivid, rich details, clear shadows and highlights, realistic, intense, enhanced contrast, highly detailed"
# N_PROMPT_DEFAULT = "worst quality, low quality, illustration, 3d, painting, cartoons, sketch , tooth, dull, blurry, watermark, low quality, (flash:1.2) , bra, hat, tatto,,sun, sunlight, daylight, day-time, sun-ray, light, day-light, day-time, day, sepia, black-and-white, black-&-white, white-and-black, gray-scale, stars, moon, starry, large-breasts, big-breasts, huge-breasts, massive-breasts, giant-breasts, gradient, noise, poorly Rendered face, poorly drawn face, poor facial details, poorly rendered hands, low resolution, head cropped, frames, frame, framed, Images cut out at the top, left, right, bottom, bad composition, mutated body parts, blurry image, disfigured, over saturated, username, watermark, signature, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, unattractive, morbid, mutated hands, amateur, cursed, dull, boring, weird, beginner,blur, text, signature, painting, cartoon, cell-shade, contour, anime, 3d render, illustration, drone-footage, fog, glow, bloom, lens-flare, glare,sun, sunlight, daylight, day-time, sun-ray, light, day-light, day-time, day, sepia, black-and-white, black-&-white, white-and-black, gray-scale, stars, moon, starry, large-breasts, big-breasts, huge-breasts, massive-breasts, giant-breasts, gradient, noise, poorly Rendered face, poorly drawn face, poor facial details, poorly rendered hands, low resolution, head cropped, frames, frame, framed, Images cut out at the top, left, right, bottom, bad composition, mutated body parts, blurry image, disfigured, over saturated, username, watermark, signature, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, unattractive, morbid, mutated hands, amateur, cursed, dull, boring, weird, beginner"
A_PROMPT_DEFAULT = ""
N_PROMPT_DEFAULT = ""

def run_sampler(
    model,
    input_image: np.ndarray,
    prompt: str,
    num_samples: int = 1,
    image_resolution: int = 512,
    seed: int = -1,
    a_prompt: str = A_PROMPT_DEFAULT,
    n_prompt: str = N_PROMPT_DEFAULT,
    guess_mode=False,
    strength=1.0,
    ddim_steps=20,
    eta=0.0,
    scale=9.0,
    show_progress: bool = True,
):
    with torch.no_grad():
        if torch.cuda.is_available():
            model = model.cuda()

        ddim_sampler = DDIMSampler(model)
        original_shape = input_image.shape
        detected_map = resize_image(HWC3(input_image), image_resolution)
        detected_map= cv2.resize(detected_map, (512, 512))
        H, W, C = detected_map.shape
        # cv2.imwrite("detected_map.png",detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            show_progress=show_progress,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]
        resized_results = [cv2.resize(result, (original_shape[1], original_shape[0])) for result in results]
    
        return resized_results
    
    
import os

def process_images(input_dir: str, output_dir: str, model, prompt: str, seed: int = -1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_image_path = os.path.join(input_dir, filename)
            output_image_path = os.path.join(output_dir, filename)
            
            inference_image = cv2.imread(input_image_path)
            results = run_sampler(model, input_image=inference_image, prompt=prompt, seed=seed)
            save_data = cv2.cvtColor(results[0], cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_image_path, save_data)

if __name__ == '__main__':
    input_dir ="./demo_input"
    output_dir = "./demo_output2"

    prompt = "indoor scene"
    
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict("/share1/timingyang/IMOV3D-OPENSOURCE/CHECKPOINT/PC_render/epoch=116-step=550718.ckpt", location='cuda'))
    model = model.cuda()

    process_images(input_dir, output_dir, model, prompt)
