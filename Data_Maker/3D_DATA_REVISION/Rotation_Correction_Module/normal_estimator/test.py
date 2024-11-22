import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from data.dataloader_custom import CustomLoader
from models.NNET import NNET
import utils.utils as utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test(model, test_loader, device, results_dir,intput_imgs_dir):
    alpha_max = 60
    kappa_max = 30

    with torch.no_grad():
        for data_dict in tqdm(test_loader):

            img = data_dict['img'].to(device)
            norm_out_list, _, _ = model(img)
            norm_out = norm_out_list[-1]

            pred_norm = norm_out[:, :3, :, :]
            pred_kappa = norm_out[:, 3:, :, :]

            # to numpy arrays
            img = img.detach().cpu().permute(0, 2, 3, 1).numpy()                    # (B, H, W, 3)
            pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()        # (B, H, W, 3)
            pred_kappa = pred_kappa.cpu().permute(0, 2, 3, 1).numpy()

            # save results
            img_name = data_dict['img_name'][0]

            # 1. save input image
            img = utils.unnormalize(img[0, ...])
            target_path = '%s/%s_img.png' % (results_dir, img_name)
            plt.imsave(target_path, img)


            # Convert to PIL image object
            # Read the original image and get its size
            from PIL import Image
            original_image_path = os.path.join(intput_imgs_dir, f'{img_name}.jpg') 
        
            print(original_image_path)
            original_image = Image.open(original_image_path)
            original_width, original_height = original_image.size

            # 2. predicted normal
            print("pred_norm",pred_norm.shape)
            pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
            
            pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
            pred_norm_rgb = pred_norm_rgb.astype(np.uint8)                  # (B, H, W, 3)

            pred_norm_rgb_single = pred_norm_rgb[0, :, :, :]

            # Convert to PIL image object
            image = Image.fromarray(pred_norm_rgb_single)

            
            resized_image = image.resize((original_width, original_height))

            # Save the adjusted image
            target_path = '%s/%s_pred_norm.png' % (results_dir, img_name)
            resized_image.save(target_path)

            # pred_norm_rgb is your RGB image
            # First, we break it down into three channels
            R = pred_norm_rgb[0, :, :, 0]
            G = pred_norm_rgb[0, :, :, 1]
            B = pred_norm_rgb[0, :, :, 2]

            # Then we set a threshold, say 128
            threshold = 230

            # We find the part where the G channel is greater than the threshold
            plane_part = np.where(G > threshold)

            # Then we can create an empty black image, the same size as the original image
            output_img = np.zeros_like(R)

            # Finally, we set the plane portion we found to green in output_img
            output_img[plane_part] = 255  # RGB 中的绿色


            # Resize output_img to the same size as the original image
            output_img_resize = Image.fromarray(output_img)
            output_img_resize = output_img_resize.resize((original_width, original_height))

            # Convert output_img to NumPy array
            output_img_array = np.array(output_img_resize)
            print(np.expand_dims(output_img_array, axis=0).shape)
            os.makedirs(results_dir+"/npy", exist_ok=True)
            output_npy_path = '%s/%s.npy' % (results_dir+"/npy", img_name)
            print(output_npy_path )
            np.save(output_npy_path, np.expand_dims(output_img_array, axis=0))
            
            output_img_2d = output_img.astype(np.uint8)
            pil_image = Image.fromarray(output_img_2d, mode='L')
       
            # Save a new grayscale image
            target_path = '%s/%s.png' % (results_dir, img_name)
            pil_image.save(target_path)

            
            # 3. predicted kappa (concentration parameter)
            target_path = '%s/%s_pred_kappa.png' % (results_dir, img_name)
            plt.imsave(target_path, pred_kappa[0, :, :, 0], vmin=0.0, vmax=kappa_max, cmap='gray')

            # 4. predicted uncertainty
            pred_alpha = utils.kappa_to_alpha(pred_kappa)
            target_path = '%s/%s_pred_alpha.png' % (results_dir, img_name)
            plt.imsave(target_path, pred_alpha[0, :, :, 0], vmin=0.0, vmax=alpha_max, cmap='jet')

            # 5. concatenated results
            image_path_list = ['img', 'pred_norm', 'pred_alpha']
            image_path_list = ['%s/%s_%s.png' % (results_dir, img_name, i) for i in image_path_list]
            target_path = '%s/%s_concat.png' % (results_dir, img_name)
            utils.concat_image(image_path_list, target_path)


import torch.nn.parallel
import torch.distributed as dist
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
def init_process(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment"""
    dist.destroy_process_group()
    
def inference(rank, world_size, num_models_per_gpu,args):
    init_process(rank, world_size)
    # Define the number of models you want to run on each GPU
    # Determine which GPU the process should run on
    gpu_id = rank // num_models_per_gpu
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # load checkpoint
    checkpoint = './checkpoints/%s.pt' % args.pretrained
    print('loading checkpoint... {}'.format(checkpoint))
    model = NNET(args).to(device)
    model = utils.load_checkpoint(checkpoint, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id])
    model.eval()


    print('loading checkpoint... / done')

    # test the model
    results_dir = args.imgs_dir + '/results_scannet_green230'
    os.makedirs(results_dir, exist_ok=True)
    test_samples = CustomLoader(args, args.intput_imgs_dir).testing_samples
    filenames = test_samples.filenames

    
    # For each process, only the corresponding file is selected to load the data
    local_filenames = filenames[rank::world_size]
    local_dataset = Subset(test_samples, [test_samples.filenames.index(f) for f in local_filenames])
    local_loader = DataLoader(local_dataset, 1, shuffle=False, num_workers=1, pin_memory=True)  # Set pin_memory=True for faster transfers to GPU
        
    test(model, local_loader, device, results_dir, args.intput_imgs_dir)
    cleanup()

if __name__ == '__main__':
    # Arguments ########################################################################################################
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args

    parser.add_argument('--architecture', required=True, type=str, help='{BN, GN}')
    parser.add_argument("--pretrained", required=True, type=str, help="{nyu, scannet}")
    parser.add_argument('--sampling_ratio', type=float, default=0.4)
    parser.add_argument('--importance_ratio', type=float, default=0.7)
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--input_width', default=640, type=int)
    parser.add_argument('--imgs_dir', default='./examples_10w', type=str)
    parser.add_argument('--intput_imgs_dir', default='/share/datasets/lvis/train2017/', type=str) 
    # read arguments from txt file
    if sys.argv.__len__() == 2 and '.txt' in sys.argv[1]:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    num_models_per_gpu = 12
    world_size = num_models_per_gpu*torch.cuda.device_count()
    mp.spawn(inference, args=(world_size,num_models_per_gpu, args), nprocs=world_size, join=True)


