# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import mss
import torch
import torchvision.transforms as T
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config

from detic.predictor import VisualizationDemo

# Fake a video capture object OpenCV style - half width, half height of first screen using MSS
class ScreenGrab:
    def __init__(self):
        self.sct = mss.mss()
        m0 = self.sct.monitors[0]
        self.monitor = {'top': 0, 'left': 0, 'width': m0['width'] / 2, 'height': m0['height'] / 2}

    def read(self):
        img =  np.array(self.sct.grab(self.monitor))
        nf = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return (True, nf)

    def isOpened(self):
        return True
    def release(self):
        return True


# constants
WINDOW_NAME = "Detic"

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=['lvis', 'openimages', 'objects365', 'coco', 'custom'],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="human,sneakers,chair,hat,lamp,bottle,cabinet/shelf,cup,car,glasses,picture/frame,desk,handbag,street lights,book,plate,helmet,leather shoes,pillow,glove,potted plant,bracelet,flower,monitor,storage box,plants pot/vase,bench,wine glass,boots,dining table,umbrella,boat,flag,speaker,trash bin/can,stool,backpack,sofa,belt,carpet,basket,towel/napkin,slippers,bowl,barrel/bucket,coffee table,suv,toy,tie,bed,traffic light,pen/pencil,microphone,sandals,canned,necklace,mirror,faucet,bicycle,bread,high heels,ring,van,watch,combine with bowl,sink,horse,fish,apple,traffic sign,camera,candle,stuffed animal,cake,motorbike/motorcycle,wild bird,laptop,knife,cellphone,paddle,truck,cow,power outlet,clock,drum,fork,bus,hanger,nightstand,pot/pan,sheep,guitar,traffic cone,tea pot,keyboard,tripod,hockey stick,fan,dog,spoon,blackboard/whiteboard,balloon,air conditioner,cymbal,mouse,telephone,pickup truck,orange,banana,airplane,luggage,skis,soccer,trolley,oven,remote,combine with glove,paper towel,refrigerator,train,tomato,machinery vehicle,tent,shampoo/shower gel,head phone,lantern,donut,cleaning products,sailboat,tangerine,pizza,kite,computer box,elephant,toiletries,gas stove,broccoli,toilet,stroller,shovel,baseball bat,microwave,skateboard,surfboard,surveillance camera,gun,Life saver,cat,lemon,liquid soap,zebra,duck,sports car,giraffe,pumpkin,Accordion/keyboard/piano,radiator,converter,tissue,carrot,washing machine,vent,cookies,cutting/chopping board,tennis racket,candy,skating and skiing shoes,scissors,folder,baseball,strawberry,bow tie,pigeon,pepper,coffee machine,bathtub,snowboard,suitcase,grapes,ladder,pear,american football,basketball,potato,paint brush,printer,billiards,fire hydrant,goose,projector,sausage,fire extinguisher,extension cord,facial mask,tennis ball,chopsticks,Electronic stove and gas stove,pie,frisbee,kettle,hamburger,golf club,cucumber,clutch,blender,tong,slide,hot dog,toothbrush,facial cleanser,mango,deer,egg,violin,marker,ship,chicken,onion,ice cream,tape,wheelchair,plum,bar soap,scale,watermelon,cabbage,router/modem,golf ball,pine apple,crane,fire truck,peach,cello,notepaper,tricycle,toaster,helicopter,green beans,brush,carriage,cigar,earphone,penguin,hurdle,swing,radio,CD,parking meter,swan,garlic,french fries,horn,avocado,saxophone,trumpet,sandwich,cue,kiwi fruit,bear,fishing rod,cherry,tablet,green vegetables,nuts,corn,key,screwdriver,globe,broom,pliers,hammer,volleyball,eggplant,trophy,board eraser,dates,rice,tape measure/ruler,dumbbell,hamimelon,stapler,camel,lettuce,goldfish,meat balls,medal,toothpaste,antelope,shrimp,rickshaw,trombone,pomegranate,coconut,jellyfish,mushroom,calculator,treadmill,butterfly,egg tart,cheese,pomelo,pig,race car,rice cooker,tuba,crosswalk sign,papaya,hair dryer,green onion,chips,dolphin,sushi,urinal,donkey,electric drill,spring rolls,tortoise/turtle,parrot,flute,measuring cup,shark,steak,poker card,binoculars,llama,radish,noodles,mop,yak,crab,microscope,barbell,Bread/bun,baozi,lion,red cabbage,polar bear,lighter,mangosteen,seal,comb,eraser,pitaya,scallop,pencil case,saw,table tennis paddle,okra,starfish,monkey,eagle,durian,rabbit,game board,french horn,ambulance,asparagus,hoverboard,pasta,target,hotair balloon,chainsaw,lobster,iron,flashlight",

        help="",    )
    parser.add_argument("--pred_all_class", action='store_true')
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp

# Specify your folders here
image_folder = "XXX/demo_output"
image_files = glob.glob(os.path.join(image_folder, "*.png"))
output_image_folder = "./demo_output_finetune/image"
output_txt_folder = "./demo_output_finetune/data_2d_bbox"
logger = setup_logger()
def is_distributed():
    if not dist.is_available() or not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_distributed():
        return 0
    return dist.get_rank()


def is_primary():
    return get_rank() == 0

def barrier():
    if not is_distributed():
        return
    torch.distributed.barrier()
# Create output directories if they don't exist
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_txt_folder, exist_ok=True)    
mp.set_start_method("spawn", force=True)
def worker(rank, world_size, image_files, output_image_folder, output_txt_folder):
    init_process(rank, world_size)
    def load_model(rank):
        # 加载模型
        rank = dist.get_rank()
        print(f"Running on rank {rank}")
        args = get_parser().parse_args()
        setup_logger(name="fvcore")
        cfg = setup_cfg(args)
        logger.info("Arguments: " + str(args))
        demo = VisualizationDemo(cfg, args)
        return demo
    
    for i in range(world_size):
        if rank == i:
            demo = load_model(rank)

            barrier()
        else:
            barrier()

    dataset = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            dataset.append(os.path.join(image_folder, filename))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_dataset = dataset[rank::world_size]
    
    for image_path in local_dataset:
        img = read_image(image_path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)

        image_filename = os.path.basename(image_path)
        output_image_path = os.path.join(output_image_folder, image_filename)
        visualized_output.save(output_image_path)

        instances = predictions["instances"]
        pred_classes = instances.pred_classes
        class_names = [demo.metadata.thing_classes[x] for x in instances.pred_classes.cpu().detach().numpy()]
        scores = instances.scores
        pred_boxes = instances.pred_boxes.tensor

        txt_filename = os.path.splitext(image_filename)[0] + ".txt"
        output_txt_path = os.path.join(output_txt_folder, txt_filename)
        with open(output_txt_path, 'w') as f:
            for idx in range(len(class_names)):
                class_name = class_names[idx].replace(" ", "_")
                bbox_coords = [round(num, 2) for num in pred_boxes[idx].tolist()]
                score = scores[idx].item()
                f.write(f"{class_name} 0 0 -10 {bbox_coords[0]} {bbox_coords[1]} {bbox_coords[2]} {bbox_coords[3]} {score}\n")

        logger.info(
            "{}: {} in {:.2f}s".format(
                image_path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )
    cleanup()

def init_process(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12111'
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment"""
    dist.destroy_process_group()


def main():
    world_size = 1 #torch.cuda.device_count() 
    mp.spawn(worker, args=(world_size, image_files, output_image_folder, output_txt_folder), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
