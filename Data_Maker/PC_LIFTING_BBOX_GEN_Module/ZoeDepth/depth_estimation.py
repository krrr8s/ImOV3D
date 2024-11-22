import os
from PIL import Image,ImageOps
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from zoedepth.utils.misc import colorize, save_raw_16bit
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model
from pprint import pprint

def init_process(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment"""
    dist.destroy_process_group()

#img2depth

def predict_depth(model, image):
    with torch.no_grad():
        depth = model.module.infer_pil(image)
        depth = torch.from_numpy(depth).unsqueeze(0).float()
    return depth.squeeze().detach().cpu().numpy() 

    
def inference(rank, world_size, input_dir, output_dir):
    init_process(rank, world_size)

    rank = dist.get_rank()
    DEVICE = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    print(f"Running on rank {rank}, device {DEVICE}")
    #model = torch.hub.load('.', "ZoeD_NK",source='local',pretrained=True).to(DEVICE).eval()

    conf = get_config("zoedepth_nk", "infer")

    print("Config:")
    pprint(conf)

    model = build_model(conf).to(DEVICE)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model.eval()

    # Divide data into processes
    dataset = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            dataset.append(os.path.join(input_dir, filename))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_dataset = dataset[rank::world_size]

    # Make predictions for each image and save the results
    for filename in local_dataset:
       
      
        image_path = os.path.join(input_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")
        except:
          
            gray_image = Image.open(image_path).convert('L')


            image = ImageOps.expand(gray_image.convert('RGB'), border=(0, 0, 0, 0))
        
        # Make depth predictions
        depth = predict_depth(model, image)
        
        # Save raw depth image
        
        output_path = os.path.join(output_dir, os.path.basename(filename).split('.')[0]+".png")

        save_raw_16bit(depth, output_path)
        
        # Save colorize depth image 
        colored = colorize(depth)
        
        # save colored output
        
        fpath_colored = os.path.join(output_color_dir, os.path.basename(filename).split('.')[0]+"_color"+".png")
        print("fpath_colored",fpath_colored) 
        Image.fromarray(colored).save(fpath_colored)
    cleanup()

    


input_dir = './image' 
output_dir = './depth' 
output_color_dir = './depth_for_visual/'
  
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
if not os.path.exists(output_color_dir):
    os.makedirs(output_color_dir) 


def main():
    world_size = 4 #torch.cuda.device_count()
    mp.spawn(inference, args=(world_size, input_dir, output_dir), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
