CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_dist_wandb.py  \
--dataset lvis  \
--log_dir log_lvis  \
--batch_size 24 \
--if_wandb  \
--use_imvotenet \
--dist_url tcp://localhost:12955 \