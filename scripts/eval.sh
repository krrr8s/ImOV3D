#using pretrain stage weight to eval sunrgbd

#If you want to evaluate the ScanNet on pretrain stage, 
#you should change the content prepare for evaluate SUNRGBD 
#to evaluate ScanNet on lvis folder 

CUDA_VISIBLE_DEVICES=2 python ./eval.py \
--dataset lvis \
--checkpoint_path /share1/timingyang/IMOV3D-OPENSOURCE/CHECKPOINT/pretrain/checkpoint_99.tar \
--dump_dir eval_lvis \
--cluster_sampling seed_fps \
--use_3d_nms \
--use_cls_nms \
--per_class_proposal \
--use_imvotenet &\


# ##using adapation stage weight to eval sunrgbd
CUDA_VISIBLE_DEVICES=2 python ./eval.py \
--dataset sunrgbd \
--checkpoint_path /share1/timingyang/IMOV3D-OPENSOURCE/CHECKPOINT/sunrgbd/checkpoint_48.tar \
--dump_dir eval_sunrgbd \
--cluster_sampling seed_fps \
--use_3d_nms \
--use_cls_nms \
--per_class_proposal \
--use_imvotenet &\


# ##using adapation stage weight to eval scannet
CUDA_VISIBLE_DEVICES=2 python ./eval.py \
--dataset scannet \
--checkpoint_path /share1/timingyang/IMOV3D-OPENSOURCE/CHECKPOINT/scannet/checkpoint_96.tar \
--dump_dir eval_scannet \
--cluster_sampling seed_fps \
--use_3d_nms \
--use_cls_nms \
--per_class_proposal \
--use_imvotenet &\
