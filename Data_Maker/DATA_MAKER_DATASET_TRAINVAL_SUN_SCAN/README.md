# 3D Data preparation for SUNRGBD and ScanNet

This step is primarily aimed at preparing data similar to [OV-3DET Data Maker](https://github.com/lyhdet/OV-3DET/tree/main/Data_Maker/Pseudo_Label_Maker), specifically by placing it in the root directory of [Detic](https://github.com/facebookresearch/Detic), and then running the `.sh` file.

After completing the first step, you still need statistics on object sizes and to generate Votes:
- `sat_lwh.py` is for calculating the meansize of objects
- `vote_create.py` is for generating votes

Note: For SUNRGBD's eval dataset, you need to use [VoteNet](https://github.com/facebookresearch/votenet)'s code to generate it. You can also directly download the Dataset from the link we share in the Homepage/Main page Link, or from the [OV-3DET](https://github.com/lyhdet/OV-3DET/)
