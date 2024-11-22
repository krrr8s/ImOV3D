# 2D branch Bbox Preparation

Please place the files from `./Detic` into the Detic project. Here we have prepared a demo in `./Detic` to help you understand what this step is doing.

## Finetune Detic for pseudo image(optional)

If you find that the original bounding boxes from Detic are not ideal, this is because the pseudo images are very noisy. Please use the code in `./Finetune_Detic_for_pseudo_image` folder to generate data, and follow the training method of [Detic](https://github.com/facebookresearch/Detic) to conduct the training.