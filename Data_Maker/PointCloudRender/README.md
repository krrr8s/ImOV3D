# Point Cloud Renderer

**⚠️ After NeurIPS 2024, we updated the weights of the point cloud render module, which means it can now generate better and more realistic pseudo images. We recommend that you generate new pseudo images yourself rather than using the data we provide, as our provided data is solely for validating the results presented in the paper.**


This step is mainly to generate colorful pseudo images. You need to install the environment according to [ControlNet](https://github.com/lllyasviel/ControlNet)'s installation process, and then run the `ControlNet` code.
We have provided a Demo in this folder, you can first understand how this process works.

### Data creater

It is used for generating data to Finetune Controlnet, you can run the demo in the folder `./Finetune_data_creater` to understand what this step is doing.

Note: Training a ControlNet may take 1-2 weeks. We have provided pre-finetuned weights on the Main page.
