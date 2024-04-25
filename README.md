# 3dgen-data-quality

# Abstract

# Requirement
- Linux is recommended for performance and compatibility reasons.
- 1-8 high-end NVIDIA GPUs. We have done all testing and development using 2 A100 GPUs.
- 64-bit Python 3.8 and PyTorch 1.9.0. See https://pytorch.org for PyTorch install instructions.
- CUDA toolkit 11.1 or later.
- CLIP from [official repo](https://github.com/openai/CLIP)
- We also recommend to install Nvdiffrast following instructions from official repo, and install Kaolin.
- We provide a script to install packages.

# Steps to Reproduce 
1. Clone all neccessary repositories
    - Clone this repository with ```gh repo clone FifthEpoch/3dgen-data-quality```
    - Then, change directory into the 3dgen-data-quality directory with ```cd 3dgen-data-quality```
    - Finally, clone TAPS3D repository with ```gh repo clone plusmultiply/TAPS3D```
2. Download shapenet dataset and produce 2D renderings
    - Download the ShapeNetCore.v1 dataset either from shapenet's [official site](https://shapenet.org/) or from [huggingface](https://huggingface.co/datasets/ShapeNet/ShapeNetCore)
    - Download Blender following the [official link](https://www.blender.org/download/releases/2-90/), we used Blender v2.90.0, we haven't tested on other versions.
    - Install required libraries:
      ```
      apt-get install -y libxi6 libgconf-2-4 libfontconfig1 libxrender1
      cd BLENDER_PATH/2.90/python/bin
      ./python3.7m -m ensurepip
      ./python3.7m -m pip install numpy 
      ```
    - Render the shapenet 3D models in the chair and table categories with
      ```
      python render_all.py --save_folder PATH_TO_SAVE_IMAGE --dataset_folder PATH_TO_3D_OBJ --blender_root PATH_TO_BLENDER
      ```
         - (Optional) The code will save the output from blender to tmp.out, this is not necessary for training, and can be removed by rm -rf tmp.out
         - This code is adopted from this [GitHub repo](https://github.com/panmari/stanford-shapenet-renderer), we thank the author for sharing the codes!
