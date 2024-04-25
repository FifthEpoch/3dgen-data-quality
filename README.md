# 3dgen-data-quality

#### Abstract

#### Requirement
- Linux is recommended for performance and compatibility reasons.
- 1-8 high-end NVIDIA GPUs. We have done all testing and development using 2 A100 GPUs.
- 64-bit Python 3.8 and PyTorch 1.9.0. See https://pytorch.org for PyTorch install instructions.
- CUDA toolkit 11.1 or later.
- CLIP from [official repo](https://github.com/openai/CLIP)
- We also recommend to install Nvdiffrast following instructions from official repo, and install Kaolin.
- We provide a script to install packages.

#### Steps to Reproduce 
1. Clone this repository with
   ```
   gh repo clone FifthEpoch/3dgen-data-quality
   ```
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
3. Prepare Compositional Split Generation for Clip-R Precision Evaluation
   - Produce 5 splits (Train, Test seen, Test unseen, Test unseen in diverse styles, Test swapped) for two attributes (colors and shapes) for each caption dataset by running the below file:
   ```
   python generate_comp_split.py
   ```
   - Be sure to organize your 3D and 2D data according to the train/test split so only data in the train split is accessed during training.
4. Training the TAPS3D model with three different caption types (pseudo captions, human-generated captions, LVM-generated captions) in two model categories (chair and table)
   - Download the pretrained model checkpoint for the chair and table categories from [this link](https://drive.google.com/drive/folders/1oJ-FmyVYjIwBZKDAQ4N1EEcE9dJjumdW) provided by the GET3D authors
   - Train the TAPS3D models. one at a time.
       - For example, to train a model with shapenet chair and human-generated caption paired dataset, run the following:
         ```
         python train.py --outdir ./data/human_captions/chair/texw_2-0_geow_0-2_lr_0-002_metrics --caption_type human_captions --num_gpus 2 --batch_size 4 --batch_gpu 2 --network <project_root>/TAPS3D/GET3D_pretrained/shapenet_chair.pt --seed 0 --snap 1000 --lr 0.002 --lambda_global 1 --lambda_direction 0 --lambda_imgcos 1 --image_root <project_root>/TAPS3D/ShapeNetCoreRendering/img --gen_class chair --mask_weight 0.05 --workers 8 --tex_weight 2 --geo_weight 0.2 --metrics fid50k_full,kid50k_full,pr50k3_full
         ```
       - With 2 A100s. our training time for each model is approxiately 10 hours.
5. Generate 3D models with the trained TAPS3D models, preparing 3D models for Clip-R Precision evaluation by running the below file:
   ```
   python end2end.py
   ```
   - This script first generates 3D models using text prompts from each of the five splits, then it renders the generated 3D models, and uses CLIP to make prediction based on the renderings.
6. Compute clip-r-precision by running the below file:
```
python compute_r_precision.py
```

