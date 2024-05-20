# need to fine tune CLIP for each caption_type, model_type, shapes/colors

import os
import json
import random
import pickle
import argparse
from PIL import Image

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import clip
from transformers import CLIPProcessor, CLIPModel


BATCH_SIZE = 32
device = "cuda:0" # if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32",device=device,jit=False)

if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)

# ------------------------------------------------------------------

class image_caption_dataset(Dataset):
    def __init__(self, list_image_path, list_caption):
        # Initialize image paths and corresponding texts
        self.image_path = list_image_path
        # Tokenize text using CLIP's tokenizer
        self.title  = clip.tokenize(list_caption, truncate=True)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        image = preprocess(Image.open(self.image_path[idx]))
        caption = self.title[idx]
        return image, caption


#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(_model):
    for p in _model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


# ------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--3dgen_root', default='./', help='directory where the root of the 3dgen_fata_quality directory')
args = parser.parse_args()
3dgen_root = args.3dgen_root

caption_types = ['pseudo_captions', 'human_captions', 'gpt4_captions']
categories = ['chair', 'table']
attrs = ['colors', 'shapes']
for caption_type in caption_types:
    for category in categories:
        for attr in attrs:
            os.makedirs(f"model_checkpoint/{caption_type}/{category}/{attr}", exist_ok=True)

            cat_id = '03001627' if category == 'chair' else '04379243'
            img_root = os.path.join(3dgen_root, f'/TAPS3D/ShapeNetCoreRendering/img/{cat_id}')
            pkl_root = os.path.join(3dgen_root, f'/home/ptclient/text_guided_3D_gen/comp-t2i-dataset/pickles/{caption_type}/{category}/{attr}/')
            split_pkl_path = os.path.join(pkl_root, 'split.pkl')
            data_pkl_path = os.path.join(pkl_root, 'data.pkl')
            
            with open(split_pkl_path, 'rb') as f:
                split = pickle.load(f)
            
            with open(data_pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            list_image_path = []
            list_caption = []
            successful_img = 0
            for uid in split['train']:
                uid_root = os.path.join(img_root, uid, 'models')
                if not os.path.exists(uid_root): continue
                
                img_list = [f for f in os.listdir(uid_root) if f.endswith('png')]
                is_valid_img = False
                img_not_found = False
                tried = 0
                img_fn = 'placeholder.png'
                
                while not is_valid_img:
                    if tried > 30:
                        print('tried > 30... skipping...')
                        img_not_found = True
                        break
                    img_fn = img_list[random.randint(0, len(img_list)-1)]
                    try:
                        im = Image.open(os.path.join(uid_root, img_fn))
                        im.verify()
                        is_valid_img = True
                    except:
                        print(f'Invalid image at path: {os.path.join(uid_root, img_fn)} for {uid}. Selecting a different image and trying again...')
                        print(f'{successful_img} successful images so far...')
                        pass
                
                if img_not_found: continue
                successful_img += 1
                img_fp = os.path.join(uid_root, img_fn)
                
                for cap_id in list(data[uid].keys()):
                    list_image_path.append(img_fp)
                    list_caption.append(data[uid][cap_id]['text'])
                
            print(f'list_image_path len: {len(list_image_path)}')
            print(f'list_caption len: {len(list_caption)}')
            
            dataset = image_caption_dataset(list_image_path, list_caption)
            train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
            loss_img = nn.CrossEntropyLoss()
            loss_txt = nn.CrossEntropyLoss()
            
            num_epochs = 30
            lowest_loss = 1000000.0
            for epoch in range(num_epochs):
                pbar = tqdm(train_dataloader, total=len(train_dataloader))
                for batch in pbar:
                    optimizer.zero_grad()
                    
                    images,captions = batch
                    
                    images = images.to(device)
                    
                    # Forward pass
                    logits_per_image, logits_per_caption = model(images.cuda(), captions.cuda())
                    
                    # Compute loss
                    ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                    total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_caption,ground_truth)) / 2.0
                    
                    # Backward pass
                    total_loss.backward()
                    if device == "cpu":
                        optimizer.step()
                    else:
                        convert_models_to_fp32(model)
                        optimizer.step()
                        clip.model.convert_weights(model)
                        
                    pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.8f}")
                    
                if total_loss < lowest_loss or epoch == (num_epochs-1):
                    if epoch != (num_epochs-1):
                        # remove previous saved checkpoint
                        f_list = os.listdir(f'model_checkpoint/{caption_type}/{category}/{attr}')
                        for f in f_list:
                            os.remove(os.path.join(f'model_checkpoint/{caption_type}/{category}/{attr}', f))
                            
                    # save checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': total_loss,
                    }, f"model_checkpoint/{caption_type}/{category}/{attr}/model_{epoch}.pt")
                    
                    print(f'New checkpoint saved at epoch {epoch} with loss {total_loss}, last lowest loss was {lowest_loss}')
                    
                    # update lowest loss
                    lowest_loss = total_loss
