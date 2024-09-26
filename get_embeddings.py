import numpy as np
from src.model.wrapper import get_model
import h5py
import json
import imageio.v2 as imageio
import glob
import os
from PIL import Image
from tqdm import tqdm

duoduoclip = get_model('Four_1to6F_bs1600_LT6.ckpt', device='cuda')

##
feature_dict = {}
image_root= "/localhome/dya78/data/mvclip_renders"
image_paths = glob.glob(os.path.join(image_root, "*.png"))

model_to_idx = {}

all_features = []

counter=0
for image_path in tqdm(image_paths):
    mv_images = Image.open(image_path).convert('RGBA')
    background = Image.new('RGBA', mv_images.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(background, mv_images)
    mv_images = composite.convert('RGB')
    # mv_images = mv_images[:,:,0:3] * mv_images[:,:,3:4] + 255 * (1 - mv_images[:,:,3:4])
    image_name = os.path.basename(image_path).split('.')[0]
    mv_images = np.array(mv_images).reshape(3, 224, 4,  224, 3)
    mv_images = mv_images.transpose(0, 2, 1, 3, 4)
    mv_images = mv_images.reshape(12, 224, 224 ,3)
    # imageio.imwrite('debug.png', mv_images[0])
    # raise Exception
    image_features = duoduoclip.encode_image(mv_images)

    all_features.append(image_features.detach().cpu().numpy())
    model_to_idx[image_name] = counter
    counter += 1



all_features = np.array(all_features).reshape(-1, 512)

## save to h5
f = h5py.File('mvclip_features.h5', 'w')
f.create_dataset('shape_feat', data=all_features)
f.close()

## save model to idx
with open('model_to_idx.json', 'w') as file:
    json.dump(model_to_idx, file)