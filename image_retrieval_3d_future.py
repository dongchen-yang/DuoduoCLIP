import json
import h5py
import faiss
import hydra
import torch
import numpy as np
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from src.model.duoduoclip import DuoduoCLIP
import imageio
import os


@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(_cfg):
    
    duoduoclip = DuoduoCLIP(_cfg)
    
    duoduoclip.eval()
    duoduoclip.cuda()

    # Get embeddings for objaverse objects
    shape_embeddings_h5 =  h5py.File('/localhome/dya78/code/DuoduoCLIP/3d-future-features/mvclip_features.h5', "r")
    shape_embeddings = shape_embeddings_h5['shape_feat'][:]
    shape_embeddings_h5.close()
    shape_embeddings = shape_embeddings.astype(np.float32)

    print("shape embedding:", shape_embeddings.shape)

    # Normalize the shape embeddings
    shape_embeddings = torch.from_numpy(shape_embeddings).cuda()
    shape_embeddings = F.normalize(shape_embeddings, dim=1)
    shape_embeddings = shape_embeddings.cpu().numpy()

    # Get index of faiss
    index = faiss.IndexFlatIP(512)
    index.add(shape_embeddings)

    # Load mapping for shape embeddings in search library
    with open('/localhome/dya78/code/DuoduoCLIP/3d-future-features/model_to_idx.json', 'r') as file:
        shape_model_to_idx = json.load(file)
    # Get reverse mappings
    shape_idx_to_model = {}
    for key, value in shape_model_to_idx.items():
        shape_idx_to_model[value] = key

    # while True:
    line_width = 60
    print('=' * line_width)
    print("importing image from: ", _cfg.image_path)
    image = np.array(imageio.imread(_cfg.image_path))[:,:, :3]
    print("image shape: ", image.shape)
    imageio.imwrite('debug.png', image)
    

    h, w, c = image.shape
    ## ensure the image is at least 208x208
    print("image shape before resizing: ", image.shape)
    if h <208:
        new_h = 208
        ratio = new_h / h
        new_w = int(w * ratio)
        ## reshape using PIL
        image = Image.fromarray(image)
        image = image.resize((new_w, new_h))
        image = np.array(image)
    if w < 208:
        new_w = 208
        ratio = new_w / w
        new_h = int(h * ratio)
        ## reshape using PIL
        image = Image.fromarray(image)
        image = image.resize((new_w, new_h))
        image = np.array(image)
    print("image shape after resizing: ", image.shape)

    image = np.transpose(image, (2, 0, 1))
        #     data_dict['mv_images'] = data_dict['mv_images'].to(torch.float16) / 255
        # for data_key in ("mv_images", "category_clip_features", "class_idx"):
        #     data_dict[data_key] = data_dict[data_key].to(model.device)
    
    c , h, w= image.shape

    bs = 1
    f = 1
    image = image.reshape(bs, f, c, h, w)
    image = image.reshape(bs * f, c, h, w)
    image = torch.from_numpy(image).cuda()
    image = image.to(torch.float16)/ 255.0
    mv_images = duoduoclip.val_norm_transform(image)



        # output_dict = model(data_dict)
    with torch.no_grad(), torch.cuda.amp.autocast():
        num_frames_list = [1] * duoduoclip.layers_threshold + [f] * (duoduoclip.duoduoclip.visual.transformer.layers - duoduoclip.layers_threshold) + [f]
        image_features = duoduoclip.duoduoclip.encode_image(mv_images, num_frames=num_frames_list)
        image_features = F.normalize(image_features, dim=1)

    query_emb = F.normalize(image_features, dim=1)
    query_emb = query_emb.cpu().numpy().astype(np.float32)
    _, I = index.search(query_emb, 5)

    print()
    print('Top 5 retrieved models:')
    for i in range(5):
        print(shape_idx_to_model[I[0][i]])
        best_match = shape_idx_to_model[I[0][i]]
        best_match_img = imageio.imread(os.path.join("/localhome/dya78/data/3D-FUTURE-model",best_match, f"image.jpg"))
        imageio.imwrite(f'best_match_{i}.png', best_match_img)
    print('=' * line_width)
    print()

    
if __name__ == '__main__':

    main()
