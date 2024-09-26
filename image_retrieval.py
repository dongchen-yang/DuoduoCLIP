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
# from .. import custom_clip
# import argparse
# import open_clip



@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(_cfg):
    
    # ckpt_path = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP', filename=_cfg.ckpt_path)
    duoduoclip = DuoduoCLIP(_cfg)
    # ckpt_path = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP', filename=_cfg.ckpt_path)
    # duoduoclip = DuoduoCLIP.load_from_checkpoint(ckpt_path)
    
    duoduoclip.eval()
    duoduoclip.cuda()

    # Get embeddings for objaverse objects
    shape_embeddings_h5 =  h5py.File('dataset/data/objaverse_embeddings/Four_1to6F_bs1600_LT6/shape_emb_objaverse.h5', "r")
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
    with open('dataset/data/objaverse_embeddings/Four_1to6F_bs1600_LT6/shape_emb_objaverse_model_to_idx.json', 'r') as file:
        shape_model_to_idx = json.load(file)
    # print("shape_model_to_idx:", shape_model_to_idx.keys())
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
    
    image = np.transpose(image, (2, 0, 1))
        #     data_dict['mv_images'] = data_dict['mv_images'].to(torch.float16) / 255
        # for data_key in ("mv_images", "category_clip_features", "class_idx"):
        #     data_dict[data_key] = data_dict[data_key].to(model.device)
    c, h, w = image.shape
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
    print('=' * line_width)
    print()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Visualize Selection') 
    # parser.add_argument('--image_path', type=str, default='/local-scratch/localhome/dya78/code/dongchen-psdr-room/obj_match/output/3d_front/00004f89-9aa5-43c2-ae3c-129586be8aaa_0/cropped_images/7a1d1b3b-4c78-4edc-8a11-c9a3f962e97f.png', help='Path to the results json file')
    # args = parser.parse_args()
    # image_path = args.image_path
    main()
