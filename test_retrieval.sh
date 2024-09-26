#!/bin/bash
# Test the retrieval model
# PATH="/localhome/dya78/data/3D-FUTURE-model/0c51740b-7fd0-4b11-91fe-fb35553d6b4e/image.jpg"
# PATH="/local-scratch/localhome/dya78/code/dongchen-psdr-room/obj_match/output/3d_front/0032b185-4914-49e5-b973-f82271674308_0/cropped_images/77185b15-66a6-45aa-8ef7-514b4b374ecb.png"
# PATH="/local-scratch/localhome/dya78/code/dongchen-psdr-room/obj_match/output/3d_front/0032b185-4914-49e5-b973-f82271674308_0/cropped_images/f350c1ba-8040-4750-9e6e-f45cfbf916a4.png"
# /localhome/dya78/software/miniconda3/envs/ddclip/bin/python image_retrieval.py ckpt_path=Four_1to6F_bs1600_LT6.ckpt image_path=$PATH
# PATH="/local-scratch/localhome/dya78/code/dongchen-psdr-room/obj_match/output/3d_front/0018b6c8-c3b6-4fb8-a640-4b9b0b763254_4/cropped_images/d7b7be7a-5ed6-419e-8d47-cfa909735db7.png"
# PATH="/local-scratch/localhome/dya78/code/dongchen-psdr-room/obj_match/output/3d_front/0032b185-4914-49e5-b973-f82271674308_10/cropped_images/82b10620-66a0-474a-9667-c87bd8ea0c7a.png"
# PATH="/local-scratch/localhome/dya78/code/dongchen-psdr-room/obj_match/output/3d_front/00004f89-9aa5-43c2-ae3c-129586be8aaa_0/cropped_images/3ab2e104-be5b-4617-a849-306ccb29c0ac.png"
# PATH="/local-scratch/localhome/dya78/code/dongchen-psdr-room/obj_match/output/3d_front/00004f89-9aa5-43c2-ae3c-129586be8aaa_0/matched_images/3ab2e104-be5b-4617-a849-306ccb29c0ac.png"
# PATH="/local-scratch/localhome/dya78/code/dongchen-psdr-room/obj_match/output/3d_front/00004f89-9aa5-43c2-ae3c-129586be8aaa_0/matched_images/7a1d1b3b-4c78-4edc-8a11-c9a3f962e97f.png"
# PATH="/local-scratch/localhome/dya78/code/dongchen-psdr-room/obj_match/output/3d_front/00004f89-9aa5-43c2-ae3c-129586be8aaa_0/matched_images/162e1aed-6e96-4503-ba3c-adfe8b07b429.png"
# PATH="/local-scratch/localhome/dya78/code/dongchen-psdr-room/obj_match/output/3d_front/0018b6c8-c3b6-4fb8-a640-4b9b0b763254_4/matched_images/1c6f1892-23cf-41ee-b707-7d79fca07cf3.png"
# PATH="/local-scratch/localhome/dya78/code/dongchen-psdr-room/obj_match/output/3d_front/0032b185-4914-49e5-b973-f82271674308_10/matched_images/7549a14b-833b-48ea-89a5-8893f67e1daa.png"
PATH="/local-scratch/localhome/dya78/code/dongchen-psdr-room/obj_match/output/3d_front/0018b6c8-c3b6-4fb8-a640-4b9b0b763254_4/cropped_images/1919418e-3329-4c71-85e4-b00c54aa2c7e.png"
/localhome/dya78/software/miniconda3/envs/ddclip/bin/python image_retrieval_3d_future.py ckpt_path=Four_1to6F_bs1600_LT6.ckpt image_path=$PATH