import json
import os
import argparse
import h5py

parser = argparse.ArgumentParser(description='Visualize Selection')
parser.add_argument('--model_id', type=str, default='e21e4188d2334babb7540fc75cfef22f', help='Path to the results json file')
args = parser.parse_args()


json_path = os.path.join("dataset","data", "lvis_model_to_idx.json")

with open(json_path, 'r') as f:
    model_to_idx = json.load(f)
print(model_to_idx['52da52f882e242aba646980e9757ef9f'])
# print(model_to_idx[args.model_id])