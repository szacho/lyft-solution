import numpy as np
import os, json, sys
import torch
import bz2, pickle, gzip
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm import tqdm

from utils import load_datasetV2, batch_get_agent_stateV2, batch_rotate
from l5kit.visualization import draw_trajectory
from l5kit.geometry import transform_points

def save_obj(obj, name, dir_cache):
    with gzip.GzipFile(f'{dir_cache}/{name}.gz', 'wb', compresslevel=1) as f:
        pickle.dump(obj, f)

def plot_multich_im(img):
    img = img.numpy().transpose(1,2,0)
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(img[:, :, 0], cmap='gray')
    ax2.imshow(img[:, :, 1], cmap='gray')
    ax3.imshow(img[:, :, 2:])
    plt.show()

MODEL_PATH = './output/a7'
NUM_BATCHES = 2960
OUTPUT_PATH = './data/rasterized'

# load model config
with open(f'{MODEL_PATH}/config.json') as cfg_file:
    cfg = json.load(cfg_file)
    bs = cfg['val_data_loader']['batch_size']

# load dataset
dataloader = load_datasetV2(cfg, 'val_data_loader', False)
dataiter = iter(dataloader)
print(f"Length of the dataset with mask: {len(dataloader)}x{bs}")

for k in tqdm(range(NUM_BATCHES)):
    data = next(dataiter)
    save_obj(data, f'sample_{k}', OUTPUT_PATH)


        

