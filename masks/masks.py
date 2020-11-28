from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv
from l5kit.data.filter import get_agents_slice_from_frames
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

dataset_name = 'train_full'
data_path = f'./data/lyft/scenes/{dataset_name}.zarr/'
interval = 19
start_frame = 11
end_frame = 200

# DO NOT CHANGE
min_history = 1
min_future = 10

cfg = {
    'format_version': 4,
    'model_params': {
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },
    
    'raster_params': {
        'raster_size': [1, 1],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.5, 0.5],
        'map_type': 'box_debug',
        "satellite_map_key": "./data/lyft/aerial_map/aerial_map.png",
        "semantic_map_key": "./data/lyft/semantic_map/semantic_map.pb",
        "dataset_meta_key": "./data/lyft/meta.json",
        'filter_agents_threshold': 0.5,
        'disable_traffic_light_faces' : False
    },
    
    'sample_data_loader': {
        'key': 'scenes/sample.zarr',
        'batch_size': 4,
        'shuffle': False,
        'num_workers': 8
    }
}


def load_chunked_dataset(ds_path):
    os.environ["L5KIT_DATA_FOLDER"] = "./"
    # local data manager
    dm = LocalDataManager()
    # set dataset path
    dataset_path = dm.require(ds_path)
    # load the dataset; this is a zarr format, chunked dataset
    chunked_dataset = ChunkedDataset(dataset_path)
    # open the dataset
    chunked_dataset.open()
    return chunked_dataset, dm

def get_frame_mask(chunked_dataset):
    n_agents = len(chunked_dataset.agents)
    frame_mask = np.zeros((n_agents,))

    # for idx in tqdm(range(3)):
    for idx in tqdm(range(len(chunked_dataset.scenes))):
        scene = chunked_dataset.scenes[idx]
        f1, _ = scene['frame_index_interval']
        for frame_no in np.arange(f1 + start_frame, f1 + end_frame, interval):
            ag_s = get_agents_slice_from_frames(chunked_dataset.frames[frame_no])
            frame_mask[ag_s] = 1
    np.savez(f'./inspection/masks/{dataset_name}_frame_{interval}_{start_frame}_{end_frame}.npz', frame_mask.astype(bool))
    return frame_mask


def get_agents_mask(chunked_dataset, dm):
    cfg["raster_params"]["map_type"] = "stub_debug"
    rasterizer = build_rasterizer(cfg, dm)
    train_dataset_full = AgentDataset(cfg, chunked_dataset, rasterizer)

    full_mask = train_dataset_full.load_agents_mask()
    past_mask = full_mask[:, 0] >= min_history
    future_mask = full_mask[:, 1] >= min_future
    del full_mask
    agents_mask = past_mask * future_mask
    np.savez(f'./inspection/masks/{dataset_name}_agents_fut_{min_future}_his_{min_history}.npz', agents_mask)
    return agents_mask

def join_masks(frame_mask, agents_mask, path):
    mask = frame_mask['arr_0']*agents_mask['arr_0']
    outfile = path+"mask_"+dataset_name +"_" + str(start_frame) + "_" + str(end_frame) + "_" + str(interval)
    return mask

if __name__ == "__main__":
    chunked, dm = load_chunked_dataset(data_path)
    # get_agents_mask(chunked, dm)
    get_frame_mask(chunked)

    fram_mask = np.load(f'./inspection/masks/{dataset_name}_frame_19_11_200.npz')
    agent_mask = np.load(f'./inspection/masks/{dataset_name}_agents_fut_10_his_1.npz')

    joined_mask = join_masks(fram_mask, agent_mask, data_path)
    print(f"{joined_mask.sum()//16}x16")
    np.savez('./inspection/masks/train_frame_19_11_200_X_agents_fut_10_his_1', joined_mask.astype(bool))



