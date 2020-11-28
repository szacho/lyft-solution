# https://github.com/lyft/l5kit/blob/master/examples/agent_motion_prediction/agent_motion_prediction.ipynb
# https://github.com/lyft/l5kit/blob/master/l5kit/l5kit/evaluation/chop_dataset.py
import os, json
import numpy as np
from pathlib import Path
from l5kit.evaluation import create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer

num_frames_to_chop = 100

MODEL_PATH = './output/b1'
# load model config
with open(f'{MODEL_PATH}/config.json') as cfg_file:
    cfg = json.load(cfg_file)

loader_cfg = cfg['val_data_loader']

os.environ["L5KIT_DATA_FOLDER"] = './'
dm = LocalDataManager(None)

rasterizer = build_rasterizer(cfg, dm)
base_path = create_chopped_dataset(dm.require(loader_cfg["key"]), cfg["raster_params"]["filter_agents_threshold"], 
                              num_frames_to_chop, cfg["model_params"]["future_num_frames"], MIN_FUTURE_STEPS)


zarr_path = str(Path(base_path) / Path(dm.require(loader_cfg["key"])).name)
mask_path = str(Path(base_path) / "mask.npz")
gt_path = str(Path(base_path) / "gt.csv")

zarr_data = ChunkedDataset(zarr_path).open()
mask = np.load(mask_path)["arr_0"]

dataset = AgentDataset(cfg, zarr_data, rasterizer, agents_mask=mask)
print(dataset)