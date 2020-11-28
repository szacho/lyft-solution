import numpy as np
import os, json, math
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer

from tqdm import tqdm

pi = torch.Tensor([math.pi])
class TrajectoryDataset(Dataset):
    def __init__(self, agents):
        self.agents = agents

    def __len__(self):
        return len(self.agents)

    def __getitem__(self, idx):
        agent = self.agents[idx]

        gt_trajectory = torch.Tensor(agent['target_positions'])
        yaw = agent['yaw']
        
        rot_angle = torch.Tensor([-yaw])
        cos, sin = torch.cos(rot_angle), torch.sin(rot_angle)
        rotmat = torch.Tensor([[cos, -sin], [sin, cos]])
        gt_rotated = torch.matmul(gt_trajectory, rotmat.transpose(0, 1))
        return gt_rotated

MODEL_PATH = './output/e1'
# load model config
with open(f'{MODEL_PATH}/config.json') as cfg_file:
    cfg = json.load(cfg_file)
    assert cfg["train_params"]["max_num_steps"] % cfg['train_params']['checkpoint_every_n_steps'] == 0

os.environ["L5KIT_DATA_FOLDER"] = './'
dm = LocalDataManager(None)

loader_cfg = cfg['train_data_loader']
rasterizer = build_rasterizer(cfg, dm)

ds_zarr = ChunkedDataset(dm.require(loader_cfg["key"])).open()
dataset = AgentDataset(cfg, ds_zarr, rasterizer, min_frame_future=50, defer_rasterizing=True)
dataset = TrajectoryDataset(dataset)

ds_length = len(dataset)

trajectories = []

for k in tqdm(range(100000)):
    idx = np.random.randint(0, ds_length)
    tr = dataset[idx]

    trajectories.append(tr.unsqueeze(0))

trajectories = torch.cat(trajectories)
torch.save(trajectories, './data/trajectories/new_trajectories_sample.pt')
