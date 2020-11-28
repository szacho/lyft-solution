import os, math
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple

from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset
from l5kit.rasterization import build_rasterizer

from dataset import FadingHistorySet

def mean_pointwise_l2_distance(lattice: torch.Tensor, ground_truth: torch.Tensor, avails: torch.Tensor) -> torch.Tensor:
    """
    Computes the index of the closest trajectory in the lattice as measured by l1 distance.
    :param lattice: Lattice of pre-generated trajectories. Shape [num_modes, n_timesteps, state_dim]
    :param ground_truth: Ground truth trajectory of agent. Shape [1, n_timesteps, state_dim].
    :return: Index of closest mode in the lattice.
    """
    arg_avails = torch.nonzero(avails, as_tuple=False).flatten()
    masked_lattice = lattice[:, arg_avails, :]
    masked_ground_truth = ground_truth[:, arg_avails, :]
    stacked_ground_truth = masked_ground_truth.repeat(masked_lattice.shape[0], 1, 1)
    return torch.pow(masked_lattice - stacked_ground_truth, 2).sum(dim=2).sqrt().mean(dim=1).argmin()

def batch_rotate(trajectories_batch: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Rotates every trajectory in batch by corresponding angle.
    :param trajectories_batch: Batch of trajectories. Shape [batch_size, n_timesteps, state_dim].
    :param angle: Tensor of rotation angles. Shape [batch_size, 1]. 
    :return: Rotated trajectories.
    """
    angle = angle.double()
    trajectories_batch = trajectories_batch.double()
    cos_vec = torch.cos(angle)
    sin_vec = torch.sin(angle)

    batch_rotmat = torch.stack([cos_vec, -sin_vec, sin_vec, cos_vec], dim=1).view(-1, 2, 2)
    rotated = torch.bmm(trajectories_batch.double(), batch_rotmat.transpose(1, 2).double())
    return rotated.float()

def angle_diff(x: float, y: float, period: float) -> float:
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity in radians for assessing angle difference.
    :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff

# agent states stats calculated on a subset of train.zarr
# speed, acceleration, yaw change rate
# MIN_AGENT_STATES = torch.Tensor([0, -12, -0.95])
# MAX_AGENT_STATES = torch.Tensor([29, 17, 0.95])
# MODE_AGENT_STATES = torch.Tensor([0, 0, 0])

AVG_SPEED_PER_LABEL = torch.zeros(17)
AVG_SPEED_PER_LABEL[3] = 3.4166
AVG_SPEED_PER_LABEL[12] = 2.3066
AVG_SPEED_PER_LABEL[14] = 0.4011

# MIN_AGENT_STATES = torch.Tensor([0, -12, -1, 0])
# MAX_AGENT_STATES = torch.Tensor([30, 18, 1, 30])
# MODE_AGENT_STATES = torch.Tensor([0, 0, 0])
MIN_AGENT_STATES = torch.Tensor([0, -12, -1])
MAX_AGENT_STATES = torch.Tensor([30, 18, 1])
MODE_AGENT_STATES = torch.Tensor([0, 0, 0])
def batch_get_agent_stateV2(data) -> torch.Tensor:
    avails = data['history_availabilities'].sum(1)
    avails = avails.float()
    bs = avails.shape[0]
    only_odd_avails = avails+(avails%2-1)
    hist_indices = torch.stack([torch.zeros(bs), avails//2 ,avails-1], dim=1)
    
    time_diffs = (avails//2*0.1).unsqueeze(-1)
    hist_indices = hist_indices.unsqueeze(-1).expand(-1, -1, 2)
    history = torch.gather(data['history_positions'], dim=1, index=hist_indices.long())

    # calculate basic properties
    prev_velocity = torch.Tensor(history[:, 1, :]-history[:, 2, :]) 
    velocity = torch.Tensor(history[:, 0, :]-history[:, 1, :]) 

    prev_velocity = torch.norm(prev_velocity, dim=1, keepdim=True) / time_diffs
    velocity = torch.norm(velocity, dim=1, keepdim=True) / time_diffs

    acceleration = (velocity-prev_velocity) 
    
    yaw_change_rate = []
    for key, yaws in enumerate(data['history_yaws']):
        yaw_change_rate.append(angle_diff(yaws[0], yaws[1], period=2*np.pi) / time_diffs[key])
    yaw_change_rate = torch.cat(yaw_change_rate).float().view(-1, 1)

    labels = torch.argmax(data['label_probabilities'], dim=1)
    avg_speeds = AVG_SPEED_PER_LABEL[labels.long()].unsqueeze(1)
    
    state = torch.cat([ velocity, acceleration, yaw_change_rate ], dim=1)

    if any(avails<3):
        bad_indices = torch.nonzero(avails<3,  as_tuple=True)[0]
        state[bad_indices] = MODE_AGENT_STATES

    # state = torch.cat([ state, avg_speeds ], dim=1)
    state_scaled = (state-MIN_AGENT_STATES)/(MAX_AGENT_STATES-MIN_AGENT_STATES)
    return state_scaled
    
def batch_get_labels(trajectories, avails, lattice, device) -> torch.Tensor:
    avails = avails.long()
    
    labels = torch.Tensor().long().to(device)
    for k in range(trajectories.shape[0]):
        label = mean_pointwise_l2_distance(lattice, trajectories[k].unsqueeze(0), avails[k]).unsqueeze(0)
        labels = torch.cat([labels, label])
        
    return labels

def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 

def get_step_schedule_with_cosine_anneal(optimizer, flat_steps_ratio, gamma, num_training_steps, last_epoch=-1):
    num_flat_steps = math.ceil(flat_steps_ratio[-1]*num_training_steps)
    milestones = np.array([math.ceil(num_training_steps*r) for r in flat_steps_ratio])

    def lr_lambda(current_step: int):
        power = (current_step >= milestones).sum()
        if current_step < num_flat_steps:
            return 1.0*gamma**power
        progress = float(current_step - num_flat_steps) / float(max(1, num_training_steps - num_flat_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(0.5) * 2.0 * progress)))*gamma**(power-1)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_cos_flat_cos_schedule(optimizer, flat_steps_ratio, gamma, num_training_steps, last_epoch=-1):
    num_flat_steps = math.ceil(flat_steps_ratio[-1]*num_training_steps)
    milestones = np.array([math.ceil(num_training_steps*r) for r in flat_steps_ratio])
    def lr_lambda(current_step: int):
        power = (current_step >= milestones).sum()
        if current_step < milestones[0]:
            lr_min = gamma**(power+1)
            lr = lr_min + (1-lr_min)/2 * (1+math.cos(math.pi*current_step/max(1, milestones[0])))
            return max(0, lr)
        elif current_step < num_flat_steps:
            return 1.0*gamma**power
        else:
            progress = float(current_step - num_flat_steps) / float(max(1, num_training_steps - num_flat_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(0.5) * 2.0 * progress)))*gamma**(power-1)

    return LambdaLR(optimizer, lr_lambda, last_epoch)
    
def load_datasetV2(cfg, dataloader_key, lattice, return_loader=True, rasterize=True):
    # set env variable for data
    lattice_path = cfg['dataset']['lattice_path'] if lattice else None
    augment_history = cfg['dataset']['drop_history_aug']
    augment_cutout = cfg['dataset']['cutout']
    if dataloader_key == 'val_data_loader' or dataloader_key == 'test_data_loader':
        augment_history, augment_cutout = False, False
    multichannel = cfg['dataset']['multichannel']

    os.environ["L5KIT_DATA_FOLDER"] = './'
    dm = LocalDataManager(None)

    loader_cfg = cfg[dataloader_key]
    rasterizer = build_rasterizer(cfg, dm)

    ds_zarr = ChunkedDataset(dm.require(loader_cfg["key"])).open()

    if dataloader_key == 'test_data_loader':
        ds_mask = np.load(loader_cfg['mask'])["arr_0"]
        dataset = AgentDataset(cfg, ds_zarr, rasterizer, agents_mask=ds_mask)
        print(dataset)
        dataset = FadingHistorySet(dataset, None, rasterize, multichannel=multichannel)
    elif not "mask" in loader_cfg:
        dataset = AgentDataset(cfg, ds_zarr, rasterizer, min_frame_future=10)
        print(dataset)
        dataset = FadingHistorySet(dataset, lattice_path, rasterize, 
                                    augment_history=augment_history, augment_cutout=augment_cutout, multichannel=multichannel)
    else:
        ds_mask = np.load(loader_cfg['mask'])["arr_0"]
        dataset = AgentDataset(cfg, ds_zarr, rasterizer, agents_mask=ds_mask)
        print(dataset)
        dataset = FadingHistorySet(dataset, lattice_path, rasterize, 
                                    augment_history=augment_history, augment_cutout=augment_cutout, multichannel=multichannel)

    if return_loader:
        loader = DataLoader(dataset,
                            shuffle=loader_cfg["shuffle"],
                            batch_size=loader_cfg["batch_size"],
                            num_workers=loader_cfg["num_workers"],
                            pin_memory=True
                            )

        return loader
    else:
        return dataset
