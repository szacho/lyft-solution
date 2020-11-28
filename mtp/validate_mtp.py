import json, os, sys
import gzip, pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.nn import functional as f
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, write_gt_csv
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace

from utils import *
from models import MTP, Backbone, pytorch_neg_multi_log_likelihood_batch

MODEL_PATH = './output/c3'
STATE_PATH = f'{MODEL_PATH}/checkpoint_1480000.pt'
LATTICE_PATH = './data/trajectories/fixed_eps_3.0.pt'
GT_PATH = './data/lyft/scenes/validate_chopped_100/gt.csv'

# load model config
with open(f'{MODEL_PATH}/config.json') as cfg_file:
    cfg = json.load(cfg_file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# load dataset
val_loader = load_datasetV2(cfg, 'val_data_loader', None)
lattice = torch.load(LATTICE_PATH).to(device)


# init model
raster_size = cfg['raster_params']['raster_size']
input_image_shape = (3 if not cfg['dataset']['multichannel'] else 5, raster_size[1], raster_size[0])
backbone = Backbone(cfg['train_params']['backbone'], input_image_shape[0])

model = MTP(backbone, 
                num_modes=3, 
                seconds=5,
                frequency_in_hz=10,
                input_shape=input_image_shape, 
                n_hidden_layers=4096,
                asv_dim=cfg['train_params']['asv_dim'])
model.to(device)

model_state = torch.load(STATE_PATH, map_location=device)['model']
model.load_state_dict(model_state)

# VALIDATION LOOP
model = model.eval()
torch.set_grad_enabled(False)
max_num_batches = np.max([len(val_loader), 150])

# store information for evaluation
future_coords_offsets_pd = []
timestamps = []
agent_ids = []
confs = []

# get ground truth
gt_path = './data/lyft/scenes/validate_chopped_100/gt.csv'
gt_iter = read_gt_csv(gt_path)

dataiter = iter(val_loader)
progress_bar = tqdm(range(max_num_batches))
loss_hist = []
labels_hist = []

for itr in progress_bar:
    data = next(dataiter)
    
    targets, avails = [], []
    for _ in range(len(data['yaw'])):
        gt = next(gt_iter)
        targets.append(gt['coord'])
        avails.append(gt['avail'])
    targets, avails = torch.Tensor(targets).to(device), torch.Tensor(avails).to(device)

    image = data["image"].to(device)
    agent_state = batch_get_agent_stateV2(data).to(device)
    # agent_state = data['state'].to(device)

    outputs = model(image, agent_state)
    rotation_angle = -data['yaw'].to(device)
    targets = batch_rotate(targets, rotation_angle).to(device)
    predicted_modes = outputs[:, 3:].reshape(-1, 3, 50, 2)
    confidence = f.softmax(outputs[:, :3], dim=1)

    loss = pytorch_neg_multi_log_likelihood_batch(targets, predicted_modes, confidence, avails, reduction=None)
    labels = batch_get_labels(targets, avails, lattice, device)
    loss_hist.append(loss.cpu())
    labels_hist.append(labels.cpu())

    # additional predictions to writte csv
    repeated_angle = -rotation_angle.repeat(3, 1).transpose(0, 1).flatten().to(device)
    predicted_modes_csv = batch_rotate(outputs[:, 3:].reshape(-1, 50, 2), repeated_angle).reshape(-1, 3, 50, 2)

    future_coords_offsets_pd.append(predicted_modes_csv.cpu().numpy().copy())
    confs.append(confidence.cpu().numpy().copy())
    timestamps.append(data["timestamp"].numpy().copy())
    agent_ids.append(data["track_id"].numpy().copy())

loss_hist = torch.cat(loss_hist, dim=0)
labels_hist = torch.cat(labels_hist, dim=0)
end_itr = int(STATE_PATH.split('_')[-1].split('.')[0])
torch.save(loss_hist, f"./inspection/losses/loss_val_{MODEL_PATH.split('/')[-1]}_{end_itr}.pt")
torch.save(labels_hist, './inspection/labels.pt')
print(f'MEAN LOSS SCORE: {loss_hist.mean()}')

pred_filename = f"val_pred_{MODEL_PATH.split('/')[-1]}_{loss_hist.mean():.2f}_{end_itr}_{max_num_batches}x{cfg['val_data_loader']['batch_size']}.csv" 
write_pred_csv('./mtp/vals/'+pred_filename,
       timestamps=np.concatenate(timestamps),
       track_ids=np.concatenate(agent_ids),
       coords=np.concatenate(future_coords_offsets_pd),
       confs=np.concatenate(confs),
      )