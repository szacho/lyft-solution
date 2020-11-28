import numpy as np
import os, json, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch import nn, optim
from ranger.ranger2020 import Ranger
from tqdm import tqdm

from utils import *
from models import MTP, NLLLoss, Backbone

MODEL_PATH = './output/c3'
STATE_PATH = './output/c2/checkpoint_1480000.pt'
# STATE_PATH = None
DEBUG = False

# load model config
with open(f'{MODEL_PATH}/config.json') as cfg_file:
    cfg = json.load(cfg_file)
CLIP_GRADIENT = cfg['train_params']['clip_gradient']

# load dataset
train_dataloader = load_datasetV2(cfg, 'full_data_loader', False)
print(f"Length of the dataset: {len(train_dataloader)}x{cfg['full_data_loader']['batch_size']}")
# training
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

input_image_shape = (3 if not cfg['dataset']['multichannel'] else 5, cfg['raster_params']['raster_size'][1], cfg['raster_params']['raster_size'][0])
backbone = Backbone(cfg['train_params']['backbone'], input_image_shape[0])

model = MTP(backbone, 
                num_modes=3, 
                seconds=5,
                frequency_in_hz=10,
                input_shape=input_image_shape, 
                n_hidden_layers=4096,
                asv_dim=cfg['train_params']['asv_dim'])
model.to(device)

# optimizer = Ranger(model.parameters(), lr=1e-3, betas=(.95, 0.999), eps=1e-5, weight_decay=1e-3)
# scheduler = get_step_schedule_with_cosine_anneal(optimizer, [0.72], 0.1, cfg["train_params"]["max_num_steps"], last_epoch=-1)
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=2e-3)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-5, total_steps=cfg["train_params"]["max_num_steps"], cycle_momentum=False)

criterion = NLLLoss(num_modes=3)

if STATE_PATH:
    start_itr = int(STATE_PATH.split('_')[-1].split('.')[0])
    checkpoint = torch.load(STATE_PATH)
    model.load_state_dict(checkpoint['model'])
else: start_itr = 0

train_iter = iter(train_dataloader)
progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
loss_history = []
if CLIP_GRADIENT: grad_history, clip_value = [], 100

for itr in progress_bar:
    try:
        data = next(train_iter)
    except StopIteration:
        train_iter = iter(train_dataloader)
        data = next(train_iter)

    model.train()
    torch.set_grad_enabled(True)

    # forward pass
    avails = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    image = data["image"].to(device)

    with torch.no_grad():
        agent_state = batch_get_agent_stateV2(data).to(device)

    outputs = model(image, agent_state)
    loss = criterion(outputs, (targets, avails))
    loss_history.extend([x.item() for x in list(loss.cpu().detach())])
    loss = loss.mean()

    # backward pass
    for param in model.parameters():
        param.grad = None
    
    loss.backward()

    if CLIP_GRADIENT:
        obs_grad_norm = get_grad_norm(model)
        grad_history.append(obs_grad_norm)
        if itr+1 >= 100:
            if (itr+1) % 100 == 0: # update clip value every 100 iters
                clip_value = np.percentile(grad_history, 15)
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)

    optimizer.step()
    scheduler.step()

    loss_history.append(loss.item())

    if (itr+1) % cfg['train_params']['checkpoint_every_n_steps'] == 0 and not DEBUG:
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
            }
        torch.save(checkpoint, f'{MODEL_PATH}/checkpoint_{start_itr+itr+1}.pt')
    if CLIP_GRADIENT:
        progress_bar.set_description(f"loss: {loss.item():.4f} loss(avg): {np.mean(loss_history[-100000:]):.4f} clip: {clip_value:.2f}")
    else:
        progress_bar.set_description(f"loss: {loss.item():.4f} loss(avg): {np.mean(loss_history[-100000:]):.4f}")

if not DEBUG:
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }
    torch.save(checkpoint, f'{MODEL_PATH}/checkpoint_{start_itr+itr+1}.pt')
    torch.save(torch.Tensor(loss_history), f'{MODEL_PATH}/loss_history.pt')
    if CLIP_GRADIENT: torch.save(torch.Tensor(grad_history), f'{MODEL_PATH}/grad_history.pt')
