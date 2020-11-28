import json, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch

from tqdm import tqdm
from torch.nn import functional as f
from l5kit.evaluation import write_pred_csv

from utils import *
from models import MTP, Backbone

MODEL_PATH = './output/a9'
STATE_PATH = f'{MODEL_PATH}/checkpoint_3396934.pt'

# load model config
with open(f'{MODEL_PATH}/config.json') as cfg_file:
    cfg = json.load(cfg_file)

# load dataset
test_dataloader = load_datasetV2(cfg, 'test_data_loader', None)

# calc model input params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

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

# INFERENCE LOOP
model.eval()

future_coords_offsets_pd = []
timestamps = []
agent_ids = []
confs = []

with torch.no_grad():
    dataiter = tqdm(test_dataloader)
    
    for batch_idx, data in enumerate(dataiter):

        image = data["image"].to(device)
        agent_state = batch_get_agent_stateV2(data).to(device)
        
        outputs = model(image, agent_state)
        confidence = f.softmax(outputs[:, :3], dim=1)

        rotation_angle = data['yaw']
        repeated_angle = rotation_angle.repeat(3, 1).transpose(0, 1).flatten().to(device)
        predicted_modes = batch_rotate(outputs[:, 3:].reshape(-1, 50, 2), repeated_angle).reshape(-1, 3, 50, 2)
        
        future_coords_offsets_pd.append(predicted_modes.cpu().numpy().copy())
        confs.append(confidence.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy())

write_pred_csv('submission.csv',
       timestamps=np.concatenate(timestamps),
       track_ids=np.concatenate(agent_ids),
       coords=np.concatenate(future_coords_offsets_pd),
       confs=np.concatenate(confs),
      )


