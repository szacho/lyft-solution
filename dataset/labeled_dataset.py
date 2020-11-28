import torch
import numpy as np
import math, random
from torch.utils.data import Dataset, DataLoader

pi = torch.Tensor([math.pi])

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

P_DROP_HISTORY = 0.03
P_CUTOUT = 0.5

class FadingHistorySet(Dataset):
    def __init__(self, agents, lattice_path=None, rasterize=True, multichannel=False, augment_history=False, augment_cutout=False):
        super(Dataset).__init__()
        self.agents = agents
        self.rasterize = rasterize
        self.return_multi_channel = multichannel
        self.augment_cutout = augment_cutout
        self.augment_history = augment_history

        if self.augment_cutout:
            print('Cutout is on!')
        if self.augment_history:
            print('Drop history augment is on!')
        if self.return_multi_channel:
            print('Returning multi-channel images!')

        if lattice_path:
            self.lattice = torch.load(lattice_path)
            print(f'Loaded lattice from {lattice_path}')
        else: 
            self.lattice = None

    def __len__(self):
        return len(self.agents)

    def get_history_drop(self, p, max_n):
        if torch.rand(1) < p:
            input_vec = 1/torch.Tensor([ k+1 for k in range(max_n) ])
            drop = torch.multinomial(input_vec, 1)
            return drop.item()+1
        else:
            return 0

    def _create_fading_effect(self, timesteps):
        n_timesteps = len(timesteps)
        delta = 0.9/n_timesteps

        if self.augment_history:
            n_timesteps -= self.get_history_drop(P_DROP_HISTORY, n_timesteps-2)        

        image_fading = timesteps[0]
        for k in range(1, n_timesteps):
            
            brightness =  1-k*delta
            previous_step = timesteps[k]*brightness
            image_fading = torch.max(previous_step, image_fading)

        return image_fading
    
    def _apply_fading(self, target_image, fading, color):
        mask = fading > 0
        target_image[:, mask==True] = 0
        color = torch.Tensor(color).view(3, 1, 1)/255
        target_image += fading.unsqueeze(0).repeat(3, 1, 1)*color
        return target_image

    def _stack_image_layers(self, image):
        num_channels = (image.shape[0]-3)//2
        agents_images = torch.Tensor(image[:num_channels])
        ego_images = torch.Tensor(image[num_channels:-3])
        semantic_map = torch.Tensor(image[-3:])

        faded_agents = self._create_fading_effect(agents_images)
        faded_ego = self._create_fading_effect(ego_images)
       
        if not self.return_multi_channel:
            semantic_map = self._apply_fading(semantic_map, faded_agents, [255, 0, 255])
            if self.augment_cutout:
                semantic_map = self._cutout(semantic_map, P_CUTOUT)
            semantic_map = self._apply_fading(semantic_map, faded_ego, [0, 0, 255])
        else:
            faded_agents = faded_agents.unsqueeze(0)
            faded_ego = faded_ego.unsqueeze(0)
            
            if self.augment_cutout:
                semantic_map = self._cutout(torch.cat([faded_agents, semantic_map], dim=0), P_CUTOUT)
                semantic_map = torch.cat([semantic_map[0].unsqueeze(0), faded_ego, semantic_map[1:]], dim=0)
            else:
                semantic_map = torch.cat([faded_agents, faded_ego, semantic_map], dim=0)

        return semantic_map.flip([1])

    def _get_label(self, agent):
        gt_trajectory = torch.Tensor(agent['target_positions'])
        avails = torch.Tensor(agent['target_availabilities'])

        label = mean_pointwise_l2_distance(self.lattice, gt_trajectory.unsqueeze(0), avails)
        return label

    def _cutout(self, image, p, n_range=(2,5)):
        def get_params(img, scale=(0.01, 0.02), ratio=(0.6, 1/0.6)):
            img_c, img_h, img_w = img.shape

            s = random.uniform(*scale)
            r = random.uniform(*ratio)
            s = s * img_h * img_w
            w = int(math.sqrt(s / r))
            h = int(math.sqrt(s * r))
            left = random.randint(0, img_w - w)
            top = random.randint(0, img_h - h)

            return left, top, h, w
        
        if random.random() < p:
            n = random.randint(*n_range)
            for _ in range(n):
                left, top, h, w = get_params(image)
                val = 0.0
                image[:, left:left + h, top:top + w] = val
        return image

    def __getitem__(self, idx):
        agent = self.agents[idx]

        if self.lattice is not None:
            label = self._get_label(agent)
            agent['lattice_label'] = label
         
        if self.rasterize:
            agent['image'] = agent['image']().transpose(2, 0, 1)
            agent['image'] = self._stack_image_layers(agent['image'])

            agent['image'] = agent['image'].clamp(0, 1)
        else:
            del agent['image']
        
        return agent
