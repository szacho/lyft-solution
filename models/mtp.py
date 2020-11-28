# nuScenes dev-kit.
# Code written by Freddy Boulton, Elena Corina Grigore 2020.

import math
import random
from typing import List, Tuple
import numpy as np

import torch
from torch import nn
from torch.nn import functional as f
from torch import Tensor
from .backbones import Mish, calculate_backbone_feature_dim

class MTP(nn.Module):
    """
    Implementation of Multiple-Trajectory Prediction (MTP) model
    based on https://arxiv.org/pdf/1809.10732.pdf
    """

    def __init__(self, backbone: nn.Module, num_modes: int,
                 seconds: float = 5, frequency_in_hz: float = 10,
                 n_hidden_layers: int = 4096, input_shape: Tuple[int, int, int] = (3, 300, 300), asv_dim: int = 3):
        """
        Inits the MTP network.
        :param backbone: CNN Backbone to use.
        :param num_modes: Number of predicted paths to estimate for each agent.
        :param seconds: Number of seconds into the future to predict.
            Default for the challenge is 5.
        :param frequency_in_hz: Frequency between timesteps in the prediction (in Hz).
        :param n_hidden_layers: Size of fully connected layer after the CNN
            backbone processes the image.
        :param input_shape: Shape of the input expected by the network.
            This is needed because the size of the fully connected layer after
            the backbone depends on the backbone and its version.
        Note:
            Although seconds and frequency_in_hz are typed as floats, their
            product should be an int.
        """

        super().__init__()

        self.backbone = backbone
        self.num_modes = num_modes
        backbone_feature_dim = calculate_backbone_feature_dim(backbone, input_shape)
        self.fc1 = nn.Linear(backbone_feature_dim + asv_dim, n_hidden_layers)
        self.drop = nn.Dropout(p=0.1)
        predictions_per_mode = int(seconds * frequency_in_hz) * 2

        self.fc2 = nn.Linear(n_hidden_layers, int(num_modes * predictions_per_mode + num_modes))

    def forward(self, image_tensor: torch.Tensor,
                agent_state_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        :param image_tensor: Tensor of images shape [batch_size, n_channels, length, width].
        :param agent_state_vector: Tensor of floats representing the agent state.
            [batch_size, 3].
        :returns: Tensor of dimension [batch_size, number_of_modes * number_of_predictions_per_mode + number_of_modes]
            storing the predicted trajectory and mode probabilities. Mode probabilities are normalized to sum
            to 1 during inference.
        """

        backbone_features = self.backbone(image_tensor)
        features = torch.cat([backbone_features, agent_state_vector], dim=1)
        x = self.fc1(features)
        x = self.drop(x)
        predictions = self.fc2(x)
        return predictions

class NLLLoss:
    def __init__(self, num_modes: int, reduction=None):
        self.num_modes = num_modes
        self.num_location_coordinates_predicted = 2  # We predict x, y coordinates at each timestep.
        self.reduction = reduction

    def _get_trajectory_and_modes(self, model_prediction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the predictions from the model into mode probabilities and trajectory.
        :param model_prediction: Tensor of shape [batch_size, n_timesteps * n_modes * 2 + n_modes].
        :return: Tuple of tensors. First item is the trajectories of shape [batch_size, n_modes, n_timesteps, 2].
            Second item are the mode probabilities of shape [batch_size, num_modes].
        """
        mode_probabilities = model_prediction[:, :self.num_modes].clone()

        desired_shape = (model_prediction.shape[0], self.num_modes, -1, self.num_location_coordinates_predicted)
        trajectories_no_modes = model_prediction[:, self.num_modes:].clone().reshape(desired_shape)

        return trajectories_no_modes, mode_probabilities
    
    def __call__(self, outputs, targets_with_avails):
        targets, avails = targets_with_avails
        trajectories, confidences = self._get_trajectory_and_modes(outputs)
        confidences = f.softmax(confidences, dim=1)
        loss = pytorch_neg_multi_log_likelihood_batch(targets, trajectories, confidences, avails, self.reduction)
        return loss


def pytorch_neg_multi_log_likelihood_batch(gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor,
reduction='mean') -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences +  1e-10) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    if reduction == 'mean':
        return torch.mean(error)
    else:
        return error