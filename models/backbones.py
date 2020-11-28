import torch
import timm
import kornia
from torch import nn
from typing import Tuple
import torch.nn.functional as F
from fastai.vision.models.xresnet import xse_resnext18, xse_resnet34, xresnet18, xresnet34

xresnet_to_version = { 'xse_resnext18': xse_resnext18, 'xse_resnet34': xse_resnet34, 'xresnet18': xresnet18, 'xresnet34': xresnet34 }

def calculate_backbone_feature_dim(backbone, input_shape: Tuple[int, int, int]) -> int:
    """ Helper to calculate the shape of the fully-connected regression layer. """
    tensor = torch.ones(2, *input_shape)
    output_feat = backbone.forward(tensor)
    return output_feat.shape[-1]

def trim_network_at_index(network: nn.Module, index: int = -1) -> nn.Module:
    """
    Returns a new network with all layers up to index from the back.
    :param network: Module to trim.
    :param index: Where to trim the network. Counted from the last layer.
    """
    assert index < 0, f"Param index must be negative. Received {index}."
    print(f'Trimming from the backbone:\n', list(network.children())[index])
    return nn.Sequential(*list(network.children())[:index])

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x*(torch.tanh(F.softplus(x)))

def change_conv_in_channels(in_channels, conv):
    return nn.Conv2d(
            in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=False
    )

def convert_MP_to_blurMP(model, layer_type_old):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_MP_to_blurMP(module, layer_type_old)

        if type(module) == layer_type_old:
            layer_old = module
            layer_new = kornia.contrib.MaxBlurPool2d(3, True)
            model._modules[name] = layer_new

    return model

class Backbone(nn.Module):
    """
    Outputs tensor after last convolution before the fully connected layer.
    """

    def __init__(self, version: str, input_channels: int = 3):
        super().__init__()
        assert version in [ 'legacy_seresnet18', 'legacy_seresnet34', 'legacy_seresnet50',
                            'rexnet_100', 'rexnet_130', 'rexnet_150',
                            'seresnext26d_32x4d',
                            'mobilenetv3_large_100', 'xresnet34_blur'
                            ] + list(xresnet_to_version.keys())
                    

        if version in list(xresnet_to_version.keys()):
            backbone_model = xresnet_to_version[version](pretrained=True, dw=True, act_cls=Mish, p=0)
            backbone_model[0][0] = change_conv_in_channels(input_channels, backbone_model[0][0])
            self.backbone = trim_network_at_index(backbone_model, -1)
        
        elif version == 'xresnet34_blur':
            backbone_model = xresnet_to_version['xresnet34'](pretrained=True, dw=True, act_cls=Mish)
            backbone_model = convert_MP_to_blurMP(backbone_model, nn.MaxPool2d)
            backbone_model[0][0] = change_conv_in_channels(input_channels, backbone_model[0][0])
            self.backbone = trim_network_at_index(backbone_model, -3)

        elif version in ['seresnext26d_32x4d']:
            backbone_model = timm.create_model(version, pretrained=True)
            backbone_model.conv1[0] = change_conv_in_channels(input_channels, backbone_model.conv1[0])
            self.backbone = trim_network_at_index(backbone_model, -1)

        elif version in ['mobilenetv3_large_100']:
            backbone_model = timm.create_model(version, pretrained=True)
            backbone_model.conv_stem = change_conv_in_channels(input_channels, backbone_model.conv_stem)
            self.backbone = trim_network_at_index(backbone_model, -3)

        elif version in ['rexnet_100', 'rexnet_130', 'rexnet_150']:
            backbone_model = timm.create_model(version, pretrained=True)
            backbone_model.stem.conv = change_conv_in_channels(input_channels, backbone_model.stem.conv)
            backbone_model.head = trim_network_at_index(backbone_model.head, -1)
            self.backbone = backbone_model

        elif version in ['legacy_seresnet18', 'legacy_seresnet34', 'legacy_seresnet50']:
            backbone_model = timm.create_model(version, pretrained=True, act_layer=Mish)
            backbone_model.layer0.conv1 = change_conv_in_channels(input_channels, backbone_model.layer0.conv1)
            self.backbone = trim_network_at_index(backbone_model, -1)

        else:
            backbone_model = timm.create_model(version, pretrained=True)
            self.backbone = trim_network_at_index(backbone_model, -1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        backbone_features = self.backbone(input_tensor)
        return torch.flatten(backbone_features, start_dim=1)


