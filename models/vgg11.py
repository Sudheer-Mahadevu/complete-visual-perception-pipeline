"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import vgg11_bn, VGG11_BN_Weights

def _conv_bn_relu(ch_in, ch_out, kernel_size = 3, padding=1):
    """3 x 3 block with Conv --> BatchNorm2d --> ReLU"""    

    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, padding=padding, 
                  bias=False),
        nn.BatchNorm2d(ch_out),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3, pretrained = True):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        self.block1 = nn.Sequential(
            _conv_bn_relu(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.block2 = nn.Sequential(
            _conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.block5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        if pretrained:
            self._load_pretrained_weights()
        else:
            self._init_weights()
    
    def _load_pretrained_weights(self):
        """Loads weights from torchvision's vgg11_bn."""
        # print("Loading ImageNet weights for VGG11Encoder...")
        vgg_imagenet = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
        
        # Flatten our custom blocks into a single list of layers
        custom_layers = []
        for block in [self.block1, self.block2, self.block3, self.block4, self.block5]:
            for layer in block.children():
                if isinstance(layer, nn.Sequential):
                    custom_layers.extend(list(layer.children()))
                else:
                    custom_layers.append(layer)
        
        # Flatten torchvision's features into a list
        std_layers = list(vgg_imagenet.features.children())
        
        # Map parameters layer by layer
        for custom_layer, std_layer in zip(custom_layers, std_layers):
            if isinstance(custom_layer, nn.Conv2d) and isinstance(std_layer, nn.Conv2d):
                custom_layer.weight.data = std_layer.weight.data.clone()
                # Note: Our convs have bias=False, standard VGG might have bias=True.
                # Since we use BN right after, ignoring the standard bias is mathematically safe.
            
            elif isinstance(custom_layer, nn.BatchNorm2d) and isinstance(std_layer, nn.BatchNorm2d):
                custom_layer.weight.data = std_layer.weight.data.clone()
                custom_layer.bias.data = std_layer.bias.data.clone()
                custom_layer.running_mean.data = std_layer.running_mean.data.clone()
                custom_layer.running_var.data = std_layer.running_var.data.clone()

    def _init_weights(self):
        """
        Do He/Kaiming Initialization for conv layers
        and (mu, sigma) = (1,0) for BN layers
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        # TODO: skip connections will be implemented later
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x
        