"""Classification components
"""

import torch
import torch.nn as nn
from models.layers import CustomDropout
from models.vgg11 import VGG11Encoder


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
        self.features = VGG11Encoder()   # initialize the Conv Layers

        # If the image size is different, to make sure that FC layers always get
        # H,W = 7,7 we do average pooling: This is not there is vgg11 architectue

        self.avgpool = nn.AdaptiveAvgPool2d((7,7))

        self.classifier = nn.Sequential(
                # FC1 : 512*7*7 -> 4096 with BN and Dropout
                nn.Linear(512* 7 * 7, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(4096),
                CustomDropout(p=dropout_p),

                # FC2 : 4096 -> 4096 with BN, Dropout
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(4096),
                CustomDropout(p=dropout_p),

                # FC3: 4096 -> num_classes
                nn.Linear(4096, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
                                      # (B, 3, H, W)
        x = self.features(x)          # (B, 512, 7, 7) if input is 224x224
        x = self.avgpool(x)           # (B, 512, 7, 7)
        x = torch.flatten(x,1)        # (B, 512* 7* 7)
        x = self.classifier(x)        # (B, num_classes)

        return x

    def get_block_outputs(self, x: torch.Tensor) -> dict:
        """
        Returns block-wise outputs in a dictionary
        """

        out = {}
        feat = self.features

        x = feat.block1(x);  out["block1"] = x
        x = feat.block2(x);  out["block2"] = x
        x = feat.block3(x);  out["block3"] = x
        x = feat.block4(x);  out["block4"] = x
        x = feat.block5(x);  out["block5"] = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        out["logits"] = x
        return out