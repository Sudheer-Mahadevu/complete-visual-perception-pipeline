"""Classification components
"""

import torch
import torch.nn as nn
from .layers import CustomDropout
from .vgg11 import VGG11Encoder

class ClassificationHead(nn.Module):
    """
    Takes the bottleneck feature map (B,512,7,7) and produces class logits.
    Mirrors the VGG11 classifier from Task 1.
    """

    def __init__(self, in_channels: int = 512, num_classes: int = 37,
                 dropout_p: float = 0.5):
        super().__init__()

        # If the image size is different, to make sure that FC layers always get
        # H,W = 7,7 we do average pooling: This is not there is vgg11 architectue

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
                # FC1 : 512 -> 256 with BN and Dropout
                nn.Flatten(),
                nn.Linear(512, 256, bias = False),
                nn.BatchNorm1d(256),
                nn.ReLU(256),
                CustomDropout(p=dropout_p),

                # FC2: 256 -> num_classes
                nn.Linear(256, num_classes),
        )
        
        # Initialize weights
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        print(f"Initialized GAP, FC(256), FC(37) Classifier Head")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        return self.classifier(x)


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5,
                 pretrained = False):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
        self.features = VGG11Encoder(pretrained=pretrained)   # initialize the Conv Layers
        self.head = ClassificationHead(dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
                                      # (B, 3, H, W)
        x = self.features(x)          # (B, 512, 7, 7) if input is 224x224
        x = self.head(x)              # (B, num_classes)
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

        x = self.head(x)
        out["logits"] = x
        return out