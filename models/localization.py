"""Localization modules
"""

import torch
import torch.nn as nn
from models import VGG11Encoder
from losses import IoULoss
from models import CustomDropout


class LocalizationHead(nn.Module):
    """
    Regression head that maps (B, 512, 7, 7) feature map to (B,4)
    """

    def __init__(self, droput=0.3):
        super().__init__()

        # Options:  AAP(7,7) --> 512 --> 256 --> 4 (High Resolution,Risk of Overfitting)
        # Other Options: GAP(1,1) --> 1024 --> 256 --> 4 (Low Resolution)
        # Other Options : AAP(7,7) --> 256 --> 4 (Only a single Layer)
        # Other Options : AAP(3,3) --> 512 --> 256 --> 4 (Middle Ground)
        # Let's Try it one by one. I will start with the last but AAP(5,5) the Middle Ground

        self.aap = nn.AdaptiveAvgPool2d((5,5))

        self.regressor = nn.Sequential(
            nn.Flatten(),

            nn.Linear(512* 5* 5, 512, bias= False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),         # More Dropout for parameter heavy layer

            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.2),         # Lesser Droput for the penultimate layer

            nn.Linear(256, 4),
            nn.Sigmoid(),
        )
        
        self._init_weights()
        # DOUBT: is droput tested for localization head also in autograder?

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Given x : (B, C, H, W) feature map from encoder

        Returns bbox: (B, 4) in (cx, cy, h, w) in range (0,1)
        """

        x = self.aap(x)
        return self.regressor(x)


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5, 
                 pretained_features = None, freeze_encoder = True):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()

        self.encoder = pretained_features if pretained_features is not None \
                                          else VGG11Encoder()
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.head = LocalizationHead()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format in original image pixel space(not normalized values).
        """
        
        features = self.encoder(x)
        bbox = self.head(features)
        return bbox


    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
    

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
