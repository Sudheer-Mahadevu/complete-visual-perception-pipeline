"""Unified multi-task model
"""

import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .classification import ClassificationHead
from .localization import LocalizationHead
from .segmentation import VGG11UNetDecoder
import gdown
import os

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, 
                 in_channels: int = 3, dropout_p = 0.5,
                 classifier_drive_id: str = "1R_eceTm-8bbKkarbxO0jkUqLrllSDVBy", 
                 localizer_drive_id: str = "1Jm-BW5SmUl_bZDZ9V7ebjgWcCTsGwoFX", 
                 unet_drive_id: str = "1Lp87qN9qp-KemomiBt369Vp4wTdQhpyl",
                 weights_dir = "checkpoints"):
        """
        Initialize the shared backbone/heads using these trained weights.
        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
        """
        super().__init__()

        # shared backbone
        self.encoder = VGG11Encoder()

        # task heads:
        self.cls_head = ClassificationHead(
            in_channels=512,
            num_classes=num_breeds,
            dropout_p=dropout_p,
        )
        self.loc_head = LocalizationHead()
        self.seg_head = VGG11UNetDecoder()

        self._load_pretrained_weights(
            classifier_drive_id=classifier_drive_id,
            localizer_drive_id=localizer_drive_id,
            unet_drive_id=unet_drive_id,
            weights_dir=weights_dir,
        )

    def _load_pretrained_weights(
        self,
        classifier_drive_id: str,
        localizer_drive_id:  str,
        unet_drive_id:       str,
        weights_dir:         str,):
        """
        Download the three individual-task checkpoints from Google Drive
        (using gdown) and copy their weights into the corresponding
        submodules of this unified model."""

        os.makedirs(weights_dir, exist_ok=True)
        
        # define local paths for downloading
        classifier_path = os.path.join(weights_dir, "classifier.pth")
        localizer_path  = os.path.join(weights_dir, "localizer.pth")
        unet_path       = os.path.join(weights_dir, "unet.pth")

        # Download wts from gdrive. skip if already cached
        if not os.path.exists(classifier_path):
            print(f"Downloading classifier weights  {classifier_path}")
            gdown.download(id=classifier_drive_id,
                           output=classifier_path, quiet=False)
        else:
            print(f"Classifier weights already cached at {classifier_path}")

        if not os.path.exists(localizer_path):
            print(f"Downloading localizer weights   {localizer_path}")
            gdown.download(id=localizer_drive_id,
                           output=localizer_path, quiet=False)
        else:
            print(f"Localizer weights already cached at {localizer_path}")

        if not os.path.exists(unet_path):
            print(f"Downloading U-Net weights      {unet_path}")
            gdown.download(id=unet_drive_id,
                           output=unet_path, quiet=False)
        else:
            print(f"U-Net weights already cached at {unet_path}")
        
        # Load the state dictionaries
        map_loc = torch.device("cpu")   # load to CPU first; model.to(device) later
        cls_state = torch.load(classifier_path, map_location=map_loc)
        loc_state = torch.load(localizer_path,  map_location=map_loc)
        seg_state = torch.load(unet_path, map_location=map_loc)

        # We load the backbone weights from backbone of classifier by default
        # ******* Backbone ***********
        # Remove features. from classifier keys
        encoder_state = {
            k.replace("features.", "", 1): v
            for k, v in cls_state.items()
            if k.startswith("features.")
        }

        missing, unexpected = self.encoder.load_state_dict(
            encoder_state, strict=True
        )

        print(f"[encoder]   loaded from classifier  | "
              f"missing={missing} | unexpected={unexpected}")

        # ******* Classification Head *******
        # head.classifier.x ---> cls_head.classifier.x
        cls_head_wts = {
            k.replace("head.", "", 1): v
            for k, v in cls_state.items()
            if k.startswith("head.")
        }

        missing, unexpected = self.cls_head.load_state_dict(
            cls_head_wts, strict=True
        )

        print(f"[cls_head]   loaded from classifier  | "
              f"missing={missing} | unexpected={unexpected}")

        # ******* Localization Head *******
        # head.regressor.x ---> loc_head.regressor.x
        loc_head_wts = {
            k.replace("head.", "", 1): v
            for k, v in loc_state.items()
            if k.startswith("head.")
        }

        missing, unexpected = self.loc_head.load_state_dict(
            loc_head_wts, strict=True
        )

        print(f"[loc_head]   loaded from classifier  | "
              f"missing={missing} | unexpected={unexpected}")

         # ******* Segmentation Head *******
        # decoder.xyz ---> seg_head.xyz
        seg_head_wts = {
            k.replace("decoder.", "", 1): v
            for k, v in seg_state.items()
            if k.startswith("decoder.")
        }

        missing, unexpected = self.seg_head.load_state_dict(
            seg_head_wts, strict=True
        )

        print(f"[seg_head]   loaded from classifier  | "
              f"missing={missing} | unexpected={unexpected}")

    # Internel Encoder with skip extraction
    def _encode_with_skips(self, x):
        """
        Run the encoder and collect feature maps before each max-pool.
        Returns (skips, bottleneck) where:
          skips[0] : (B,  64, 112, 112)
          skips[1] : (B, 128,  56,  56)
          skips[2] : (B, 256,  28,  28)
          skips[3] : (B, 512,  14,  14)
          bottleneck: (B, 512,   7,   7)
        """

        enc = self.encoder

        def pre_pool_feat(block, x_in):
            """Return (feature_before_pool, output_after_pool)."""

            layers = list(block.children())
            pre = nn.Sequential(*layers[:-1]) # everything except MaxPool
            pool = layers[-1]
            feat = pre(x_in)

            return feat, pool(feat)

        _, s1 = pre_pool_feat(enc.block1, x)   # s1: (B, 64,  112, 112)
        _, s2 = pre_pool_feat(enc.block2, s1)   # s2: (B, 128,  56,  56)
        _, s3 = pre_pool_feat(enc.block3, s2)   # s3: (B, 256,  28,  28)
        _, s4 = pre_pool_feat(enc.block4, s3)   # s4: (B, 512,  14,  14)
        _, bottleneck = pre_pool_feat(enc.block5, s4)   # bottleneck: (B, 512, 7, 7)

        return [s1, s2, s3, s4], bottleneck


    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        
        skips, bottleneck = self._encode_with_skips(x)

        logits = {}
        logits["classification"] = self.cls_head(bottleneck)          
        logits["localization"]       = self.loc_head(bottleneck)    
        logits["segmentation"] = self.seg_head(bottleneck, skips)

        return logits