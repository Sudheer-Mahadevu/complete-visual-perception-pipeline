"""Segmentation model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg11 import VGG11Encoder

class DecoderBlock(nn.Module):

    def __init__(self, ch_in, ch_skip, ch_out):
        super().__init__()

        # Learnable 2x upsampling
        self.upsample = nn.ConvTranspose2d(
            ch_in, ch_in//2, kernel_size=2, stride=2
        )

        fused = ch_in //2 + ch_skip
        self.conv = nn.Sequential(
            nn.Conv2d(fused, ch_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, skip):
        """
        Args:
        x: (B, ch_in, H, W)
        skip : (B, ch_skip, 2H, 2W) from encoder

        Returns:
        (B, ch_out, 2H, 2W)

        """

        x = self.upsample(x)  #(B, ch_in//2, 2H, 2W)

        # Handle odd spatial sizes (skip may be 1px larger due to rounding in // operation)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size = skip.shape[-2:],
                              mode = "bilinear", align_corners=False)
        
        x = torch.cat([x, skip], dim=1) # channel-wise fusion

        return self.conv(x)

class VGG11UNetDecoder(nn.Module):
    """
    Unet style VGG11 Decoder Block
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        """
        Channel Layout (in ---> skip ---> out)
        d5 : 512 ---> 512 ---> 256 (upsample from 7 to 14, skip from b4 pre-pool)
        d4 : 256 ---> 512 ---> 128 (upsample from 14 to 28, skip from b3 pre-pool)
        d3 : 128 ---> 256 ---> 128 (upsample from 28 to 56, skip from b2 pre-pool)
        d2 : 128 ---> 128 ---> 64 (upsample from 56 to 112, skip from b1 pre-pool)
        d1: 64 ---> 64 ---> 32 (upsample from 112 to 224, no encoder skip here)
        """

        # At d3, reducing the skips to 64 and continuing is another option.
        # It reduces number of parameters and can prevent overfitting
        self.dec5 = DecoderBlock(512, 512, 256)
        self.dec4 = DecoderBlock(256, 512, 128)
        self.dec3 = DecoderBlock(128, 256, 128)
        self.dec2 = DecoderBlock(128, 128, 64)
        self.dec1 = DecoderBlock(64, 64, 32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1) #1x1 Projection

        # Final Upsample (d1) without skip connections
        # self.d1_ups = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # self.d1_conv = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, num_classes, kernel_size=1) # 1x1 projection
        # )

        # do init
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, bottleneck, skips):
        """
        skips[0] : (B,  64, 224, 224)
        skips[1] : (B, 128,  112,  112)
        skips[2] : (B, 256, 56,  56)
        skips[3] : (B, 512,  28,  28)
        skips[4]: (B, 512,   14,   14)
        bottleneck : (B, 512,   7,   7)
        """

        s1, s2, s3, s4, s5 = skips

        # Decode
        d = self.dec5(bottleneck, s5)   # (B, 256,  14, 14)
        d = self.dec4(d, s4)            # (B, 128,  28, 28)
        d = self.dec3(d, s3)            # (B,  64,  56, 56)
        d = self.dec2(d, s2)            # (B,  64, 112, 112)
        d = self.dec1(d, s1)            # (B, 32, 224, 224)
        d = self.final_conv(d)          # (B, 3, 224, 224)

        return d

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5,
                 pretrained_features: VGG11Encoder | None = None,
                 freeze_encoder=False):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the segmentation head.
        """

        super().__init__()

        # Encoder

        self.encoder = pretrained_features if pretrained_features is not None \
                                           else VGG11Encoder()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # ********Decoder******
        self.decoder = VGG11UNetDecoder(num_classes=num_classes)
        self.num_classes = num_classes


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

        s1, out = pre_pool_feat(enc.block1, x)   # s1: (B, 64,  224, 224)
        s2, out = pre_pool_feat(enc.block2, out)   # s2: (B, 128,  112,  112)
        s3, out = pre_pool_feat(enc.block3, out)   # s3: (B, 256,  56,  56)
        s4, out = pre_pool_feat(enc.block4, out)   # s4: (B, 512,  28,  28)
        s5, bottleneck = pre_pool_feat(enc.block5, out)   # s5: (B, 512, 14, 14)

        # bottleneck (B, 512, 7, 7)
        return [s1, s2, s3, s4, s5], bottleneck

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, 3, H, W].

        Returns:
            Segmentation logits [B, num_classes(3), H, W].
        """
        
        # collect encoder skips
        skips, bottleneck = self._encode_with_skips(x)
        logits = self.decoder(bottleneck, skips)

        return logits

    def unfreeze_last_encoder_blocks(self, num_blocks: int = 2):
        """Partial fine-tuning: unfreeze the last 'num_blocks' encoder blocks."""
        blocks = [
            self.encoder.block1,
            self.encoder.block2,
            self.encoder.block3,
            self.encoder.block4,
            self.encoder.block5,
        ]
        # Unfreeze the last num_blocks
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False



class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.

    Dice = (2 * |P (intersection) G|) / (|P| + |G|)
    Loss = 1 - mean_over_classes(Dice)

    Soft Dice uses predicted probabilities instead of hard binary masks,
    making it fully differentiable.

    Args:
    smooth   : Laplace smoothing constant (avoids 0/0 when both
                prediction and target are empty)
    ignore_bg: if True, class 0 (background) is excluded from the
                Dice average, which prevents the easy background class
                from dominating.
    """
    
    def __init__(self, smooth: float = 1.0, ignore_bg: bool = False):
        super().__init__()
        self.smooth    = smooth
        self.ignore_bg = ignore_bg
    
    def forward(self, logits, targets):
        """
        Args:
        logits : (B, C, H, W) raw (un-normalized) class scores
        tragets: (B, H, W) integer class indices in [0, C)
        """

        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim = 1) # (B, C, H, W)

        targs_one_hot = F.one_hot(targets, num_classes) #(B, H, W, C)
        targs_one_hot = targs_one_hot.permute(0, 3, 1, 2).float() #(B, C, H, W)

        # Flatten Spatial dims
        probs_flat = probs.view(probs.shape[0], num_classes, -1)
        targs_flat = targs_one_hot.view(targs_one_hot.shape[0], num_classes, -1)

        intersection = (probs_flat * targs_flat).sum(-1)
        union = probs_flat.sum(-1) + targs_flat.sum(-1)

        dice = (2* intersection + self.smooth) / (union + self.smooth)

        if self.ignore_bg:
            dice = dice[:, 1:] # skip bckg class
        
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """
    Combined Cross-Entropy + Dice Loss
    """

    def __init__(self, ce_wt=1.0, dice_wt  = 1.0, class_weights = None):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(weight= class_weights)
        self.dice = DiceLoss(smooth=1.0, ignore_bg=False)
        self.ce_weight = ce_wt
        self.dice_weight = dice_wt

    def forward(self, logits, targets):

        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
