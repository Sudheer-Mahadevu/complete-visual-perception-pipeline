"""
vgg11_no_bn.py  [NEW FILE]
==========================
VGG11 WITHOUT BatchNorm — used exclusively for W&B Experiment 2.1 to compare
activation distributions and training dynamics with the standard BN version.

Architecturally identical to VGG11Encoder / VGG11Classifier, but every
BatchNorm2d and BatchNorm1d layer is removed so the raw effect of BN on
activation distributions can be studied in isolation.

Usage (see EXPERIMENT_GUIDE.md):
    python train_cls.py --no_bn --run_name "cls_no_bn" --use_wandb ...
"""

import torch
import torch.nn as nn
from .layers import CustomDropout


# ─────────────────────────────────────────────────────────────────────────────
# Building block: Conv → ReLU  (no BN)
# ─────────────────────────────────────────────────────────────────────────────

def _conv_relu(ch_in: int, ch_out: int, kernel_size: int = 3,
               padding: int = 1) -> nn.Sequential:
    """Conv2d → ReLU without BatchNorm.  bias=True because there is no BN to
    absorb it."""
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,
                  padding=padding, bias=True),
        nn.ReLU(inplace=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Encoder without BN
# ─────────────────────────────────────────────────────────────────────────────

class VGG11EncoderNoBN(nn.Module):
    """
    VGG11 convolutional backbone WITHOUT BatchNorm.

    Spatial dimensions are identical to VGG11Encoder (same blocks, same pools),
    so the autograder dimension checks still pass when the same image is fed.
    """

    def __init__(self):
        super().__init__()

        # Each block mirrors VGG11Encoder but uses _conv_relu instead of _conv_bn_relu
        self.block1 = nn.Sequential(
            _conv_relu(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            _conv_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            _conv_relu(128, 256),
            _conv_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            _conv_relu(256, 512),
            _conv_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block5 = nn.Sequential(
            _conv_relu(512, 512),
            _conv_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Classification head without BN
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHeadNoBN(nn.Module):
    """FC classifier WITHOUT BatchNorm1d — pairs with VGG11EncoderNoBN."""

    def __init__(self, in_channels: int = 512, num_classes: int = 37,
                 dropout_p: float = 0.5):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # bias=True (no BN to absorb it)
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),

            nn.Linear(256, num_classes, bias=True),
        )

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full classifier without BN
# ─────────────────────────────────────────────────────────────────────────────

class VGG11ClassifierNoBN(nn.Module):
    """
    Full VGG11 classifier WITHOUT BatchNorm anywhere.

    Drop-in replacement for VGG11Classifier when --no_bn flag is set in
    train_cls.py.  Used to produce the 'without BN' run in W&B Experiment 2.1.
    """

    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()
        self.features = VGG11EncoderNoBN()
        self.head = ClassificationHeadNoBN(num_classes=num_classes,
                                           dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x

    def get_block_outputs(self, x: torch.Tensor) -> dict:
        """Returns intermediate block outputs for dimension verification."""
        out = {}
        f = self.features
        x = f.block1(x);  out["block1"] = x
        x = f.block2(x);  out["block2"] = x
        x = f.block3(x);  out["block3"] = x
        x = f.block4(x);  out["block4"] = x
        x = f.block5(x);  out["block5"] = x
        x = self.head(x); out["logits"]  = x
        return out