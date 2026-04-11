import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader
import random
from torchvision.transforms import v2


def _parse_bbox_xml(xml_path: str, img_w: int, img_h: int):
    """Return (cx, cy, w, h) in [0,1] from a VOC-style XML annotation."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bndbox = root.find(".//bndbox")
    if bndbox is None:
        return (0.5, 0.5, 1.0, 1.0)
    
    xmin = float(bndbox.find("xmin").text)
    ymin = float(bndbox.find("ymin").text)
    xmax = float(bndbox.find("xmax").text)
    ymax = float(bndbox.find("ymax").text)
    
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    w  = (xmax - xmin) / img_w
    h  = (ymax - ymin) / img_h
    
    return (cx, cy, w, h)


class MultiTaskPetDataset(OxfordIIITPet):
    _TRIMAP_MAP = {1: 1, 2: 0, 3: 2}

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        img_size: int = 224,
        transform = None,
        augment: bool = False,
        download: bool = True,
        task = 'classification'
    ):
        super().__init__(root, split=split, target_types=['category', 'segmentation'], download=download)
        self.img_size  = img_size
        self.custom_transform = transform
        self.augment   = augment and (transform is None)
        self.base_folder = Path(self._base_folder)
        self.xml_dir = self.base_folder / "annotations" / "xmls"
        self.task = task

    def __getitem__(self, idx: int):
        img, (class_idx, mask) = super().__getitem__(idx)
        img_path = Path(self._images[idx])
        stem = img_path.stem
        orig_w, orig_h = img.size

        xml_path = self.xml_dir / f"{stem}.xml"
        if not xml_path.exists():
            bbox = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
        else:
            cx, cy, w, h = _parse_bbox_xml(str(xml_path), orig_w, orig_h)
            bbox = torch.tensor([
                np.clip(cx, 0.0, 1.0),
                np.clip(cy, 0.0, 1.0),
                np.clip(w,  0.0, 1.0),
                np.clip(h,  0.0, 1.0)
            ], dtype=torch.float32)

        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)
        remapped_mask = np.zeros_like(mask_np)
        for src, dst in self._TRIMAP_MAP.items():
            remapped_mask[mask_np == src] = dst
        mask_tensor = torch.from_numpy(remapped_mask)

        if self.custom_transform is not None:
            img_tensor = self.custom_transform(img)
        else:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            
            if self.augment:
                if torch.rand(1).item() > 0.5:
                    img = TF.hflip(img)
                    if self.task == 'segmentation':
                        mask_tensor = TF.hflip(mask_tensor)
                    if self.task == 'localization':
                        bbox[0] = 1.0 - bbox[0]

                # --- 2. Task-Specific Geometric: Random Resized Crop ---
                if self.task == 'classification':
                    cropper = T.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0))
                    img = cropper(img)
                
                # --- 3. random  rotation +- 15 degrees
                if self.task in ['classification', 'segmentation']:
                    angle = random.uniform(-15, 15)
                    
                    img = TF.rotate(img, angle)
                    if self.task == 'segmentation':
                    # Add batch dim: [C, H, W] -> [1, C, H, W]
                        mask_tensor = mask_tensor.unsqueeze(0)
                        mask_tensor = TF.rotate(
                            mask_tensor, 
                            angle, 
                            interpolation=v2.InterpolationMode.NEAREST
                        )
                        mask_tensor = mask_tensor.squeeze(0) # Back to [C, H, W]

                # --- 4. Photometric: Color Jitter & Grayscale ---
                if self.task != 'segmentation':
                    jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
                    img = jitter(img)
                    
                    if torch.rand(1).item() > 0.8:
                        img = TF.to_grayscale(img, num_output_channels=3)
            
        
        #Convert to Tensor and Normalize
        img_tensor = TF.to_tensor(img)
        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # --- 5. Tensor-level: Random Erasing ---
        if self.augment and self.task == 'classification':
            eraser = T.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
            img_tensor = eraser(img_tensor)

        return {
            "image"     : img_tensor,
            "label"     : torch.tensor(class_idx, dtype=torch.long),
            "bbox"      : bbox,
            "mask"      : mask_tensor,
            "stem"      : stem,
        }


def build_dataloaders(root: str, img_size: int = 224, batch_size: int = 32, num_workers: int = 4, task = 'classification'):
    """
    Downloads/loads data and returns (train_loader, val_loader, test_loader).
    """
    # --- Fixed the Subset Bug by creating two independent dataset instances ---
    # Previously, setting val_ds.dataset.augment = False turned off train augmentation too!
    trainval_base_aug   = MultiTaskPetDataset(root=root, split="trainval", img_size=img_size, augment=True, download=True, task = task)
    trainval_base_noaug = MultiTaskPetDataset(root=root, split="trainval", img_size=img_size, augment=False, download=False, task = task)
    # ---------------------------------------------------------------------------------

    total_trainval = len(trainval_base_aug)
    train_len = int(0.85 * total_trainval)
    
    indices = list(range(total_trainval))
    torch.manual_seed(42)
    indices = torch.randperm(total_trainval).tolist()
    
    train_idx = indices[:train_len]
    val_idx = indices[train_len:]
    
    # --- Assign indices to the correct base dataset ---
    train_ds = torch.utils.data.Subset(trainval_base_aug, train_idx)
    val_ds   = torch.utils.data.Subset(trainval_base_noaug, val_idx)
    # -----------------------------------------------------------
    
    test_ds = MultiTaskPetDataset(root=root, split="test", img_size=img_size, augment=False, download=True)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)
    
    return train_loader, val_loader, test_loader