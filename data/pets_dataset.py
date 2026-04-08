import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, random_split


def _parse_bbox_xml(xml_path: str, img_w: int, img_h: int):
    """Return (cx, cy, w, h) in [0,1] from a VOC-style XML annotation."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bndbox = root.find(".//bndbox")
    if bndbox is None:
        # fall back to full-image box if annotation is missing
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
    """Oxford-IIIT Pet multi-task dataset loader wrapping torchvision's implementation.

    Args:
        root         : path where the dataset will be saved/downloaded.
        split        : 'trainval' | 'test' (standard torchvision splits).
        img_size     : square size to resize images to (default 224).
        transform    : optional callable applied to the PIL image BEFORE tensor conversion.
        augment      : if True applies random horizontal flip (only meaningful when transform is None).
        download     : if True, downloads the dataset from the internet and puts it in root.
    """

    # Trimap pixel values → class index mapping
    _TRIMAP_MAP = {1: 1, 2: 0, 3: 2}   # 1→foreground, 2→background, 3→boundary

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        img_size: int = 224,
        transform = None,
        augment: bool = False,
        download: bool = True
    ):
        # target_types=['category', 'segmentation'] fetches both label and mask
        super().__init__(root, split=split, target_types=['category', 'segmentation'], download=download)
        
        self.img_size  = img_size
        self.custom_transform = transform
        self.augment   = augment and (transform is None)

        # torchvision extracts files to root/oxford-iiit-pet/
        self.base_folder = Path(self._base_folder)
        self.xml_dir = self.base_folder / "annotations" / "xmls"

    def __getitem__(self, idx: int):
        # 1. Fetch image, class index, and raw trimap mask from base torchvision class
        img, (class_idx, mask) = super().__getitem__(idx)
        
        # We extract the stem from torchvision's internal image path list to find the matching XML
        img_path = Path(self._images[idx])
        stem = img_path.stem
        orig_w, orig_h = img.size

        # 2. Parse bounding box
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

        # 3. Process mask: resize to target size & remap trimap values {1,2,3} → {1,0,2}
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)
        remapped_mask = np.zeros_like(mask_np)
        for src, dst in self._TRIMAP_MAP.items():
            remapped_mask[mask_np == src] = dst
        mask_tensor = torch.from_numpy(remapped_mask)

        # 4. Image transforms & augmentations
        if self.custom_transform is not None:
            # Caller handles complex logic (e.g. albumentations bounding box / mask tracking)
            img_tensor = self.custom_transform(img)
        else:
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            
            # Basic data augmentation
            if self.augment and torch.rand(1).item() > 0.5:
                img = TF.hflip(img)
                mask_tensor = torch.flip(mask_tensor, dims=[1]) # FIXED: Mask needs flipping too!
                bbox[0] = 1.0 - bbox[0]                         # Flip bbox cx
                
            img_tensor = TF.to_tensor(img)
            img_tensor = TF.normalize(
                img_tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

        return {
            "image"     : img_tensor,                               # (3, H, W) float
            "label"     : torch.tensor(class_idx, dtype=torch.long),# scalar long
            "bbox"      : bbox,                                     # (4,) float [cx, cy, w, h]
            "mask"      : mask_tensor,                              # (H, W) long {0, 1, 2}
            "stem"      : stem,
        }


# Convenience factory
def build_dataloaders(root: str, img_size: int = 224, batch_size: int = 32, num_workers: int = 4):
    """
    Downloads/loads data and returns (train_loader, val_loader, test_loader).
    Splits the official 'trainval' dataset into 85% train and 15% validation.
    """
    # Load full train/val split (with download=True so it fetches it if missing)
    full_trainval_ds = MultiTaskPetDataset(
        root=root, split="trainval", img_size=img_size, augment=True, download=True
    )

    # Calculate lengths for 85/15 split
    total_trainval = len(full_trainval_ds)
    train_len = int(0.85 * total_trainval)
    val_len = total_trainval - train_len

    # 1. Create indices
    indices = list(range(len(full_trainval_ds)))
    
    # 2. Explicitly shuffle indices
    torch.manual_seed(42)
    indices = torch.randperm(len(full_trainval_ds)).tolist()
    
    # 3. Create subsets based on shuffled indices
    train_idx = indices[:train_len]
    val_idx = indices[train_len:]
    
    train_ds = torch.utils.data.Subset(full_trainval_ds, train_idx)
    val_ds = torch.utils.data.Subset(full_trainval_ds, val_idx)
    
    # Load test split
    test_ds = MultiTaskPetDataset(
        root=root, split="test", img_size=img_size, augment=False, download=True
    )

    # # Split dataset
    # generator = torch.Generator().manual_seed(42) # Ensure reproducible splits
    # train_ds, val_ds = random_split(full_trainval_ds, [train_len, val_len], generator=generator)

    # Disable augmentation for the validation subset
    val_ds.dataset.augment = False

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)
    
    return train_loader, val_loader, test_loader
