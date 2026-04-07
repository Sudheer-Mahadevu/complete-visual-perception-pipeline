"""
Oxford-IIIT Pet Dataset loader.

Loads images along with:
  - class label  (37 breed indices, 0-based)
  - bounding box (cx, cy, w, h) normalised to [0,1]
  - segmentation trimap (H×W long tensor; values 0=background, 1=foreground, 2=boundary)

Directory layout expected (standard Oxford-IIIT download):
    root/
      images/          *.jpg
      annotations/
        list.txt       (filename class_id species breed_id)
        trimaps/       *.png
        xmls/          *.xml  (head bounding boxes)
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# helpers

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


# dataset class

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Args:
        root         : path to the dataset root (contains images/ and annotations/)
        split        : 'train' | 'val' | 'test'
        img_size     : square size to resize images to (default 224)
        transform    : optional callable applied to the PIL image BEFORE
                       conversion to tensor; e.g. albumentations pipeline.
        augment      : if True applies random horizontal flip + colour jitter
                       (only meaningful when transform is None)
    """

    # Trimap pixel values → class index mapping
    _TRIMAP_MAP = {1: 1, 2: 0, 3: 2}   # 1→foreground, 2→background, 3→boundary

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 224,
        transform=None,
        augment: bool = False,
    ):
        super().__init__()
        self.root      = Path(root)
        self.split     = split
        self.img_size  = img_size
        self.transform = transform
        self.augment   = augment and (transform is None)

        self.img_dir     = self.root / "images"
        self.mask_dir    = self.root / "annotations" / "trimaps"
        self.xml_dir     = self.root / "annotations" / "xmls"
        list_file        = self.root / "annotations" / "list.txt"

        self.samples   = []   # list of (stem, class_idx)
        self._load_list(list_file, split)

    #  internal 

    def _load_list(self, list_file: Path, split: str):
        """Parse list.txt; first 6000 samples → train, rest → val/test split."""
        all_samples = []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                # list.txt format: <Image CLASS-ID SPECIES BREED-ID>
                stem     = parts[0]          # e.g. Abyssinian_1
                class_id = int(parts[1]) - 1 # convert to 0-based (1..37 → 0..36)
                all_samples.append((stem, class_id))

        # deterministic split: 85% train, 15% val, 0% test
        n = len(all_samples)
        train_end = int(0.85 * n)
        val_end   = int(n)

        if split == "train":
            self.samples = all_samples[:train_end]
        elif split == "val":
            self.samples = all_samples[train_end:val_end]
        else:  # test
            self.samples = all_samples[val_end:]

    def _load_image(self, stem: str) -> Image.Image:
        img_path = self.img_dir / f"{stem}.jpg"
        img = Image.open(img_path).convert("RGB")
        return img

    def _load_mask(self, stem: str, target_size: int) -> torch.Tensor:
        """Returns a (H,W) long tensor with values 0,1,2."""
        mask_path = self.mask_dir / f"{stem}.png"
        if not mask_path.exists():
            # return all-background mask if annotation is absent
            return torch.zeros(target_size, target_size, dtype=torch.long)
        mask = Image.open(mask_path)
        mask = mask.resize((target_size, target_size), Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)
        # Remap trimap values {1,2,3} → {1,0,2}
        remapped = np.zeros_like(mask_np)
        for src, dst in self._TRIMAP_MAP.items():
            remapped[mask_np == src] = dst
        return torch.from_numpy(remapped)

    def _load_bbox(self, stem: str, img_w: int, img_h: int):
        xml_path = self.xml_dir / f"{stem}.xml"
        if not xml_path.exists():
            return torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
        cx, cy, w, h = _parse_bbox_xml(str(xml_path), img_w, img_h)
        # Clamp to [0,1] — some annotations slightly exceed image bounds
        cx = float(np.clip(cx, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))
        w  = float(np.clip(w,  0.0, 1.0))
        h  = float(np.clip(h,  0.0, 1.0))
        return torch.tensor([cx, cy, w, h], dtype=torch.float32)

    #  public API 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, class_idx = self.samples[idx]

        # --- image ---
        img = self._load_image(stem)
        orig_w, orig_h = img.size
        bbox = self._load_bbox(stem, orig_w, orig_h)

        # --- optional custom transform (e.g. albumentations) ---
        if self.transform is not None:
            # caller must handle resizing inside their transform
            img_tensor = self.transform(img)
        else:
            # basic resize + optional augmentation
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            if self.augment and torch.rand(1).item() > 0.5:
                img = TF.hflip(img)
                # flip bbox cx
                bbox[0] = 1.0 - bbox[0]
            img_tensor = TF.to_tensor(img)
            img_tensor = TF.normalize(
                img_tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

        # --- mask ---
        mask = self._load_mask(stem, self.img_size)

        return {
            "image"     : img_tensor,            # (3, H, W) float
            "label"     : torch.tensor(class_idx, dtype=torch.long),
            "bbox"      : bbox,                  # (4,) float, (cx,cy,w,h) in [0,1]
            "mask"      : mask,                  # (H, W) long, values {0,1,2}
            "stem"      : stem,
        }


#  convenience factory 

def build_dataloaders(root: str, img_size: int = 224, batch_size: int = 32,
                      num_workers: int = 4):
    """Return (train_loader, val_loader, test_loader)."""
    from torch.utils.data import DataLoader
    train_ds = OxfordIIITPetDataset(root, split="train",
                                    img_size=img_size, augment=True)
    val_ds   = OxfordIIITPetDataset(root, split="val",   img_size=img_size)
    test_ds  = OxfordIIITPetDataset(root, split="test",  img_size=img_size)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


## Note: This dataloader class is generated by LLMs. All other code in the project
# is written manually
