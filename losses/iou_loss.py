"""Custom IoU loss 
"""

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum'.
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    @staticmethod
    def _cxcywh_to_xyxy(boxes):
        """
        Change the format from center,h,w to corner points format
        """
        cx, cy, w, h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2

        return x1, y1, x2, y2

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.
        Args:
            pred_boxes: [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes in (x_center, y_center, width, height) format."""
        
        # convert to corner format
        p_x1, p_y1, p_x2, p_y2 = self._cxcywh_to_xyxy(pred_boxes)
        t_x1, t_y1, t_x2, t_y2 = self._cxcywh_to_xyxy(target_boxes)

        # Calculate the area of intersection Rectangle
        inter_x1 = torch.max(p_x1, t_x1)
        inter_y1 = torch.max(p_y1, t_y1)
        inter_x2 = torch.min(p_x2, t_x2)
        inter_y2 = torch.min(p_y2, t_y2)

        # Calculate Area of Intersection
        # Clip it to zero if no overlap happens
        inter_w = (inter_x2-inter_x1).clip(min=0.0)
        inter_h = (inter_y2-inter_y1).clip(min=0.0)
        inter_area = inter_w * inter_h

        # Calculate Area of Union
        # Clip to Zero for cases when inside-out coordinate are given 
        # (i.e, h<0 or w<0)
        pred_area = pred_boxes[:,2].clip(min=0.0) * pred_boxes[:,3].clip(min=0.0)
        targ_area = target_boxes[:,2].clip(min=0.0) * target_boxes[:,3].clip(min=0.0)

        #Union Area
        union_area = pred_area+targ_area - inter_area + self.eps

        iou_loss = 1 - (inter_area/union_area)

        if self.reduction == 'mean':
            return iou_loss.mean()
        elif self.reduction == 'sum':
            return iou_loss.sum()
        else:
            raise AttributeError(f"reduction should be 'mean' or 'sum' got {self.reduction}")