import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def calculate_ciou(pred_boxes, target_boxes):
    """
    Calculates Complete Intersection over Union (CIoU).
    Assumes boxes are in format [x1, y1, x2, y2]
    """
    # 1. Calculate Intersection Area
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 2. Calculate Union Area
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_area + target_area - inter_area
    
    iou = inter_area / (union_area + 1e-6)
    
    # 3. Center Distance
    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    
    center_distance = (pred_center_x - target_center_x)**2 + (pred_center_y - target_center_y)**2
    
    # 4. Enclosing Box Diagonal
    enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    
    diagonal_distance = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2
    
    # 5. Aspect Ratio Penalty (v)
    w_pred, h_pred = pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]
    w_target, h_target = target_boxes[:, 2] - target_boxes[:, 0], target_boxes[:, 3] - target_boxes[:, 1]
    
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(w_target / h_target) - torch.atan(w_pred / h_pred), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)
        
    # Final CIoU
    ciou = iou - (center_distance / (diagonal_distance + 1e-6)) - alpha * v
    return ciou

class UAVCompositeLoss(nn.Module):
    def __init__(self, lambda_box=1.5, alpha=0.25, gamma=2.0):
        super(UAVCompositeLoss, self).__init__()
        self.lambda_box = lambda_box
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_classes, pred_boxes, target_classes, target_boxes):
        # 1. Classification: Focal Loss (handles extreme class imbalance)
        ce_loss = F.cross_entropy(pred_classes, target_classes, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        cls_loss = focal_loss.mean()
        
        # 2. Bounding Box: CIoU Loss
        ciou = calculate_ciou(pred_boxes, target_boxes)
        box_loss = (1 - ciou).mean()
        
        # 3. Composite Loss Matrix
        total_loss = cls_loss + (self.lambda_box * box_loss)
        
        return total_loss, cls_loss, box_loss
