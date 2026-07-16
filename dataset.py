import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class UAVInfrastructureDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, tile_size=512, is_training=True):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.tile_size = tile_size
        


        valid_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
        self.images = [f for f in os.listdir(image_dir) if f.endswith(valid_extensions)]
        # Phase I: Advanced Data Pipeline Engineering
        # Geometric Coordinate Invariance & Custom Tiling
        if is_training:
            self.transform = A.Compose([
                # 1. Custom Tiling: Extract a localized patch to preserve microscopic targets
                A.RandomCrop(width=tile_size, height=tile_size),
                
                # 2. Spatial Manipulations: Scale-invariant shifts and perspective warps
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                
                # 3. Normalization and Tensor Conversion
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='pascal_voc', # [xmin, ymin, xmax, ymax] format matches our CIoU loss
                label_fields=['class_labels'], 
                min_area=10.0,       # Drop boxes that become too small after cropping
                min_visibility=0.3   # Drop boxes if 70% of the defect is cropped out
            ))
        else:
            # Validation transform (no random shifts, just tiling/resizing and normalization)
            self.transform = A.Compose([
                A.CenterCrop(width=tile_size, height=tile_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
            
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        # 1. Load Image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Load Annotations (Assuming a standard text/CSV format for this example)
        label_name = img_name.replace('.png', '.txt').replace('.jpg', '.txt')
        label_path = os.path.join(self.annotation_dir, label_name)
        
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # Parse assuming: class xmin ymin xmax ymax
                    parts = line.strip().split()
                    labels.append(int(parts[0]))
                    boxes.append([float(x) for x in parts[1:5]])
                    
        # 3. Apply Transformations synchronously to image and metadata
        if len(boxes) > 0:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image_tensor = transformed['image']
            boxes_tensor = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels_tensor = torch.tensor(transformed['class_labels'], dtype=torch.int64)
        else:
            # Handle empty images (no defects)
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image_tensor = transformed['image']
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.int64)
            
        return image_tensor, boxes_tensor, labels_tensor
