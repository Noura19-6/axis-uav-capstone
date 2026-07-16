import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class UAVInfrastructureDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, tile_size=512):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.tile_size = tile_size
        valid_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
        self.images = [f for f in os.listdir(image_dir) if f.endswith(valid_extensions)]
        
        # Phase I: Geometric Coordinate Invariance & Tiling
        self.transform = A.Compose([
            A.RandomCrop(width=tile_size, height=tile_size),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_area=10.0, min_visibility=0.3))
            
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        # 1. Load Raw Image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ==========================================
        # 2. DYNAMIC MASK-TO-BOX CONVERTER
        # Converts segmentation masks to bounding boxes on the fly
        # ==========================================
        boxes = []
        labels = []
        
        # CRACK500 uses .png images for annotations instead of .txt files
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.annotation_dir, base_name + '.png')
        
        if os.path.exists(label_path):
            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Isolate the crack (white pixels) and draw geometric contours
                _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Filter out microscopic noise / tiny specks
                    if w > 10 and h > 10: 
                        boxes.append([x, y, x + w, y + h])
                        labels.append(0) # Class 0: Crack
                    
        # 3. Apply Transformations
        if len(boxes) > 0:
            transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image_tensor = transformed['image']
            boxes_tensor = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels_tensor = torch.tensor(transformed['class_labels'], dtype=torch.int64)
        else:
            # Handle perfectly smooth concrete patches securely
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image_tensor = transformed['image']
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.int64)
            
        return image_tensor, boxes_tensor, labels_tensor
