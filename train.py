import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Import your custom modules
from dataset import UAVInfrastructureDataset
from losses import UAVCompositeLoss
from models.custom_backbone import UAVDefectDetector

def set_deterministic_seeds(seed=42):
    """Phase III: Global System Determinism"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[SYSTEM] Deterministic mode locked with seed {seed}")

# ==========================================
# CUSTOM BATCH COLLATION
# ==========================================
def uav_collate_fn(batch):
    """
    Intercepts the batch before stacking to handle variable-length target lists.
    Filters out empty concrete tiles to prevent model poisoning.
    """
    valid_images = []
    valid_boxes = []
    valid_classes = []

    for img, boxes, labels in batch:
        if len(labels) > 0: # We only want to train on tiles that actually contain defects
            valid_images.append(img)
            valid_boxes.append(boxes[0])   # Isolate the primary defect to align with our 1-prediction head
            valid_classes.append(labels[0])

    # If by random chance every image in this batch was empty concrete, return None
    if len(valid_images) == 0:
        return None, None, None

    # Safely stack the aligned tensors
    return torch.stack(valid_images, 0), torch.stack(valid_boxes, 0), torch.stack(valid_classes, 0)

def verify_overfit(model, dataloader, criterion, optimizer, device):
    """Network Capacity Overfit Verification"""
    print("\n[AUDIT] Initiating Network Capacity Overfit Verification...")
    model.train()
    
    print("Hunting for a valid defect crop...")
    for images, target_boxes, target_classes in dataloader:
        if images is not None: # Handled seamlessly by our new collate_fn!
            print("Defect found! Locking in data.")
            break
            
    images = images.to(device)
    target_boxes = target_boxes.to(device)
    target_classes = target_classes.to(device)
    
    # Flatten targets to perfectly match the 1D model outputs
    target_classes = target_classes.long().view(-1)
    target_boxes = target_boxes.view(-1, 4)
    
    for epoch in range(50): 
        optimizer.zero_grad()
        pred_classes, pred_boxes = model(images)
        loss, cls_loss, box_loss = criterion(pred_classes, pred_boxes, target_classes, target_boxes)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Overfit Epoch {epoch} | Total Loss: {loss.item():.4f} | Cls: {cls_loss.item():.4f} | Box: {box_loss.item():.4f}")
            
    print("[AUDIT] Overfit Verification Complete. If loss approached 0, architecture is sound.\n")

def main(args):
    set_deterministic_seeds()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[SYSTEM] Executing on hardware: {device}")
    
    # 1. Initialize Data Pipeline with the custom collate function
    dataset = UAVInfrastructureDataset(image_dir=args.image_path, annotation_dir=args.label_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=uav_collate_fn)
    
    # 2. Initialize Model Architecture 
    model = UAVDefectDetector(num_classes=3).to(device)
    
    # 3. Initialize Optimization Mechanics
    criterion = UAVCompositeLoss(lambda_box=1.5, alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 4. Perform Network Overfit Check (if triggered)
    if args.overfit_check:
        verify_overfit(model, dataloader, criterion, optimizer, device)
        return 
        
    # 5. Main Training Loop 
    print("\n[SYSTEM] Initiating Main Optimization Loop...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        
        for batch_idx, batch_data in enumerate(dataloader):
            images, target_boxes, target_classes = batch_data
            
            # Skip the iteration if the entire batch was empty background concrete
            if images is None:
                continue
                
            images = images.to(device)
            target_boxes = target_boxes.to(device)
            target_classes = target_classes.to(device)
            
            optimizer.zero_grad()
            pred_classes, pred_boxes = model(images)
            
            # Align shapes
            target_classes = target_classes.long().view(-1)
            target_boxes = target_boxes.view(-1, 4)
            
            loss, _, _ = criterion(pred_classes, pred_boxes, target_classes, target_boxes)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            valid_batches += 1
            
        scheduler.step()
        
        if valid_batches > 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {epoch_loss/valid_batches:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AXIS Capstone: UAV Training Pipeline")
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--label_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8) 
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--overfit_check', action='store_true')
    
    args = parser.parse_args()
    main(args)


if valid_batches > 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {epoch_loss/valid_batches:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # ==========================================
    # ADD THIS TO THE VERY END OF main(args):
    # ==========================================
torch.save(model.state_dict(), 'axis_uav_final_weights.pth')
print("\n[SYSTEM] Model weights successfully saved to 'axis_uav_final_weights.pth'")
