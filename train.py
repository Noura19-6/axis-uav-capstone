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
    """Required by Phase III: Global System Determinism"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[SYSTEM] Deterministic mode locked with seed {seed}")

def verify_overfit(model, dataloader, criterion, optimizer, device):
    """
    Phase III: Network Capacity Overfit Verification.
    Isolates 1 mini-batch (2-5 samples) and trains until loss nears zero.
    """
    print("\n[AUDIT] Initiating Network Capacity Overfit Verification...")
    model.train()
    
    # ==========================================
    # 1. FIND VALID DATA
    # Keep pulling batches until we find a crop that ACTUALLY contains a defect
    # ==========================================
    print("Hunting for a valid defect crop...")
    for images, target_boxes, target_classes in dataloader:
        if target_classes.numel() > 0: # We found a crack!
            print("Defect found! Locking in data.")
            break
            
    images = images.to(device)
    target_boxes = target_boxes.to(device)
    target_classes = target_classes.to(device)
    
    # ==========================================
    # 2. SHAPE ALIGNMENT FIX
    # Extract the primary box/class and flatten to 1D to match ResNet heads
    # ==========================================
    if target_classes.dim() > 1 and target_classes.size(1) > 0:
        target_classes = target_classes[:, 0]
    if target_boxes.dim() > 2 and target_boxes.size(1) > 0:
        target_boxes = target_boxes[:, 0, :]
        
    target_classes = target_classes.long().view(-1)
    target_boxes = target_boxes.view(-1, 4)
    
    # ==========================================
    # 3. OVERFIT LOOP
    # ==========================================
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
    
    # 1. Initialize Data Pipeline
    dataset = UAVInfrastructureDataset(image_dir=args.image_path, annotation_dir=args.label_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Initialize Model Architecture (3 classes for Track I: fracture, corrosion, missing fastener)
    model = UAVDefectDetector(num_classes=3).to(device)
    
    # 3. Initialize Optimization Mechanics
    criterion = UAVCompositeLoss(lambda_box=1.5, alpha=0.25, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Phase III: Adaptive Scheduler Execution (Cosine Annealing with Warm Restarts)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 4. Perform Network Overfit Check before main loop
    if args.overfit_check:
        verify_overfit(model, dataloader, criterion, optimizer, device)
        return # Stop execution after check so we can evaluate the logs
        
    # 5. Main Training Loop (Simplified for boilerplate)
    print("\n[SYSTEM] Initiating Main Optimization Loop...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, target_boxes, target_classes) in enumerate(dataloader):
            images, target_boxes, target_classes = images.to(device), target_boxes.to(device), target_classes.to(device)
            
            optimizer.zero_grad()
            pred_classes, pred_boxes = model(images)
            loss, _, _ = criterion(pred_classes, pred_boxes, target_classes, target_boxes)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {epoch_loss/len(dataloader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AXIS Capstone: UAV Training Pipeline")
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--label_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4) # Keep batch size small for testing
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--overfit_check', action='store_true', help="Run the structural overfit audit")
    
    args = parser.parse_args()
    main(args)
