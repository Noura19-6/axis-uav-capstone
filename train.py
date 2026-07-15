import argparse
import torch
from torch.utils.data import DataLoader
from dataset import UAVInfrastructureDataset

def set_deterministic_seeds(seed=42):
    # Required by rubric: Global System Determinism
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Deterministic mode enabled with seed {seed}")

def main(args):
    set_deterministic_seeds()
    
    print(f"Loading data from Kaggle path: {args.image_path}")
    
    # Initialize dataset using the paths passed from the command line
    train_dataset = UAVInfrastructureDataset(
        image_dir=args.image_path,
        annotation_dir=args.label_path
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # TODO: Initialize model, composite loss, and dynamic scheduler here
    print("Training pipeline initialized. Ready for Phase II.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AXIS Capstone: UAV Training Pipeline")
    
    # Kaggle data paths (Required)
    parser.add_argument('--image_path', type=str, required=True, help="Path to training images")
    parser.add_argument('--label_path', type=str, required=True, help="Path to training labels")
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    main(args)