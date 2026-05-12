import torch
import os
import json
from dataset import CelebA_IdentityBased_Dataset
from model import FaceResNet18, ArcFaceLoss
from torch.utils.data import DataLoader
import torch.optim as optim

def debug_client_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    root_dir = "c:/Users/root/project/federetad_learning"
    partition_json = os.path.join(root_dir, "fl_partition.json")
    
    # 1. Load Dataset for a specific client
    client_id = "client_0"
    print(f"Loading data for {client_id}...")
    ds = CelebA_IdentityBased_Dataset(root_dir, partition_json, split="train", client_id=client_id)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    
    num_classes = 10177
    print(f"Client has {len(ds)} images.")
    
    # 2. Initialize Model and Local Criterion
    model = FaceResNet18(pretrained=True).to(device)
    criterion = ArcFaceLoss(num_classes=num_classes).to(device)
    
    # 3. Optimizers
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    crit_optimizer = optim.Adam(criterion.parameters(), lr=0.005)
    
    # 4. Train for 1 epoch
    model.train()
    criterion.train()
    
    print("Starting 1 epoch of training...")
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        crit_optimizer.zero_grad()
        
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        
        loss.backward()
        optimizer.step()
        crit_optimizer.step()
        
        if i % 5 == 0:
            print(f"  Batch {i}, Loss: {loss.item():.4f}")
            
    # 5. Check Accuracy
    model.eval()
    criterion.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            
            import torch.nn.functional as F
            cosine = F.linear(F.normalize(embeddings), F.normalize(criterion.weight))
            _, predicted = torch.max(cosine.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print a few predictions
            if total < 10:
                print(f"  Labels: {labels[:5].tolist()}")
                print(f"  Preds:  {predicted[:5].tolist()}")

    print(f"Final Debug Accuracy: {100.0 * correct / total:.2f}%")

if __name__ == "__main__":
    debug_client_training()
