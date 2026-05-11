import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# Add parent directory to path to import model and dataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import FaceResNet18
from dataset import CelebA_IdentityBased_Dataset

def generate_verification_pairs(dataset, num_pairs=1000):
    """
    Kişi bazlı veri setinden pozitif ve negatif çiftler oluşturur.
    Positive pairs: Aynı kişiye ait iki farklı resim.
    Negative pairs: Farklı kişilere ait iki resim.
    """
    print(f"Generating {num_pairs} verification pairs from test set...")
    
    # Organize images by identity
    identity_groups = {}
    for i in range(len(dataset)):
        img_name = dataset.images[i]
        lbl = dataset.labels[i] # Integer mapped identity
        if lbl not in identity_groups:
            identity_groups[lbl] = []
        identity_groups[lbl].append(i)
        
    # Filter identities with at least 2 images for positive pairs
    valid_ids = [id_ for id_, imgs in identity_groups.items() if len(imgs) >= 2]
    
    pairs = []
    labels = [] # 1 for same, 0 for different
    
    # Generate Positive Pairs
    for _ in range(num_pairs // 2):
        id_ = random.choice(valid_ids)
        idx1, idx2 = random.sample(identity_groups[id_], 2)
        pairs.append((idx1, idx2))
        labels.append(1)
        
    # Generate Negative Pairs
    all_ids = list(identity_groups.keys())
    for _ in range(num_pairs // 2):
        id1, id2 = random.sample(all_ids, 2)
        idx1 = random.choice(identity_groups[id1])
        idx2 = random.choice(identity_groups[id2])
        pairs.append((idx1, idx2))
        labels.append(0)
        
    return pairs, labels

def evaluate_model(model_path, root_dir, partition_json, num_pairs=2000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluating Model: {os.path.basename(model_path)}")
    print(f"Using Device: {device}")
    
    # Load model
    model = FaceResNet18(pretrained=False).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None
    model.eval()
    
    # Load dataset
    test_ds = CelebA_IdentityBased_Dataset(root_dir, partition_json, split="test")
    
    # Generate pairs
    pairs, ground_truth = generate_verification_pairs(test_ds, num_pairs=num_pairs)
    
    similarities = []
    
    with torch.no_grad():
        for idx1, idx2 in tqdm(pairs, desc="Computing similarities"):
            img1, _ = test_ds[idx1]
            img2, _ = test_ds[idx2]
            
            img1 = img1.unsqueeze(0).to(device)
            img2 = img2.unsqueeze(0).to(device)
            
            emb1 = model(img1)
            emb2 = model(img2)
            
            # Cosine Similarity
            cos_sim = F.cosine_similarity(emb1, emb2).item()
            similarities.append(cos_sim)
            
    similarities = np.array(similarities)
    ground_truth = np.array(ground_truth)
    
    # Calculate Metrics
    auc = roc_auc_score(ground_truth, similarities)
    
    # Find best threshold for accuracy
    best_acc = 0
    best_th = 0
    for th in np.linspace(-1, 1, 200):
        acc = accuracy_score(ground_truth, similarities > th)
        if acc > best_acc:
            best_acc = acc
            best_th = th
            
    print(f"Results for {os.path.basename(model_path)}:")
    print(f"  - ROC-AUC: {auc:.4f}")
    print(f"  - Best Accuracy: {best_acc*100:.2f}% (at threshold {best_th:.2f})")
    
    return {
        "model": os.path.basename(model_path),
        "auc": auc,
        "best_acc": best_acc,
        "best_threshold": best_th
    }

if __name__ == "__main__":
    root = "c:/Users/root/project/federetad_learning"
    partition = os.path.join(root, "fl_partition.json")
    
    # Test all models found in root
    model_files = [f for f in os.listdir(root) if f.startswith("model_") and f.endswith(".pth")]
    
    all_stats = []
    for m in model_files:
        m_path = os.path.join(root, m)
        stats = evaluate_model(m_path, root, partition)
        if stats:
            all_stats.append(stats)
            
    # Save comparison report
    report_path = os.path.join(root, "test", "verification_report.json")
    with open(report_path, 'w') as f:
        json.dump(all_stats, f, indent=4)
        
    # Generate Comparison Plot
    if all_stats:
        names = [s['model'].replace('model_', '').replace('.pth', '') for s in all_stats]
        accs = [s['best_acc'] * 100 for s in all_stats]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, accs, color='skyblue')
        plt.axhline(y=90, color='r', linestyle='--', label='High Performance Target (90%)')
        plt.title("Face Verification Accuracy Comparison (Unseen Identities)")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')
            
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(root, "test", "comparison_plot.png")
        plt.savefig(plot_path)
        print(f"Comparison plot saved to {plot_path}")
        
    print(f"\nFinal report saved to {report_path}")
