import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

class DLGAttack:
    """
    Deep Leakage from Gradients (DLG) Saldırısı.
    Sunucu (veya aracı bir saldırgan), client'tan gelen gradient'leri yakaladığında
    bunu kullanarak orijinal eğitim görüntüsünü (yüzü) geri üretmeye çalışır.
    """
    def __init__(self, model, image_size=(112, 112), num_channels=3, device="cpu"):
        self.model = copy.deepcopy(model).to(device)
        self.model.eval()
        self.image_size = image_size
        self.num_channels = num_channels
        self.device = device
        
    def reconstruct(self, original_dy_dx, target_label=None, num_iterations=300, lr=0.1):
        """
        original_dy_dx: Client'ın gönderdiği (yakalanan) orijinal gradient listesi.
        target_label: (Opsiyonel) Etiket bilindiği varsayılırsa (iDLG). 
                      Bilinmiyorsa gradient'ten çıkarılabilir veya iteratif optimize edilebilir.
        """
        # Create a dummy image starting from random noise
        dummy_data = torch.randn(1, self.num_channels, self.image_size[0], self.image_size[1]).to(self.device).requires_grad_(True)
        
        # If label is unknown, we just pick a dummy label (this requires iDLG optimizations if truly unknown)
        # For simplicity in this demo, we assume the malicious server knows the label or tries a specific one
        if target_label is None:
            # iDLG trick: the gradient of the loss w.r.t the classification weights 
            # usually has the minimum value at the true label index.
            # But since our ArcFace is local, we don't even have classification gradients globally!
            # Wait! In FL Face Recog, if ArcFace is local, the server NEVER sees classification gradients!
            # It only sees Backbone gradients. This makes DLG much harder but still possible via feature matching.
            target_label = torch.tensor([0]).to(self.device)
        else:
            target_label = torch.tensor([target_label]).to(self.device)
            
        optimizer = optim.LBFGS([dummy_data], lr=lr)
        
        # We need a loss function criteria. Since we only have feature extractor in global model:
        # If the server only has ResNet18 (without ArcFace), they match the gradients of the features.
        # Let's assume the gradient matching is done on the backbone layers.
        
        history = []
        
        for iters in range(num_iterations):
            def closure():
                optimizer.zero_grad()
                
                # Forward pass dummy data
                dummy_pred = self.model(dummy_data) 
                # To get gradients, we need a loss. If we don't have the classification layer,
                # the "intercepted gradients" might be gradients of the features, or we just 
                # compute a dummy loss (e.g., mean of features) to get gradients.
                # In standard DLG, model includes the classifier.
                # Assuming dummy_pred is the embeddings. 
                # Let's compute a simple dummy loss to get gradients
                dummy_loss = dummy_pred.mean()
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)
                
                # Match gradients
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                    
                grad_diff.backward()
                return grad_diff
            
            optimizer.step(closure)
            
            if iters % 50 == 0:
                current_loss = closure()
                print(f"DLG Iteration {iters}: Gradient Difference = {current_loss.item():.4f}")
                history.append(dummy_data.clone().detach().cpu())
                
        return dummy_data.detach().cpu(), history

def apply_differential_privacy(parameters, clip_bound=1.0, noise_multiplier=0.01, device="cpu"):
    """
    Client'ın güncellemelerine (gradients/deltas) DP uygulamak için:
    1. Güncellemeleri kırp (Clip)
    2. Gauss gürültüsü (Gaussian Noise) ekle
    """
    total_norm = 0.0
    for p in parameters:
        param_norm = p.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = clip_bound / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.mul_(clip_coef)
            
    # Add noise
    for p in parameters:
        noise = torch.normal(mean=0.0, std=noise_multiplier * clip_bound, size=p.shape).to(device)
        p.add_(noise)
        
    return parameters
