import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

class FLClient:
    def __init__(self, client_id, dataloader, device="cpu", local_epochs=1, lr=0.01, mu=0.01, num_classes=None, dp_noise=None, dp_clip=1.0):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.mu = mu  # for FedProx
        self.num_classes = num_classes
        self.dp_noise = dp_noise # If provided, add Gaussian noise multiplier to gradients
        self.dp_clip = dp_clip # Gradient clipping bound
        
        # In Federated Face Recognition with Non-IID identities, 
        # the ArcFace loss layer is kept strictly LOCAL because class IDs 
        # don't overlap. We only federate/aggregate the backbone (FaceResNet18).
        from model import ArcFaceLoss
        self.local_criterion = ArcFaceLoss(num_classes=num_classes).to(self.device)
        # ArcFace weight'leri sıfırdan başladığı için ResNet'e göre x10 daha agresif (hızlı) öğrenmesi gerekir
        self.local_criterion_optimizer = optim.Adam(self.local_criterion.parameters(), lr=self.lr * 5.0, weight_decay=1e-4) # Higher weight decay for stability
        self.opt_state = None # To be loaded by server

    def train(self, global_model, algo="fedavg", c_global=None, c_local=None, current_round=1):
        """
        global_model: Global FaceResNet18 modeli (Sadece feature extractor kısmı)
        algo: "fedavg", "fedprox", "scaffold", "fednova" vs.
        c_global: Global control variate (for SCAFFOLD)
        c_local: Local control variate (for SCAFFOLD)
        Returns:
            delta_weights: The difference between trained local model and initial global model.
            num_samples: Total samples trained on.
            avg_loss: Average loss over local epochs.
            avg_acc: Local accuracy.
            local_state_to_save: Local ArcFace weights.
            extra_metrics: {
                "local_steps": int,
                "new_c_local": dict (if scaffold),
                "drift_norm": float
            }
        """
        # Create a local copy of the global model
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()
        
        # Only the backbone is optimized here globally + local ArcFace weights
        optimizer = optim.Adam(local_model.parameters(), lr=self.lr, weight_decay=1e-5)
        
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        # Sabit kalmasın, model ezberlemeye başlayınca (plateau) öğrenme oranını düşürsün
        scheduler_model = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        scheduler_criterion = ReduceLROnPlateau(self.local_criterion_optimizer, mode='min', factor=0.5, patience=2)
        
        global_weights = copy.deepcopy(list(global_model.parameters()))

        # --- ARC FACE MARGIN WARMUP ---
        # Raunt 1-5 arası margin 0.1 (Softmax'a yakın), sonra kademeli artırıyoruz.
        # Bu modelin "patlamadan" öğrenmeye başlamasını sağlar.
        target_margin = 0.3
        if current_round <= 5:
            self.local_criterion.margin = 0.1
        elif current_round <= 10:
            self.local_criterion.margin = 0.2
        else:
            self.local_criterion.margin = target_margin
            
        # Load local optimizer state if exists
        if self.opt_state is not None:
            self.local_criterion_optimizer.load_state_dict(self.opt_state)

        epoch_loss = []
        local_steps = 0
        
        # Prepare Scaffold control variates on device
        if algo == "scaffold" and c_global is not None and c_local is not None:
            c_global_dev = {k: v.to(self.device) for k, v in c_global.items()}
            c_local_dev = {k: v.to(self.device) for k, v in c_local.items()}
        else:
            c_global_dev, c_local_dev = None, None
            batch_loss = []
            for images, labels in self.dataloader:
                if len(images) < 2:
                    continue
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                self.local_criterion_optimizer.zero_grad()
                
                embeddings = local_model(images)
                loss = self.local_criterion(embeddings, labels)

                # Proximal term for FedProx
                if algo == "fedprox":
                    proximal_term = 0.0
                    for local_param, global_param in zip(local_model.parameters(), global_weights):
                        proximal_term += ((local_param - global_param.to(self.device)) ** 2).sum()
                    loss += (self.mu / 2) * proximal_term

                loss.backward()
                
                # Gradient Clipping: Gradyan patlamasını önlemek için (NaN koruması)
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.local_criterion.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # SCAFFOLD: Gradient correction using control variates
                if algo == "scaffold" and c_global_dev is not None:
                    with torch.no_grad():
                        for name, param in local_model.named_parameters():
                            if name in c_global_dev:
                                # y = y - lr * (g - c_i + c)
                                # optimizer.step() already did y = y - lr * g
                                # So we need to add lr * (c_i - c)
                                param.add_(self.lr * (c_local_dev[name] - c_global_dev[name]))

                self.local_criterion_optimizer.step()
                local_steps += 1
                
                # NaN Check
                if torch.isnan(loss):
                    print(f"  [WARNING] Client {self.client_id} hit NaN loss!")
                    break
                    
                batch_loss.append(loss.item())
            
            
            if len(batch_loss) > 0:
                ep_loss = sum(batch_loss) / len(batch_loss)
                epoch_loss.append(ep_loss)
                # Epoch sonunda öğrenme oranlarını ezberlemeye göre dinamik olarak düşür
                scheduler_model.step(ep_loss)
                scheduler_criterion.step(ep_loss)

        # Calculate Accuracy on the last training data (Representative of local convergence)
        correct = 0
        total = 0
        local_model.eval()
        with torch.no_grad():
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                embeddings = local_model(images)
                # ArcFace accuracy calculation: Compare embeddings with local head weights
                cosine = F.linear(F.normalize(embeddings), F.normalize(self.local_criterion.weight))
                _, predicted = torch.max(cosine.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_acc = 100.0 * correct / total if total > 0 else 0.0

        # Calculate weight delta (Update)
        delta_weights = {}
        global_state = global_model.state_dict()
        local_state = local_model.state_dict()

        for key in global_state.keys():
            delta_weights[key] = local_state[key].cpu() - global_state[key].cpu()

        # Apply Differential Privacy if enabled
        if self.dp_noise is not None:
            from privacy import apply_differential_privacy
            delta_list = list(delta_weights.values())
            delta_list = apply_differential_privacy(delta_list, clip_bound=self.dp_clip, noise_multiplier=self.dp_noise, device="cpu")

        num_samples = len(self.dataloader.dataset)
        avg_loss = sum(epoch_loss)/len(epoch_loss) if len(epoch_loss) > 0 else 0.0
        
        # Return local state so server can persist it
        local_state_to_save = {
            "weights": {k: v.cpu() for k, v in self.local_criterion.state_dict().items()},
            "opt_state": self.local_criterion_optimizer.state_dict()
        }
        
        # SCAFFOLD: Update local control variate
        new_c_local = None
        if algo == "scaffold":
            new_c_local = {}
            with torch.no_grad():
                # c_i+ = c_i - c + (x - y_i) / (K * lr)
                # K = local_steps
                for name, param in local_model.named_parameters():
                    if name in global_state:
                        # global_state[name] is x, param.cpu() is y_i
                        diff = (global_state[name] - param.cpu()) / (local_steps * self.lr)
                        if c_local is not None and name in c_local and c_global is not None and name in c_global:
                            new_c_local[name] = c_local[name] - c_global[name] + diff
                        else:
                            new_c_local[name] = diff

        # Calculate Drift Norm (L2 distance from global model)
        drift_norm = 0.0
        for k in delta_weights.keys():
            if torch.is_floating_point(delta_weights[k]):
                drift_norm += torch.norm(delta_weights[k], p=2).item() ** 2
        drift_norm = drift_norm ** 0.5

        extra_metrics = {
            "local_steps": local_steps,
            "new_c_local": new_c_local,
            "drift_norm": drift_norm
        }
        
        return delta_weights, num_samples, avg_loss, avg_acc, local_state_to_save, extra_metrics

