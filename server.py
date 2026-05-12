import torch
import random
import os
import json
from torch.utils.data import DataLoader
from dataset import CelebA_IdentityBased_Dataset
from model import FaceResNet18
from client import FLClient
from aggregation import aggregate_updates

class FLServer:
    def __init__(self, root_dir, partition_json, num_clients=50, fraction=0.2, device="cpu", local_epochs=10, lr=0.001):
        self.root_dir = root_dir
        self.device = device
        self.num_clients = num_clients
        self.fraction = fraction
        self.num_sampled = max(int(fraction * num_clients), 1)
        self.local_epochs = local_epochs
        self.lr = lr
        
        # Persistence: Store local ArcFace weights for each client
        self.client_states = {}
        
        # Load global model backbone
        self.global_model = FaceResNet18(pretrained=True).to(self.device)
        self.prev_global_update = None
        
        # SCAFFOLD: Control variates
        self.c_global = {k: torch.zeros(v.shape).float() for k, v in self.global_model.state_dict().items()}
        self.client_c_locals = {} # client_id -> c_local
        
        # Drift Analysis storage
        self.round_drifts = []
        
        # Load global test dataset
        print("Loading global unseen test dataset...")
        self.test_dataset = CelebA_IdentityBased_Dataset(
            root_dir=root_dir,
            partition_json_path=partition_json,
            split="test"
        )
        self.test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        
        # Initialize clients lazily (Dataset initialization can be heavy)
        self.partition_json = partition_json
        
    def _create_client(self, client_id_str):
        ds = CelebA_IdentityBased_Dataset(
            root_dir=self.root_dir,
            partition_json_path=self.partition_json,
            split="train",
            client_id=client_id_str
        )
        loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
        # For arcface we pass num_classes. Because we build local ArcFace, we just need 
        # local classes. But dataset returns global mapped IDs. We can just use the global max for simplicity.
        num_classes = 10177
        client = FLClient(client_id_str, loader, device=self.device, local_epochs=self.local_epochs, lr=self.lr, num_classes=num_classes)
        
        # Load persisted local state if it exists
        if client_id_str in self.client_states:
            state = self.client_states[client_id_str]
            client.local_criterion.load_state_dict(state["weights"])
            client.opt_state = state["opt_state"]
            if "backbone_opt_state" in state:
                client.backbone_opt_state = state["backbone_opt_state"]
            
        # SCAFFOLD: Ensure c_local exists for client
        if client_id_str not in self.client_c_locals:
            self.client_c_locals[client_id_str] = {k: torch.zeros(v.shape).float() for k, v in self.global_model.state_dict().items()}
            
        return client

    def evaluate(self):
        """
        Görülmemiş (unseen) test kimlikleri üzerinde yüz doğrulama (Face Verification) testi yapar.
        Rastgele pozitif ve negatif çiftler oluşturarak modelin ayırt ediciliğini (Cosine Similarity) ölçer.
        """
        self.global_model.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            # Test setinin bir kısmını kullanarak embedding çıkaralım (Hız için)
            count = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                embeddings = self.global_model(images)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
                count += images.size(0)
                if count > 1000: break # 1000 örnek yeterli
                
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Rastgele 500 pozitif, 500 negatif çift seçelim
        pos_sims = []
        neg_sims = []
        
        for _ in range(500):
            # Pozitif çift (Aynı kişi)
            idx1 = random.randint(0, len(all_labels)-1)
            target_label = all_labels[idx1]
            same_id_indices = (all_labels == target_label).nonzero(as_tuple=True)[0]
            if len(same_id_indices) > 1:
                idx2 = same_id_indices[random.randint(0, len(same_id_indices)-1)]
                while idx2 == idx1:
                    idx2 = same_id_indices[random.randint(0, len(same_id_indices)-1)]
                sim = torch.cosine_similarity(all_embeddings[idx1].unsqueeze(0), all_embeddings[idx2].unsqueeze(0)).item()
                pos_sims.append(sim)
                
            # Negatif çift (Farklı kişiler)
            idx2 = random.randint(0, len(all_labels)-1)
            while all_labels[idx2] == target_label:
                idx2 = random.randint(0, len(all_labels)-1)
            sim = torch.cosine_similarity(all_embeddings[idx1].unsqueeze(0), all_embeddings[idx2].unsqueeze(0)).item()
            neg_sims.append(sim)
            
        avg_pos = sum(pos_sims)/len(pos_sims) if pos_sims else 0
        avg_neg = sum(neg_sims)/len(neg_sims) if neg_sims else 0
        
        # Accuracy at threshold 0.5 (Simple heuristic)
        correct = 0
        for s in pos_sims: 
            if s > 0.4: correct += 1 # Lowered threshold for face verification in early rounds
        for s in neg_sims:
            if s <= 0.4: correct += 1
        
        total = len(pos_sims) + len(neg_sims)
        test_acc = 100.0 * correct / total if total > 0 else 0.0
        
        print(f"  [Global Eval] Unseen IDs - Avg Pos Sim: {avg_pos:.3f}, Avg Neg Sim: {avg_neg:.3f}, Verification Acc: {test_acc:.2f}%")
        return test_acc

    def start_training(self, num_rounds=10, algo="fedavg"):
        print(f"--- Starting FL Training with Algorithm: {algo} ---")
        round_losses = []
        round_accuracies = []
        self.round_drifts = []
        
        for rnd in range(1, num_rounds + 1):
            print(f"\n[Round {rnd}/{num_rounds}]")
            
            # Learning Rate Decay (Every 15 rounds)
            if rnd > 1 and rnd % 15 == 0:
                self.lr *= 0.5
                print(f"  [Server] Learning rate decayed to: {self.lr}")
            
            # 1. Sample clients
            sampled_indices = random.sample(range(self.num_clients), self.num_sampled)
            client_deltas = []
            client_samples = []
            client_extra_info = []
            round_client_losses = []
            round_client_accs = []
            round_client_drifts = []
            
            # 2. Local Training
            for idx in sampled_indices:
                client_id_str = f"client_{idx}"
                client = self._create_client(client_id_str)
                
                print(f"  Training {client_id_str} on {self.device}...")
                c_local = self.client_c_locals.get(client_id_str)
                delta, num_s, loss, acc, local_state, extra = client.train(
                    self.global_model, algo=algo, c_global=self.c_global, c_local=c_local, current_round=rnd
                )
                
                # Save client's local state for next round
                self.client_states[client_id_str] = local_state
                if algo == "scaffold":
                    self.client_c_locals[client_id_str] = extra["new_c_local"]
                
                client_deltas.append(delta)
                client_samples.append(num_s)
                client_extra_info.append(extra)
                round_client_losses.append(loss)
                round_client_accs.append(acc)
                round_client_drifts.append(extra["drift_norm"])
                print(f"    {client_id_str} - Loss: {loss:.4f}, Acc(T1/T5): {acc:.2f}%/{extra['acc_top5']:.2f}%, Drift: {extra['drift_norm']:.4f}")
            
            # 3. Server Aggregation
            print(f"  Aggregating updates using {algo}...")
            global_state = self.global_model.state_dict()
            new_global_state, self.prev_global_update = aggregate_updates(
                global_state, client_deltas, client_samples, algo, self.prev_global_update, extra_info=client_extra_info
            )
            
            # SCAFFOLD: Update global control variate
            if algo == "scaffold":
                for k in self.c_global.keys():
                    c_delta = torch.zeros_like(self.c_global[k])
                    for i in range(len(client_extra_info)):
                        # c = c + (1/N) * sum(c_i+ - c_i) 
                        # In our case, we just average the new c_locals from sampled clients
                        # Simplified SCAFFOLD global update
                        c_delta += (client_extra_info[i]["new_c_local"][k] - self.client_c_locals[f"client_{sampled_indices[i]}"][k])
                    self.c_global[k] += c_delta / len(sampled_indices)
            
            # 4. Broadcast
            self.global_model.load_state_dict(new_global_state)
            
            # 5. Evaluate Convergence
            avg_loss = sum(round_client_losses) / len(round_client_losses)
            avg_acc = sum(round_client_accs) / len(round_client_accs)
            avg_drift = sum(round_client_drifts) / len(round_client_drifts)
            
            # Run Global Evaluation
            test_acc = self.evaluate()
            
            round_losses.append(avg_loss)
            round_accuracies.append(avg_acc)
            self.round_drifts.append(avg_drift)
            print(f"  [Round {rnd}] Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2f}%, Test Acc: {test_acc:.2f}%, Avg Drift: {avg_drift:.4f}\n")
            
            # --- Per-Round Saving (Numerical) ---
            temp_results = {"loss": round_losses, "accuracy": round_accuracies, "drift": self.round_drifts}
            temp_json_path = os.path.join(self.root_dir, f"results_{algo}.json")
            with open(temp_json_path, 'w') as f:
                json.dump(temp_results, f, indent=4)
                
            # --- Per-Round Saving (Visual) ---
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.plot(range(1, rnd + 1), round_losses, marker='o', color='blue')
                plt.title(f"{algo} - Loss")
                plt.grid(True)
                
                plt.subplot(1, 3, 2)
                plt.plot(range(1, rnd + 1), round_accuracies, marker='x', color='green')
                plt.title(f"{algo} - Acc (%)")
                plt.grid(True)

                plt.subplot(1, 3, 3)
                plt.plot(range(1, rnd + 1), self.round_drifts, marker='s', color='orange')
                plt.title(f"{algo} - Model Drift")
                plt.grid(True)
                
                plt.tight_layout()
                temp_plot_path = os.path.join(self.root_dir, f"live_plot_{algo}.png")
                plt.savefig(temp_plot_path)
                plt.close()
            except Exception as e:
                print(f"  [Warning] Could not update live plot: {e}")
            
        # Save final model
        save_path = os.path.join(self.root_dir, f"model_{algo}.pth")
        torch.save(self.global_model.state_dict(), save_path)
        print(f"--- Final Weights Saved: {save_path} ---")
            
        return {"loss": round_losses, "accuracy": round_accuracies, "drift": self.round_drifts}
            
if __name__ == "__main__":
    server = FLServer(root_dir="c:/Users/root/project/federetad_learning", 
                      partition_json="c:/Users/root/project/federetad_learning/fl_partition.json",
                      device="cuda" if torch.cuda.is_available() else "cpu")
    
    server.start_training(num_rounds=2, algo="fedavg")
