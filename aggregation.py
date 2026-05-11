import torch
import torch.nn.functional as F

def aggregate_updates(global_state, client_deltas, client_samples, algo="fedavg", prev_global_update=None, extra_info=None):
    """
    global_state: dict of global model weights
    client_deltas: list of dicts containing weight differences (local - global)
    client_samples: list of int, number of samples per client
    algo: aggregation strategy string
    prev_global_update: The previous round's aggregated delta (useful for momentum or cosine similarity)
    Returns:
        new_global_state: Updated global model weights
        current_global_update: The aggregated delta for this round
    """
    total_samples = sum(client_samples)
    num_clients = len(client_deltas)
    
    # Initialize aggregated delta with zeros on CPU to ensure compatibility with client.py deltas
    aggregated_delta = {k: torch.zeros(v.shape, device='cpu').float() for k, v in global_state.items()}
    
    # 1. FedAvg / FedProx / SCAFFOLD (SCAFFOLD uses basic averaging of corrected deltas)
    if algo in ["fedavg", "fedprox", "scaffold"]:
        for i in range(num_clients):
            weight = client_samples[i] / total_samples
            for k in aggregated_delta.keys():
                aggregated_delta[k] += client_deltas[i][k].float() * weight
                
    # 2. FedNova: Normalized Averaging
    elif algo == "fednova":
        # Weight based on (n_i / N) * (avg_tau / tau_i)
        local_steps = [info["local_steps"] for info in extra_info]
        avg_tau = sum(local_steps) / len(local_steps)
        for i in range(num_clients):
            # Normalization factor for FedNova
            weight = (client_samples[i] / total_samples) * (avg_tau / local_steps[i])
            for k in aggregated_delta.keys():
                aggregated_delta[k] += client_deltas[i][k].float() * weight
                
    # 2. Proposed Ağırlıklandırma Yöntemleri (Cosine, Norm, Combined)
    elif algo.startswith("proposed_"):
        # Calculate metric vectors per client
        cosine_scores = torch.zeros(num_clients)
        norm_scores = torch.zeros(num_clients)
        
        # Flatten previous global update for cosine similarity
        if prev_global_update is None:
            # If no previous update, use uniform distribution for the first round
            prev_flat = None
            cosine_scores += 1.0
        else:
            prev_flat = torch.cat([v.flatten() for v in prev_global_update.values()])
            
        for i in range(num_clients):
            current_flat = torch.cat([v.flatten() for v in client_deltas[i].values()])
            
            # Gradient Norm Score (Inverse mapping: smaller norm drift is typically better, or 
            # we suppress extreme norms to prevent drift. Assume suppression of high drift)
            # Normalize to avoid extreme numerical values
            gnorm = torch.norm(current_flat, p=2)
            norm_scores[i] = 1.0 / (gnorm + 1e-8)  # Inverse norm weighting
            
            # Cosine Similarity Score
            if prev_flat is not None:
                cos_sim = F.cosine_similarity(current_flat.unsqueeze(0), prev_flat.unsqueeze(0)).item()
                # Shift to strictly positive range [0, 2] assuming cos_sim is in [-1, 1]
                cosine_scores[i] = cos_sim + 1.0
                
        # Normalize scores to sum to 1
        # Normalize scores using Softmax for better numerical stability
        temperature = 1.0 # Can be tuned
        if "cosine" in algo:
            weights = F.softmax(cosine_scores / temperature, dim=0)
        elif "norm" in algo:
            weights = F.softmax(norm_scores / temperature, dim=0)
        elif "combined" in algo:
            # Combined score using log-sum-exp implicitly or just adding normalized scores
            # multiplication in probability space is addition in log space
            combined = (cosine_scores + norm_scores) / 2.0
            weights = F.softmax(combined / temperature, dim=0)
        else:
            weights = torch.ones(num_clients) / num_clients

        # Apply weights
        for i in range(num_clients):
            w = weights[i].item()
            for k in aggregated_delta.keys():
                aggregated_delta[k] += client_deltas[i][k].float() * w

    else:
        raise ValueError(f"Unknown aggregation algorithm: {algo}")

    # Update global state
    new_global_state = {}
    has_nan = False
    for k in global_state.keys():
        update = aggregated_delta[k]
        if torch.isnan(update).any():
            has_nan = True
            break
        new_global_state[k] = global_state[k].cpu() + update
        
    if has_nan:
        print(f"!!! [CRITICAL] NaN detected in aggregated updates for {algo}. Skipping update for this round. !!!")
        return global_state, prev_global_update

    return new_global_state, aggregated_delta

