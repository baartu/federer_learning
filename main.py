import torch
import matplotlib.pyplot as plt
import json
import os
from server import FLServer

def run_experiment(algo_name, num_rounds, num_clients=50, fraction=0.1):
    print(f"==============================================")
    print(f" Starting Experiment: {algo_name}")
    print(f"==============================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using Device for Training: {device} ---")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"*** AKTIF EKRAN KARTI (GPU): {props.name} ***")
        print(f"*** VRAM (BELLEK): {props.total_memory / 1024**3:.2f} GB ***")
        print(f"*** Streaming Multiprocessors (SMs): {props.multi_processor_count} ***")
        print(f"*** PyTorch Cuda Versiyonu: {torch.version.cuda} ***\n")
    
    server = FLServer(
        root_dir="c:/Users/root/project/federetad_learning", 
        partition_json="c:/Users/root/project/federetad_learning/fl_partition.json",
        num_clients=num_clients,
        fraction=fraction,
        device=device,
        local_epochs=3, # Fixed epoch loop in client.py means 3 is plenty
        lr=0.001 # Fine-tuning için daha stabil bir değer
    )
    
    # Run the training and get the loss and accuracy history
    metrics = server.start_training(num_rounds=num_rounds, algo=algo_name)
    
    return metrics

if __name__ == "__main__":
    algorithms_to_test = [
        "fedavg",
        "fedprox",
        "fednova",
        "scaffold",
        "proposed_cosine",
        "proposed_norm",
        "proposed_combined"
    ]
    
    # --- Configuration ---
    rounds = 25 # Hedef doğruluğa ulaşmak için raunt sayısı artırıldı
    num_clients = 50
    fraction = 0.2 # Stability için her rauntta daha fazla client (10 client)
    
    # --- Resume/Skip Logic ---
    results = {}
    for algo in algorithms_to_test:
        algo_json_path = os.path.join("c:/Users/root/project/federetad_learning", f"results_{algo}.json")
        if os.path.exists(algo_json_path):
            try:
                with open(algo_json_path, 'r') as f:
                    results[algo] = json.load(f)
                print(f"--- Loaded existing results for: {algo} ---")
                # If the loaded results don't have enough rounds, we might want to re-run, 
                # but for now we assume they are complete if the file exists.
                continue
            except:
                print(f"--- Error loading {algo_json_path}, will re-run. ---")
            
        metrics = run_experiment(algo, num_rounds=rounds, num_clients=num_clients, fraction=fraction)
        results[algo] = metrics
        
    # Plotting
    plt.figure(figsize=(20, 6))
    
    # 1. Loss Plot
    plt.subplot(1, 3, 1)
    for algo, metrics in results.items():
        plt.plot(range(1, len(metrics["loss"]) + 1), metrics["loss"], marker='o', label=algo)
    plt.title("Federated Face Recognition - Convergence Loss")
    plt.xlabel("Communication Round")
    plt.ylabel("Average Training Loss")
    plt.legend()
    plt.grid(True)
    
    # 2. Error Rate Plot (100 - Accuracy)
    plt.subplot(1, 3, 2)
    for algo, metrics in results.items():
        error_rates = [100.0 - acc for acc in metrics["accuracy"]]
        plt.plot(range(1, len(error_rates) + 1), error_rates, marker='x', label=algo)
    plt.title("Federated Face Recognition - Error Rate (%)")
    plt.xlabel("Communication Round")
    plt.ylabel("Error Rate % (Lower is Better)")
    plt.ylim(0, 100) # Full scale for perspective
    plt.axhline(y=10, color='r', linestyle='--', label='Target Max Error (10%)')
    plt.legend()
    plt.grid(True)
    
    # 3. Drift Analysis Plot
    plt.subplot(1, 3, 3)
    for algo, metrics in results.items():
        if "drift" in metrics:
            plt.plot(range(1, len(metrics["drift"]) + 1), metrics["drift"], marker='s', label=algo)
    plt.title("Federated Learning - Drift Analysis")
    plt.xlabel("Communication Round")
    plt.ylabel("Avg Model Drift (L2 Norm)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join("c:/Users/root/project/federetad_learning", "results_plot.png")
    plt.savefig(plot_path)
    
    # Tüm metrikleri (Loss ve Accuracy) JSON dosyasına kaydetme
    results_json_path = os.path.join("c:/Users/root/project/federetad_learning", "training_results.json")
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nAll experiments finished.")
    print(f"*** Grafikler kaydedildi: {plot_path} ***")
    print(f"*** Tüm sayısal veriler (Loss/Acc) kaydedildi: {results_json_path} ***")
