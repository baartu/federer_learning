import os
import random
import json
import numpy as np

def create_partition():
    identity_file = os.path.join("c:/Users/root/project/federetad_learning", "identity_CelebA.txt")
    output_meta = os.path.join("c:/Users/root/project/federetad_learning", "fl_partition.json")
    
    if not os.path.exists(identity_file):
        print(f"Error: {identity_file} not found.")
        return
        
    print("Reading identities...")
    id_to_images = {}
    with open(identity_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_name, person_id = parts
                if person_id not in id_to_images:
                    id_to_images[person_id] = []
                id_to_images[person_id].append(img_name)
                
    all_identities = list(id_to_images.keys())
    print(f"Total unique identities: {len(all_identities)}")
    
    # Shuffle for random train/test split
    random.seed(42)
    np.random.seed(42)
    random.shuffle(all_identities)
    
    num_test = int(0.20 * len(all_identities))
    num_train = len(all_identities) - num_test
    
    test_identities = all_identities[:num_test]
    train_identities = all_identities[num_test:]
    
    print(f"Train identities (80%): {len(train_identities)}")
    print(f"Test identities (20% unseen): {len(test_identities)}")
    
    # Extreme non-IID: 50 clients. No identity overlap.
    num_clients = 50
    client_identities = [[] for _ in range(num_clients)]
    
    # To make it "unbalanced", we use a Dirichlet distribution to determine
    # the proportions of identities assigned to each client.
    diri_alpha = 1.0 # A smaller alpha creates a more unbalanced distribution
    proportions = np.random.dirichlet([diri_alpha] * num_clients)
    
    # Convert proportions to actual counts of identities per client
    identity_counts = (proportions * len(train_identities)).astype(int)
    
    # Ensure every client has at least 1 identity
    identity_counts[identity_counts == 0] = 1
    
    # Fix rounding errors to ensure all identities are allocated
    diff = len(train_identities) - identity_counts.sum()
    if diff > 0:
        for _ in range(diff):
            identity_counts[np.random.randint(num_clients)] += 1
    elif diff < 0:
        for _ in range(abs(diff)):
            max_idx = np.argmax(identity_counts)
            identity_counts[max_idx] -= 1
        
    # Assign identities to clients
    idx = 0
    for client_id in range(num_clients):
        count = identity_counts[client_id]
        client_identities[client_id] = train_identities[idx : idx + count]
        idx += count
        
    # Create final mapping structure
    partition_data = {
        "train_clients": {},
        "test_unseen": {
            "identities": test_identities,
            "images": []
        }
    }
    
    # Populate test images
    for pid in test_identities:
        partition_data["test_unseen"]["images"].extend(id_to_images[pid])
        
    # Populate client train images
    total_train_images = 0
    for client_id in range(num_clients):
        client_imgs = []
        client_ids = client_identities[client_id]
        for pid in client_ids:
            client_imgs.extend(id_to_images[pid])
            
        partition_data["train_clients"][f"client_{client_id}"] = {
            "identities": client_ids,
            "images": client_imgs
        }
        total_train_images += len(client_imgs)
        print(f"Client {client_id}: {len(client_ids)} identities, {len(client_imgs)} images")
        
    total_test_images = len(partition_data["test_unseen"]["images"])
    print(f"Total train images across clients: {total_train_images}")
    print(f"Total test (unseen) images: {total_test_images}")
    
    # Save the output
    print(f"Saving partition metadata to {output_meta}...")
    with open(output_meta, 'w') as f:
        json.dump(partition_data, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    create_partition()
