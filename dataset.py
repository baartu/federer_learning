import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CelebA_IdentityBased_Dataset(Dataset):
    """
    Kişi bazlı (identity-based) ayarlanmış Extreme Non-IID CelebA veri setini
    yüklemek için kullanılan PyTorch Dataset sınıfıdır.
    """
    def __init__(self, root_dir, partition_json_path, split="train", client_id=None, transform=None):
        """
        Args:
            root_dir (str): `img_align_celeba` klasörünün bulunduğu ana dizin.
            partition_json_path (str): partition_data.py çıktısı olan json.
            split (str): "train" veya "test". "train" ise client_id verilmelidir.
                         "test" ise %20'lik sıfır-atış (unseen) kimlikler yüklenir.
            client_id (str): "client_0", "client_1" ... formatında. (Sadece train için)
            transform: Uygulanacak torchvision augmentasyon/transform işlemleri.
        """
        self.root_dir = root_dir
        # ZIP dosyasının iç içe çıktığı anlaşıldı (img_align_celeba/img_align_celeba kalıbı)
        self.img_dir = os.path.join(root_dir, "img_align_celeba", "img_align_celeba")
        if split == "train":
            self.transform = transform if transform else transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform if transform else transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
        with open(partition_json_path, 'r') as f:
            self.partition_data = json.load(f)
            
        self.images = []
        self.labels = []
        
        # We need a global mapping of identity to standard integer class labels
        # because the face recognition loss usually needs integer class numbers [0, num_classes-1]
        self._build_class_mapping()
        
        if split == "train":
            if client_id is None:
                raise ValueError("client_id must be provided for train split")
            if client_id not in self.partition_data["train_clients"]:
                raise ValueError(f"Unknown client_id {client_id}")
                
            client_data = self.partition_data["train_clients"][client_id]
            self.images = client_data["images"]
            # To get labels, we need to map the image back to its identity.
            # We can parse it from identity_CelebA.txt if needed, but since we parsed it
            # once, let's read the identity file quickly to get per-image id
            self._load_image_to_id(os.path.join(root_dir, "identity_CelebA.txt"))
            
        elif split == "test":
            self.images = self.partition_data["test_unseen"]["images"]
            self._load_image_to_id(os.path.join(root_dir, "identity_CelebA.txt"))
            
        else:
            raise ValueError("split must be 'train' or 'test'")

    def _build_class_mapping(self):
        # We assign an integer index [0, 10176] to each unique identity ID
        # Since identity IDs in CelebA are 1-10177, we could just subtract 1, 
        # but let's be safe and map them dynamically.
        pass # Will implement inside _load_image_to_id to ensure consistency

    def _load_image_to_id(self, identity_file):
        self.img_to_id = {}
        self.id_to_int_label = {}
        curr_label = 0
        
        with open(identity_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_name, person_id = parts
                    self.img_to_id[img_name] = person_id
                    if person_id not in self.id_to_int_label:
                        self.id_to_int_label[person_id] = curr_label
                        curr_label += 1
                        
        # Now populate self.labels based on self.images
        self.labels = [self.id_to_int_label[self.img_to_id[img]] for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Handle potential unzipping/image reading errors
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (112, 112), color='black')
            
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

# Example usage test
if __name__ == "__main__":
    test_ds = CelebA_IdentityBased_Dataset(
        root_dir="c:/Users/root/project/federetad_learning",
        partition_json_path="c:/Users/root/project/federetad_learning/fl_partition.json",
        split="train",
        client_id="client_0"
    )
    print(f"Client 0 has {len(test_ds)} images.")
    img, lbl = test_ds[0]
    print(f"Sample image shape: {img.shape}, Label (int mapped): {lbl}")

    test_unseen = CelebA_IdentityBased_Dataset(
        root_dir="c:/Users/root/project/federetad_learning",
        partition_json_path="c:/Users/root/project/federetad_learning/fl_partition.json",
        split="test"
    )
    print(f"Total unseen test images: {len(test_unseen)}")
