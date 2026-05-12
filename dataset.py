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
                transforms.RandomRotation(15), # Yüz açılarındaki küçük sapmalar için
                transforms.RandomGrayscale(p=0.1), # Işık değişimlerine direnç için
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
            
        # Cache identity mapping at class level to avoid re-reading 200k lines for 50 clients
        if not hasattr(CelebA_IdentityBased_Dataset, '_global_img_to_id'):
            self._load_global_mappings(os.path.join(root_dir, "identity_CelebA.txt"))
            
        if split == "train":
            if client_id is None:
                raise ValueError("client_id must be provided for train split")
            if client_id not in self.partition_data["train_clients"]:
                raise ValueError(f"Unknown client_id {client_id}")
                
            self.images = self.partition_data["train_clients"][client_id]["images"]
        elif split == "test":
            self.images = self.partition_data["test_unseen"]["images"]
        
        self.labels = [CelebA_IdentityBased_Dataset._global_id_to_int[CelebA_IdentityBased_Dataset._global_img_to_id[img]] for img in self.images]

    def _load_global_mappings(self, identity_file):
        print(f"--- Parsing identity file (Once): {identity_file} ---")
        CelebA_IdentityBased_Dataset._global_img_to_id = {}
        CelebA_IdentityBased_Dataset._global_id_to_int = {}
        curr_label = 0
        
        with open(identity_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_name, person_id = parts
                    CelebA_IdentityBased_Dataset._global_img_to_id[img_name] = person_id
                    if person_id not in CelebA_IdentityBased_Dataset._global_id_to_int:
                        CelebA_IdentityBased_Dataset._global_id_to_int[person_id] = curr_label
                        curr_label += 1
        print(f"--- Global identity mapping built: {len(CelebA_IdentityBased_Dataset._global_id_to_int)} classes ---")

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
