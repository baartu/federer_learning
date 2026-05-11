import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models import resnet18, ResNet18_Weights

class FaceResNet18(nn.Module):
    """
    Standart ResNet-18 modelinin yüz tanıma (Face Recognition) görevine uyarlanmış hali.
    Son FC katmanı kaldırılarak, ArcFace/CosFace için normalize edilmiş feature (embedding) çıkarır.
    """
    def __init__(self, embedding_size=512, pretrained=True):
        super(FaceResNet18, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.backbone = resnet18(weights=weights)
        else:
            self.backbone = resnet18(weights=None)
            
        # Remove the classification head (FC layer)
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add a custom embedding layer
        self.embedding_layer = nn.Linear(512 * self.backbone.fc.weight.shape[1] // 512, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.embedding_layer(x)
        x = self.bn(x)
        # Normalize the embedding (crucial for ArcFace/CosFace)
        x = F.normalize(x, p=2, dim=1)
        return x

class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss)
    """
    def __init__(self, num_classes, embedding_size=512, margin=0.3, scale=64.0):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # Weight parameter for identities
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        # Normalize weights
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7) # Numerical stability
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Condition to keep phi in range
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encoding for targets
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin to the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        loss = F.cross_entropy(output, labels)
        return loss

class CosFaceLoss(nn.Module):
    """
    CosFace (Large Margin Cosine Loss)
    """
    def __init__(self, num_classes, embedding_size=512, margin=0.35, scale=64.0):
        super(CosFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        # Weight parameter for identities
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize weights
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        
        # One-hot encoding for targets
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply cosine margin to the target class
        output = cosine - (one_hot * self.margin)
        output *= self.scale
        
        loss = F.cross_entropy(output, labels)
        return loss

if __name__ == "__main__":
    # Quick sanity check
    model = FaceResNet18(pretrained=False)
    # Let's say we have 10177 classes in total
    arcface = ArcFaceLoss(num_classes=10177)
    
    dummy_input = torch.randn(4, 3, 112, 112)
    dummy_labels = torch.tensor([0, 1, 2, 3])
    
    embeddings = model(dummy_input)
    loss = arcface(embeddings, dummy_labels)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Initial ArcFace loss: {loss.item():.4f}")
