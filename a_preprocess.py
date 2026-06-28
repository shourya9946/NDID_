import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
# import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.models import ResNet50_Weights


device = "cuda" if torch.cuda.is_available() else "cpu"

# return emedding(from my model) and expects file path
class GeMPooling(nn.Module):
    def __init__(self, p=3):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)

    def forward(self, x):
        return F.adaptive_avg_pool2d(
            x.clamp(min=1e-6).pow(self.p), (1, 1)
        ).pow(1.0 / self.p)


class NearDuplicateModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove avgpool and FC — keep feature map output
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze layer1, layer2, layer3 — only fine-tune layer4
        # This saves significant GPU memory on a 4GB card
        for name, param in self.backbone.named_parameters():
            if "layer1" in name or "layer2" in name or "layer3" in name:
                param.requires_grad = False

        self.pool = GeMPooling(p=3)

        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        x = self.backbone(x)        # (B, 2048, 7, 7)
        x = self.pool(x)            # (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 2048)
        x = self.projection(x)      # (B, 128)
        x = F.normalize(x, p=2, dim=1)  # unit hypersphere
        return x


model = NearDuplicateModel(embedding_dim=128).to(device)

checkpoint = torch.load(
    "best_model.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state"])
model.eval()



# Freeze backbone
# for p in embedding_model.parameters():
#     p.requires_grad = False
# fa = embedding_model(X).squeeze(-1).squeeze(-1)
# return model(fa)

@torch.no_grad()
def get_embedding(img_path):

    img = Image.open(img_path).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)

    embedding = model(x)

    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze(0)


transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)  # add batch dimension
    return img



# def return_embedding(X):
#     fa = embedding_model(X).squeeze(-1).squeeze(-1)
#     return model(fa)