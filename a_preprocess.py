import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

# return emedding(from my model) and expects file path
def return_embedding(X):
    class siamese_triple(nn.Module):
        def __init__(self, num_feature):
            super().__init__()
            self.siamese_model = nn.Sequential(
                nn.Linear(num_feature, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                # nn.ReLU(),
                # nn.Linear(128, 100)
            )

        def forward(self, X):
            k = self.siamese_model(X)
            k = k / (torch.norm(k, p=2, dim=-1, keepdim=True) + 1e-9)
            return k



    model = siamese_triple(num_feature=512)
    model.load_state_dict(torch.load("siamese_triple.pth", map_location=device,weights_only=True))
    model.to(device)
    model.eval()


    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    embedding_model = nn.Sequential(*list(backbone.children())[:-1])  # remove last FC
    embedding_model.to(device)
    embedding_model.eval()

    # Freeze backbone if you want
    for p in embedding_model.parameters():
        p.requires_grad = False
    fa = embedding_model(X).squeeze(-1).squeeze(-1)
    return model(fa)


preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                            std=[0.229,0.224,0.225])
    ])




def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)  # add batch dimension
    return img
