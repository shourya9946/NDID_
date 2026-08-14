
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1. MODEL
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────
# Expected folder structure:
#
#   ukbench_train/
#   ├── img_0000/
#   │   ├── original.jpg
#   │   ├── aug_1.jpg
#   │   ├── aug_2.jpg
#   │   ├── aug_3.jpg
#   │   └── aug_4.jpg
#   ├── img_0001/
#   │   └── ...
#
# Each folder = one class (one unique image + its near-duplicates).
# Run generate_augmentations.py first to create this structure.

class NearDuplicateDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []   # list of (img_path, class_id)

        classes = sorted(os.listdir(root_dir))
        for class_id, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), class_id)
                    )

        print(f"Dataset loaded: {len(self.samples)} images, "
            f"{len(classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────
# 3. TRANSFORMS
# ─────────────────────────────────────────────

# Training: standard ImageNet normalize only
# Augmentations were already applied when building the dataset folders.
# If you want on-the-fly augmentation instead, swap train_transform below.
train_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# 4. TRAINING CONFIG
# ─────────────────────────────────────────────

CONFIG = {
    "train_dir":     "ukbench_train",   # folder with class subfolders
    "val_dir":       "ukbench_val",
    "embedding_dim": 128,
    "batch_size":    16,   # reduced for 4GB GPU (was 64)
    "num_epochs":    50,
    "margin":        0.2,
    "checkpoint_dir": "checkpoints",
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"Using device: {CONFIG['device']}")


# ─────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, miner, loss_fn, device):
    model.train()
    total_loss = 0
    total_triplets = 0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device)

        embeddings = model(imgs)

        # Online mining — finds hard/semi-hard triplets in this batch
        hard_pairs = miner(embeddings, labels)
        num_triplets = hard_pairs[0].numel()

        if num_triplets == 0:
            continue  # skip batches with no valid triplets

        loss = loss_fn(embeddings, labels, hard_pairs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss     += loss.item()
        total_triplets += num_triplets

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, total_triplets


@torch.no_grad()
def validate(model, loader, device):
    """
    Simple validation: for each image compute embedding,
    then check cosine similarity between same-class pairs.
    Reports mean intra-class sim and mean inter-class sim.
    A good model has high intra, low inter.
    """
    model.eval()
    all_embeddings = []
    all_labels     = []

    for imgs, labels in tqdm(loader, desc="Val  ", leave=False):
        emb = model(imgs.to(device))
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings)   # (N, 128)
    all_labels     = torch.cat(all_labels)       # (N,)

    # Cosine similarity matrix (already L2-normalised, so dot product = cosine)
    sim_matrix = all_embeddings @ all_embeddings.T  # (N, N)

    same_mask = (all_labels.unsqueeze(0) == all_labels.unsqueeze(1))
    # Exclude diagonal (self-similarity)
    diag_mask = torch.eye(len(all_labels), dtype=torch.bool)
    same_mask &= ~diag_mask
    diff_mask  = ~same_mask & ~diag_mask

    intra_sim = sim_matrix[same_mask].mean().item()
    inter_sim = sim_matrix[diff_mask].mean().item()

    return intra_sim, inter_sim


def train():
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # ── Datasets & loaders ──────────────────────────────────────────────
    train_ds = NearDuplicateDataset(CONFIG["train_dir"], train_transform)
    val_ds   = NearDuplicateDataset(CONFIG["val_dir"],   val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,        # reduced — high num_workers eats RAM
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────
    device = CONFIG["device"]

    # Free any leftover GPU memory before starting
    if device == "cuda":
        torch.cuda.empty_cache()

    model  = NearDuplicateModel(embedding_dim=CONFIG["embedding_dim"]).to(device)

    # ── Optimizer with differential LR ───────────────────────────────────
    # Backbone (layer3, layer4) gets smaller LR than the projection head
    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(),   "lr": 1e-5},
        {"params": model.pool.parameters(),       "lr": 1e-4},
        {"params": model.projection.parameters(), "lr": 1e-3},
    ])

    # Cosine annealing: LR smoothly decays to near-zero by last epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-6
    )

    # ── Loss & Miner ─────────────────────────────────────────────────────
    # Phase 1 (epochs 1-10):  semi-hard — safer, prevents collapse
    # Phase 2 (epochs 11+):   hard      — more aggressive, better final model
    miner_semihard = miners.TripletMarginMiner(
        margin=CONFIG["margin"], type_of_triplets="semihard"
    )
    miner_hard = miners.TripletMarginMiner(
        margin=CONFIG["margin"], type_of_triplets="hard"
    )
    loss_fn = losses.TripletMarginLoss(margin=CONFIG["margin"])

    # ── Training ─────────────────────────────────────────────────────────
    history = {"train_loss": [], "intra_sim": [], "inter_sim": []}
    best_score = -1.0

    for epoch in range(1, CONFIG["num_epochs"] + 1):

        # Switch miner strategy after epoch 10
        active_miner = miner_semihard if epoch <= 10 else miner_hard
        phase = "semi-hard" if epoch <= 10 else "hard"

        train_loss, n_triplets = train_one_epoch(
            model, train_loader, optimizer, active_miner, loss_fn, device
        )
        intra_sim, inter_sim = validate(model, val_loader, device)
        scheduler.step()

        # Score = gap between intra and inter similarity (higher = better)
        gap = intra_sim - inter_sim
        history["train_loss"].append(train_loss)
        history["intra_sim"].append(intra_sim)
        history["inter_sim"].append(inter_sim)

        print(
            f"Epoch {epoch:02d}/{CONFIG['num_epochs']} "
            f"[{phase:9s}] "
            f"loss={train_loss:.4f}  "
            f"triplets={n_triplets:6d}  "
            f"intra={intra_sim:.4f}  inter={inter_sim:.4f}  "
            f"gap={gap:.4f}"
        )

        # Save best checkpoint
        if gap > best_score:
            best_score = gap
            ckpt_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "gap":         gap,
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (gap={gap:.4f})")

    # ── Plot training curves ──────────────────────────────────────────────
    epochs = range(1, CONFIG["num_epochs"] + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Triplet Loss"); ax1.legend()

    ax2.plot(epochs, history["intra_sim"], label="Intra-class sim ↑")
    ax2.plot(epochs, history["inter_sim"], label="Inter-class sim ↓")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Cosine similarity")
    ax2.set_title("Embedding quality"); ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Training curves saved to training_curves.png")
    plt.show()


if __name__ == "__main__":
    train()
