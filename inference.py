"""
inference.py

Load trained model, build an embedding database from a folder of images,
then query it with a new image to find near-duplicates.

Usage:
    python inference.py --db_dir ukbench_val \
                        --query  path/to/query.jpg \
                        --checkpoint checkpoints/best_model.pth \
                        --threshold 0.85
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np


# ── Copy model definition here (same as train.py) ───────────────────────────

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
        self.backbone   = nn.Sequential(*list(resnet.children())[:-2])
        self.pool       = GeMPooling(p=3)
        self.projection = nn.Sequential(
            nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),  nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, embedding_dim), nn.BatchNorm1d(embedding_dim),
        )
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)


# ── Helpers ──────────────────────────────────────────────────────────────────

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path, device):
    model = NearDuplicateModel().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(gap={ckpt['gap']:.4f})")
    return model


@torch.no_grad()
def embed_image(model, img_path, device):
    img = Image.open(img_path).convert("RGB")
    x   = transform(img).unsqueeze(0).to(device)
    return model(x).squeeze(0).cpu()


@torch.no_grad()
def build_database(model, db_dir, device):
    """Walk db_dir, embed every image, return (embeddings, paths)."""
    paths      = []
    embeddings = []

    for root, _, files in os.walk(db_dir):
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                p   = os.path.join(root, f)
                emb = embed_image(model, p, device)
                paths.append(p)
                embeddings.append(emb)

    embeddings = torch.stack(embeddings)   # (N, 128)
    print(f"Database built: {len(paths)} images")
    return embeddings, paths


def query(model, query_path, db_embeddings, db_paths, threshold, device, top_k=5):
    query_emb = embed_image(model, query_path, device)

    # Cosine similarity (embeddings are L2-normalised → dot product = cosine)
    sims = db_embeddings @ query_emb   # (N,)

    # Sort by similarity descending
    ranked = torch.argsort(sims, descending=True)

    print(f"\nQuery: {query_path}")
    print(f"{'Rank':<6} {'Similarity':<12} {'Near-dup?':<12} Path")
    print("─" * 80)

    results = []
    for rank, idx in enumerate(ranked[:top_k], start=1):
        sim   = sims[idx].item()
        is_dup = sim >= threshold
        flag  = "✓ DUPLICATE" if is_dup else "✗"
        print(f"{rank:<6} {sim:<12.4f} {flag:<12} {db_paths[idx]}")
        results.append((db_paths[idx], sim, is_dup))

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_dir",     required=True,  help="Folder of DB images")
    parser.add_argument("--query",      required=True,  help="Query image path")
    parser.add_argument("--checkpoint", required=True,  help="Path to best_model.pth")
    parser.add_argument("--threshold",  type=float, default=0.85,
                        help="Cosine similarity threshold (0-1)")
    parser.add_argument("--top_k",      type=int,   default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(args.checkpoint, device)

    db_embeddings, db_paths = build_database(model, args.db_dir, device)
    query(model, args.query, db_embeddings, db_paths,
          args.threshold, device, args.top_k)


if __name__ == "__main__":
    main()
