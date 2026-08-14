"""
generate_augmentations.py

Run this ONCE before training to build the folder structure:

  ukbench_train/
  ├── img_0000/
  │   ├── original.jpg
  │   ├── aug_1.jpg  (crop + JPEG)
  │   ├── aug_2.jpg  (rotation + blur)
  │   ├── aug_3.jpg  (color jitter + flip)
  │   └── aug_4.jpg  (all combined — hardest positive)
  └── ...

  ukbench_val/   (same structure, different images)

Usage:
    python generate_augmentations.py \
        --src "C:/Users/shour/Downloads/ukbench/full" \
        --out_train ukbench_train \
        --out_val   ukbench_val \
        --train_ratio 0.8
"""

import os
import argparse
import random
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ─────────────────────────────────────────────
# Augmentation pipelines (near-duplicate style)
# ─────────────────────────────────────────────

def aug_1(img):
    """Heavy crop + low JPEG quality — simulates stolen/resaved image"""
    w, h = img.size
    # Crop to 65-80% of the image
    scale = random.uniform(0.65, 0.80)
    new_w, new_h = int(w * scale), int(h * scale)
    left = random.randint(0, w - new_w)
    top  = random.randint(0, h - new_h)
    img  = img.crop((left, top, left + new_w, top + new_h))
    img  = img.resize((224, 224), Image.BILINEAR)
    # Re-save at low quality to add JPEG artefacts
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(25, 45))
    buf.seek(0)
    return Image.open(buf).copy()


def aug_2(img):
    """Rotation + Gaussian blur — simulates scanned/photographed copy"""
    angle = random.uniform(10, 50) * random.choice([-1, 1])
    img = TF.rotate(img, angle, expand=False)
    img = T.GaussianBlur(kernel_size=5, sigma=(1.5, 3.0))(img)
    return img


def aug_3(img):
    """Color jitter + horizontal flip — simulates color-graded repost"""
    img = T.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1
    )(img)
    if random.random() > 0.5:
        img = TF.hflip(img)
    return img


def aug_4(img):
    """All transforms chained — hardest positive (simulates professional theft)"""
    img = aug_1(img)   # crop + JPEG
    img = aug_3(img)   # color grade + flip
    # Light blur on top
    img = img.resize((224, 224), Image.BILINEAR)
    img = T.GaussianBlur(kernel_size=3, sigma=(0.8, 1.5))(img)
    return img


AUGMENTATIONS = [aug_1, aug_2, aug_3, aug_4]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def generate(src_dir, out_train, out_val, train_ratio=0.8):
    # Collect all image paths from ukbench/full
    all_images = sorted([
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Found {len(all_images)} images in {src_dir}")

    # We use only the first image of each 4-group as our anchor
    # (ignore the other 3 angles — they are NOT near-duplicates for our task)
    # ukbench images are named ukbench00000 ... ukbench10199
    # Groups of 4: 00000-00003, 00004-00007, ...
    # We take every 4th image (index 0 of each group) as anchor
    anchors = all_images[::1]   # [ukbench00000, ukbench00004, ...]
    print(f"Using {len(anchors)} unique anchor images (1 per 4-group)")

    # Train/val split
    random.shuffle(anchors)
    split = int(len(anchors) * train_ratio)
    train_anchors = anchors[:split]
    val_anchors   = anchors[split:]
    print(f"Train: {len(train_anchors)} | Val: {len(val_anchors)}")

    def process_split(anchor_list, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        for idx, fname in enumerate(anchor_list):
            img_path  = os.path.join(src_dir, fname)
            class_name = f"img_{idx:05d}"
            class_dir  = os.path.join(out_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize((224, 224), Image.BILINEAR)

            # Save original
            img_resized.save(os.path.join(class_dir, "original.jpg"), quality=95)

            # Save 4 augmented versions
            for aug_idx, aug_fn in enumerate(AUGMENTATIONS, start=1):
                aug_img = aug_fn(img.copy())
                aug_img = aug_img.resize((224, 224), Image.BILINEAR)
                aug_img.save(
                    os.path.join(class_dir, f"aug_{aug_idx}.jpg"), quality=90
                )

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx+1}/{len(anchor_list)}")

        print(f"Done → {out_dir}  ({len(anchor_list)} classes × 5 images)")

    process_split(train_anchors, out_train)
    process_split(val_anchors,   out_val)
    print("\nAll done! Now run: python train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",         default=r"C:\Users\shour\Downloads\Images_shourya\Images_shourya")
    parser.add_argument("--out_train",   default="my_train")
    parser.add_argument("--out_val",     default="my_val")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    generate(args.src, args.out_train, args.out_val, args.train_ratio)
