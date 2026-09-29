"""
MOT17-05-FRCNN → YOLOv8 Dataset Preparation Script
====================================================
IBVAP Stage 1: Baseline person detector dataset preparation.

Converts MOT17 ground-truth annotations to YOLO format,
performs a chronological 70/30 train/val split, copies images,
preserves tracking ground truth, and validates the result.

Author: IBVAP Pipeline
Date: 2026-09-29
"""

import os
import shutil
import math
from pathlib import Path

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

SRC_ROOT = Path(r"C:\Users\Atharva\Downloads\SeemaDrishti\MOT17\MOT17\train\MOT17-05-FRCNN")
DST_ROOT = Path(r"C:\Users\Atharva\Downloads\SeemaDrishti\IBVAP\datasets\MOT17_05_YOLO")

IMG_DIR = SRC_ROOT / "img1"
GT_FILE = SRC_ROOT / "gt" / "gt.txt"

IMG_WIDTH = 640
IMG_HEIGHT = 480
TOTAL_FRAMES = 837
TRAIN_RATIO = 0.70

# Chronological split
TRAIN_END = math.floor(TOTAL_FRAMES * TRAIN_RATIO)  # frame 585
VAL_START = TRAIN_END + 1                             # frame 586

print("=" * 65)
print("MOT17-05-FRCNN → YOLO Dataset Preparation")
print("=" * 65)
print(f"Train frames: 1 – {TRAIN_END}  ({TRAIN_END} frames)")
print(f"Val   frames: {VAL_START} – {TOTAL_FRAMES}  ({TOTAL_FRAMES - TRAIN_END} frames)")
print()

# ─── STEP 1: Parse GT and convert to YOLO format ────────────────────────────

print("[1/5] Parsing MOT ground truth and converting to YOLO format...")

# Data structures: frame_id → list of YOLO label strings
yolo_labels = {}          # frame_id → ["0 x_c y_c w h", ...]
tracking_annotations = [] # preserve original lines for tracking GT

valid_count = 0
skipped_count = 0

with open(GT_FILE, "r") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 9:
            print(f"  WARNING: Malformed line {line_num}: {line}")
            skipped_count += 1
            continue

        frame_id = int(parts[0])
        track_id = int(parts[1])
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        mark = int(parts[6])
        cls = int(parts[7])
        visibility = float(parts[8])

        # Keep only valid pedestrian annotations (class=1, mark=1)
        if cls != 1 or mark != 1:
            skipped_count += 1
            continue

        # Convert MOT (x, y, w, h) → YOLO normalized (x_center, y_center, w, h)
        x_center = (x + w / 2.0) / IMG_WIDTH
        y_center = (y + h / 2.0) / IMG_HEIGHT
        w_norm = w / IMG_WIDTH
        h_norm = h / IMG_HEIGHT

        # Clamp to [0, 1] — handle any edge annotations
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))

        # Skip zero-area boxes
        if w_norm <= 0 or h_norm <= 0:
            skipped_count += 1
            continue

        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

        if frame_id not in yolo_labels:
            yolo_labels[frame_id] = []
        yolo_labels[frame_id].append(yolo_line)
        valid_count += 1

print(f"  Valid pedestrian annotations converted: {valid_count}")
print(f"  Skipped annotations (non-pedestrian/invalid): {skipped_count}")
print(f"  Frames with labels: {len(yolo_labels)}")
print()

# ─── STEP 2: Copy images and write labels ───────────────────────────────────

print("[2/5] Copying images and writing YOLO label files...")

train_img_count = 0
val_img_count = 0
train_label_instances = 0
val_label_instances = 0

for frame_id in range(1, TOTAL_FRAMES + 1):
    img_name = f"{frame_id:06d}.jpg"
    label_name = f"{frame_id:06d}.txt"

    src_img = IMG_DIR / img_name

    if not src_img.exists():
        print(f"  WARNING: Missing source image: {src_img}")
        continue

    # Determine split
    if frame_id <= TRAIN_END:
        split = "train"
        train_img_count += 1
    else:
        split = "val"
        val_img_count += 1

    # Copy image
    dst_img = DST_ROOT / "images" / split / img_name
    shutil.copy2(src_img, dst_img)

    # Write label file
    dst_label = DST_ROOT / "labels" / split / label_name
    labels = yolo_labels.get(frame_id, [])
    with open(dst_label, "w") as lf:
        lf.write("\n".join(labels))
        if labels:
            lf.write("\n")

    instance_count = len(labels)
    if frame_id <= TRAIN_END:
        train_label_instances += instance_count
    else:
        val_label_instances += instance_count

print(f"  Train: {train_img_count} images, {train_label_instances} instances")
print(f"  Val:   {val_img_count} images, {val_label_instances} instances")
print()

# ─── STEP 3: Copy tracking ground truth ─────────────────────────────────────

print("[3/5] Copying tracking ground truth...")

tracking_dst = DST_ROOT / "tracking_gt" / "MOT17-05-FRCNN_gt.txt"
shutil.copy2(GT_FILE, tracking_dst)
print(f"  Copied to: {tracking_dst}")
print()

# ─── STEP 4: Create data.yaml ───────────────────────────────────────────────

print("[4/5] Creating data.yaml...")

data_yaml_path = DST_ROOT / "data.yaml"
data_yaml_content = f"""# YOLOv8 Dataset Configuration
# IBVAP Stage 1 — MOT17-05-FRCNN Baseline
# Generated: 2026-09-29

path: {DST_ROOT.as_posix()}
train: images/train
val: images/val

names:
  0: person

# Dataset info:
# Source: MOT17-05-FRCNN
# Resolution: {IMG_WIDTH}x{IMG_HEIGHT}
# Train frames: 1–{TRAIN_END} ({train_img_count} frames)
# Val frames: {VAL_START}–{TOTAL_FRAMES} ({val_img_count} frames)
# Total pedestrian instances: {valid_count}
# Split method: chronological 70/30
"""

with open(data_yaml_path, "w") as f:
    f.write(data_yaml_content)

print(f"  Written to: {data_yaml_path}")
print()

# ─── STEP 5: Validate dataset ───────────────────────────────────────────────

print("[5/5] Validating dataset...")
print()

errors = []
warnings = []
total_instances = 0
invalid_labels = 0
missing_labels = 0
missing_images = 0

for split in ["train", "val"]:
    img_dir = DST_ROOT / "images" / split
    lbl_dir = DST_ROOT / "labels" / split

    images = sorted(img_dir.glob("*.jpg"))
    labels = sorted(lbl_dir.glob("*.txt"))

    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}

    # Check for labels without images
    for stem in lbl_stems - img_stems:
        errors.append(f"[{split}] Label without image: {stem}")
        missing_images += 1

    # Check for images without labels
    for stem in img_stems - lbl_stems:
        warnings.append(f"[{split}] Image without label: {stem}")
        missing_labels += 1

    # Validate each label file
    for lbl_path in labels:
        with open(lbl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"[{split}] {lbl_path.name}:{line_num} malformed ({len(parts)} fields)")
                    invalid_labels += 1
                    continue

                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    errors.append(f"[{split}] {lbl_path.name}:{line_num} non-numeric values")
                    invalid_labels += 1
                    continue

                if cls_id != 0:
                    errors.append(f"[{split}] {lbl_path.name}:{line_num} unexpected class {cls_id}")
                    invalid_labels += 1

                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    errors.append(f"[{split}] {lbl_path.name}:{line_num} out-of-range bbox: {xc:.4f} {yc:.4f} {w:.4f} {h:.4f}")
                    invalid_labels += 1

                total_instances += 1

# ─── VALIDATION REPORT ──────────────────────────────────────────────────────

print("=" * 50)
print("         DATASET VALIDATION REPORT")
print("=" * 50)
print(f"  Total images:             {train_img_count + val_img_count}")
print(f"  Train images:             {train_img_count}")
print(f"  Validation images:        {val_img_count}")
print(f"  Train label instances:    {train_label_instances}")
print(f"  Val label instances:      {val_label_instances}")
print(f"  Total instances verified: {total_instances}")
print(f"  Invalid labels:           {invalid_labels}")
print(f"  Missing labels:           {missing_labels}")
print(f"  Missing images:           {missing_images}")
print(f"  Errors:                   {len(errors)}")
print(f"  Warnings:                 {len(warnings)}")
print("=" * 50)

if errors:
    print("\nERRORS:")
    for e in errors[:20]:
        print(f"  ✗ {e}")
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more errors")

if warnings:
    print("\nWARNINGS:")
    for w in warnings[:10]:
        print(f"  ⚠ {w}")

if not errors and not warnings:
    print("\n✓ Dataset is VALID and ready for YOLOv8 training.")
else:
    if not errors:
        print("\n✓ No errors found. Dataset is usable (with warnings noted).")
    else:
        print("\n✗ Dataset has errors that should be fixed before training.")

print()
print("Done.")
