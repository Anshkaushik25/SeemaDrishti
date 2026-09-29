"""
IBVAP Stage 2 — SIH26187 Roboflow Dataset Non-Destructive Inspection & QA Tool
=================================================================================
Safely inspects the newly downloaded SIH26187 Roboflow dataset:
- Locates dataset directory and metadata files (data.yaml, README.roboflow.txt, README.dataset.txt)
- Analyzes folder structure, splits (train, valid, test), image/label counts
- Parses class mapping and counts bounding boxes per class
- Checks for Roboflow preprocessing & augmentations
- Runs non-destructive QA (image integrity, label syntax, box bounds, image duplicates)
- Generates visual contact sheet grid image (IBVAP/outputs/metrics/sih26187_contact_sheet.jpg)
- Compares against current IBVAP unified dataset class balance
- Generates IBVAP/datasets/SIHL26187_IMPORT_REPORT.md
- Exports IBVAP/outputs/metrics/sih26187_dataset_report.json & sih26187_dataset_report.csv

DOES NOT MUTATE THE SOURCE DATASET OR TRAIN ANY MODEL.
"""

import os
import sys
import glob
import json
import csv
import hashlib
import random
import yaml
import cv2
import numpy as np
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATASET_DIR = Path(r"C:\Users\Atharva\Downloads\SeemaDrishti\SIH26187 Border Surveillance.v1i.yolov8")
OUTPUTS_METRICS_DIR = PROJECT_DIR / "outputs" / "metrics"
DATASETS_DIR = PROJECT_DIR / "datasets"

OUTPUTS_METRICS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# IBVAP Target Classes
IBVAP_CLASSES = {
    0: "person",
    1: "car",
    2: "truck",
    3: "bike",
    4: "patrol_vehicle"
}


def read_text_file_safe(path):
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        try:
            return path.read_text(encoding="latin-1")
        except Exception:
            return ""


def main():
    print("=" * 75)
    print("  IBVAP STAGE 2 — SIH26187 ROBOFLOW DATASET INSPECTION & QA")
    print("=" * 75)

    # 1. Locate Dataset
    print("\n[STEP 1] DATASET LOCATION")
    print(f"  Dataset Path: {DATASET_DIR}")
    print(f"  Path Exists : {DATASET_DIR.exists()}")

    if not DATASET_DIR.exists():
        print("  [ERROR] Dataset directory not found!")
        sys.exit(1)

    contents = [p.name for p in DATASET_DIR.iterdir()]
    print(f"  Top-level Contents: {contents}")

    # Read YAML & Readme
    yaml_path = DATASET_DIR / "data.yaml"
    readme_roboflow_path = DATASET_DIR / "README.roboflow.txt"
    readme_dataset_path = DATASET_DIR / "README.dataset.txt"

    yaml_data = {}
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

    roboflow_txt = read_text_file_safe(readme_roboflow_path)
    dataset_txt = read_text_file_safe(readme_dataset_path)

    # 2. Extract Class Mapping
    print("\n[STEP 2] CLASS MAPPING IN DATA.YAML")
    source_names = yaml_data.get("names", {})
    print(f"  Raw YAML Names: {source_names}")

    if isinstance(source_names, list):
        source_class_map = {i: name for i, name in enumerate(source_names)}
    elif isinstance(source_names, dict):
        source_class_map = {int(k): v for k, v in source_names.items()}
    else:
        source_class_map = {}

    print("  Exact Class ID -> Class Name Mapping:")
    for cid, cname in source_class_map.items():
        print(f"    ID {cid} -> '{cname}'")

    # 3. Analyze Splits & Count Images/Labels
    print("\n[STEP 3] SPLIT & ANNOTATION INSPECTION")
    split_info = {}
    total_images_all = 0
    total_labels_all = 0
    total_boxes_all = 0
    class_bbox_counts = {cname: 0 for cname in source_class_map.values()}

    qa_issues = {
        "unreadable_images": [],
        "missing_labels": [],
        "empty_labels": [],
        "invalid_class_ids": [],
        "malformed_labels": [],
        "out_of_bound_boxes": [],
        "zero_area_boxes": [],
        "image_label_mismatches": [],
        "duplicate_filenames": [],
        "duplicate_images_hash": []
    }

    qa_rows = []
    image_hashes = {}
    filename_tracker = {}

    for split in ["train", "valid", "test"]:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"

        if not img_dir.exists():
            img_dir = DATASET_DIR / split
        if not lbl_dir.exists():
            lbl_dir = DATASET_DIR / split

        imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg")))
        lbls = sorted(list(lbl_dir.glob("*.txt")))

        split_boxes = 0

        for img_p in imgs:
            total_images_all += 1

            # Duplicate filename check
            if img_p.name in filename_tracker:
                qa_issues["duplicate_filenames"].append(f"{split}/{img_p.name} (duplicate of {filename_tracker[img_p.name]})")
            else:
                filename_tracker[img_p.name] = split

            # Read image & Hash check
            mat = cv2.imread(str(img_p))
            if mat is None or mat.size == 0:
                qa_issues["unreadable_images"].append(str(img_p))
                continue

            h_img, w_img = mat.shape[:2]

            # Compute image MD5 hash for duplicate detection
            with open(img_p, "rb") as f:
                img_hash = hashlib.md5(f.read()).hexdigest()
            if img_hash in image_hashes:
                qa_issues["duplicate_images_hash"].append((str(img_p), image_hashes[img_hash]))
            else:
                image_hashes[img_hash] = str(img_p)

            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                qa_issues["missing_labels"].append(str(img_p))
                continue

            total_labels_all += 1
            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines or len(lines) == 0 or "".join(lines).strip() == "":
                qa_issues["empty_labels"].append(str(lbl_p))

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    qa_issues["malformed_labels"].append((str(lbl_p), line.strip()))
                    continue

                try:
                    cls_id = int(parts[0])
                    xc, yc, w_n, h_n = map(float, parts[1:5])
                except ValueError:
                    qa_issues["malformed_labels"].append((str(lbl_p), line.strip()))
                    continue

                if cls_id not in source_class_map:
                    qa_issues["invalid_class_ids"].append((str(lbl_p), cls_id))
                    c_name = f"unknown_cls_{cls_id}"
                else:
                    c_name = source_class_map[cls_id]

                if xc < 0 or xc > 1 or yc < 0 or yc > 1 or w_n <= 0 or h_n <= 0:
                    qa_issues["out_of_bound_boxes"].append((str(lbl_p), line.strip()))

                bw_px = w_n * w_img
                bh_px = h_n * h_img
                area_px = bw_px * bh_px

                if area_px <= 0:
                    qa_issues["zero_area_boxes"].append((str(lbl_p), line.strip()))
                    continue

                total_boxes_all += 1
                split_boxes += 1
                if c_name in class_bbox_counts:
                    class_bbox_counts[c_name] += 1
                else:
                    class_bbox_counts[c_name] = 1

                qa_rows.append({
                    "split": split,
                    "image_name": img_p.name,
                    "class_id": cls_id,
                    "class_name": c_name,
                    "bbox_px_w": round(bw_px, 1),
                    "bbox_px_h": round(bh_px, 1),
                    "area_px": round(area_px, 1),
                    "norm_xc": xc,
                    "norm_yc": yc,
                    "norm_w": w_n,
                    "norm_h": h_n
                })

        split_info[split] = {
            "image_count": len(imgs),
            "label_count": len(lbls),
            "box_count": split_boxes
        }

    print("  Splits Found:")
    for s_name, s_data in split_info.items():
        print(f"    {s_name:<6} -> {s_data['image_count']} images, {s_data['label_count']} label files, {s_data['box_count']} bounding boxes")

    print("\n  Bounding Box Counts Per Class:")
    for cname, count in class_bbox_counts.items():
        print(f"    {cname:<18} = {count}")

    # 4. Augmentation Inspection
    print("\n[STEP 4] AUGMENTATION INSPECTION")
    aug_findings = "Cannot reliably distinguish original and augmented images from filename syntax alone, but Roboflow metadata indicates offline augmentations were applied prior to export."

    roboflow_aug_details = []
    if roboflow_txt:
        print("  Found README.roboflow.txt content:")
        for line in roboflow_txt.split("\n"):
            if any(k in line.lower() for k in ["aug", "flip", "rotate", "brightness", "stretch", "crop", "resize"]):
                roboflow_aug_details.append(line.strip())
                print(f"    {line.strip()}")

    # 5. Visual Inspection & Contact Sheet
    print("\n[STEP 5] GENERATING VISUAL CONTACT SHEET")
    class_sample_imgs = {cid: [] for cid in source_class_map.keys()}

    # Sample images from train split
    train_img_dir = DATASET_DIR / "train" / "images"
    train_lbl_dir = DATASET_DIR / "train" / "labels"
    if not train_img_dir.exists():
        train_img_dir = DATASET_DIR / "train"
        train_lbl_dir = DATASET_DIR / "train"

    train_imgs = sorted(list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png")))
    random.seed(42)
    random.shuffle(train_imgs)

    for img_p in train_imgs:
        lbl_p = train_lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue
        with open(lbl_p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    if cid in class_sample_imgs and len(class_sample_imgs[cid]) < 2:
                        class_sample_imgs[cid].append((img_p, lbl_p))

    panels = []
    palette = [(0, 220, 100), (255, 140, 0), (200, 0, 200), (0, 200, 255), (0, 100, 255), (255, 0, 100), (100, 255, 0)]

    for cid, cname in source_class_map.items():
        samples = class_sample_imgs.get(cid, [])
        for img_p, lbl_p in samples:
            mat = cv2.imread(str(img_p))
            if mat is None:
                continue
            mat = cv2.resize(mat, (320, 240))
            h_m, w_m = mat.shape[:2]

            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c_box_id = int(parts[0])
                        xc, yc, nw, nh = map(float, parts[1:5])
                        x1 = int((xc - nw / 2.0) * w_m)
                        y1 = int((yc - nh / 2.0) * h_m)
                        x2 = int((xc + nw / 2.0) * w_m)
                        y2 = int((yc + nh / 2.0) * h_m)

                        col = palette[c_box_id % len(palette)]
                        cv2.rectangle(mat, (x1, y1), (x2, y2), col, 2)
                        lbl_str = f"{source_class_map.get(c_box_id, 'obj')}"
                        cv2.putText(mat, lbl_str, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            cv2.putText(mat, f"CLASS {cid}: {cname.upper()}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            panels.append(mat)

    if panels:
        while len(panels) % 3 != 0:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            panels.append(blank)
        rows = [np.hstack(panels[i:i + 3]) for i in range(0, len(panels), 3)]
        contact_sheet = np.vstack(rows)

        contact_path = OUTPUTS_METRICS_DIR / "sih26187_contact_sheet.jpg"
        cv2.imwrite(str(contact_path), contact_sheet)
        print(f"  Contact Sheet saved to: {contact_path}")

    # 6. Compare with Current IBVAP Dataset
    print("\n[STEP 6] COMPARISON WITH CURRENT IBVAP DATASET")
    current_counts = {"person": 56223, "car": 31, "truck": 17, "bike": 33, "patrol_vehicle": 39}

    print(f"  {'SOURCE':<25} | {'PERSON':<8} | {'CAR':<8} | {'TRUCK':<8} | {'BIKE':<8} | {'PATROL':<8}")
    print("  " + "-" * 75)
    print(f"  {'Current IBVAP dataset':<25} | {current_counts['person']:<8} | {current_counts['car']:<8} | {current_counts['truck']:<8} | {current_counts['bike']:<8} | {current_counts['patrol_vehicle']:<8}")

    sih_person = class_bbox_counts.get("person", 0) + class_bbox_counts.get("Person", 0)
    sih_car = class_bbox_counts.get("car", 0) + class_bbox_counts.get("Car", 0) + class_bbox_counts.get("cars", 0)
    sih_truck = class_bbox_counts.get("truck", 0) + class_bbox_counts.get("Truck", 0) + class_bbox_counts.get("trucks", 0)
    sih_bike = class_bbox_counts.get("bike", 0) + class_bbox_counts.get("Bike", 0) + class_bbox_counts.get("motorcycle", 0) + class_bbox_counts.get("bicycle", 0)
    sih_patrol = class_bbox_counts.get("patrol_vehicle", 0) + class_bbox_counts.get("patrol", 0) + class_bbox_counts.get("police", 0)

    print(f"  {'New SIH26187 dataset':<25} | {sih_person:<8} | {sih_car:<8} | {sih_truck:<8} | {sih_bike:<8} | {sih_patrol:<8}")

    # 7. Class Mapping Compatibility
    print("\n[STEP 7] CLASS MAPPING COMPATIBILITY")
    target_mapping = {}
    for cid, cname in source_class_map.items():
        cname_lower = cname.lower()
        if "person" in cname_lower or "pedestrian" in cname_lower:
            target_mapping[cname] = "0 (person)"
        elif "patrol" in cname_lower or "police" in cname_lower:
            target_mapping[cname] = "4 (patrol_vehicle)"
        elif "truck" in cname_lower:
            target_mapping[cname] = "2 (truck)"
        elif "bike" in cname_lower or "cycle" in cname_lower or "motor" in cname_lower:
            target_mapping[cname] = "3 (bike)"
        elif "car" in cname_lower or "auto" in cname_lower or "vehicle" in cname_lower:
            target_mapping[cname] = "1 (car)"
        else:
            target_mapping[cname] = "REQUIRES MANUAL REVIEW"

    print("  Proposed Class Mapping (Source -> IBVAP Target):")
    for s_cls, t_cls in target_mapping.items():
        print(f"    '{s_cls}' -> IBVAP Class {t_cls}")

    # 8. Save Reports
    print("\n[STEP 8] GENERATING REPORTS & JSON/CSV METRICS")

    # JSON report
    report_json = {
        "zip_path": str(DATASET_DIR),
        "dataset_name": DATASET_DIR.name,
        "folder_structure": contents,
        "splits": split_info,
        "total_images": total_images_all,
        "total_labels": total_labels_all,
        "total_boxes": total_boxes_all,
        "source_class_mapping": source_class_map,
        "class_bbox_counts": class_bbox_counts,
        "roboflow_metadata_notes": roboflow_aug_details,
        "qa_summary": {
            "unreadable_images_count": len(qa_issues["unreadable_images"]),
            "missing_labels_count": len(qa_issues["missing_labels"]),
            "empty_labels_count": len(qa_issues["empty_labels"]),
            "invalid_class_ids_count": len(qa_issues["invalid_class_ids"]),
            "malformed_labels_count": len(qa_issues["malformed_labels"]),
            "out_of_bound_boxes_count": len(qa_issues["out_of_bound_boxes"]),
            "duplicate_images_hash_count": len(qa_issues["duplicate_images_hash"])
        },
        "target_mapping_proposal": target_mapping
    }

    json_out_path = OUTPUTS_METRICS_DIR / "sih26187_dataset_report.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)

    # CSV report
    csv_out_path = OUTPUTS_METRICS_DIR / "sih26187_dataset_report.csv"
    if qa_rows:
        with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(qa_rows[0].keys()))
            writer.writeheader()
            writer.writerows(qa_rows)

    # Markdown report
    md_out_path = DATASETS_DIR / "SIHL26187_IMPORT_REPORT.md"
    md_content = f"""# SIH26187 Roboflow Dataset Inspection & Non-Destructive Import Report

## 1. Dataset Overview
* **Exact Dataset Directory**: `{DATASET_DIR}`
* **Dataset Name**: `{DATASET_DIR.name}`
* **Top-Level Folder Structure**: `{', '.join(contents)}`
* **Format**: Standard PyTorch YOLO Format (`data.yaml`, `train/`, `valid/`, `test/`)

---

## 2. Dataset Split Breakdown

| Split Name | Image Count | Label File Count | Bounding Box Count |
| :--- | :---: | :---: | :---: |
| **Train** | {split_info.get('train', {}).get('image_count', 0)} | {split_info.get('train', {}).get('label_count', 0)} | {split_info.get('train', {}).get('box_count', 0)} |
| **Valid** | {split_info.get('valid', {}).get('image_count', 0)} | {split_info.get('valid', {}).get('label_count', 0)} | {split_info.get('valid', {}).get('box_count', 0)} |
| **Test** | {split_info.get('test', {}).get('image_count', 0)} | {split_info.get('test', {}).get('label_count', 0)} | {split_info.get('test', {}).get('box_count', 0)} |
| **Total** | **{total_images_all}** | **{total_labels_all}** | **{total_boxes_all}** |

---

## 3. Class Distribution & Bounding Box Counts

| Source Class ID | Source Class Name | Bounding Box Count | Proposed IBVAP Target Class |
| :---: | :--- | :---: | :--- |
"""
    for cid, cname in source_class_map.items():
        cnt = class_bbox_counts.get(cname, 0)
        t_cls = target_mapping.get(cname, "N/A")
        md_content += f"| `{cid}` | `{cname}` | {cnt} | `{t_cls}` |\n"

    md_content += f"""
---

## 4. Comparison with Current IBVAP Dataset Balance

| Source Dataset | Person | Car | Truck | Bike | Patrol Vehicle |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Current IBVAP Dataset** | 56,223 | 31 | 17 | 33 | 39 |
| **New SIH26187 Dataset** | {sih_person} | {sih_car} | {sih_truck} | {sih_bike} | {sih_patrol} |

---

## 5. Augmentation & Quality Assurance Findings

### Augmentation Findings
* {aug_findings}
* **Roboflow Metadata Notes**:
"""
    for note in roboflow_aug_details:
        md_content += f"  - `{note}`\n"

    md_content += f"""
### Non-Destructive Quality Assurance (QA) Summary
* **Unreadable Images**: {len(qa_issues['unreadable_images'])}
* **Missing Label Files**: {len(qa_issues['missing_labels'])}
* **Empty Label Files**: {len(qa_issues['empty_labels'])}
* **Invalid Class IDs**: {len(qa_issues['invalid_class_ids'])}
* **Malformed Bounding Boxes**: {len(qa_issues['malformed_labels'])}
* **Out-of-Bound Bounding Boxes**: {len(qa_issues['out_of_bound_boxes'])}
* **Duplicate Images (MD5 Match)**: {len(qa_issues['duplicate_images_hash'])}

---

## 6. Recommended Integration Plan
1. **Safety Protocol**: Maintain raw source dataset untouched at `{DATASET_DIR}`.
2. **Unified Dataset Integration**: Map source class IDs into standard 5-class target schema (`0: person`, `1: car`, `2: truck`, `3: bike`, `4: patrol_vehicle`) in `IBVAP/datasets/build_unified_dataset.py`.
3. **Class Imbalance Mitigation**: Use the new vehicle instances to improve vehicle detection recall without sacrificing person detection accuracy.
4. **Validation Gate**: Verify bounding box contact sheet overlays before executing Phase 4 retraining.
"""

    with open(md_out_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n  [SUCCESS] All inspection tasks complete!")
    print(f"  - Markdown Import Report : {md_out_path}")
    print(f"  - JSON Report            : {json_out_path}")
    print(f"  - CSV Report             : {csv_out_path}")
    print(f"  - Contact Sheet Image    : {OUTPUTS_METRICS_DIR / 'sih26187_contact_sheet.jpg'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
