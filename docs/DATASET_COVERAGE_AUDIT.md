# WasteWise-AI — Dataset Coverage & Validation Audit

## 1. Executive Summary

This document presents a comprehensive, evidence-based audit of all datasets present, extracted, referenced in configuration files, or processed by training and evaluation scripts across the `WasteWise-AI` repository.

No code modifications or training runs were initiated during this audit. All statistics and findings reflect verified disk assets, directory manifests, configuration files, and evaluation outputs.

---

## 2. Master Dataset Inventory & Metadata

The repository contains **6 primary source datasets** and **7 derived/processed variations**, spanning classification, object detection, and negative/OOD domain gating.

### Summary Table of All Datasets

| Dataset Identifier | Filesystem Path | Source Origin | Total Images | Annotation Format | Roles in Pipeline |
|:---|:---|:---|:---|:---|:---|
| **Garbage Dataset (Core)** | `data/raw/garbage-dataset` | Kaggle Garbage Dataset | 19,762 | ImageFolder (directories) | Classification Train / Val / Test (Core 10-Class & Expanded 19-Class) |
| **Medical Waste 4.0 (Dataset A)** | `data/external/medical_waste_4_0/.../DATASET A` | Roboflow / Git | 5,520 | ImageFolder (12 subfolders) | Classification Train / Val (Expanded 19-Class) |
| **Medical Waste 4.0 (Dataset B)** | `data/external/medical_waste_4_0/.../DATASET B` | Roboflow / Git | 801 | ImageFolder (14 subfolders) | Independent Classification External Test Split |
| **TrashNet++ (Classwise)** | `data/external/trashnet_plus_plus/TrashNet/Classwise Formet` | GitHub/Roboflow | 1,344 | ImageFolder (7 subfolders) | Selective Train / Val / Review / Excluded |
| **TrashNet++ (YOLO Format)** | `data/external/trashnet_plus_plus/TrashNet/YOLO Formet` | GitHub/Roboflow | 2,613 | YOLO TXT (7 classes) | Independent YOLO Detector Evaluation |
| **TACO (Raw Extracted)** | `data/external/taco/extracted` | TACO GitHub / COCO | 1,514 | COCO JSON (`annotations.json`) | Source for Detection Splits |
| **TACO Practical (Processed)** | `data/processed/taco_detection_practical` | Derived from TACO | 1,499 | YOLO TXT (6 classes) | YOLO Train / Val / Test (Standard 6-class) |
| **Garbage Object Detection** | `data/external/garbage_object_detection` | Roboflow Universe | 10,464 | YOLO TXT (6 classes) | Source for External YOLO Detector |
| **Garbage External Wastewise** | `data/processed/garbage_external_wastewise` | Derived from Roboflow | 10,464 | YOLO TXT (5 classes) | YOLO Train (Combined) & Test (External) |
| **TACO + External Combined** | `data/processed/taco_external_combined` | TACO Train + External Train | 8,523 | YOLO TXT (6 classes) | Final YOLO11s Training Set |
| **MS-COCO val2017** | `data/external/coco/val2017/val2017` | Common Objects in Context | 5,000 | COCO JSON (80 categories) | Negative / Out-Of-Distribution (OOD) Domain Gate |
| **Splits Core (10-Class)** | `data/splits` | Derived from Garbage Dataset | 19,762 | CSV Manifests | Legacy Baseline 10-Class Benchmark |
| **Splits Expanded (19-Class)** | `data/splits_expanded` | Core + Med A + TrashNet | 27,247 | CSV Manifests & JSON | Production ResNet50 Train / Val / Test / MedB |

---

## 3. Dataset-by-Dataset Detailed Breakdown

### 3.1 Main Garbage Dataset (Core 10-Class)
- **Path**: `data/raw/garbage-dataset/`
- **Total Images**: 19,762
- **Annotation Format**: Directory hierarchy (ImageFolder)
- **Class Breakdown**:
  - `battery`: 944
  - `biological`: 997
  - `cardboard`: 1,825
  - `clothes`: 5,327
  - `glass`: 3,061
  - `metal`: 1,020
  - `paper`: 1,680
  - `plastic`: 1,984
  - `shoes`: 1,977
  - `trash`: 947
- **Usage**:
  - 10-class baseline: `data/splits/` (Train: 15,809, Val: 1,976, Test: 1,977).
  - 19-class expanded: Provided 10 core classes (`trash` mapped canonically to `general_trash`).

---

### 3.2 Medical Waste 4.0
- **Path**: `data/external/medical_waste_4_0/Medical-Waste-4.0-Dataset/Medical Waste 4.0/`
- **Total Images**: 6,321 (Dataset A: 5,520; Dataset B: 801)
- **Annotation Format**: Subfolder ImageFolder
- **Taxonomy & Class Counts**:
  - **Dataset A (5,520 images — Used for Training & Validation)**:
    - `gauze`: 537
    - `glove_pair_latex`: 429
    - `glove_pair_nitrile`: 519
    - `glove_pair_surgery`: 375
    - `glove_single_latex`: 390
    - `glove_single_nitrile`: 453
    - `glove_single_surgery`: 369
    - `medical_cap`: 381
    - `medical_glasses`: 393
    - `shoe_cover_pair`: 432
    - `shoe_cover_single`: 405
    - `test_tube`: 429
    - `urine_bag`: 408
    *(All glove subcategories mapped to canonical `medical_gloves`; shoe covers mapped to canonical `shoe_cover`)*.
  - **Dataset B (801 images — Isolated External Test Set)**:
    - `gauzes`: 93
    - `glove_pair_latex`: 42
    - `glove_pair_nitrile`: 33
    - `glove_pair_surgery`: 54
    - `glove_single_latex`: 48
    - `glove_single_nitrile`: 30
    - `glove_single_surgery`: 45
    - `medical_cap`: 57
    - `medical_filter`: 72 *(Unsupported / out-of-taxonomy class)*
    - `medical_glasses`: 93
    - `shoe_cover_pair`: 45
    - `shoe_cover_single`: 90
    - `test_tube`: 54
    - `urine_bag`: 45

---

### 3.3 TrashNet++
- **Path**: `data/external/trashnet_plus_plus/TrashNet/`
- **Total Images**: 3,960
- **Formats Available**:
  1. `Classwise Formet` (1,344 images):
     - `Cable`: 162 (Mapped to `cable`, `expanded_train`)
     - `Can`: 100 (Isolated to `expanded_train_review`)
     - `E-wast`: 334 (Mapped to `electronic_device`, `expanded_train`)
     - `Glass`: 201 (Mapped to `glass`, `expanded_train`)
     - `Medical Wast`: 80 (Flagged `excluded_from_clean_training` due to noisy coarse labels)
     - `Metal`: 213 (Mapped to `metal`, `expanded_train`)
     - `Plastic`: 254 (Mapped to `plastic`, `expanded_train`)
  2. `YOLO Formet` (2,613 images):
     - Train: 1,833 images
     - Valid: 515 images
     - Test: 265 images
     - Original 7 classes: `['Cable', 'E-waste', 'Glass', 'Medical Waste', 'Metal', 'Plastics', 'can']`
     - Evaluated via `scripts/evaluation/evaluate_trashnet_plus_plus.py`.

---

### 3.4 TACO (Trash Annotations in Context)
- **Path**: `data/external/taco/`
- **Total Images**: 1,514 raw images
- **Annotation Format**: Original COCO JSON format (`annotations.json`, 60 distinct litter subcategories)
- **Derived Processed Variant**: `data/processed/taco_detection_practical/` (1,499 valid images)
  - Split: Train = 1,199, Val = 149, Test = 151
  - Label format: YOLO normalized bounding boxes
  - 6 Practical Classes:
    - 0: `plastic`
    - 1: `glass`
    - 2: `metal`
    - 3: `paper_cardboard`
    - 4: `biological`
    - 5: `other_trash`

---

### 3.5 External Garbage Object Detection (Roboflow)
- **Path**: `data/external/garbage_object_detection/` and `data/processed/garbage_external_wastewise/`
- **Total Images**: 10,464
  - Train: 7,324 images
  - Valid: 2,098 images
  - Test: 1,042 images
- **Classes in Processed Wastewise Version**:
  - 0: `plastic`
  - 1: `glass`
  - 2: `metal`
  - 3: `paper_cardboard`
  - 4: `biological`
  *(Note: Class 5 `other_trash` is absent from this dataset)*.

---

### 3.6 TACO + External Combined Detection Dataset
- **Path**: `data/processed/taco_external_combined/`
- **Total Images**: 8,523 images (7,324 external train + 1,199 TACO practical train)
- **Purpose**: High-diversity training corpus used to train the final `yolo11s_balanced` detector.
- **Evaluation Target**: Tested against `taco_detection_practical/images/val` and held-out test splits.

---

### 3.7 MS-COCO val2017 (Negative / Out-of-Distribution Domain Gate)
- **Path**: `data/external/coco/val2017/val2017/`
- **Total Images**: 5,000 images
- **Annotation Format**: MS-COCO object instances (`instances_val2017.json`, 80 standard categories: person, dog, car, etc.)
- **Critical Domain Role**: Strictly used as **negative non-waste out-of-distribution evaluation data**. Any waste detection or classification in this set represents an in-the-wild false positive.

---

## 4. Train / Validation / Test Splits & Leakage Audit

### 4.1 Classification Split Construction (`data/splits_expanded`)
- **Script**: `scripts/create_expanded_splits.py`
- **Split Counts**:
  - Train: 21,155 samples ($80.0\%$)
  - Validation: 2,645 samples ($10.0\%$)
  - Test: 2,646 samples ($10.0\%$)
  - Medical External Test: 801 samples (100% held out)
  - Review: 100 samples (isolated)
- **Leakage Prevention**:
  - Grouping strategy: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by **SHA-256 image content hashes**.
  - Duplicate or perceptual duplicate images with identical content hashes are strictly constrained to the same fold.
  - Verification: `split_summary.json` reports `"sha256_group_leakage": false`.
- **Genuinely Held Out**:
  - The evaluation in `scripts/evaluate_expanded_resnet50_stage2.py` strictly loads `data/splits_expanded/test.csv` (2,646 samples). Reported accuracy ($95.73\%$) and macro F1 ($95.99\%$) are **genuinely held out**.
  - `Medical Waste Dataset B` (801 images) was NEVER part of training or validation. The reported $95.20\%$ accuracy on supported classes is completely out-of-domain held-out generalization.

### 4.2 Leakage Risk Identified in `scripts/evaluation/evaluate_all.py`
> [!WARNING] Train-Test Leakage in Legacy Evaluation Script
> `scripts/evaluation/evaluate_all.py` loads the entire raw directory `data/raw/garbage-dataset` (all 19,762 images) and computes classification metrics across all files. Because 80% of those images were present in `train.csv`, `evaluate_all.py` **suffers from severe training data leakage**.
> **Action**: Never cite `evaluate_all.py` as a held-out test metric. Only cite `evaluate_expanded_resnet50_stage2.py` which evaluates on `data/splits_expanded/test.csv`.

---

## 5. Dataset Evaluation Status & Scope Matrix

### 5.1 Independent vs. Combined Evaluation Audit

1. **Evaluated Individually**:
   - `TACO Practical Test`: Evaluated independently on its 151-image test set via `scripts/evaluation/evaluate_taco_detector.py`.
   - `Garbage External Test`: Evaluated independently on its 1,042-image test set via `scripts/evaluation/evaluate_external_detector.py`.
   - `Medical Waste Dataset B`: Evaluated independently on its 801-image set via `scripts/evaluate_medical_b_external_stage2.py`.
   - `TrashNet++ YOLO Test`: Evaluated independently on 265 test images via `scripts/evaluation/evaluate_trashnet_plus_plus.py`.
   - `MS-COCO val2017`: Evaluated independently on 5,000 images via `scripts/evaluation/evaluate_coco_negative.py` and `scripts/evaluate_coco_ood.py`.

2. **Evaluated Only in Combined Sets**:
   - `TrashNet++ Classwise Crops`: The classification crops (`Cable`, `E-wast`, `Glass`, `Metal`, `Plastic`) were pooled into `data/splits_expanded/train.csv` and `test.csv`. They were never reported as an isolated single-source test metric for TrashNet++ classification.
   - `Medical Waste Dataset A`: Pooled into `data/splits_expanded/train.csv` and `test.csv`; evaluated as part of the overall 19-class test set rather than an isolated Dataset A report.

---

## 6. Pipeline Subsystem Testing Coverage (A, B, C, D)

For each dataset, testing coverage is audited across the 4 core tiers:
- **A. YOLO Detection**
- **B. ResNet50 Classification**
- **C. OOD Detection**
- **D. Complete Integrated Pipeline** (YOLO $\to$ Crop $\to$ ResNet50 $\to$ OOD $\to$ Compatibility)

| Dataset | Tier A (YOLO) | Tier B (ResNet50) | Tier C (OOD) | Tier D (Full Pipeline) |
|:---|:---:|:---:|:---:|:---:|
| **Garbage Dataset (Raw 10-Class)** | ❌ No | ✅ Yes (95.7%) | ✅ Yes | ❌ Partial (10 sanity images) |
| **Splits Expanded Test (19-Class)** | ❌ No | ✅ Yes (95.73% Acc, 95.99% F1) | ✅ Yes | ❌ No (crops only, no scene boxes) |
| **Medical Waste Dataset A** | ❌ No | ✅ Yes (in 19-class split) | ✅ Yes | ❌ No |
| **Medical Waste Dataset B** | ❌ No | ✅ Yes (95.20% Acc, 95.43% F1) | ✅ Yes (OOD on `medical_filter`) | ❌ No |
| **TrashNet++ (YOLO Format)** | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **TrashNet++ (Classwise)** | ❌ No | ✅ Yes (in 19-class split) | ✅ Yes | ❌ No |
| **TACO Practical Test** | ✅ Yes (mAP50: 0.461, F1: 0.468) | ❌ No (full scenes) | ❌ No | ❌ Partial (10 sanity images) |
| **Garbage External Test** | ✅ Yes | ❌ No (full scenes) | ❌ No | ❌ No |
| **TACO + External Combined** | ✅ Yes (Train Set) | ❌ No | ❌ No | ❌ No |
| **MS-COCO val2017** | ✅ Yes (FPR: 65.9% at 0.30) | ✅ Yes (full-frame OOD) | ✅ Yes (Rejection: 42.24%) | ✅ Yes (`integrated_coco_results.json`) |

---

## 7. Taxonomy & Class Alignment Checks

### 7.1 Canonical Target Taxonomies
1. **YOLO 6 Classes**: `0: plastic`, `1: glass`, `2: metal`, `3: paper_cardboard`, `4: biological`, `5: other_trash`.
2. **ResNet 19 Classes**: `battery`, `biological`, `cable`, `cardboard`, `clothes`, `electronic_device`, `gauze`, `general_trash`, `glass`, `medical_cap`, `medical_glasses`, `medical_gloves`, `metal`, `paper`, `plastic`, `shoe_cover`, `shoes`, `test_tube`, `urine_bag`.

### 7.2 Label Alignment Findings & Flags

1. **`medical_filter` in Medical Waste Dataset B**:
   - *Status*: **Unsupported Class**.
   - *Detail*: Dataset B contains 72 images of `medical_filter`. This class was intentionally not added to the 19-class classifier.
   - *Verification*: Evaluated in `scripts/evaluate_medical_b_external_stage2.py` as an unsupported test class; must be rejected by the OOD centroid detector.
2. **Missing `other_trash` in `garbage_external_wastewise`**:
   - *Status*: **Missing Ground-Truth Class**.
   - *Detail*: Contains only classes 0 to 4 (`plastic`, `glass`, `metal`, `paper_cardboard`, `biological`). Class 5 (`other_trash`) is missing in this external test split.
3. **TrashNet++ `Cable` and `can` Mappings in YOLO Evaluation**:
   - *Status*: **Imperfect Detector Mapping**.
   - *Detail*: In `scripts/evaluation/evaluate_trashnet_plus_plus.py`, `Cable` is mapped to `0: plastic` and `can` to `2: metal`. While fine for bounding box localization, `cable` is an independent category in the ResNet classifier under `e_waste`.
4. **COCO Domain Integrity**:
   - *Status*: **Valid Negative Gate, Invalid Waste Dataset**.
   - *Detail*: MS-COCO has no waste ground-truth annotations. Using COCO to compute detection recall or classification precision is mathematically invalid. It is correctly utilized exclusively as a negative baseline to measure false positive rate (FPR) and OOD rejection efficiency.

---

## 8. Missing Evaluations & Execution Commands

To achieve 100% full-pipeline coverage across all datasets, the following evaluations should be run using existing scripts:

1. **Full-Pipeline Evaluation on TACO Practical Test**:
   - *Missing*: Running the complete end-to-end `WasteWisePredictor` (YOLO $\to$ ResNet $\to$ OOD $\to$ Gate) on all 151 test images of `taco_detection_practical`.
   - *Execution Command*:
     ```powershell
     .venv\Scripts\python.exe -c "from scripts.evaluation.evaluate_integrated import evaluate_images; evaluate_images('data/processed/taco_detection_practical/images/test')"
     ```
2. **Full-Pipeline Evaluation on Garbage External Test**:
   - *Missing*: End-to-end multi-stage pipeline evaluation on the 1,042 images of `data/processed/garbage_external_wastewise/test/images`.
   - *Execution Command*:
     ```powershell
     .venv\Scripts\python.exe -c "from scripts.evaluation.evaluate_integrated import evaluate_images; evaluate_images('data/processed/garbage_external_wastewise/test/images')"
     ```
3. **Isolated Classification Benchmark on TrashNet++ Held-Out Crops**:
   - *Missing*: Benchmark ResNet50 strictly on the held-out test crops originating from TrashNet++ (`Cable`, `E-wast`, `Glass`, `Metal`, `Plastic`).
   - *Execution Command*: Filter `data/splits_expanded/test.csv` where source is TrashNet++ and compute accuracy.
