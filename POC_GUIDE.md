# POC Guide: CV Model Bias Monitoring

This guide walks you through demonstrating that your bias monitoring repo works with CV models.

## Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  STEP 1: Data       │ ──► │  STEP 2: Train      │ ──► │  STEP 3: Monitor    │
│  (This Repo)        │     │  (This Repo)        │     │  (Your Monitoring   │
│                     │     │                     │     │   Repo)             │
│  100 images with    │     │  CV model outputs   │     │  Clarify detects    │
│  intentional bias   │     │  predictions.csv    │     │  bias in results    │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

---

## Step 1: Create Minimal Dataset (5 minutes)

```bash
# Create 100 images with intentional bias
python scripts/prepare_minimal_poc.py --output ./data --num-images 100
```

**What this creates:**
```
./data/
├── train/
│   ├── no_rust/     (40 images)
│   └── rust/        (40 images)
├── test/
│   ├── no_rust/     (10 images)
│   └── rust/        (10 images)
└── metadata.csv     (with sensitive attributes)
```

**Intentional Bias in Data:**
| Environment | Rust Images | No-Rust Images | Expected Model Behavior |
|-------------|-------------|----------------|------------------------|
| coastal     | 60%         | 10%            | Predicts rust more often |
| outdoor     | 30%         | 30%            | Balanced |
| indoor      | 10%         | 60%            | Predicts no-rust more often |

This bias will be **detectable by Clarify**.

---

## Step 2: Train CV Model (10 minutes)

```bash
# Train locally (no SageMaker needed for POC)
python scripts/train_local.py --data-dir ./data --epochs 5
```

**Outputs:**
```
./output/
├── predictions.csv                  # Basic predictions
├── predictions_with_metadata.csv    # With sensitive attributes (for Clarify)
└── model.pth                        # Trained model
```

**predictions_with_metadata.csv format:**
```csv
image_path,label,environment,surface_type,prediction,ground_truth,confidence,prob_no_rust,prob_rust
./data/test/rust/rust_0002.jpg,1,coastal,steel,1,1,0.87,0.13,0.87
./data/test/no_rust/no_rust_0003.jpg,0,indoor,aluminum,0,0,0.92,0.92,0.08
```

---

## Step 3: Upload to S3

```bash
# Upload predictions for your monitoring repo
aws s3 cp ./output/predictions_with_metadata.csv \
    s3://tdsp-ml-products-dev/tdspds/rust-detection/predictions.csv
```

---

## Step 4: Run Clarify (Your Monitoring Repo)

Your monitoring repo should use these settings:

```python
# Clarify configuration
clarify_config = {
    "dataset_uri": "s3://tdsp-ml-products-dev/tdspds/rust-detection/predictions.csv",
    "label_column": "ground_truth",
    "predicted_label_column": "prediction", 
    "facet_columns": ["environment", "surface_type"],  # Sensitive attributes
    "positive_label_values": [1],  # 1 = rust (positive class)
}
```

---

## Expected Clarify Results

Clarify should detect bias like this:

### Disparate Impact (DI) by Environment
| Facet | Value | Interpretation |
|-------|-------|----------------|
| coastal vs indoor | ~0.6-0.8 | Model favors rust prediction for coastal |
| outdoor vs indoor | ~0.9-1.0 | Relatively balanced |

### Accuracy Difference (AD) by Environment
| Group | Accuracy | Difference |
|-------|----------|------------|
| indoor | ~90% | Baseline |
| outdoor | ~85% | -5% |
| coastal | ~75% | -15% (BIAS!) |

---

## Quick Commands Summary

```bash
# 1. Create data
python scripts/prepare_minimal_poc.py --output ./data --num-images 100

# 2. Train
python scripts/train_local.py --data-dir ./data --epochs 5

# 3. Upload
aws s3 cp ./output/predictions_with_metadata.csv \
    s3://tdsp-ml-products-dev/tdspds/rust-detection/predictions.csv

# 4. (In your monitoring repo) Run Clarify on the S3 path
```

---

## Scaling Up After POC

Once POC is validated, increase data:

| Stage | Images | Command |
|-------|--------|---------|
| POC | 100 | `--num-images 100` |
| Dev | 500 | `--num-images 500` |
| Production | Use real NEU dataset | See below |

### Using Real NEU Dataset

1. Download from: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

2. Process:
```bash
python scripts/prepare_dataset.py --dataset neu --output ./data
```

---

## Files Summary

| File | Purpose |
|------|---------|
| `scripts/prepare_minimal_poc.py` | Create small biased dataset |
| `scripts/train_local.py` | Train model locally |
| `src/train_cv.py` | SageMaker training script |
| `deploy/dev/tdspds-config.yml` | Configuration |
