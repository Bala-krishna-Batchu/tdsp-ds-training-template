# POC Guide: CV Model Bias Monitoring

This guide walks you through demonstrating that your bias monitoring repo works with CV models.

## Overview

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  STEP 1: Create  │ ──► │  STEP 2: Upload  │ ──► │  STEP 3: Train   │ ──► │  STEP 4: Monitor │
│  Local Data      │     │  to S3           │     │  (SageMaker)     │     │  (Your Repo)     │
│                  │     │                  │     │                  │     │                  │
│  100 images      │     │  S3 bucket       │     │  CV model        │     │  Clarify         │
│  + metadata      │     │                  │     │  predictions.csv │     │  bias report     │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

## Data Flow (Same as Churn Model)

```
S3 (Input)                           SageMaker                         S3 (Output)
──────────                           ─────────                         ───────────
s3://bucket/.../train/      ──►     /opt/ml/input/data/train/    
s3://bucket/.../test/       ──►     /opt/ml/input/data/test/     ──►  Training
s3://bucket/.../metadata/   ──►     /opt/ml/input/data/metadata/      
                                                                       │
                                                                       ▼
                                                               s3://bucket/.../predictions.csv
                                                                       │
                                                                       ▼
                                                               Your Monitoring Repo
                                                               (Clarify Bias Analysis)
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

## Step 2: Upload Data to S3 (5 minutes)

```bash
# Upload to S3 (same pattern as churn model)
python scripts/upload_data_to_s3.py \
    --local-dir ./data \
    --bucket tdsp-data-products-dev \
    --prefix tdspds/rust-detection/data
```

**S3 Structure (after upload):**
```
s3://tdsp-data-products-dev/tdspds/rust-detection/data/
├── train/
│   ├── no_rust/*.jpg
│   └── rust/*.jpg
├── test/
│   ├── no_rust/*.jpg
│   └── rust/*.jpg
└── metadata/
    └── metadata.csv
```

**This matches pipelines-config.yml:**
```yaml
InputDataConfig:
  - ChannelName: "train"
    S3Uri: "s3://tdsp-data-products-dev/tdspds/rust-detection/data/train"
  - ChannelName: "test"
    S3Uri: "s3://tdsp-data-products-dev/tdspds/rust-detection/data/test"
  - ChannelName: "metadata"
    S3Uri: "s3://tdsp-data-products-dev/tdspds/rust-detection/data/metadata"
```

---

## Step 3: Train CV Model

### Option A: Train with SageMaker (Production)

SageMaker will:
1. Download S3 data to `/opt/ml/input/data/`
2. Run `train_cv.py`
3. Upload results to S3

```bash
# Trigger via your CI/CD pipeline or manually
# SageMaker reads from S3 paths in pipelines-config.yml
```

### Option B: Train Locally (For Testing)

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

## Step 4: Upload Predictions to S3 (For Monitoring Repo)

```bash
# Upload predictions for your monitoring repo
aws s3 cp ./output/predictions_with_metadata.csv \
    s3://tdsp-ml-products-dev/tdspds/rust-detection/predictions.csv
```

---

## Step 5: Run Clarify (Your Monitoring Repo)

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
# 1. Create local dataset (100 images with bias)
python scripts/prepare_minimal_poc.py --output ./data --num-images 100

# 2. Upload data to S3 (same as churn model pattern)
python scripts/upload_data_to_s3.py \
    --local-dir ./data \
    --bucket tdsp-data-products-dev \
    --prefix tdspds/rust-detection/data

# 3. Train locally (for testing) OR trigger SageMaker
python scripts/train_local.py --data-dir ./data --epochs 5

# 4. Upload predictions to S3
aws s3 cp ./output/predictions_with_metadata.csv \
    s3://tdsp-ml-products-dev/tdspds/rust-detection/predictions.csv

# 5. (In your monitoring repo) Run Clarify on the S3 path
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
