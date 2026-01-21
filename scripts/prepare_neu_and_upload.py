"""
Prepare NEU Dataset and Upload to S3

Downloads from Kaggle, processes for rust detection, and uploads to S3.

STEPS:
1. Download NEU dataset from Kaggle
2. Organize into rust/no_rust classes
3. Create metadata with bias attributes
4. Upload to S3

Usage:
    # If you have Kaggle CLI configured:
    python scripts/prepare_neu_and_upload.py --download --upload

    # If you downloaded manually:
    python scripts/prepare_neu_and_upload.py --neu-dir ./NEU-DET --upload
"""

import argparse
import os
import random
import shutil
import subprocess

import pandas as pd
from PIL import Image


def download_from_kaggle(output_dir: str):
    """Download NEU dataset from Kaggle."""
    print("Downloading NEU dataset from Kaggle...")
    print("Make sure you have Kaggle CLI configured (~/.kaggle/kaggle.json)")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Download
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "kaustubhdikshit/neu-surface-defect-database",
        "-p", output_dir
    ], check=True)
    
    # Unzip
    zip_file = os.path.join(output_dir, "neu-surface-defect-database.zip")
    if os.path.exists(zip_file):
        subprocess.run(["unzip", "-o", zip_file, "-d", output_dir], check=True)
        os.remove(zip_file)
    
    print(f"Downloaded to {output_dir}")


def process_neu_dataset(neu_dir: str, output_dir: str, max_images: int = None):
    """
    Process NEU dataset for rust detection.
    
    NEU Classes:
    - RS (Rolled-in Scale) → rust (looks like rust/oxidation)
    - Pa (Patches) → rust (surface damage similar to corrosion)
    - Cr (Crazing) → no_rust
    - In (Inclusion) → no_rust
    - PS (Pitted Surface) → no_rust
    - Sc (Scratches) → no_rust
    """
    print(f"Processing NEU dataset from {neu_dir}...")
    
    # Find the images directory
    images_dir = None
    for root, dirs, files in os.walk(neu_dir):
        if any(f.endswith('.bmp') for f in files):
            images_dir = root
            break
    
    if not images_dir:
        # Try common paths
        possible_paths = [
            os.path.join(neu_dir, "NEU-DET", "IMAGES"),
            os.path.join(neu_dir, "IMAGES"),
            os.path.join(neu_dir, "images"),
            neu_dir,
        ]
        for path in possible_paths:
            if os.path.exists(path):
                images_dir = path
                break
    
    if not images_dir or not os.path.exists(images_dir):
        print(f"ERROR: Could not find images directory in {neu_dir}")
        print("Please check the dataset structure.")
        return None
    
    print(f"Found images in: {images_dir}")
    
    # Class mapping: which NEU classes are "rust-like"
    RUST_CLASSES = ["RS", "Pa"]  # Rolled-in Scale, Patches
    NO_RUST_CLASSES = ["Cr", "In", "PS", "Sc"]  # Crazing, Inclusion, Pitted Surface, Scratches
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    for class_name in ["no_rust", "rust"]:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    # Collect images by class
    rust_images = []
    no_rust_images = []
    
    for filename in os.listdir(images_dir):
        if filename.endswith(('.bmp', '.jpg', '.png')):
            prefix = filename.split('_')[0]
            filepath = os.path.join(images_dir, filename)
            
            if prefix in RUST_CLASSES:
                rust_images.append(filepath)
            elif prefix in NO_RUST_CLASSES:
                no_rust_images.append(filepath)
    
    print(f"Found {len(rust_images)} rust images (RS, Pa)")
    print(f"Found {len(no_rust_images)} no_rust images (Cr, In, PS, Sc)")
    
    # Limit images if specified
    if max_images:
        max_per_class = max_images // 2
        rust_images = rust_images[:max_per_class]
        no_rust_images = no_rust_images[:max_per_class]
        print(f"Limited to {len(rust_images)} rust, {len(no_rust_images)} no_rust")
    
    # Shuffle and split
    random.shuffle(rust_images)
    random.shuffle(no_rust_images)
    
    metadata_records = []
    
    # Sensitive attributes for bias (simulated based on image properties)
    environments = ["indoor", "outdoor", "coastal"]
    surface_types = ["steel", "iron"]
    
    def process_class(images, class_name, label):
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        test_images = images[split_idx:]
        
        for i, src_path in enumerate(train_images + test_images):
            split = "train" if i < len(train_images) else "test"
            
            # Convert BMP to JPG and resize
            img = Image.open(src_path).convert("RGB")
            img = img.resize((224, 224), Image.LANCZOS)
            
            # Save as JPG
            filename = f"{class_name}_{i:04d}.jpg"
            dst_path = os.path.join(output_dir, split, class_name, filename)
            img.save(dst_path, "JPEG", quality=95)
            
            # Create bias in metadata (intentional for Clarify to detect)
            if class_name == "rust":
                env = random.choices(environments, weights=[0.1, 0.3, 0.6])[0]
                surface = random.choices(surface_types, weights=[0.7, 0.3])[0]
            else:
                env = random.choices(environments, weights=[0.6, 0.3, 0.1])[0]
                surface = random.choices(surface_types, weights=[0.3, 0.7])[0]
            
            metadata_records.append({
                "image_path": dst_path,
                "label": label,
                "label_name": class_name,
                "environment": env,
                "surface_type": surface,
                "split": split,
                "original_file": os.path.basename(src_path),
            })
        
        return len(train_images), len(test_images)
    
    # Process both classes
    rust_train, rust_test = process_class(rust_images, "rust", 1)
    no_rust_train, no_rust_test = process_class(no_rust_images, "no_rust", 0)
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    
    # Save manifests
    train_df = metadata_df[metadata_df["split"] == "train"]
    train_df.to_csv(os.path.join(train_dir, "manifest.csv"), index=False)
    
    test_df = metadata_df[metadata_df["split"] == "test"]
    test_df.to_csv(os.path.join(test_dir, "manifest.csv"), index=False)
    
    print()
    print("=" * 50)
    print("DATASET PREPARED")
    print("=" * 50)
    print(f"Location: {output_dir}")
    print(f"Training: {rust_train + no_rust_train} images")
    print(f"  - rust: {rust_train}")
    print(f"  - no_rust: {no_rust_train}")
    print(f"Test: {rust_test + no_rust_test} images")
    print(f"  - rust: {rust_test}")
    print(f"  - no_rust: {no_rust_test}")
    print()
    print("Bias in data (for Clarify to detect):")
    print(metadata_df.groupby(["environment", "label_name"]).size().unstack(fill_value=0))
    
    return output_dir


def upload_to_s3(local_dir: str, bucket: str, prefix: str):
    """Upload dataset to S3."""
    print()
    print("=" * 50)
    print("UPLOADING TO S3")
    print("=" * 50)
    print(f"From: {local_dir}")
    print(f"To: s3://{bucket}/{prefix}/")
    print()
    
    # Use AWS CLI for faster upload
    subprocess.run([
        "aws", "s3", "sync",
        os.path.join(local_dir, "train"),
        f"s3://{bucket}/{prefix}/train",
        "--quiet"
    ], check=True)
    print("✓ Uploaded train/")
    
    subprocess.run([
        "aws", "s3", "sync",
        os.path.join(local_dir, "test"),
        f"s3://{bucket}/{prefix}/test",
        "--quiet"
    ], check=True)
    print("✓ Uploaded test/")
    
    # Upload metadata
    subprocess.run([
        "aws", "s3", "cp",
        os.path.join(local_dir, "metadata.csv"),
        f"s3://{bucket}/{prefix}/metadata/metadata.csv",
        "--quiet"
    ], check=True)
    print("✓ Uploaded metadata/")
    
    print()
    print("=" * 50)
    print("UPLOAD COMPLETE")
    print("=" * 50)
    print()
    print("S3 Structure:")
    print(f"  s3://{bucket}/{prefix}/")
    print(f"  ├── train/")
    print(f"  │   ├── no_rust/*.jpg")
    print(f"  │   └── rust/*.jpg")
    print(f"  ├── test/")
    print(f"  │   ├── no_rust/*.jpg")
    print(f"  │   └── rust/*.jpg")
    print(f"  └── metadata/")
    print(f"      └── metadata.csv")
    print()
    print("These paths match your pipelines-config.yml!")


def main():
    parser = argparse.ArgumentParser(description="Prepare NEU dataset and upload to S3")
    parser.add_argument("--download", action="store_true", help="Download from Kaggle")
    parser.add_argument("--neu-dir", default="./neu-download", help="NEU dataset directory")
    parser.add_argument("--output-dir", default="./data", help="Processed output directory")
    parser.add_argument("--max-images", type=int, default=None, help="Limit total images (for POC)")
    parser.add_argument("--upload", action="store_true", help="Upload to S3")
    parser.add_argument("--bucket", default="tdsp-data-products-dev", help="S3 bucket")
    parser.add_argument("--prefix", default="tdspds/rust-detection/data", help="S3 prefix")
    
    args = parser.parse_args()
    
    # Step 1: Download if requested
    if args.download:
        download_from_kaggle(args.neu_dir)
    
    # Step 2: Process dataset
    if os.path.exists(args.neu_dir) or args.download:
        process_neu_dataset(args.neu_dir, args.output_dir, args.max_images)
    else:
        print(f"ERROR: NEU dataset not found at {args.neu_dir}")
        print()
        print("Please either:")
        print("  1. Use --download flag to download from Kaggle")
        print("  2. Download manually and specify --neu-dir")
        print()
        print("Download links:")
        print("  Kaggle: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database")
        print("  Direct: https://drive.google.com/file/d/1qrdZlaDi272eA79FX4ZVSY0VzWOcXS8G/view")
        return
    
    # Step 3: Upload to S3 if requested
    if args.upload:
        upload_to_s3(args.output_dir, args.bucket, args.prefix)
    
    print()
    print("=" * 50)
    print("NEXT STEPS")
    print("=" * 50)
    print()
    if not args.upload:
        print("1. Upload to S3:")
        print(f"   python scripts/prepare_neu_and_upload.py --neu-dir {args.neu_dir} --upload")
        print()
    print("2. Train the model:")
    print("   python scripts/train_local.py --data-dir ./data --epochs 5")
    print()
    print("3. Upload predictions to S3:")
    print("   aws s3 cp ./output/predictions_with_metadata.csv \\")
    print(f"       s3://{args.bucket}/tdspds/rust-detection/predictions.csv")


if __name__ == "__main__":
    main()
