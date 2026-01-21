"""
Dataset Preparation Script for Rust Detection

This script downloads and prepares datasets for rust detection training
and creates metadata for bias monitoring demonstration.

DATASETS AVAILABLE:
==================

1. NEU Surface Defect Dataset (Recommended - Free)
   - 1,800 grayscale images (300x300)
   - 6 defect types including rust/oxidation
   - Link: http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html

2. Kaggle Corrosion Dataset (Requires Kaggle account)
   - ~1,000 images of corrosion/no-corrosion
   - Link: https://www.kaggle.com/datasets/harshulgupta001/corrosion

3. Synthetic Demo Dataset (Generated - for quick testing)
   - Creates sample images for pipeline testing

Usage:
    python scripts/prepare_dataset.py --dataset neu --output ./data
    python scripts/prepare_dataset.py --dataset demo --output ./data
"""

import argparse
import json
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


def download_file(url: str, output_path: str) -> None:
    """Download a file with progress."""
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")


def create_demo_dataset(output_dir: str, num_images: int = 500) -> Tuple[List[str], List[int]]:
    """
    Create a synthetic demo dataset for testing.
    
    Creates simple images that simulate rust/no-rust classification.
    """
    print(f"Creating demo dataset with {num_images} images...")
    
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    for class_name in ["no_rust", "rust"]:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    image_paths = []
    labels = []
    metadata_records = []
    
    # Attributes for bias analysis
    surface_types = ["metal", "steel", "iron", "aluminum"]
    environments = ["indoor", "outdoor", "coastal", "industrial"]
    lighting_conditions = ["bright", "dim", "natural", "artificial"]
    
    for i in range(num_images):
        # Determine class
        is_rust = random.random() > 0.5
        class_name = "rust" if is_rust else "no_rust"
        label = 1 if is_rust else 0
        
        # Randomly assign to train (80%) or test (20%)
        split = "train" if random.random() < 0.8 else "test"
        
        # Create image
        img = create_synthetic_image(is_rust)
        
        # Save image
        img_filename = f"img_{i:04d}.jpg"
        img_path = os.path.join(output_dir, split, class_name, img_filename)
        img.save(img_path, quality=95)
        
        image_paths.append(img_path)
        labels.append(label)
        
        # Generate metadata for bias analysis
        # Introduce some bias: rust more common in certain conditions
        if is_rust:
            # Bias: rust more likely in coastal/outdoor environments
            env = random.choices(environments, weights=[0.1, 0.3, 0.4, 0.2])[0]
            surface = random.choices(surface_types, weights=[0.2, 0.3, 0.3, 0.2])[0]
        else:
            # No rust more likely in indoor/aluminum
            env = random.choices(environments, weights=[0.4, 0.2, 0.1, 0.3])[0]
            surface = random.choices(surface_types, weights=[0.2, 0.2, 0.2, 0.4])[0]
        
        metadata_records.append({
            "image_path": img_path,
            "label": class_name,
            "surface_type": surface,
            "environment": env,
            "lighting": random.choice(lighting_conditions),
            "image_quality": random.choice(["high", "medium", "low"]),
        })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    
    # Create train manifest
    train_records = [r for r in metadata_records if "/train/" in r["image_path"]]
    pd.DataFrame(train_records).to_csv(os.path.join(train_dir, "manifest.csv"), index=False)
    
    # Create test manifest
    test_records = [r for r in metadata_records if "/test/" in r["image_path"]]
    pd.DataFrame(test_records).to_csv(os.path.join(test_dir, "manifest.csv"), index=False)
    
    print(f"Created {len(train_records)} training images")
    print(f"Created {len(test_records)} test images")
    print(f"Metadata saved to {output_dir}/metadata.csv")
    
    return image_paths, labels


def create_synthetic_image(is_rust: bool, size: int = 224) -> Image.Image:
    """Create a synthetic image for rust/no-rust classification."""
    
    # Base metal color
    if is_rust:
        # Rusty colors: orange, brown, reddish
        base_color = random.choice([
            (139, 69, 19),    # saddle brown
            (160, 82, 45),    # sienna
            (205, 133, 63),   # peru
            (210, 105, 30),   # chocolate
            (178, 34, 34),    # firebrick
        ])
    else:
        # Clean metal colors: gray, silver
        base_color = random.choice([
            (169, 169, 169),  # dark gray
            (192, 192, 192),  # silver
            (211, 211, 211),  # light gray
            (119, 136, 153),  # light slate gray
            (176, 196, 222),  # light steel blue
        ])
    
    # Create base image
    img = Image.new("RGB", (size, size), base_color)
    draw = ImageDraw.Draw(img)
    
    # Add texture/noise
    pixels = img.load()
    for x in range(size):
        for y in range(size):
            noise = random.randint(-20, 20)
            r = max(0, min(255, base_color[0] + noise))
            g = max(0, min(255, base_color[1] + noise))
            b = max(0, min(255, base_color[2] + noise))
            pixels[x, y] = (r, g, b)
    
    if is_rust:
        # Add rust spots
        num_spots = random.randint(3, 10)
        for _ in range(num_spots):
            x = random.randint(10, size - 10)
            y = random.randint(10, size - 10)
            radius = random.randint(5, 30)
            rust_color = random.choice([
                (139, 69, 19),
                (160, 82, 45),
                (205, 92, 0),
                (178, 34, 34),
            ])
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=rust_color)
    
    # Apply slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    return img


def download_neu_dataset(output_dir: str) -> None:
    """
    Download and prepare NEU Surface Defect Dataset.
    
    Note: The official dataset requires manual download from:
    http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
    
    This function provides instructions and prepares the directory structure.
    """
    print("=" * 60)
    print("NEU Surface Defect Dataset")
    print("=" * 60)
    print()
    print("Please download the dataset manually from:")
    print("http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html")
    print()
    print("Alternative mirrors:")
    print("- https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database")
    print("- https://github.com/abin24/Surface-Inspection-defect-detection-dataset")
    print()
    print(f"After downloading, extract to: {output_dir}/NEU-DET/")
    print()
    print("Expected structure:")
    print("  NEU-DET/")
    print("  ├── IMAGES/")
    print("  │   ├── Cr_1.bmp (Crazing)")
    print("  │   ├── In_1.bmp (Inclusion)")
    print("  │   ├── Pa_1.bmp (Patches)")
    print("  │   ├── PS_1.bmp (Pitted Surface)")
    print("  │   ├── RS_1.bmp (Rolled-in Scale - similar to rust)")
    print("  │   └── Sc_1.bmp (Scratches)")
    print()
    
    # Create directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a script to process NEU dataset after download
    process_script = '''#!/usr/bin/env python3
"""
Run this after downloading NEU dataset to prepare it for training.
"""
import os
import shutil
from sklearn.model_selection import train_test_split

NEU_DIR = "./NEU-DET/IMAGES"
OUTPUT_DIR = "./data"

# Map NEU classes to rust/no_rust
# RS (Rolled-in Scale) and Pa (Patches) are similar to rust/corrosion
RUST_CLASSES = ["RS", "Pa"]  # Treat as rust
NO_RUST_CLASSES = ["Cr", "In", "PS", "Sc"]  # Treat as no rust

def process():
    train_dir = os.path.join(OUTPUT_DIR, "train")
    test_dir = os.path.join(OUTPUT_DIR, "test")
    
    for label in ["rust", "no_rust"]:
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label), exist_ok=True)
    
    rust_images = []
    no_rust_images = []
    
    for filename in os.listdir(NEU_DIR):
        if filename.endswith(".bmp"):
            prefix = filename.split("_")[0]
            src_path = os.path.join(NEU_DIR, filename)
            
            if prefix in RUST_CLASSES:
                rust_images.append(src_path)
            elif prefix in NO_RUST_CLASSES:
                no_rust_images.append(src_path)
    
    # Split and copy
    for images, label in [(rust_images, "rust"), (no_rust_images, "no_rust")]:
        train, test = train_test_split(images, test_size=0.2, random_state=42)
        
        for img_path in train:
            shutil.copy(img_path, os.path.join(train_dir, label))
        for img_path in test:
            shutil.copy(img_path, os.path.join(test_dir, label))
    
    print(f"Prepared {len(rust_images)} rust images")
    print(f"Prepared {len(no_rust_images)} no_rust images")

if __name__ == "__main__":
    process()
'''
    
    script_path = os.path.join(output_dir, "process_neu_dataset.py")
    with open(script_path, "w") as f:
        f.write(process_script)
    print(f"Created processing script: {script_path}")


def download_kaggle_corrosion(output_dir: str) -> None:
    """
    Instructions for Kaggle Corrosion Dataset.
    
    Requires Kaggle account and API key.
    """
    print("=" * 60)
    print("Kaggle Corrosion Dataset")
    print("=" * 60)
    print()
    print("Dataset: https://www.kaggle.com/datasets/harshulgupta001/corrosion")
    print()
    print("To download:")
    print("1. Install kaggle: pip install kaggle")
    print("2. Set up Kaggle API key (~/.kaggle/kaggle.json)")
    print("3. Run: kaggle datasets download -d harshulgupta001/corrosion")
    print()
    print("Alternative datasets on Kaggle:")
    print("- https://www.kaggle.com/datasets/harshulgupta001/corrosion")
    print("- https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia (different domain)")
    print()
    
    # Create download script
    download_script = '''#!/bin/bash
# Download Kaggle Corrosion Dataset

# Ensure kaggle is installed
pip install kaggle

# Download dataset
kaggle datasets download -d harshulgupta001/corrosion -p ./data/

# Unzip
cd ./data/
unzip corrosion.zip
rm corrosion.zip

echo "Dataset downloaded to ./data/"
'''
    
    script_path = os.path.join(output_dir, "download_kaggle.sh")
    with open(script_path, "w") as f:
        f.write(download_script)
    os.chmod(script_path, 0o755)
    print(f"Created download script: {script_path}")


def create_metadata_for_existing_dataset(data_dir: str) -> None:
    """
    Create metadata CSV for an existing dataset (for bias monitoring).
    
    This adds synthetic sensitive attributes to demonstrate bias analysis.
    """
    print("Creating metadata for bias monitoring...")
    
    records = []
    
    # Attributes for bias analysis
    surface_types = ["metal", "steel", "iron", "aluminum"]
    environments = ["indoor", "outdoor", "coastal", "industrial"]
    lighting_conditions = ["bright", "dim", "natural", "artificial"]
    
    for split in ["train", "test"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            is_rust = class_name.lower() in ["rust", "corrosion", "defect", "positive"]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    
                    # Introduce bias in metadata distribution
                    if is_rust:
                        env = random.choices(environments, weights=[0.1, 0.3, 0.4, 0.2])[0]
                        surface = random.choices(surface_types, weights=[0.2, 0.3, 0.3, 0.2])[0]
                    else:
                        env = random.choices(environments, weights=[0.4, 0.2, 0.1, 0.3])[0]
                        surface = random.choices(surface_types, weights=[0.2, 0.2, 0.2, 0.4])[0]
                    
                    records.append({
                        "image_path": img_path,
                        "label": class_name,
                        "surface_type": surface,
                        "environment": env,
                        "lighting": random.choice(lighting_conditions),
                        "image_quality": random.choice(["high", "medium", "low"]),
                    })
    
    if records:
        df = pd.DataFrame(records)
        metadata_path = os.path.join(data_dir, "metadata.csv")
        df.to_csv(metadata_path, index=False)
        print(f"Created metadata with {len(records)} records: {metadata_path}")
        
        # Show bias in data
        print("\nData distribution (for bias analysis):")
        print(df.groupby(["label", "environment"]).size().unstack(fill_value=0))
    else:
        print("No images found in the dataset directory.")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for rust detection")
    parser.add_argument(
        "--dataset",
        choices=["demo", "neu", "kaggle", "metadata"],
        default="demo",
        help="Dataset to prepare: demo (synthetic), neu, kaggle, or metadata (add to existing)"
    )
    parser.add_argument(
        "--output",
        default="./data",
        help="Output directory"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=500,
        help="Number of images for demo dataset"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.dataset == "demo":
        create_demo_dataset(args.output, args.num_images)
        print("\n" + "=" * 60)
        print("Demo dataset created successfully!")
        print("=" * 60)
        print(f"\nData location: {args.output}")
        print("\nTo train:")
        print(f"  python src/train_cv.py")
        print("\nTo upload to S3:")
        print(f"  aws s3 sync {args.output} s3://your-bucket/rust-detection/data/")
        
    elif args.dataset == "neu":
        download_neu_dataset(args.output)
        
    elif args.dataset == "kaggle":
        download_kaggle_corrosion(args.output)
        
    elif args.dataset == "metadata":
        create_metadata_for_existing_dataset(args.output)


if __name__ == "__main__":
    main()
