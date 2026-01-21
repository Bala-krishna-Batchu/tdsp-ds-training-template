"""
Minimal POC Dataset Preparation

Creates a small dataset (~100-200 images) sufficient to demonstrate:
1. CV model training works
2. Predictions are generated correctly
3. Bias monitoring repo can detect bias

This intentionally introduces bias in the data so Clarify can detect it.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


def create_minimal_poc_dataset(output_dir: str, num_images: int = 100):
    """
    Create minimal POC dataset with intentional bias for demonstration.
    
    Args:
        output_dir: Where to save the dataset
        num_images: Total number of images (default 100)
    """
    print(f"Creating minimal POC dataset with {num_images} images...")
    
    # Create directories
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    for class_name in ["no_rust", "rust"]:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    metadata_records = []
    
    # Sensitive attributes for bias analysis
    # We'll intentionally create bias: model will perform worse on "coastal" environment
    environments = ["indoor", "outdoor", "coastal"]
    surface_types = ["steel", "aluminum"]
    
    # Distribution to create detectable bias
    # Rust is more common in coastal, less in indoor
    # This will make the model biased toward certain environments
    
    for i in range(num_images):
        # 50/50 split between classes
        is_rust = i % 2 == 0
        class_name = "rust" if is_rust else "no_rust"
        label = 1 if is_rust else 0
        
        # 80% train, 20% test
        split = "train" if i < int(num_images * 0.8) else "test"
        
        # Create INTENTIONAL BIAS in the data:
        # - "coastal" environment: mostly rust images
        # - "indoor" environment: mostly no_rust images
        # This will cause the model to associate environment with prediction
        if is_rust:
            # Rust images: 60% coastal, 30% outdoor, 10% indoor
            env = random.choices(environments, weights=[0.1, 0.3, 0.6])[0]
            surface = random.choices(surface_types, weights=[0.7, 0.3])[0]
        else:
            # No rust images: 60% indoor, 30% outdoor, 10% coastal
            env = random.choices(environments, weights=[0.6, 0.3, 0.1])[0]
            surface = random.choices(surface_types, weights=[0.3, 0.7])[0]
        
        # Create image
        img = create_image(is_rust, env)
        
        # Save
        img_filename = f"{class_name}_{i:04d}.jpg"
        img_path = os.path.join(output_dir, split, class_name, img_filename)
        img.save(img_path, quality=90)
        
        metadata_records.append({
            "image_path": img_path,
            "label": label,
            "label_name": class_name,
            "environment": env,
            "surface_type": surface,
            "split": split,
        })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    
    # Full metadata
    metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    
    # Train manifest
    train_df = metadata_df[metadata_df["split"] == "train"]
    train_df.to_csv(os.path.join(train_dir, "manifest.csv"), index=False)
    
    # Test manifest  
    test_df = metadata_df[metadata_df["split"] == "test"]
    test_df.to_csv(os.path.join(test_dir, "manifest.csv"), index=False)
    
    # Print summary
    print("\n" + "=" * 50)
    print("DATASET CREATED")
    print("=" * 50)
    print(f"Total images: {num_images}")
    print(f"Training: {len(train_df)}")
    print(f"Test: {len(test_df)}")
    print(f"Location: {output_dir}")
    
    print("\n" + "=" * 50)
    print("INTENTIONAL BIAS IN DATA (for Clarify to detect)")
    print("=" * 50)
    print("\nClass distribution by environment:")
    print(metadata_df.groupby(["environment", "label_name"]).size().unstack(fill_value=0))
    
    print("\nClass distribution by surface_type:")
    print(metadata_df.groupby(["surface_type", "label_name"]).size().unstack(fill_value=0))
    
    print("\n" + "=" * 50)
    print("EXPECTED BIAS RESULTS")
    print("=" * 50)
    print("""
When you run Clarify on this data, it should detect:

1. Environment Bias:
   - Model likely performs BETTER on "indoor" (more no_rust samples)
   - Model likely performs WORSE on "coastal" (more rust samples)
   - Expected DI (Disparate Impact) deviation for coastal vs indoor

2. Surface Type Bias:
   - Steel associated more with rust
   - Aluminum associated more with no_rust
   - Expected accuracy difference between groups

These biases are INTENTIONAL to demonstrate your monitoring works!
""")
    
    return metadata_df


def create_image(is_rust: bool, environment: str, size: int = 224) -> Image.Image:
    """Create a synthetic metal surface image."""
    
    # Base color based on class and environment
    if is_rust:
        # Rust colors vary by environment (adds correlation for bias)
        if environment == "coastal":
            base_color = (180, 80, 40)   # Strong rust (coastal = more rust)
        elif environment == "outdoor":
            base_color = (160, 90, 50)   # Medium rust
        else:
            base_color = (140, 100, 60)  # Light rust (indoor = less rust)
    else:
        # Clean metal colors
        if environment == "indoor":
            base_color = (200, 200, 200)  # Bright silver (indoor = cleaner)
        elif environment == "outdoor":
            base_color = (170, 170, 175)  # Slightly darker
        else:
            base_color = (150, 155, 160)  # Darkest (coastal = some wear)
    
    # Create image with texture
    img = Image.new("RGB", (size, size), base_color)
    pixels = img.load()
    
    # Add noise/texture
    for x in range(size):
        for y in range(size):
            noise = random.randint(-25, 25)
            r = max(0, min(255, base_color[0] + noise))
            g = max(0, min(255, base_color[1] + noise))
            b = max(0, min(255, base_color[2] + noise))
            pixels[x, y] = (r, g, b)
    
    # Add rust spots if rust class
    if is_rust:
        draw = ImageDraw.Draw(img)
        num_spots = random.randint(2, 8)
        for _ in range(num_spots):
            x = random.randint(20, size - 20)
            y = random.randint(20, size - 20)
            radius = random.randint(5, 25)
            rust_color = (
                random.randint(120, 180),
                random.randint(40, 80),
                random.randint(20, 50)
            )
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=rust_color)
    
    # Blur slightly for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Create minimal POC dataset")
    parser.add_argument("--output", default="./data", help="Output directory")
    parser.add_argument("--num-images", type=int, default=100, help="Number of images")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    create_minimal_poc_dataset(args.output, args.num_images)
    
    print("\n" + "=" * 50)
    print("NEXT STEPS")
    print("=" * 50)
    print(f"""
1. Train the model:
   python scripts/train_local.py --data-dir {args.output} --epochs 5

2. Check the outputs:
   ls ./output/
   - predictions.csv (for Clarify)
   - predictions_with_metadata.csv (with sensitive attributes)

3. Upload to S3:
   aws s3 cp ./output/predictions_with_metadata.csv \\
       s3://tdsp-ml-products-dev/tdspds/rust-detection/predictions.csv

4. Your monitoring repo runs Clarify on this S3 path
""")


if __name__ == "__main__":
    main()
