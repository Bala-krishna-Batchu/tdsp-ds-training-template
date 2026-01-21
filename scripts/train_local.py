"""
Local Training Script for Testing

This script allows you to train the CV model locally (without SageMaker)
for testing and demonstration purposes.

Usage:
    # First, prepare the dataset
    python scripts/prepare_dataset.py --dataset demo --output ./data --num-images 500
    
    # Then train locally
    python scripts/train_local.py --data-dir ./data --epochs 5
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class ImageDataset(Dataset):
    """Simple image classification dataset."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, 224, 224)
        return image, self.labels[idx]


def load_data(data_dir, split="train"):
    """Load data from directory structure."""
    split_dir = os.path.join(data_dir, split)
    image_paths = []
    labels = []
    
    class_names = sorted([d for d in os.listdir(split_dir) 
                          if os.path.isdir(os.path.join(split_dir, d))])
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(split_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_idx)
    
    return image_paths, labels, class_names


def train_model(model, train_loader, test_loader, num_epochs, device):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% "
              f"Val Acc: {val_acc:.2f}%")
    
    return model


def generate_predictions(model, data_loader, class_names, device):
    """Generate predictions for bias monitoring."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    
    # Create DataFrame
    df = pd.DataFrame({
        "prediction": all_preds,
        "ground_truth": all_labels,
        "confidence": [max(p) for p in all_probs],
    })
    
    for i, class_name in enumerate(class_names):
        df[f"prob_{class_name}"] = [p[i] for p in all_probs]
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Train CV model locally")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--architecture", default="resnet18", help="Model architecture")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    train_images, train_labels, class_names = load_data(args.data_dir, "train")
    test_images, test_labels, _ = load_data(args.data_dir, "test")
    
    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(test_images)}")
    print(f"Classes: {class_names}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Datasets
    train_dataset = ImageDataset(train_images, train_labels, train_transform)
    test_dataset = ImageDataset(test_images, test_labels, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    print(f"\nCreating {args.architecture} model...")
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    model = train_model(model, train_loader, test_loader, args.epochs, device)
    
    # Generate predictions for monitoring repo
    print("\nGenerating predictions for bias monitoring...")
    predictions_df = generate_predictions(model, test_loader, class_names, device)
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")
    
    # Print metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    y_true = predictions_df["ground_truth"]
    y_pred = predictions_df["prediction"]
    
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Save model
    model_path = os.path.join(args.output_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Merge with metadata if available
    metadata_path = os.path.join(args.data_dir, "metadata.csv")
    if os.path.exists(metadata_path):
        print("\nMerging predictions with metadata for bias analysis...")
        metadata_df = pd.read_csv(metadata_path)
        
        # Filter to test set
        test_metadata = metadata_df[metadata_df["image_path"].str.contains("/test/")]
        
        if len(test_metadata) == len(predictions_df):
            merged_df = test_metadata.reset_index(drop=True)
            merged_df["prediction"] = predictions_df["prediction"]
            merged_df["ground_truth"] = predictions_df["ground_truth"]
            merged_df["confidence"] = predictions_df["confidence"]
            
            for col in predictions_df.columns:
                if col.startswith("prob_"):
                    merged_df[col] = predictions_df[col]
            
            merged_path = os.path.join(args.output_dir, "predictions_with_metadata.csv")
            merged_df.to_csv(merged_path, index=False)
            print(f"Saved predictions with metadata to {merged_path}")
            
            # Show bias in predictions by group
            print("\n" + "=" * 60)
            print("BIAS ANALYSIS PREVIEW")
            print("=" * 60)
            
            for attr in ["environment", "surface_type"]:
                if attr in merged_df.columns:
                    print(f"\nAccuracy by {attr}:")
                    for group in merged_df[attr].unique():
                        group_df = merged_df[merged_df[attr] == group]
                        acc = accuracy_score(group_df["ground_truth"], group_df["prediction"])
                        print(f"  {group}: {acc:.4f} (n={len(group_df)})")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"\n1. Upload to S3 for your monitoring repo:")
    print(f"   aws s3 cp {predictions_path} s3://your-bucket/rust-detection/predictions.csv")
    print(f"\n2. Your monitoring repo can now run Clarify on this S3 path")
    print(f"\n3. The metadata columns (environment, surface_type) can be used")
    print(f"   as sensitive attributes for bias analysis")


if __name__ == "__main__":
    main()
