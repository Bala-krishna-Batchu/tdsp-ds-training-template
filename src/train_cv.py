"""
Computer Vision Model Training Script

Trains a CV model (e.g., rust detection) and saves:
1. Model artifacts to S3 (/opt/ml/model/)
2. Predictions and metrics to W&B
3. Results CSV to S3 for downstream bias monitoring

This script replaces the churn prediction model while maintaining
the existing structure for TDSPDS integration.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

logger = logging.getLogger("root")
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)


# =============================================================================
# DATASET
# =============================================================================

class ImageDataset(Dataset):
    """Simple image classification dataset."""

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            logger.warning(f"Error loading {self.image_paths[idx]}: {e}")
            image = torch.zeros(3, 224, 224)
        return image, self.labels[idx]


# =============================================================================
# MODEL
# =============================================================================

def get_model(num_classes: int, architecture: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """Get pre-trained model for transfer learning."""
    
    weights = "IMAGENET1K_V1" if pretrained else None
    
    if architecture == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        logger.warning(f"Unknown architecture {architecture}, using resnet18")
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_from_directory(data_path: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Load images from directory structure: data_path/class_name/image.jpg
    
    Returns:
        image_paths, labels, class_names
    """
    image_paths = []
    labels = []
    
    class_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    
    if not class_dirs:
        raise ValueError(f"No class directories found in {data_path}")
    
    for class_idx, class_name in enumerate(class_dirs):
        class_dir = os.path.join(data_path, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_idx)
    
    logger.info(f"Loaded {len(image_paths)} images from {len(class_dirs)} classes: {class_dirs}")
    return image_paths, labels, class_dirs


def load_data_from_manifest(manifest_path: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Load images from manifest CSV with columns: image_path, label
    
    Returns:
        image_paths, labels, class_names
    """
    df = pd.read_csv(manifest_path)
    image_paths = df["image_path"].tolist()
    
    if df["label"].dtype == object:
        class_names = sorted(df["label"].unique().tolist())
        label_map = {name: idx for idx, name in enumerate(class_names)}
        labels = [label_map[l] for l in df["label"]]
    else:
        labels = df["label"].tolist()
        class_names = [str(i) for i in range(max(labels) + 1)]
    
    logger.info(f"Loaded {len(image_paths)} images from manifest")
    return image_paths, labels, class_names


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train the model and return training history."""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_accuracy = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
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
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Metrics
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        
        history["train_loss"].append(train_loss / len(train_loader))
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss / len(test_loader))
        history["val_acc"].append(val_acc)
        
        # Log to W&B
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "train_accuracy": train_acc,
            "val_loss": val_loss / len(test_loader),
            "val_accuracy": val_acc,
        })
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), "/opt/ml/model/best_model.pth")
    
    history["best_accuracy"] = best_accuracy
    return model, history


# =============================================================================
# GENERATE RESULTS FOR MONITORING REPO
# =============================================================================

def generate_results(
    model: nn.Module,
    data_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate predictions CSV for your monitoring repo to consume.
    
    Output format (what your Clarify monitoring repo expects):
    - prediction: predicted class index
    - ground_truth: actual class index  
    - confidence: max probability
    - prob_class_0, prob_class_1, ...: per-class probabilities
    """
    
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
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        "prediction": all_preds,
        "ground_truth": all_labels,
        "confidence": [max(p) for p in all_probs],
    })
    
    # Add per-class probabilities
    for i, class_name in enumerate(class_names):
        results_df[f"prob_{class_name}"] = [p[i] for p in all_probs]
    
    # Save locally
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "predictions.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved predictions to {results_path}")
    
    return results_df


def calculate_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    y_true = results_df["ground_truth"]
    y_pred = results_df["prediction"]
    
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# =============================================================================
# S3 UPLOAD
# =============================================================================

def upload_to_s3(local_path: str, s3_bucket: str, s3_key: str) -> str:
    """Upload file to S3 and return S3 URI."""
    s3 = boto3.client("s3")
    s3.upload_file(local_path, s3_bucket, s3_key)
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    logger.info(f"Uploaded {local_path} to {s3_uri}")
    return s3_uri


# =============================================================================
# CONFIG
# =============================================================================

def read_config(env: str) -> Dict[str, Any]:
    """Read configuration file."""
    file_path = f"/opt/ml/code/deploy/{env}/tdspds-config.yml"
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {file_path}")
    return config


def get_hyperparameters() -> Dict[str, Any]:
    """Get hyperparameters from SageMaker config."""
    path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


# =============================================================================
# MAIN
# =============================================================================

def train_cv(env: str):
    """Main training function."""
    
    config = read_config(env)
    hyperparams = get_hyperparameters()
    
    # Config values
    project_name = config.get("Model", {}).get("ProjectName", "CV Model")
    model_name = config.get("Model", {}).get("ModelName", "cv_model")
    pod_name = config.get("TeamDetails", {}).get("PodName", "tdspds")
    
    cv_config = config.get("CVModel", {})
    s3_config = config.get("S3Output", {})
    
    # Hyperparameters (from SageMaker or defaults)
    architecture = hyperparams.get("model_architecture", cv_config.get("Architecture", "resnet18"))
    num_classes = int(hyperparams.get("num_classes", cv_config.get("NumClasses", 2)))
    batch_size = int(hyperparams.get("batch_size", 32))
    num_epochs = int(hyperparams.get("num_epochs", 10))
    learning_rate = float(hyperparams.get("learning_rate", 0.001))
    pretrained = hyperparams.get("pretrained", "true").lower() == "true"
    
    # Initialize W&B
    wandb.init(
        entity=pod_name,
        project=project_name,
        name=f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "architecture": architecture,
            "num_classes": num_classes,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
        },
    )
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    train_path = "/opt/ml/input/data/train"
    test_path = "/opt/ml/input/data/test"
    
    manifest_path = os.path.join(train_path, "manifest.csv")
    if os.path.exists(manifest_path):
        train_images, train_labels, class_names = load_data_from_manifest(manifest_path)
    else:
        train_images, train_labels, class_names = load_data_from_directory(train_path)
    
    # Test data
    test_manifest = os.path.join(test_path, "manifest.csv")
    if os.path.exists(test_path):
        if os.path.exists(test_manifest):
            test_images, test_labels, _ = load_data_from_manifest(test_manifest)
        else:
            test_images, test_labels, _ = load_data_from_directory(test_path)
    else:
        # Split training data
        from sklearn.model_selection import train_test_split
        train_images, test_images, train_labels, test_labels = train_test_split(
            train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )
    
    # Data loaders
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
    
    train_dataset = ImageDataset(train_images, train_labels, train_transform)
    test_dataset = ImageDataset(test_images, test_labels, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    logger.info(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    logger.info(f"Classes: {class_names}")
    
    # Create model
    model = get_model(num_classes, architecture, pretrained)
    
    # Train
    logger.info("Starting training...")
    model, history = train_model(model, train_loader, test_loader, num_epochs, learning_rate, device)
    
    # Generate results for monitoring repo
    output_dir = "/opt/ml/output"
    results_df = generate_results(model, test_loader, class_names, device, output_dir)
    
    # Calculate final metrics
    metrics = calculate_metrics(results_df)
    logger.info(f"Final metrics: {metrics}")
    
    # Log to W&B
    wandb.log({"final_" + k: v for k, v in metrics.items()})
    wandb.log({"best_val_accuracy": history["best_accuracy"]})
    
    # Save model info
    model_info = {
        "model_name": model_name,
        "architecture": architecture,
        "num_classes": num_classes,
        "class_names": class_names,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    
    model_info_path = os.path.join(output_dir, "model_info.json")
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Upload to S3 (for your monitoring repo)
    s3_bucket = s3_config.get("Bucket", "tdsp-ml-products-dev")
    s3_prefix = s3_config.get("Prefix", f"{pod_name}/{model_name}")
    
    # Upload predictions CSV - THIS IS WHAT YOUR MONITORING REPO READS
    predictions_s3_uri = upload_to_s3(
        os.path.join(output_dir, "predictions.csv"),
        s3_bucket,
        f"{s3_prefix}/predictions.csv"
    )
    
    # Upload model info
    upload_to_s3(model_info_path, s3_bucket, f"{s3_prefix}/model_info.json")
    
    # Upload model
    upload_to_s3("/opt/ml/model/best_model.pth", s3_bucket, f"{s3_prefix}/model/best_model.pth")
    
    # Register with W&B
    artifact = wandb.Artifact(model_name, type="model")
    artifact.add_file("/opt/ml/model/best_model.pth")
    artifact.add_file(os.path.join(output_dir, "predictions.csv"))
    artifact.add_file(model_info_path)
    wandb.log_artifact(artifact)
    wandb.run.link_artifact(artifact, f"{pod_name}/{project_name}/{model_name}")
    
    wandb.finish()
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved to: s3://{s3_bucket}/{s3_prefix}/model/")
    logger.info(f"Predictions saved to: {predictions_s3_uri}")
    logger.info("Your monitoring repo can now use this S3 path for Clarify bias analysis")
    logger.info("=" * 60)


if __name__ == "__main__":
    env = os.getenv("ENV", "dev")
    train_cv(env)
