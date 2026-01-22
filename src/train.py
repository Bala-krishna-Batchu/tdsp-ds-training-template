"""
Computer Vision Model Training Script

Trains a CV model (e.g., rust detection) for image classification.
Saves model and predictions to S3 for downstream bias monitoring.

Data is read from S3 via SageMaker:
- S3: s3://bucket/data/train/ → /opt/ml/input/data/train/
- S3: s3://bucket/data/test/  → /opt/ml/input/data/test/
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import boto3
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

logger = logging.getLogger("root")
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

TEST_SIZE = 0.2
RANDOM_STATE = 42


class ImageDataset(Dataset):
    """Image classification dataset."""

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
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


def read_config(env: str) -> Dict[str, Any]:
    """Read configuration file."""
    file_path = f"/opt/ml/code/deploy/{env}/tdspds-config.yml"
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configurations = {config}")
    return config


def get_hyperparameters() -> Dict[str, Any]:
    """Get hyperparameters from SageMaker."""
    path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def get_resource_config():
    """Get resource configuration."""
    path = "/opt/ml/input/config/resourceconfig.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            config = json.load(f)
            print(config)


def load_data_from_directory(data_path: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Load images from directory structure: data_path/class_name/image.jpg
    """
    image_paths = []
    labels = []

    class_dirs = sorted([d for d in os.listdir(data_path) 
                         if os.path.isdir(os.path.join(data_path, d))])

    if not class_dirs:
        raise ValueError(f"No class directories found in {data_path}")

    for class_idx, class_name in enumerate(class_dirs):
        class_dir = os.path.join(data_path, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_idx)

    logger.info(f"Loaded {len(image_paths)} images from {len(class_dirs)} classes: {class_dirs}")
    return image_paths, labels, class_dirs


def get_model(num_classes: int, architecture: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """Get pre-trained model for transfer learning."""
    weights = "IMAGENET1K_V1" if pretrained else None

    if architecture == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def train_model(model, train_loader, test_loader, num_epochs, learning_rate, device):
    """Train the model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0

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

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        })

        logger.info(f"Epoch [{epoch+1}/{num_epochs}] Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), "/opt/ml/model/best_model.pth")

    return model, best_accuracy


def generate_predictions(model, data_loader, class_names, device) -> pd.DataFrame:
    """Generate predictions CSV for bias monitoring."""
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

    # Add per-class probabilities
    for i, class_name in enumerate(class_names):
        df[f"prob_{class_name}"] = [p[i] for p in all_probs]

    return df


def register_model(model_path: str, model_name: str, pod_name: str, project_name: str):
    """Register model to W&B."""
    artifact = wandb.Artifact(model_name, type="model")
    artifact.add_file(model_path)
    arti = wandb.log_artifact(artifact)
    arti.wait()
    wandb.run.link_artifact(artifact, f"{pod_name}/{project_name}/{model_name}")
    logger.info("Model registered to W&B")


def upload_to_s3(local_path: str, s3_bucket: str, s3_key: str) -> str:
    """Upload file to S3."""
    s3 = boto3.client("s3")
    s3.upload_file(local_path, s3_bucket, s3_key)
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    logger.info(f"Uploaded to {s3_uri}")
    return s3_uri


def train(env: str):
    """Main training function."""
    config = read_config(env)
    hyperparams = get_hyperparameters()
    get_resource_config()

    # Config
    project_name = config.get("Model", {}).get("ProjectName", "CV Model")
    model_name = config.get("Model", {}).get("ModelName", "cv_model")
    pod_name = config.get("TeamDetails", {}).get("PodName", "tdspds")
    cv_config = config.get("CVModel", {})
    s3_config = config.get("S3Output", {})

    # Hyperparameters
    architecture = hyperparams.get("model_architecture", cv_config.get("Architecture", "resnet18"))
    num_classes = int(hyperparams.get("num_classes", cv_config.get("NumClasses", 2)))
    batch_size = int(hyperparams.get("batch_size", 32))
    num_epochs = int(hyperparams.get("num_epochs", 10))
    learning_rate = float(hyperparams.get("learning_rate", 0.001))
    pretrained = str(hyperparams.get("pretrained", "true")).lower() == "true"

    # Initialize W&B
    wandb.init(
        entity=pod_name,
        project=project_name,
        name=f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data from SageMaker paths (downloaded from S3)
    train_path = "/opt/ml/input/data/train"
    test_path = "/opt/ml/input/data/test"

    train_images, train_labels, class_names = load_data_from_directory(train_path)

    # Load test data or split
    if os.path.exists(test_path) and os.listdir(test_path):
        test_images, test_labels, _ = load_data_from_directory(test_path)
    else:
        train_images, test_images, train_labels, test_labels = train_test_split(
            train_images, train_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=train_labels
        )

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

    # Datasets and loaders
    train_dataset = ImageDataset(train_images, train_labels, train_transform)
    test_dataset = ImageDataset(test_images, test_labels, test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}, Classes: {class_names}")

    # Model
    model = get_model(num_classes, architecture, pretrained)

    # Train
    logger.info("Starting training...")
    model, best_accuracy = train_model(model, train_loader, test_loader, num_epochs, learning_rate, device)

    # Generate predictions for metrics logging
    predictions_df = generate_predictions(model, test_loader, class_names, device)

    # Log metrics
    accuracy = accuracy_score(predictions_df["ground_truth"], predictions_df["prediction"])
    wandb.log({"test_accuracy": accuracy, "best_val_accuracy": best_accuracy})
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    # S3 config
    s3_bucket = s3_config.get("Bucket", "tdsp-ml-products-dev")
    s3_prefix = s3_config.get("Prefix", f"{pod_name}/{model_name}")

    # =========================================================================
    # OUTPUT FOR MONITORING REPO (Clarify Bias Analysis)
    # =========================================================================
    # Clarify needs:
    #   1. model.tar.gz - Trained model (Clarify runs inference)
    #   2. test_data.csv - Raw test data with image paths + ground truth labels
    # =========================================================================

    os.makedirs("/opt/ml/output", exist_ok=True)

    # 1. Create model.tar.gz (SageMaker format for Clarify)
    import tarfile
    model_tar_path = "/opt/ml/model/model.tar.gz"
    with tarfile.open(model_tar_path, "w:gz") as tar:
        tar.add("/opt/ml/model/best_model.pth", arcname="model.pth")
        # Save model config for inference
        model_config = {
            "architecture": architecture,
            "num_classes": num_classes,
            "class_names": class_names,
        }
        config_path = "/opt/ml/model/model_config.json"
        with open(config_path, "w") as f:
            json.dump(model_config, f)
        tar.add(config_path, arcname="model_config.json")
    
    logger.info(f"Created model.tar.gz at {model_tar_path}")

    # 2. Create test_data.csv (raw data for Clarify to run inference on)
    #    Format: image_path, ground_truth_label
    test_data_df = pd.DataFrame({
        "image_path": test_images,
        "ground_truth_label": test_labels,
        "ground_truth_name": [class_names[l] for l in test_labels],
    })
    test_data_path = "/opt/ml/output/test_data.csv"
    test_data_df.to_csv(test_data_path, index=False)
    logger.info(f"Created test_data.csv with {len(test_data_df)} samples")

    # 3. Upload to S3 for monitoring repo
    # Model artifact
    upload_to_s3(model_tar_path, s3_bucket, f"{s3_prefix}/model/model.tar.gz")
    
    # Test data (raw - Clarify runs inference)
    upload_to_s3(test_data_path, s3_bucket, f"{s3_prefix}/test_data.csv")
    
    # Also save predictions (for reference/validation)
    predictions_path = "/opt/ml/output/predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    upload_to_s3(predictions_path, s3_bucket, f"{s3_prefix}/predictions.csv")

    # Register model
    register_model("/opt/ml/model/best_model.pth", model_name, pod_name, project_name)

    wandb.finish()
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE - OUTPUTS FOR MONITORING REPO:")
    logger.info("=" * 60)
    logger.info(f"Model artifact: s3://{s3_bucket}/{s3_prefix}/model/model.tar.gz")
    logger.info(f"Test data CSV:  s3://{s3_bucket}/{s3_prefix}/test_data.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    env = os.getenv("ENV", "dev")
    train(env)
