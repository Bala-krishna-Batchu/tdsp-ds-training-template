"""
Computer Vision Data Utilities

Utilities for loading, preprocessing, and preparing CV data for training
and Clarify bias analysis.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import boto3
import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger("root")


class CVDataLoader:
    """Data loader for computer vision datasets with Clarify support."""

    def __init__(self, s3_client: Optional[boto3.client] = None):
        self.s3_client = s3_client or boto3.client("s3")

    def load_dataset_from_s3(
        self,
        s3_bucket: str,
        s3_prefix: str,
        local_path: str,
        include_metadata: bool = True,
    ) -> Tuple[List[str], List[int], Optional[pd.DataFrame]]:
        """
        Download and load dataset from S3.

        Args:
            s3_bucket: S3 bucket name
            s3_prefix: S3 prefix for the dataset
            local_path: Local path to download files
            include_metadata: Whether to load metadata for bias analysis

        Returns:
            Tuple of (image_paths, labels, metadata_df)
        """
        os.makedirs(local_path, exist_ok=True)

        # List objects in S3
        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix)

        downloaded_files = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                local_file = os.path.join(local_path, key.replace(s3_prefix, "").lstrip("/"))

                # Create directories if needed
                os.makedirs(os.path.dirname(local_file), exist_ok=True)

                # Download file
                self.s3_client.download_file(s3_bucket, key, local_file)
                downloaded_files.append(local_file)

        logger.info(f"Downloaded {len(downloaded_files)} files from s3://{s3_bucket}/{s3_prefix}")

        # Load dataset
        return self.load_dataset_from_local(local_path, include_metadata)

    def load_dataset_from_local(
        self,
        data_path: str,
        include_metadata: bool = True,
    ) -> Tuple[List[str], List[int], Optional[pd.DataFrame]]:
        """
        Load dataset from local directory.

        Supports two formats:
        1. Directory structure: data_path/class_name/image.jpg
        2. Manifest file: data_path/manifest.csv with columns [image_path, label, ...]

        Args:
            data_path: Path to dataset
            include_metadata: Whether to load metadata

        Returns:
            Tuple of (image_paths, labels, metadata_df)
        """
        manifest_path = os.path.join(data_path, "manifest.csv")

        if os.path.exists(manifest_path):
            return self._load_from_manifest(manifest_path, include_metadata)
        else:
            return self._load_from_directory(data_path)

    def _load_from_directory(
        self, data_path: str
    ) -> Tuple[List[str], List[int], None]:
        """Load from directory structure."""
        image_paths = []
        labels = []

        class_dirs = sorted([
            d for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ])

        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}

        for class_name in class_dirs:
            class_dir = os.path.join(data_path, class_name)
            for img_name in os.listdir(class_dir):
                if self._is_image(img_name):
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(class_to_idx[class_name])

        logger.info(f"Loaded {len(image_paths)} images from {len(class_dirs)} classes")
        return image_paths, labels, None

    def _load_from_manifest(
        self, manifest_path: str, include_metadata: bool
    ) -> Tuple[List[str], List[int], Optional[pd.DataFrame]]:
        """Load from manifest CSV."""
        df = pd.read_csv(manifest_path)

        image_paths = df["image_path"].tolist()

        # Handle string or integer labels
        if df["label"].dtype == object:
            unique_labels = sorted(df["label"].unique())
            label_map = {name: idx for idx, name in enumerate(unique_labels)}
            labels = [label_map[l] for l in df["label"]]
        else:
            labels = df["label"].tolist()

        # Extract metadata columns
        metadata = None
        if include_metadata:
            meta_cols = [c for c in df.columns if c not in ["image_path", "label"]]
            if meta_cols:
                metadata = df[meta_cols].copy()
                metadata["label"] = labels

        return image_paths, labels, metadata

    def _is_image(self, filename: str) -> bool:
        """Check if file is an image."""
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))


class ClarifyDataFormatter:
    """Format CV data for Clarify bias analysis."""

    @staticmethod
    def create_predictions_file(
        predictions: List[int],
        ground_truth: List[int],
        probabilities: List[List[float]],
        class_names: List[str],
        output_path: str,
    ) -> str:
        """
        Create Clarify-compatible predictions file.

        Args:
            predictions: List of predicted class indices
            ground_truth: List of ground truth class indices
            probabilities: List of probability distributions
            class_names: List of class names
            output_path: Output file path

        Returns:
            Path to created file
        """
        df = pd.DataFrame({
            "prediction": predictions,
            "ground_truth": ground_truth,
            "confidence": [max(p) for p in probabilities],
        })

        # Add per-class probabilities
        for i, class_name in enumerate(class_names):
            df[f"prob_{class_name}"] = [p[i] for p in probabilities]

        df.to_csv(output_path, index=False)
        logger.info(f"Created predictions file: {output_path}")
        return output_path

    @staticmethod
    def create_metadata_file(
        metadata: pd.DataFrame,
        predictions: List[int],
        ground_truth: List[int],
        output_path: str,
        sensitive_attributes: Optional[List[str]] = None,
    ) -> str:
        """
        Create metadata file with predictions for bias analysis.

        Args:
            metadata: DataFrame with sensitive attributes
            predictions: List of predictions
            ground_truth: List of ground truth labels
            output_path: Output file path
            sensitive_attributes: List of columns to mark as sensitive

        Returns:
            Path to created file
        """
        df = metadata.copy()
        df["prediction"] = predictions
        df["ground_truth"] = ground_truth

        # Add indicator for sensitive attributes
        if sensitive_attributes:
            df.attrs["sensitive_attributes"] = sensitive_attributes

        df.to_csv(output_path, index=False)
        logger.info(f"Created metadata file: {output_path}")
        return output_path

    @staticmethod
    def create_clarify_config(
        predictions_file: str,
        facet_columns: List[str],
        label_column: str = "ground_truth",
        predicted_label_column: str = "prediction",
        output_path: str = "clarify_config.json",
        positive_label_values: Optional[List[int]] = None,
    ) -> str:
        """
        Create Clarify analysis configuration.

        Args:
            predictions_file: Path to predictions file
            facet_columns: Columns containing sensitive attributes
            label_column: Name of ground truth column
            predicted_label_column: Name of prediction column
            output_path: Output config file path
            positive_label_values: Values considered as positive outcome

        Returns:
            Path to config file
        """
        config = {
            "version": "1.0",
            "dataset_uri": predictions_file,
            "label_column": label_column,
            "predicted_label_column": predicted_label_column,
            "facet_columns": facet_columns,
            "positive_label_values": positive_label_values or [1],
            "methods": {
                "post_training_bias": {
                    "metrics": [
                        "DPPL",  # Difference in Positive Proportions in Predicted Labels
                        "DI",    # Disparate Impact
                        "AD",    # Accuracy Difference
                        "RD",    # Recall Difference
                        "DAR",   # Difference in Acceptance Rates
                        "DRR",   # Difference in Rejection Rates
                        "TE",    # Treatment Equality
                        "CDDPL", # Conditional Demographic Disparity in Predicted Labels
                    ]
                }
            }
        }

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Created Clarify config: {output_path}")
        return output_path


class ImagePreprocessor:
    """Image preprocessing utilities."""

    @staticmethod
    def validate_image(image_path: str) -> bool:
        """Validate that image can be loaded."""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.warning(f"Invalid image {image_path}: {e}")
            return False

    @staticmethod
    def get_image_stats(image_paths: List[str]) -> Dict[str, Any]:
        """Get statistics about a set of images."""
        widths = []
        heights = []
        channels = []
        formats = []

        for path in image_paths:
            try:
                with Image.open(path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
                    channels.append(len(img.getbands()))
                    formats.append(img.format)
            except Exception:
                continue

        return {
            "count": len(widths),
            "avg_width": np.mean(widths) if widths else 0,
            "avg_height": np.mean(heights) if heights else 0,
            "min_width": min(widths) if widths else 0,
            "max_width": max(widths) if widths else 0,
            "min_height": min(heights) if heights else 0,
            "max_height": max(heights) if heights else 0,
            "channels": list(set(channels)),
            "formats": list(set(formats)),
        }

    @staticmethod
    def create_manifest_from_directory(
        data_path: str,
        output_path: str,
        metadata: Optional[Dict[str, List[Any]]] = None,
    ) -> str:
        """
        Create manifest CSV from directory structure.

        Args:
            data_path: Path to image directory
            output_path: Output manifest file path
            metadata: Additional metadata columns {column_name: [values]}

        Returns:
            Path to manifest file
        """
        records = []

        class_dirs = sorted([
            d for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ])

        for class_name in class_dirs:
            class_dir = os.path.join(data_path, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    records.append({
                        "image_path": os.path.join(class_dir, img_name),
                        "label": class_name,
                    })

        df = pd.DataFrame(records)

        # Add metadata columns if provided
        if metadata:
            for col_name, values in metadata.items():
                if len(values) == len(df):
                    df[col_name] = values
                else:
                    logger.warning(f"Metadata column {col_name} has wrong length, skipping")

        df.to_csv(output_path, index=False)
        logger.info(f"Created manifest with {len(df)} entries: {output_path}")
        return output_path
