"""
Clarify Repository Integration

This module provides the integration layer between this training repo and
your existing Clarify bias monitoring repo.

ARCHITECTURE:
=============

    ┌─────────────────────────────┐
    │   THIS REPO (Training)      │
    │   - Train rust detection    │
    │   - Generate artifacts      │
    │   - Upload to S3            │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │         S3 BUCKET           │
    │   /model/best_model.pth     │
    │   /artifacts/predictions.csv│
    │   /artifacts/baseline.csv   │
    │   /artifacts/metadata.csv   │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  YOUR CLARIFY REPO          │
    │  - Read S3 artifacts        │
    │  - Run Clarify analysis     │
    │  - Generate bias reports    │
    │  - Compute PSI              │
    └─────────────────────────────┘

"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger("root")


class S3ArtifactPublisher:
    """
    Publishes model artifacts to S3 in a format consumable by your Clarify repo.
    """

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str,
        region: str = "us-east-1",
    ):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client("s3", region_name=region)

    def publish_training_artifacts(
        self,
        model_path: str,
        predictions: List[int],
        ground_truth: List[int],
        probabilities: List[List[float]],
        class_names: List[str],
        metadata: Optional[pd.DataFrame] = None,
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Publish all artifacts needed by your Clarify repo.

        Args:
            model_path: Path to trained model file
            predictions: Model predictions
            ground_truth: Ground truth labels
            probabilities: Prediction probabilities
            class_names: Class names (e.g., ["no_rust", "rust"])
            metadata: DataFrame with sensitive attributes and image metadata
            image_paths: List of image file paths

        Returns:
            Dictionary of S3 URIs for each artifact
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        s3_paths = {}

        # 1. Upload model
        model_key = f"{self.s3_prefix}/model/model.pth"
        self._upload_file(model_path, model_key)
        s3_paths["model"] = f"s3://{self.s3_bucket}/{model_key}"

        # 2. Create and upload predictions file (for bias analysis)
        predictions_df = self._create_predictions_dataframe(
            predictions, ground_truth, probabilities, class_names, image_paths
        )
        predictions_key = f"{self.s3_prefix}/artifacts/predictions.csv"
        self._upload_dataframe(predictions_df, predictions_key)
        s3_paths["predictions"] = f"s3://{self.s3_bucket}/{predictions_key}"

        # 3. Create and upload baseline file (for PSI calculation)
        baseline_key = f"{self.s3_prefix}/artifacts/baseline.csv"
        self._upload_dataframe(predictions_df, baseline_key)
        s3_paths["baseline"] = f"s3://{self.s3_bucket}/{baseline_key}"

        # 4. Upload metadata if available
        if metadata is not None:
            metadata_df = metadata.copy()
            metadata_df["prediction"] = predictions
            metadata_df["ground_truth"] = ground_truth
            if image_paths:
                metadata_df["image_path"] = image_paths

            metadata_key = f"{self.s3_prefix}/artifacts/metadata.csv"
            self._upload_dataframe(metadata_df, metadata_key)
            s3_paths["metadata"] = f"s3://{self.s3_bucket}/{metadata_key}"

        # 5. Create and upload manifest for Clarify repo
        manifest = self._create_clarify_manifest(s3_paths, class_names, timestamp)
        manifest_key = f"{self.s3_prefix}/artifacts/manifest.json"
        self._upload_json(manifest, manifest_key)
        s3_paths["manifest"] = f"s3://{self.s3_bucket}/{manifest_key}"

        # 6. Upload versioned copy for historical tracking
        versioned_prefix = f"{self.s3_prefix}/history/{timestamp}"
        for name, local_key in [
            ("predictions", predictions_key),
            ("metadata", f"{self.s3_prefix}/artifacts/metadata.csv"),
        ]:
            if name in s3_paths:
                versioned_key = f"{versioned_prefix}/{name}.csv"
                self.s3_client.copy_object(
                    Bucket=self.s3_bucket,
                    CopySource=f"{self.s3_bucket}/{local_key.replace(self.s3_prefix, self.s3_prefix)}",
                    Key=versioned_key,
                )

        logger.info(f"Published artifacts to s3://{self.s3_bucket}/{self.s3_prefix}/")
        return s3_paths

    def _create_predictions_dataframe(
        self,
        predictions: List[int],
        ground_truth: List[int],
        probabilities: List[List[float]],
        class_names: List[str],
        image_paths: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Create predictions DataFrame in Clarify-compatible format."""
        df = pd.DataFrame({
            "prediction": predictions,
            "ground_truth": ground_truth,
            "confidence": [max(p) for p in probabilities],
        })

        # Add per-class probabilities (important for PSI calculation)
        for i, class_name in enumerate(class_names):
            df[f"prob_{class_name}"] = [p[i] if i < len(p) else 0 for p in probabilities]

        # Add image paths if available
        if image_paths:
            df["image_path"] = image_paths

        return df

    def _create_clarify_manifest(
        self,
        s3_paths: Dict[str, str],
        class_names: List[str],
        timestamp: str,
    ) -> Dict[str, Any]:
        """Create manifest file for your Clarify repo to consume."""
        return {
            "version": "1.0",
            "model_type": "rust_detection",
            "task_type": "image_classification",
            "timestamp": timestamp,
            "class_names": class_names,
            "num_classes": len(class_names),
            "artifacts": s3_paths,
            "columns": {
                "prediction": "prediction",
                "ground_truth": "ground_truth",
                "probability_prefix": "prob_",
            },
            # Configuration for your Clarify repo
            "clarify_config": {
                "label_column": "ground_truth",
                "predicted_label_column": "prediction",
                "probability_columns": [f"prob_{c}" for c in class_names],
            },
            # PSI configuration
            "psi_config": {
                "baseline_path": s3_paths.get("baseline"),
                "feature_columns": [f"prob_{c}" for c in class_names],
                "bins": 10,
            },
        }

    def _upload_file(self, local_path: str, s3_key: str) -> None:
        """Upload a file to S3."""
        self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
        logger.info(f"Uploaded {local_path} to s3://{self.s3_bucket}/{s3_key}")

    def _upload_dataframe(self, df: pd.DataFrame, s3_key: str) -> None:
        """Upload a DataFrame as CSV to S3."""
        csv_buffer = df.to_csv(index=False)
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=csv_buffer.encode("utf-8"),
        )
        logger.info(f"Uploaded DataFrame to s3://{self.s3_bucket}/{s3_key}")

    def _upload_json(self, data: Dict, s3_key: str) -> None:
        """Upload a dictionary as JSON to S3."""
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=json.dumps(data, indent=2).encode("utf-8"),
        )
        logger.info(f"Uploaded JSON to s3://{self.s3_bucket}/{s3_key}")


class PSICalculator:
    """
    Calculate Population Stability Index (PSI) for model drift monitoring.
    
    PSI measures how much the distribution of predictions has shifted
    from the baseline (training) distribution.
    
    PSI Interpretation:
    - PSI < 0.1: No significant shift
    - 0.1 <= PSI < 0.25: Moderate shift, investigate
    - PSI >= 0.25: Significant shift, action required
    """

    @staticmethod
    def calculate_psi(
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
        eps: float = 1e-6,
    ) -> float:
        """
        Calculate PSI between baseline and current distributions.

        Args:
            baseline: Baseline distribution (e.g., training predictions)
            current: Current distribution (e.g., inference predictions)
            bins: Number of bins for histogram
            eps: Small value to avoid division by zero

        Returns:
            PSI value
        """
        # Create bins based on baseline distribution
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        # Calculate proportions in each bin
        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)

        baseline_props = (baseline_counts + eps) / (len(baseline) + eps * bins)
        current_props = (current_counts + eps) / (len(current) + eps * bins)

        # Calculate PSI
        psi = np.sum((current_props - baseline_props) * np.log(current_props / baseline_props))

        return float(psi)

    @staticmethod
    def calculate_psi_report(
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        probability_columns: List[str],
        bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive PSI report for model monitoring.

        Args:
            baseline_df: Baseline predictions DataFrame
            current_df: Current predictions DataFrame
            probability_columns: Columns containing prediction probabilities
            bins: Number of bins for PSI calculation

        Returns:
            PSI report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "baseline_size": len(baseline_df),
            "current_size": len(current_df),
            "psi_by_column": {},
            "overall_psi": 0.0,
            "drift_detected": False,
            "recommendations": [],
        }

        psi_values = []
        for col in probability_columns:
            if col in baseline_df.columns and col in current_df.columns:
                psi = PSICalculator.calculate_psi(
                    baseline_df[col].values,
                    current_df[col].values,
                    bins=bins,
                )
                report["psi_by_column"][col] = {
                    "psi": psi,
                    "status": PSICalculator._get_psi_status(psi),
                }
                psi_values.append(psi)

        # Overall PSI (average across columns)
        if psi_values:
            report["overall_psi"] = float(np.mean(psi_values))
            report["max_psi"] = float(max(psi_values))

        # Determine if drift is detected
        if report["overall_psi"] >= 0.1:
            report["drift_detected"] = True
            if report["overall_psi"] >= 0.25:
                report["recommendations"].append(
                    "CRITICAL: Significant distribution shift detected. "
                    "Consider retraining the model with recent data."
                )
            else:
                report["recommendations"].append(
                    "WARNING: Moderate distribution shift detected. "
                    "Monitor closely and prepare for potential retraining."
                )

        return report

    @staticmethod
    def _get_psi_status(psi: float) -> str:
        """Get status string for PSI value."""
        if psi < 0.1:
            return "STABLE"
        elif psi < 0.25:
            return "WARNING"
        else:
            return "CRITICAL"


# =============================================================================
# BEST AUTOMATION APPROACHES
# =============================================================================

class AutomationRecommendations:
    """
    Recommended automation approaches for model bias and PSI monitoring.
    """

    @staticmethod
    def get_best_approach() -> Dict[str, Any]:
        """
        Returns the recommended automation architecture.
        """
        return {
            "recommended_approach": "Event-Driven with Scheduled Backup",
            "description": """
            RECOMMENDED ARCHITECTURE:
            ========================
            
            1. EVENT-DRIVEN (Primary):
               - Trigger Clarify analysis after each training run
               - S3 Event → EventBridge → Lambda → Your Clarify Repo
               
            2. SCHEDULED (Backup for Production):
               - Run PSI checks on inference data daily/weekly
               - EventBridge Schedule → Lambda → PSI Check → Alert
               
            This hybrid approach ensures:
            - Immediate feedback after training (catches bias before deployment)
            - Continuous monitoring in production (catches drift over time)
            """,
            
            "architecture": {
                "training_trigger": {
                    "type": "S3 Event",
                    "trigger": "PutObject on predictions.csv",
                    "action": "Invoke Clarify repo analysis",
                },
                "production_monitoring": {
                    "type": "EventBridge Schedule",
                    "frequency": "Daily (cron: 0 0 * * ? *)",
                    "action": "Run PSI check on inference logs",
                },
                "alerting": {
                    "channels": ["SNS", "Slack", "Email"],
                    "thresholds": {
                        "bias_alert": 0.1,
                        "psi_warning": 0.1,
                        "psi_critical": 0.25,
                    },
                },
            },
            
            "implementation_steps": [
                "1. Configure S3 event notification on artifacts bucket",
                "2. Create EventBridge rule to route events to Lambda",
                "3. Lambda invokes your Clarify repo (via API/Step Functions)",
                "4. Clarify repo processes artifacts and generates reports",
                "5. Reports stored in S3, alerts sent via SNS",
                "6. Set up scheduled job for production PSI monitoring",
            ],
        }

    @staticmethod
    def get_eventbridge_rule_config(
        s3_bucket: str,
        artifacts_prefix: str,
        lambda_arn: str,
    ) -> Dict[str, Any]:
        """
        Returns EventBridge rule configuration for automation.
        """
        return {
            "Name": "rust-detection-clarify-trigger",
            "EventPattern": json.dumps({
                "source": ["aws.s3"],
                "detail-type": ["Object Created"],
                "detail": {
                    "bucket": {"name": [s3_bucket]},
                    "object": {
                        "key": [{"prefix": f"{artifacts_prefix}/artifacts/predictions.csv"}]
                    }
                }
            }),
            "State": "ENABLED",
            "Targets": [{
                "Id": "clarify-lambda",
                "Arn": lambda_arn,
            }],
        }

    @staticmethod
    def get_step_functions_workflow() -> Dict[str, Any]:
        """
        Returns Step Functions workflow for complete automation.
        
        This is the MOST ROBUST approach for production.
        """
        return {
            "Comment": "Automated Bias and PSI Monitoring Workflow",
            "StartAt": "CheckArtifactsExist",
            "States": {
                "CheckArtifactsExist": {
                    "Type": "Task",
                    "Resource": "arn:aws:states:::lambda:invoke",
                    "Parameters": {
                        "FunctionName": "check-artifacts",
                        "Payload.$": "$"
                    },
                    "Next": "RunBiasAnalysis"
                },
                "RunBiasAnalysis": {
                    "Type": "Task",
                    "Resource": "arn:aws:states:::sagemaker:createProcessingJob.sync",
                    "Parameters": {
                        "ProcessingJobName.$": "States.Format('bias-analysis-{}', $$.Execution.Name)",
                        "AppSpecification": {
                            "ImageUri": "${CLARIFY_IMAGE_URI}",
                        },
                        # ... processing job config
                    },
                    "Next": "RunPSICheck"
                },
                "RunPSICheck": {
                    "Type": "Task",
                    "Resource": "arn:aws:states:::lambda:invoke",
                    "Parameters": {
                        "FunctionName": "calculate-psi",
                        "Payload.$": "$"
                    },
                    "Next": "EvaluateResults"
                },
                "EvaluateResults": {
                    "Type": "Choice",
                    "Choices": [
                        {
                            "Variable": "$.bias_detected",
                            "BooleanEquals": True,
                            "Next": "SendBiasAlert"
                        },
                        {
                            "Variable": "$.psi_critical",
                            "BooleanEquals": True,
                            "Next": "SendDriftAlert"
                        }
                    ],
                    "Default": "Success"
                },
                "SendBiasAlert": {
                    "Type": "Task",
                    "Resource": "arn:aws:states:::sns:publish",
                    "Parameters": {
                        "TopicArn": "${ALERT_TOPIC_ARN}",
                        "Message.$": "States.Format('Bias detected in rust detection model: {}', $.bias_report)"
                    },
                    "Next": "Success"
                },
                "SendDriftAlert": {
                    "Type": "Task",
                    "Resource": "arn:aws:states:::sns:publish",
                    "Parameters": {
                        "TopicArn": "${ALERT_TOPIC_ARN}",
                        "Message.$": "States.Format('PSI drift detected: {}', $.psi_report)"
                    },
                    "Next": "Success"
                },
                "Success": {
                    "Type": "Succeed"
                }
            }
        }


# =============================================================================
# INTEGRATION EXAMPLE
# =============================================================================

def integrate_with_clarify_repo():
    """
    Example showing how to integrate this training repo with your Clarify repo.
    """
    
    # After training completes, publish artifacts
    publisher = S3ArtifactPublisher(
        s3_bucket="tdsp-ml-products-dev",
        s3_prefix="tdspds/rust-detection",
    )
    
    # These would come from your training run
    predictions = [0, 1, 1, 0, 1, 0, 0, 1]
    ground_truth = [0, 1, 0, 0, 1, 0, 1, 1]
    probabilities = [
        [0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.85, 0.15],
        [0.1, 0.9], [0.95, 0.05], [0.4, 0.6], [0.15, 0.85]
    ]
    class_names = ["no_rust", "rust"]
    
    # Metadata for bias analysis (e.g., vehicle type, environment)
    metadata = pd.DataFrame({
        "vehicle_type": ["sedan", "truck", "suv", "sedan", "truck", "suv", "sedan", "truck"],
        "environment": ["urban", "rural", "coastal", "urban", "rural", "coastal", "urban", "rural"],
        "lighting": ["day", "day", "night", "day", "night", "day", "night", "night"],
    })
    
    # Publish to S3
    s3_paths = publisher.publish_training_artifacts(
        model_path="/opt/ml/model/best_model.pth",
        predictions=predictions,
        ground_truth=ground_truth,
        probabilities=probabilities,
        class_names=class_names,
        metadata=metadata,
    )
    
    print("Artifacts published to S3:")
    for name, path in s3_paths.items():
        print(f"  {name}: {path}")
    
    # Your Clarify repo can now:
    # 1. Read the manifest.json to find all artifacts
    # 2. Run bias analysis on predictions.csv + metadata.csv
    # 3. Calculate PSI comparing baseline.csv vs new inference data
    # 4. Generate reports and alerts
    
    return s3_paths


if __name__ == "__main__":
    # Print recommended approach
    recommendations = AutomationRecommendations.get_best_approach()
    print(json.dumps(recommendations, indent=2))
