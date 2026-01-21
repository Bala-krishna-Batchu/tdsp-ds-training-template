"""
Clarify Integration Utilities

Utilities for integrating with Amazon SageMaker Clarify for bias monitoring
of computer vision models.

This module provides:
1. Artifact generation for Clarify analysis
2. Bias report parsing and alerting
3. Automation helpers for scheduled bias monitoring
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import boto3
import pandas as pd

logger = logging.getLogger("root")


class ClarifyArtifactGenerator:
    """Generate artifacts compatible with SageMaker Clarify for CV models."""

    def __init__(self, output_dir: str = "/opt/ml/output/clarify"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_all_artifacts(
        self,
        predictions: List[int],
        ground_truth: List[int],
        probabilities: List[List[float]],
        class_names: List[str],
        metadata: Optional[pd.DataFrame] = None,
        model_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Generate all artifacts needed for Clarify bias analysis.

        Args:
            predictions: List of predicted class indices
            ground_truth: List of ground truth labels
            probabilities: List of probability distributions
            class_names: List of class names
            metadata: DataFrame with sensitive attributes
            model_info: Dictionary with model information

        Returns:
            Dictionary mapping artifact names to file paths
        """
        artifacts = {}

        # 1. Predictions file
        artifacts["predictions"] = self.create_predictions_file(
            predictions, ground_truth, probabilities, class_names
        )

        # 2. Metadata with predictions (for bias analysis)
        if metadata is not None:
            artifacts["metadata"] = self.create_metadata_file(
                metadata, predictions, ground_truth
            )

        # 3. Model info
        artifacts["model_info"] = self.create_model_info_file(
            model_info or {}, class_names
        )

        # 4. Clarify config template
        sensitive_attrs = []
        if metadata is not None:
            sensitive_attrs = [c for c in metadata.columns 
                            if c not in ["prediction", "ground_truth", "label"]]
        artifacts["clarify_config"] = self.create_clarify_config(sensitive_attrs)

        # 5. Metrics file
        artifacts["metrics"] = self.create_metrics_file(
            predictions, ground_truth, class_names
        )

        return artifacts

    def create_predictions_file(
        self,
        predictions: List[int],
        ground_truth: List[int],
        probabilities: List[List[float]],
        class_names: List[str],
    ) -> str:
        """Create predictions CSV file."""
        df = pd.DataFrame({
            "prediction": predictions,
            "ground_truth": ground_truth,
            "confidence": [max(p) for p in probabilities],
        })

        # Add per-class probabilities
        for i, class_name in enumerate(class_names):
            df[f"prob_{class_name}"] = [p[i] if i < len(p) else 0 for p in probabilities]

        output_path = os.path.join(self.output_dir, "predictions.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Created predictions file: {output_path}")
        return output_path

    def create_metadata_file(
        self,
        metadata: pd.DataFrame,
        predictions: List[int],
        ground_truth: List[int],
    ) -> str:
        """Create metadata file with predictions."""
        df = metadata.copy()
        df["prediction"] = predictions
        df["ground_truth"] = ground_truth

        output_path = os.path.join(self.output_dir, "metadata_with_predictions.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Created metadata file: {output_path}")
        return output_path

    def create_model_info_file(
        self,
        model_info: Dict[str, Any],
        class_names: List[str],
    ) -> str:
        """Create model information JSON file."""
        info = {
            "model_type": "computer_vision_classifier",
            "framework": "pytorch",
            "num_classes": len(class_names),
            "class_names": class_names,
            "timestamp": datetime.now().isoformat(),
            "clarify_compatible": True,
            **model_info,
        }

        output_path = os.path.join(self.output_dir, "model_info.json")
        with open(output_path, "w") as f:
            json.dump(info, f, indent=2)
        logger.info(f"Created model info file: {output_path}")
        return output_path

    def create_clarify_config(
        self,
        sensitive_attributes: List[str],
        label_column: str = "ground_truth",
        predicted_label_column: str = "prediction",
    ) -> str:
        """Create Clarify analysis configuration template."""
        config = {
            "version": "1.0",
            "dataset_type": "computer_vision",
            "label_column": label_column,
            "predicted_label_column": predicted_label_column,
            "facet_columns": sensitive_attributes,
            "probability_threshold": 0.5,
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
                        "FT",    # Flip Test
                    ]
                }
            },
            "report_config": {
                "include_visualizations": True,
                "output_format": ["json", "html"],
            }
        }

        output_path = os.path.join(self.output_dir, "clarify_config.json")
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created Clarify config: {output_path}")
        return output_path

    def create_metrics_file(
        self,
        predictions: List[int],
        ground_truth: List[int],
        class_names: List[str],
    ) -> str:
        """Create metrics JSON file."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report
        )

        metrics = {
            "accuracy": float(accuracy_score(ground_truth, predictions)),
            "precision_macro": float(precision_score(ground_truth, predictions, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(ground_truth, predictions, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(ground_truth, predictions, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(ground_truth, predictions).tolist(),
            "per_class_metrics": classification_report(ground_truth, predictions, output_dict=True),
            "timestamp": datetime.now().isoformat(),
        }

        output_path = os.path.join(self.output_dir, "metrics.json")
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Created metrics file: {output_path}")
        return output_path


class ClarifyBiasAnalyzer:
    """Analyze and parse Clarify bias reports."""

    def __init__(self, sagemaker_client: Optional[boto3.client] = None):
        self.sagemaker_client = sagemaker_client or boto3.client("sagemaker")

    def parse_bias_report(self, report_path: str) -> Dict[str, Any]:
        """
        Parse Clarify bias report and extract key metrics.

        Args:
            report_path: Path to bias report JSON file

        Returns:
            Dictionary with parsed bias metrics
        """
        with open(report_path, "r") as f:
            report = json.load(f)

        parsed = {
            "timestamp": datetime.now().isoformat(),
            "overall_bias_detected": False,
            "facet_metrics": {},
            "alerts": [],
        }

        # Parse facet-level metrics
        for facet_name, facet_data in report.get("facets", {}).items():
            facet_metrics = {}
            for metric_name, metric_value in facet_data.get("metrics", {}).items():
                facet_metrics[metric_name] = {
                    "value": metric_value.get("value", 0),
                    "threshold": metric_value.get("threshold", 0.1),
                    "exceeded": abs(metric_value.get("value", 0)) > metric_value.get("threshold", 0.1),
                }
                
                if facet_metrics[metric_name]["exceeded"]:
                    parsed["overall_bias_detected"] = True
                    parsed["alerts"].append({
                        "facet": facet_name,
                        "metric": metric_name,
                        "value": metric_value.get("value", 0),
                        "threshold": metric_value.get("threshold", 0.1),
                    })

            parsed["facet_metrics"][facet_name] = facet_metrics

        return parsed

    def check_bias_threshold(
        self,
        metrics: Dict[str, float],
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Check if bias metrics exceed threshold.

        Args:
            metrics: Dictionary of metric_name -> value
            threshold: Threshold for bias detection

        Returns:
            Dictionary with threshold check results
        """
        results = {
            "threshold": threshold,
            "metrics_checked": [],
            "violations": [],
            "pass": True,
        }

        for metric_name, value in metrics.items():
            check = {
                "metric": metric_name,
                "value": value,
                "threshold": threshold,
                "passed": abs(value) <= threshold,
            }
            results["metrics_checked"].append(check)

            if not check["passed"]:
                results["violations"].append(check)
                results["pass"] = False

        return results


class BiasMonitoringAutomation:
    """
    Automation utilities for scheduled bias monitoring.
    
    This class provides methods to:
    1. Set up scheduled Clarify jobs
    2. Configure alerts and notifications
    3. Track bias drift over time
    """

    def __init__(
        self,
        sagemaker_client: Optional[boto3.client] = None,
        sns_client: Optional[boto3.client] = None,
        events_client: Optional[boto3.client] = None,
    ):
        self.sagemaker_client = sagemaker_client or boto3.client("sagemaker")
        self.sns_client = sns_client or boto3.client("sns")
        self.events_client = events_client or boto3.client("events")

    def create_monitoring_schedule(
        self,
        schedule_name: str,
        model_name: str,
        processing_job_config: Dict[str, Any],
        schedule_expression: str = "rate(1 day)",
        role_arn: str = "",
    ) -> Dict[str, Any]:
        """
        Create a scheduled Clarify bias monitoring job.

        Args:
            schedule_name: Name for the monitoring schedule
            model_name: Name of the model to monitor
            processing_job_config: Configuration for Clarify processing job
            schedule_expression: Cron or rate expression for scheduling
            role_arn: IAM role ARN for the job

        Returns:
            Dictionary with schedule creation response
        """
        # Create EventBridge rule for scheduling
        rule_response = self.events_client.put_rule(
            Name=f"{schedule_name}-rule",
            ScheduleExpression=schedule_expression,
            State="ENABLED",
            Description=f"Schedule for {model_name} bias monitoring",
        )

        logger.info(f"Created monitoring schedule: {schedule_name}")
        
        return {
            "schedule_name": schedule_name,
            "rule_arn": rule_response.get("RuleArn"),
            "schedule_expression": schedule_expression,
        }

    def setup_bias_alerts(
        self,
        topic_name: str,
        email_endpoints: List[str],
        threshold: float = 0.1,
    ) -> Dict[str, str]:
        """
        Set up SNS topic and subscriptions for bias alerts.

        Args:
            topic_name: Name for the SNS topic
            email_endpoints: List of email addresses for notifications
            threshold: Bias threshold for alerting

        Returns:
            Dictionary with SNS topic ARN and subscription ARNs
        """
        # Create SNS topic
        topic_response = self.sns_client.create_topic(Name=topic_name)
        topic_arn = topic_response["TopicArn"]

        # Subscribe email endpoints
        subscription_arns = []
        for email in email_endpoints:
            sub_response = self.sns_client.subscribe(
                TopicArn=topic_arn,
                Protocol="email",
                Endpoint=email,
            )
            subscription_arns.append(sub_response.get("SubscriptionArn", "pending"))

        logger.info(f"Created SNS topic for bias alerts: {topic_arn}")

        return {
            "topic_arn": topic_arn,
            "subscription_arns": subscription_arns,
        }

    def send_bias_alert(
        self,
        topic_arn: str,
        model_name: str,
        violations: List[Dict[str, Any]],
        report_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send bias alert notification.

        Args:
            topic_arn: SNS topic ARN
            model_name: Name of the model
            violations: List of bias violations
            report_url: URL to full bias report

        Returns:
            SNS publish response
        """
        message = f"""
        BIAS ALERT: {model_name}
        
        Time: {datetime.now().isoformat()}
        
        The following bias metrics exceeded thresholds:
        
        """
        
        for violation in violations:
            message += f"""
        - {violation['metric']} for facet '{violation.get('facet', 'unknown')}':
          Value: {violation['value']:.4f}
          Threshold: {violation['threshold']:.4f}
        """

        if report_url:
            message += f"\n\nFull report: {report_url}"

        response = self.sns_client.publish(
            TopicArn=topic_arn,
            Subject=f"Bias Alert: {model_name}",
            Message=message,
        )

        logger.info(f"Sent bias alert for {model_name}")
        return response

    def track_bias_drift(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        drift_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Track bias drift compared to baseline.

        Args:
            current_metrics: Current bias metrics
            baseline_metrics: Baseline bias metrics
            drift_threshold: Threshold for significant drift

        Returns:
            Dictionary with drift analysis results
        """
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "metrics": {},
        }

        for metric_name in current_metrics:
            if metric_name in baseline_metrics:
                current = current_metrics[metric_name]
                baseline = baseline_metrics[metric_name]
                drift = abs(current - baseline)

                drift_results["metrics"][metric_name] = {
                    "current": current,
                    "baseline": baseline,
                    "drift": drift,
                    "significant": drift > drift_threshold,
                }

                if drift > drift_threshold:
                    drift_results["drift_detected"] = True

        return drift_results

    def generate_automation_config(
        self,
        model_name: str,
        s3_bucket: str,
        s3_prefix: str,
        sensitive_attributes: List[str],
        schedule: str = "rate(1 day)",
        alert_emails: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete automation configuration for bias monitoring.

        Args:
            model_name: Name of the model
            s3_bucket: S3 bucket for artifacts
            s3_prefix: S3 prefix for artifacts
            sensitive_attributes: List of sensitive attribute column names
            schedule: Monitoring schedule expression
            alert_emails: Email addresses for alerts

        Returns:
            Complete automation configuration dictionary
        """
        config = {
            "model_name": model_name,
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            
            "data_config": {
                "predictions_uri": f"s3://{s3_bucket}/{s3_prefix}/predictions.csv",
                "metadata_uri": f"s3://{s3_bucket}/{s3_prefix}/metadata_with_predictions.csv",
                "output_uri": f"s3://{s3_bucket}/{s3_prefix}/bias-reports/",
            },
            
            "analysis_config": {
                "facet_columns": sensitive_attributes,
                "label_column": "ground_truth",
                "predicted_label_column": "prediction",
                "probability_threshold": 0.5,
                "metrics": [
                    "DPPL", "DI", "AD", "RD", "DAR", "DRR", "TE", "CDDPL"
                ],
            },
            
            "schedule_config": {
                "expression": schedule,
                "enabled": True,
            },
            
            "alert_config": {
                "enabled": alert_emails is not None and len(alert_emails) > 0,
                "email_endpoints": alert_emails or [],
                "bias_threshold": 0.1,
                "drift_threshold": 0.05,
            },
            
            "resources": {
                "instance_type": "ml.m4.xlarge",
                "instance_count": 1,
                "volume_size_gb": 30,
            },
        }

        return config
