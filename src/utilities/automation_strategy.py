"""
Automated Model Bias Monitoring Strategy

This module documents and implements the automation strategy for continuous
model bias monitoring using Amazon SageMaker Clarify for Computer Vision models.

===================================================================================
AUTOMATION ARCHITECTURE OVERVIEW
===================================================================================

The automated bias monitoring system consists of the following components:

1. TRAINING PIPELINE (This Repo)
   - Trains CV model
   - Generates Clarify-compatible artifacts:
     * predictions.csv - Model predictions with probabilities
     * metadata_with_predictions.csv - Predictions + sensitive attributes
     * model_info.json - Model configuration and metadata
     * metrics.json - Performance metrics
     * clarify_config.json - Clarify analysis configuration
   - Uploads artifacts to S3

2. CLARIFY ANALYSIS (External Repo)
   - Consumes artifacts from S3
   - Runs bias analysis using SageMaker Clarify
   - Generates bias reports

3. MONITORING & ALERTING
   - Scheduled jobs for continuous monitoring
   - SNS notifications for bias threshold violations
   - Dashboard for bias metric trends

===================================================================================
AUTOMATION STRATEGIES
===================================================================================

Strategy 1: Event-Driven (Recommended)
--------------------------------------
Trigger bias analysis automatically after each training run.

    Training Job Completion
            │
            ▼
    S3 Event Notification (artifacts uploaded)
            │
            ▼
    EventBridge Rule
            │
            ▼
    Lambda Function (trigger Clarify job)
            │
            ▼
    SageMaker Processing Job (Clarify)
            │
            ▼
    Bias Report → S3
            │
            ▼
    SNS Notification (if violations detected)

Strategy 2: Scheduled Monitoring
--------------------------------
Run bias analysis on a schedule (daily/weekly) for deployed models.

    EventBridge Scheduler (cron)
            │
            ▼
    Lambda Function
            │
            ├──► Fetch latest predictions from inference endpoint
            │
            ▼
    SageMaker Processing Job (Clarify)
            │
            ▼
    Compare with baseline metrics
            │
            ├──► If drift detected → SNS Alert
            │
            ▼
    Update dashboard metrics

Strategy 3: CI/CD Integration
-----------------------------
Integrate bias checks into deployment pipeline.

    Code Commit
            │
            ▼
    CI Pipeline (Training)
            │
            ▼
    Generate Clarify Artifacts
            │
            ▼
    Run Bias Analysis
            │
            ├──► If bias > threshold → Fail Pipeline
            │
            ▼
    Deploy Model (if passed)

===================================================================================
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import boto3

logger = logging.getLogger("root")


class AutomatedBiasMonitoringPipeline:
    """
    Implements automated bias monitoring pipeline for CV models.
    
    This class provides utilities to:
    1. Set up event-driven bias analysis
    2. Configure scheduled monitoring
    3. Integrate with CI/CD pipelines
    """

    def __init__(
        self,
        region: str = "us-east-1",
        account_id: Optional[str] = None,
    ):
        self.region = region
        self.account_id = account_id or self._get_account_id()
        
        # AWS clients
        self.s3 = boto3.client("s3", region_name=region)
        self.events = boto3.client("events", region_name=region)
        self.lambda_client = boto3.client("lambda", region_name=region)
        self.sns = boto3.client("sns", region_name=region)
        self.sagemaker = boto3.client("sagemaker", region_name=region)

    def _get_account_id(self) -> str:
        """Get AWS account ID."""
        sts = boto3.client("sts")
        return sts.get_caller_identity()["Account"]

    # =========================================================================
    # STRATEGY 1: Event-Driven Automation
    # =========================================================================

    def setup_event_driven_monitoring(
        self,
        s3_bucket: str,
        artifacts_prefix: str,
        clarify_job_name_prefix: str,
        lambda_function_arn: str,
        sns_topic_arn: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Set up event-driven bias monitoring.
        
        This creates:
        1. S3 event notification for artifact uploads
        2. EventBridge rule to trigger Lambda
        3. Lambda function configuration (assumed pre-created)
        
        Args:
            s3_bucket: S3 bucket containing artifacts
            artifacts_prefix: S3 prefix for Clarify artifacts
            clarify_job_name_prefix: Prefix for Clarify job names
            lambda_function_arn: ARN of Lambda function to trigger Clarify
            sns_topic_arn: Optional SNS topic for notifications
        
        Returns:
            Configuration details
        """
        
        # Create EventBridge rule for S3 object creation
        rule_name = f"{clarify_job_name_prefix}-artifact-trigger"
        
        event_pattern = {
            "source": ["aws.s3"],
            "detail-type": ["Object Created"],
            "detail": {
                "bucket": {"name": [s3_bucket]},
                "object": {
                    "key": [{"prefix": f"{artifacts_prefix}/predictions.csv"}]
                }
            }
        }
        
        rule_response = self.events.put_rule(
            Name=rule_name,
            EventPattern=json.dumps(event_pattern),
            State="ENABLED",
            Description=f"Trigger Clarify bias analysis when new artifacts are uploaded",
        )
        
        # Add Lambda as target
        self.events.put_targets(
            Rule=rule_name,
            Targets=[{
                "Id": "clarify-lambda-trigger",
                "Arn": lambda_function_arn,
                "Input": json.dumps({
                    "s3_bucket": s3_bucket,
                    "artifacts_prefix": artifacts_prefix,
                    "job_name_prefix": clarify_job_name_prefix,
                    "sns_topic_arn": sns_topic_arn,
                }),
            }]
        )
        
        logger.info(f"Created event-driven monitoring rule: {rule_name}")
        
        return {
            "rule_name": rule_name,
            "rule_arn": rule_response["RuleArn"],
            "lambda_arn": lambda_function_arn,
            "event_pattern": event_pattern,
        }

    # =========================================================================
    # STRATEGY 2: Scheduled Monitoring
    # =========================================================================

    def setup_scheduled_monitoring(
        self,
        schedule_name: str,
        schedule_expression: str,  # e.g., "rate(1 day)" or "cron(0 0 * * ? *)"
        lambda_function_arn: str,
        monitoring_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Set up scheduled bias monitoring.
        
        Args:
            schedule_name: Name for the schedule
            schedule_expression: CloudWatch Events schedule expression
            lambda_function_arn: Lambda function to trigger
            monitoring_config: Configuration for bias monitoring
        
        Returns:
            Schedule configuration details
        """
        
        rule_response = self.events.put_rule(
            Name=schedule_name,
            ScheduleExpression=schedule_expression,
            State="ENABLED",
            Description=f"Scheduled bias monitoring for CV models",
        )
        
        # Add Lambda as target with config
        self.events.put_targets(
            Rule=schedule_name,
            Targets=[{
                "Id": "scheduled-bias-monitor",
                "Arn": lambda_function_arn,
                "Input": json.dumps(monitoring_config),
            }]
        )
        
        logger.info(f"Created scheduled monitoring: {schedule_name}")
        
        return {
            "schedule_name": schedule_name,
            "rule_arn": rule_response["RuleArn"],
            "schedule_expression": schedule_expression,
            "config": monitoring_config,
        }

    # =========================================================================
    # STRATEGY 3: CI/CD Integration
    # =========================================================================

    def generate_cicd_config(
        self,
        model_name: str,
        s3_bucket: str,
        artifacts_prefix: str,
        bias_threshold: float = 0.1,
        fail_on_bias: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate configuration for CI/CD pipeline integration.
        
        This config can be used by:
        - GitHub Actions
        - AWS CodePipeline
        - Jenkins
        - GitLab CI
        
        Args:
            model_name: Name of the model
            s3_bucket: S3 bucket for artifacts
            artifacts_prefix: S3 prefix for artifacts
            bias_threshold: Threshold for bias metrics
            fail_on_bias: Whether to fail pipeline on bias detection
        
        Returns:
            CI/CD configuration dictionary
        """
        
        config = {
            "pipeline_name": f"{model_name}-bias-check",
            "version": "1.0",
            
            "stages": {
                "train": {
                    "script": "python src/train_cv.py",
                    "environment": {
                        "MODEL_TYPE": "computer_vision",
                        "ENV": "dev",
                    },
                    "outputs": {
                        "model": f"s3://{s3_bucket}/{artifacts_prefix}/model/",
                        "artifacts": f"s3://{s3_bucket}/{artifacts_prefix}/",
                    }
                },
                
                "bias_analysis": {
                    "depends_on": "train",
                    "script": "python -c 'from bias_analysis import run_clarify; run_clarify()'",
                    "inputs": {
                        "predictions": f"s3://{s3_bucket}/{artifacts_prefix}/predictions.csv",
                        "metadata": f"s3://{s3_bucket}/{artifacts_prefix}/metadata_with_predictions.csv",
                        "config": f"s3://{s3_bucket}/{artifacts_prefix}/clarify_config.json",
                    },
                    "outputs": {
                        "report": f"s3://{s3_bucket}/{artifacts_prefix}/bias-reports/",
                    },
                    "thresholds": {
                        "max_bias": bias_threshold,
                        "fail_on_violation": fail_on_bias,
                    }
                },
                
                "deploy": {
                    "depends_on": "bias_analysis",
                    "condition": "bias_analysis.passed == true",
                    "script": "python deploy_model.py",
                }
            },
            
            "notifications": {
                "on_bias_detected": {
                    "type": "sns",
                    "topic": f"arn:aws:sns:{self.region}:{self.account_id}:bias-alerts",
                },
                "on_failure": {
                    "type": "email",
                    "recipients": ["ml-team@example.com"],
                }
            }
        }
        
        return config

    # =========================================================================
    # LAMBDA FUNCTION CODE TEMPLATE
    # =========================================================================

    @staticmethod
    def get_lambda_function_template() -> str:
        """
        Returns Lambda function code template for triggering Clarify analysis.
        
        This function can be deployed to AWS Lambda to:
        1. Parse S3 event
        2. Start SageMaker Processing Job for Clarify
        3. Send notifications on completion
        """
        
        return '''
import boto3
import json
import os
from datetime import datetime

sagemaker = boto3.client("sagemaker")
sns = boto3.client("sns")


def handler(event, context):
    """
    Lambda handler for triggering Clarify bias analysis.
    
    Can be triggered by:
    - S3 event (new artifact uploaded)
    - EventBridge schedule
    - Direct invocation
    """
    
    print(f"Received event: {json.dumps(event)}")
    
    # Extract configuration
    if "detail" in event:  # S3 event via EventBridge
        s3_bucket = event["detail"]["bucket"]["name"]
        s3_key = event["detail"]["object"]["key"]
        artifacts_prefix = "/".join(s3_key.split("/")[:-1])
    else:  # Direct invocation or scheduled
        s3_bucket = event.get("s3_bucket")
        artifacts_prefix = event.get("artifacts_prefix")
    
    job_name_prefix = event.get("job_name_prefix", "cv-bias-analysis")
    sns_topic_arn = event.get("sns_topic_arn")
    
    # Generate unique job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"{job_name_prefix}-{timestamp}"
    
    # Clarify processing job configuration
    processing_job_config = {
        "ProcessingJobName": job_name,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": 1,
                "InstanceType": "ml.m5.xlarge",
                "VolumeSizeInGB": 30,
            }
        },
        "AppSpecification": {
            "ImageUri": f"{os.environ['CLARIFY_IMAGE_URI']}",
            "ContainerEntrypoint": ["python", "/opt/ml/processing/input/code/clarify_analysis.py"],
        },
        "ProcessingInputs": [
            {
                "InputName": "predictions",
                "S3Input": {
                    "S3Uri": f"s3://{s3_bucket}/{artifacts_prefix}/predictions.csv",
                    "LocalPath": "/opt/ml/processing/input/predictions",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                }
            },
            {
                "InputName": "config",
                "S3Input": {
                    "S3Uri": f"s3://{s3_bucket}/{artifacts_prefix}/clarify_config.json",
                    "LocalPath": "/opt/ml/processing/input/config",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                }
            },
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "bias_report",
                    "S3Output": {
                        "S3Uri": f"s3://{s3_bucket}/{artifacts_prefix}/bias-reports/{timestamp}/",
                        "LocalPath": "/opt/ml/processing/output/bias_report",
                        "S3UploadMode": "EndOfJob",
                    }
                }
            ]
        },
        "RoleArn": os.environ["SAGEMAKER_ROLE_ARN"],
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 3600,
        },
    }
    
    # Start processing job
    response = sagemaker.create_processing_job(**processing_job_config)
    
    print(f"Started Clarify job: {job_name}")
    
    # Optionally send notification
    if sns_topic_arn:
        sns.publish(
            TopicArn=sns_topic_arn,
            Subject=f"Bias Analysis Started: {job_name}",
            Message=json.dumps({
                "job_name": job_name,
                "s3_bucket": s3_bucket,
                "artifacts_prefix": artifacts_prefix,
                "timestamp": timestamp,
            }),
        )
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "job_name": job_name,
            "status": "STARTED",
        }),
    }
'''

    # =========================================================================
    # MONITORING DASHBOARD CONFIGURATION
    # =========================================================================

    def generate_cloudwatch_dashboard(
        self,
        dashboard_name: str,
        model_name: str,
        metrics_namespace: str = "CVBiasMonitoring",
    ) -> Dict[str, Any]:
        """
        Generate CloudWatch dashboard configuration for bias monitoring.
        
        Args:
            dashboard_name: Name for the dashboard
            model_name: Name of the model
            metrics_namespace: CloudWatch metrics namespace
        
        Returns:
            Dashboard configuration
        """
        
        widgets = [
            {
                "type": "metric",
                "properties": {
                    "title": "Bias Metrics Over Time",
                    "metrics": [
                        [metrics_namespace, "DPPL", "ModelName", model_name],
                        [metrics_namespace, "DI", "ModelName", model_name],
                        [metrics_namespace, "AD", "ModelName", model_name],
                        [metrics_namespace, "RD", "ModelName", model_name],
                    ],
                    "period": 86400,  # Daily
                    "stat": "Average",
                    "region": self.region,
                }
            },
            {
                "type": "metric",
                "properties": {
                    "title": "Bias Threshold Violations",
                    "metrics": [
                        [metrics_namespace, "BiasViolations", "ModelName", model_name],
                    ],
                    "period": 86400,
                    "stat": "Sum",
                    "region": self.region,
                }
            },
            {
                "type": "metric",
                "properties": {
                    "title": "Model Accuracy by Group",
                    "metrics": [
                        [metrics_namespace, "Accuracy", "ModelName", model_name, "Group", "All"],
                    ],
                    "period": 86400,
                    "stat": "Average",
                    "region": self.region,
                }
            },
        ]
        
        dashboard_body = {
            "widgets": widgets
        }
        
        return {
            "dashboard_name": dashboard_name,
            "dashboard_body": json.dumps(dashboard_body),
        }


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_event_driven_setup():
    """Example: Set up event-driven bias monitoring."""
    
    pipeline = AutomatedBiasMonitoringPipeline(region="us-east-1")
    
    config = pipeline.setup_event_driven_monitoring(
        s3_bucket="tdsp-ml-products-dev",
        artifacts_prefix="tdspds/cv-model/clarify-artifacts",
        clarify_job_name_prefix="cv-bias-analysis",
        lambda_function_arn="arn:aws:lambda:us-east-1:123456789:function:clarify-trigger",
        sns_topic_arn="arn:aws:sns:us-east-1:123456789:bias-alerts",
    )
    
    print(f"Event-driven monitoring configured: {config}")


def example_scheduled_setup():
    """Example: Set up scheduled bias monitoring."""
    
    pipeline = AutomatedBiasMonitoringPipeline(region="us-east-1")
    
    config = pipeline.setup_scheduled_monitoring(
        schedule_name="daily-cv-bias-monitor",
        schedule_expression="cron(0 0 * * ? *)",  # Daily at midnight
        lambda_function_arn="arn:aws:lambda:us-east-1:123456789:function:clarify-trigger",
        monitoring_config={
            "s3_bucket": "tdsp-ml-products-dev",
            "artifacts_prefix": "tdspds/cv-model/clarify-artifacts",
            "job_name_prefix": "scheduled-bias-analysis",
            "bias_threshold": 0.1,
        }
    )
    
    print(f"Scheduled monitoring configured: {config}")


def example_cicd_integration():
    """Example: Generate CI/CD pipeline configuration."""
    
    pipeline = AutomatedBiasMonitoringPipeline(region="us-east-1")
    
    config = pipeline.generate_cicd_config(
        model_name="cv-image-classifier",
        s3_bucket="tdsp-ml-products-dev",
        artifacts_prefix="tdspds/cv-model/image-classifier",
        bias_threshold=0.1,
        fail_on_bias=True,
    )
    
    print(f"CI/CD configuration: {json.dumps(config, indent=2)}")


if __name__ == "__main__":
    # Print documentation
    print(__doc__)
    
    # Show examples
    print("\n" + "=" * 80)
    print("EXAMPLE CONFIGURATIONS")
    print("=" * 80)
    
    example_cicd_integration()
