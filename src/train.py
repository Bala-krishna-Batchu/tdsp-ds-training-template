"""
Training script for churn prediction model.

Uses W&B for experiment tracking and model registry with safe/optional logging.
See src/wandb_tracking/schema.py for the shared run-tracker schema.
"""

import ast
import json
import logging
import os
import subprocess
from typing import Any, Dict, Optional, Tuple

import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

from src.utilities.fetch_features import fetch_features
from src.wandb_tracking import (
    WandbLogger,
    build_training_config,
    build_model_artifact_metadata,
)

logger = logging.getLogger("root")
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

TEST_SIZE = 0.5
RANDOM_STATE = 0
TARGET_COLUMN = "Exited"


def read_config(env: str) -> Dict[str, Any]:
    """Read TDSPDS configuration for the given environment."""
    file_path = f"/opt/ml/code/deploy/{env}/tdspds-config.yml"
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configurations = {config}")
    return config


def get_git_sha() -> Optional[str]:
    """Get current git commit SHA, or None if not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_dataset_uri_from_pipelines_config(env: str) -> Optional[str]:
    """Extract training data S3 URI from pipelines config."""
    try:
        file_path = f"/opt/ml/code/deploy/{env}/pipelines-config.yml"
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
        # Get S3 URI from first input channel (train)
        input_config = config.get("Training", {}).get("InputDataConfig", [])
        for channel in input_config:
            if channel.get("ChannelName") == "train":
                return channel.get("DataSource", {}).get("S3DataSource", {}).get("S3Uri")
    except Exception as e:
        logger.warning(f"Could not read pipelines config: {e}")
    return None


def train(env: str) -> None:
    """
    Main training function.
    
    Trains a RandomForestClassifier on churn data, logs metrics and model
    to W&B, and registers the model artifact with linkage metadata.
    
    Args:
        env: Environment name (dev|qa|prod)
    """
    # Load configuration
    config = read_config(env)
    project_name = config.get("Model", {}).get("ProjectName", "unknown-project")
    model_name = config.get("Model", {}).get("ModelName", "unknown-model")
    model_version = config.get("Model", {}).get("ModelVersion", "v1")
    model_alias = config.get("Model", {}).get("ModelAlias", env)
    pod_name = config.get("TeamDetails", {}).get("PodName", "unknown-pod")
    feature_group_arn = config.get("FeatureStore", {}).get("FeatureGroupName")
    
    # Get additional metadata
    code_git_sha = get_git_sha()
    dataset_uri = get_dataset_uri_from_pipelines_config(env)
    hyperparameters = get_hyperparameters()
    
    # Build standardized W&B config
    wandb_config = build_training_config(
        env=env,
        project_name=project_name,
        model_name=model_name,
        model_version=model_version,
        model_alias=model_alias,
        pod_name=pod_name,
        dataset_uri=dataset_uri,
        feature_group_arn=feature_group_arn,
        hyperparameters=hyperparameters,
        code_git_sha=code_git_sha,
        image_uri=os.getenv("SAGEMAKER_IMAGE_URI"),
    )
    
    # Initialize W&B with safe wrapper
    with WandbLogger(
        project=project_name,
        entity=pod_name,
        name=f"{model_name}-{model_version}-{env}",
        job_type="training",
        notes="Training churn prediction model",
        tags=[env, model_version, model_alias] if model_alias else [env, model_version],
        config=wandb_config,
    ) as wb:
        
        # Log resource config
        get_resource_config()
        
        # Fetch and prepare data
        logger.info("Fetching features from feature store...")
        df = fetch_features(config["FeatureStore"])
        logger.info("Features fetched successfully.")
        df = df.dropna()
        x = df.drop([TARGET_COLUMN, "EventTime", "CustomRecordId"], axis=1)
        y = df[TARGET_COLUMN]
        
        # Split the data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Log dataset size
        wb.log({
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "n_features": x_train.shape[1],
        })
        
        # Hyperparameter tuning
        logger.info("Starting hyperparameter tuning...")
        grid_search = hyperparameter_tuning(hyperparameters, x_train, y_train)
        best_params = grid_search.best_params_
        wb.log({"best_hyperparameters": best_params})
        
        # Train model with best parameters
        logger.info("Training final model...")
        rf_reg = train_model(grid_search, x_test, x_train, y_train)
        
        # Evaluate model
        test_score, train_score = evaluate_model(rf_reg, x_test, x_train, y_test, y_train)
        
        # Log metrics (both step metrics and summary)
        wb.log({"Training Score": train_score, "Test Score": test_score})
        wb.set_summary("train_score", train_score)
        wb.set_summary("test_score", test_score)
        wb.set_summary("best_hyperparameters", best_params)
        wb.set_summary("status", "success")
        
        logger.info(f"Training Score: {train_score:.4f}, Test Score: {test_score:.4f}")
        
        # Register model with linkage metadata
        register_model(
            wb=wb,
            model=rf_reg,
            model_name=model_name,
            model_version=model_version,
            model_alias=model_alias,
            pod_name=pod_name,
            project_name=project_name,
            env=env,
            hyperparameters=best_params,
            train_score=train_score,
            test_score=test_score,
        )
        
        logger.info("Training complete. Model saved to W&B Registry.\n")


def evaluate_model(
    rf_reg: RandomForestClassifier,
    x_test,
    x_train,
    y_test,
    y_train,
) -> Tuple[float, float]:
    """
    Evaluate model on train and test sets.
    
    Returns:
        Tuple of (test_score, train_score)
    """
    y_pred_train = rf_reg.predict(x_train)
    y_pred_test = rf_reg.predict(x_test)
    train_score = accuracy_score(y_train, y_pred_train)
    test_score = accuracy_score(y_test, y_pred_test)
    return test_score, train_score


def train_model(
    grid_search: GridSearchCV,
    x_test,
    x_train,
    y_train,
) -> RandomForestClassifier:
    """Train final model with best hyperparameters from grid search."""
    rf_reg = RandomForestClassifier(**(grid_search.best_params_))
    rf_reg.fit(x_train, y_train)
    rf_reg.predict(x_test)  # Warm up prediction
    return rf_reg


def hyperparameter_tuning(
    hyperparameters: Dict[str, Any],
    x_train,
    y_train,
) -> GridSearchCV:
    """Perform grid search hyperparameter tuning."""
    c = ast.literal_eval(str(hyperparameters))
    for k, v in c.items():
        # Convert single element strings to int if possible
        if isinstance(v, str):
            try:
                c[k] = [int(v)]
            except ValueError:
                c[k] = ast.literal_eval(v)
        # Convert int to list of int
        elif isinstance(v, int):
            c[k] = [v]
        # Handle other types as they are
        else:
            c[k] = v
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid=c)
    grid_search.fit(x_train, y_train)
    return grid_search


def register_model(
    wb: WandbLogger,
    model: RandomForestClassifier,
    model_name: str,
    model_version: str,
    model_alias: str,
    pod_name: str,
    project_name: str,
    env: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    train_score: Optional[float] = None,
    test_score: Optional[float] = None,
) -> None:
    """
    Save model and register to W&B Model Registry with linkage metadata.
    
    The artifact metadata includes trained_from_run_id so inference repos
    can link back to this training run.
    """
    # Save model locally
    joblib.dump(model, model_name)
    logger.info("Model saved locally.")
    
    # Build artifact metadata for inference linkage
    artifact_metadata = build_model_artifact_metadata(
        trained_from_run_id=wb.run_id or "unknown",
        model_version=model_version,
        model_alias=model_alias,
        model_registry_path=f"{pod_name}/{project_name}/{model_name}",
        env=env,
        hyperparameters=hyperparameters,
        train_score=train_score,
        test_score=test_score,
    )
    
    # Create and log artifact
    artifact = wb.create_artifact(
        name=model_name,
        type="model",
        metadata=artifact_metadata,
    )
    
    if artifact:
        artifact.add_file(model_name)
        logged_artifact = wb.log_artifact(artifact)
        
        if logged_artifact:
            # Link to model registry
            registry_path = f"{pod_name}/{project_name}/{model_name}"
            wb.link_artifact(logged_artifact, registry_path)
            logger.info(f"Model registered to: {registry_path}")
        else:
            logger.warning("Failed to log artifact to W&B")
    else:
        logger.warning("W&B disabled or failed - model saved locally only")


def get_resource_config() -> Dict[str, Any]:
    """Load SageMaker resource configuration."""
    try:
        with open("/opt/ml/input/config/resourceconfig.json", "r") as json_file:
            resourceconfig = json.load(json_file)
        logger.info(f"Resource config: {resourceconfig}")
        return resourceconfig
    except FileNotFoundError:
        logger.warning("Resource config not found (not running in SageMaker)")
        return {}


def get_hyperparameters() -> Dict[str, Any]:
    """Load hyperparameters from SageMaker config."""
    try:
        with open("/opt/ml/input/config/hyperparameters.json", "r") as json_file:
            hyperparameters = json.load(json_file)
            logger.info(f"Hyperparameters: {hyperparameters}")
        return hyperparameters
    except FileNotFoundError:
        logger.warning("Hyperparameters file not found, using defaults")
        return {
            "n_estimators": "[50, 100]",
            "max_depth": "[3, 6]",
            "max_leaf_nodes": "[3, 6]",
        }


if __name__ == "__main__":
    env = os.getenv("ENV", "dev")
    train(env)
