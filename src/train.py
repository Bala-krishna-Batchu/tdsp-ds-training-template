import ast
import json
import logging
import os
import time

import joblib
import wandb
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

from src.utilities.fetch_features import fetch_features
from src.wandb_tracking import (
    RUN_KIND_TRAINING,
    WandbLogger,
    build_linkage_metadata,
    build_run_config,
)

logger = logging.getLogger("root")
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

TEST_SIZE = 0.5
RANDOM_STATE = 0
TARGET_COLUMN = "Exited"


def read_config(env):
    file_path = f"/opt/ml/code/deploy/{env}/tdspds-config.yml"
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configurations = {config}")
    return config


def train(env):
    env = env or "unknown"
    config = read_config(env)
    model_config = config.get("Model", {})
    team_config = config.get("TeamDetails", {})
    feature_store_config = config.get("FeatureStore", {})

    project_name = model_config.get("ProjectName", "unknown_project")
    model_name = model_config.get("ModelName", "unknown_model")
    model_version = model_config.get("ModelVersion", "unknown")
    model_alias = model_config.get("ModelAlias", "unknown")
    pod_name = team_config.get("PodName")
    model_registry_path = None
    if pod_name and project_name and model_name:
        model_registry_path = f"{pod_name}/{project_name}/{model_name}"

    run_config = build_run_config(
        run_kind=RUN_KIND_TRAINING,
        env=env,
        service_name="training",
        model_name=model_name,
        model_version=model_version,
        model_alias=model_alias,
        model_registry_path=model_registry_path,
        dataset_name=feature_store_config.get("FeatureGroupName"),
        dataset_uri=feature_store_config.get("FeatureStoreOutputPath"),
        code_git_sha=os.getenv("CODE_GIT_SHA") or os.getenv("GIT_SHA"),
        image_uri=os.getenv("IMAGE_URI"),
        image_digest=os.getenv("IMAGE_DIGEST"),
    )

    wandb_logger = WandbLogger(
        entity=pod_name,
        project=project_name,
        run_name=f"train-{model_name}-{model_version}",
        notes="Training run",
        tags=["training"],
        config=run_config,
    )

    # Helper functions to get hyperparameters and resource details
    hyperparameters = get_hyperparameters()
    resource_config = get_resource_config()
    wandb_logger.update_config({"hyperparameters": hyperparameters, "resource_config": resource_config})

    # Load the data
    """
    The following snippet will pull the data from S3

    training_data_path = '/opt/ml/input/data/train'
    validation_data_path = '/opt/ml/input/data/validation'
    test_data_path = '/opt/ml/input/data/test'
    df_train = pd.read_csv(os.path.join(training_data_path, 'churn_data.csv'))
    df_validation = pd.read_csv(os.path.join(training_data_path, 'churn_data.csv'))
    df_test = pd.read_csv(os.path.join(training_data_path, 'churn_data.csv'))
    """

    """
    The following code will fetch the data from feature store
    """
    start_time = time.monotonic()
    status = "success"
    train_score = None
    test_score = None
    try:
        df = fetch_features(feature_store_config)
        logger.info("Features fetched sucessfully.")
        df = df.dropna()
        x = df.drop([TARGET_COLUMN, "EventTime", "CustomRecordId"], axis=1)
        y = df[TARGET_COLUMN]

        # Split the data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Hyperparameter Tuning
        grid_search = hyperparameter_tuning(hyperparameters, x_train, y_train)
        # Train Model
        rf_reg = train_model(grid_search, x_test, x_train, y_train)
        # Evaluate Model
        test_score, train_score = evaluate_model(rf_reg, x_test, x_train, y_test, y_train)

        wandb_logger.log_metrics({"train_accuracy": train_score, "test_accuracy": test_score})

        linkage_metadata = build_linkage_metadata(
            trained_from_run_id=wandb_logger.run_id,
            model_registry_path=model_registry_path,
        )
        model_artifact_ref = register_model(
            rf_reg,
            model_name,
            pod_name,
            project_name,
            wandb_logger,
            linkage_metadata,
        )
        if model_artifact_ref:
            wandb_logger.update_config({"model_artifact_ref": model_artifact_ref})
    except Exception:
        status = "failed"
        logger.exception("Training failed.")
        raise
    finally:
        duration_ms = int((time.monotonic() - start_time) * 1000)
        summary = {"status": status, "duration_ms": duration_ms}
        if train_score is not None:
            summary["train_accuracy"] = train_score
        if test_score is not None:
            summary["test_accuracy"] = test_score
        wandb_logger.set_summary(summary)
        wandb_logger.finish()

    logger.info("Model saved Wandb Registry\n")


def evaluate_model(rf_reg, x_test, x_train, y_test, y_train):
    y_pred_train = rf_reg.predict(x_train)
    y_pred_test = rf_reg.predict(x_test)
    train_score = accuracy_score(y_train, y_pred_train)
    test_score = accuracy_score(y_test, y_pred_test)
    return test_score, train_score


def train_model(grid_search, x_test, x_train, y_train):
    rf_reg = RandomForestClassifier(**(grid_search.best_params_))
    rf_reg.fit(x_train, y_train)
    rf_reg.predict(x_test)
    return rf_reg


def hyperparameter_tuning(hyperparameters, x_train, y_train):

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


def register_model(model, model_name, pod_name, project_name, wandb_logger, linkage_metadata=None):
    joblib.dump(model, model_name)
    if not wandb_logger or not wandb_logger.enabled:
        logger.info("W&B disabled; skipping artifact logging.")
        return None
    # Create an artifact
    artifact = wandb.Artifact(model_name, type="model")
    if linkage_metadata:
        artifact.metadata.update(linkage_metadata)
    # Add the model file to the artifact
    artifact.add_file(model_name)
    # Log the artifact to the W&B run
    logged_artifact = wandb_logger.log_artifact(artifact)
    if not logged_artifact:
        return None
    logged_artifact.wait()
    # Link the artifact to the model registry
    if pod_name and project_name and model_name:
        registry_path = f"{pod_name}/{project_name}/{model_name}"
        wandb_logger.link_artifact(logged_artifact, registry_path)
    return getattr(logged_artifact, "qualified_name", None)


def get_resource_config():
    try:
        with open("/opt/ml/input/config/resourceconfig.json", "r") as json_file:
            resourceconfig = json.load(json_file)
    except FileNotFoundError:
        logger.warning("Resource config not found at /opt/ml/input/config/resourceconfig.json")
        return {}
    logger.info("Resource config loaded.")
    return resourceconfig


def get_hyperparameters():
    try:
        with open("/opt/ml/input/config/hyperparameters.json", "r") as json_file:
            hyperparameters = json.load(json_file)
    except FileNotFoundError:
        logger.warning("Hyperparameters not found at /opt/ml/input/config/hyperparameters.json")
        return {}
    logger.info("Hyperparameters loaded.")
    return hyperparameters


if __name__ == "__main__":
    env = os.getenv("ENV")
    train(env)
