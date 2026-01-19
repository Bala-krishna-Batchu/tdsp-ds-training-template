import ast
import json
import logging
import os
import time

import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

from src.utilities.fetch_features import fetch_features
from src.wandb_tracking import WandbLogger
from src.wandb_tracking.schema import (
    RUN_KIND_TRAINING,
    RUN_TRACKER_SCHEMA_VERSION,
    build_model_artifact_metadata,
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


def build_training_run_config(env, config, hyperparameters):
    model_config = config.get("Model", {})
    team_config = config.get("TeamDetails", {})
    feature_config = config.get("FeatureStore", {})
    return {
        "run_tracker_schema_version": RUN_TRACKER_SCHEMA_VERSION,
        "run_kind": RUN_KIND_TRAINING,
        "env": env,
        "project_name": model_config.get("ProjectName"),
        "model_name": model_config.get("ModelName"),
        "model_version": model_config.get("ModelVersion"),
        "model_alias": model_config.get("ModelAlias"),
        "pod_name": team_config.get("PodName"),
        "feature_group_name": feature_config.get("FeatureGroupName"),
        "feature_store_output_path": feature_config.get("FeatureStoreOutputPath"),
        "hyperparameters": hyperparameters,
    }


def train(env):

    start_time = time.monotonic()
    config = read_config(env)
    model_config = config.get("Model", {})
    project_name = model_config.get("ProjectName")
    model_name = model_config.get("ModelName")
    model_version = model_config.get("ModelVersion")
    model_alias = model_config.get("ModelAlias")
    pod_name = config.get("TeamDetails", {}).get("PodName")

    wandb_logger = WandbLogger.from_env(logger)
    run_name = model_name
    if model_name and model_version:
        run_name = f"{model_name}-{model_version}"
    wandb_logger.init(
        entity=pod_name,
        project=project_name,
        job_type="training",
        name=run_name,
        notes="Training run",
    )

    # Helper functions to get hyperparameters and resource details
    hyperparameters = get_hyperparameters()
    get_resource_config()
    wandb_logger.config_update(build_training_run_config(env, config, hyperparameters))

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
    df = fetch_features(config["FeatureStore"])
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

    wandb_logger.log({"Training Score": train_score, "Test Score": test_score})
    wandb_logger.set_summary(
        {
            "Training Score": train_score,
            "Test Score": test_score,
            "status": "success",
            "duration_ms": int((time.monotonic() - start_time) * 1000),
        }
    )

    # Model registry
    model_registry_path = f"{pod_name}/{project_name}/{model_name}"
    artifact_metadata = build_model_artifact_metadata(
        model_name=model_name,
        model_version=model_version,
        model_alias=model_alias,
        model_registry_path=model_registry_path,
        trained_from_run_id=wandb_logger.run_id,
        project_name=project_name,
        pod_name=pod_name,
    )
    register_model(
        rf_reg,
        model_name,
        model_registry_path,
        wandb_logger,
        artifact_metadata,
    )

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


def register_model(model, model_name, model_registry_path, wandb_logger, artifact_metadata):
    joblib.dump(model, model_name)
    logger.info("Model serialized to %s", model_name)
    wandb_logger.log_artifact(
        name=model_name,
        artifact_type="model",
        files=[model_name],
        metadata=artifact_metadata,
        registry_path=model_registry_path,
    )
    logger.info("Model artifact logged to W&B registry")


def get_resource_config():
    with open("/opt/ml/input/config/resourceconfig.json", "r") as json_file:
        resourceconfig = json.load(json_file)
    logger.info("Resource config = %s", resourceconfig)
    return resourceconfig


def get_hyperparameters():
    with open("/opt/ml/input/config/hyperparameters.json", "r") as json_file:
        hyperparameters = json.load(json_file)
        logger.info("Hyperparameters = %s", hyperparameters)
    return hyperparameters


if __name__ == "__main__":
    env = os.getenv("ENV")
    train(env)
