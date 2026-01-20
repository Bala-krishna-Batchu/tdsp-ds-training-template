import ast
import json
import logging
import os

import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

from src.utilities.fetch_features import fetch_features
from src.utilities.wandb_utils import (
    safe_wandb_init,
    safe_wandb_log,
    safe_wandb_config_update,
    safe_wandb_summary,
    safe_wandb_artifact,
    safe_wandb_log_artifact,
    safe_wandb_link_artifact,
    safe_wandb_finish,
    get_run_id,
)

logger = logging.getLogger("root")
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.INFO)

TEST_SIZE = 0.5
RANDOM_STATE = 0
TARGET_COLUMN = "Exited"

# Run-tracker schema version for cross-repo consistency
RUN_TRACKER_SCHEMA_VERSION = "1.0.0"


def read_config(env):
    file_path = f"/opt/ml/code/deploy/{env}/tdspds-config.yml"
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configurations = {config}")
    return config


def train(env):

    config = read_config(env)
    project_name = config.get("Model").get("ProjectName")
    model_name = config.get("Model").get("ModelName")
    model_version = config.get("Model", {}).get("ModelVersion", "v1")
    model_alias = config.get("Model", {}).get("ModelAlias", env)
    pod_name = config.get("TeamDetails").get("PodName")

    # Initialize W&B with safe wrapper (no-op if disabled)
    safe_wandb_init(
        entity=pod_name,
        project=project_name,
        notes="Testing with additional parameters",
        name="test churn model run",
        job_type="training",
        tags=[env, model_version],
    )

    # Log standardized run-tracker config fields
    safe_wandb_config_update({
        "run_tracker_schema_version": RUN_TRACKER_SCHEMA_VERSION,
        "run_kind": "training",
        "env": env,
        "model_name": model_name,
        "model_version": model_version,
        "model_alias": model_alias,
        "model_registry_path": f"{pod_name}/{project_name}/{model_name}",
    })

    # Helper functions to get hyperparameters and resource details
    hyperparameters = get_hyperparameters()
    get_resource_config()

    # Log hyperparameters to W&B config
    safe_wandb_config_update({"hyperparameters": hyperparameters})

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

    # Log metrics to W&B (safe wrapper)
    safe_wandb_log({"Training Score": train_score, "Test Score": test_score})

    # Set summary metrics for run-tracker table
    safe_wandb_summary("train_score", train_score)
    safe_wandb_summary("test_score", test_score)
    safe_wandb_summary("status", "success")

    # Model registry
    register_model(rf_reg, model_name, pod_name, project_name, model_version, model_alias, env)

    # Finish W&B run (safe wrapper)
    safe_wandb_finish()

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


def register_model(model, model_name, pod_name, project_name, model_version, model_alias, env):
    joblib.dump(model, model_name)

    # Build artifact metadata for inference run linkage
    artifact_metadata = {
        "trained_from_run_id": get_run_id() or "unknown",
        "model_version": model_version,
        "model_alias": model_alias,
        "model_registry_path": f"{pod_name}/{project_name}/{model_name}",
        "trained_in_env": env,
        "schema_version": RUN_TRACKER_SCHEMA_VERSION,
    }

    # Create artifact with safe wrapper
    artifact = safe_wandb_artifact(model_name, type="model", metadata=artifact_metadata)

    if artifact:
        # Add the model file to the artifact
        artifact.add_file(model_name)
        logger.info("done wandb artifact add\n")

        # Log the artifact to the W&B run
        safe_wandb_log_artifact(artifact)
        logger.info("done wandb artifact log\n")

        # Link the artifact to the model registry
        safe_wandb_link_artifact(artifact, f"{pod_name}/{project_name}/{model_name}")
        logger.info("done with wandb model registry\n")
    else:
        logger.info("W&B disabled - model saved locally only\n")


def get_resource_config():
    try:
        with open("/opt/ml/input/config/resourceconfig.json", "r") as json_file:
            resourceconfig = json.load(json_file)
        print(resourceconfig)
    except FileNotFoundError:
        logger.warning("Resource config not found (not running in SageMaker)")


def get_hyperparameters():
    try:
        with open("/opt/ml/input/config/hyperparameters.json", "r") as json_file:
            hyperparameters = json.load(json_file)
            print(hyperparameters)
        return hyperparameters
    except FileNotFoundError:
        logger.warning("Hyperparameters file not found, using defaults")
        return {
            "n_estimators": "[50, 100]",
            "max_depth": "[3, 6]",
            "max_leaf_nodes": "[3, 6]",
        }


if __name__ == "__main__":
    env = os.getenv("ENV")
    train(env)
