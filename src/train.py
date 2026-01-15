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
from src.wandb_tracking import (
    WandbLogger,
    TrainingRunConfig,
    TrainingRunSummary,
    RunStatus,
    get_run_tags,
    get_model_artifact_metadata,
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


def get_git_sha() -> str:
    """Get git SHA from environment or return empty string."""
    # SageMaker/CI may set these
    return os.getenv("GIT_SHA", os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION", ""))


def get_git_branch() -> str:
    """Get git branch from environment or return empty string."""
    return os.getenv("GIT_BRANCH", os.getenv("CODEBUILD_SOURCE_VERSION", ""))


def get_image_uri() -> str:
    """Get container image URI from environment or return empty string."""
    return os.getenv("IMAGE_URI", os.getenv("SAGEMAKER_CONTAINER_IMAGE", ""))


def train(env):
    start_time = time.time()

    config = read_config(env)
    project_name = config.get("Model").get("ProjectName")
    model_name = config.get("Model").get("ModelName")
    model_version = config.get("Model").get("ModelVersion", "")
    model_alias = config.get("Model").get("ModelAlias", "")
    pod_name = config.get("TeamDetails").get("PodName")
    
    # Feature store config
    feature_store_config = config.get("FeatureStore", {})
    feature_group_arn = feature_store_config.get("FeatureGroupName", "")
    dataset_uri = feature_store_config.get("FeatureStoreOutputPath", "")

    # Build registry path
    model_registry_path = f"{pod_name}/{project_name}/{model_name}"

    # Build standardized run config
    run_config = TrainingRunConfig(
        env=env,
        service_name=f"{model_name}-training",
        model_name=model_name,
        model_version=model_version,
        model_alias=model_alias,
        model_registry_path=model_registry_path,
        pod_name=pod_name,
        project_name=project_name,
        code_git_sha=get_git_sha(),
        code_git_branch=get_git_branch(),
        image_uri=get_image_uri(),
        feature_group_arn=feature_group_arn,
        dataset_uri=dataset_uri,
        training_job_name=os.getenv("TRAINING_JOB_NAME", ""),
        training_job_arn=os.getenv("TRAINING_JOB_ARN", ""),
    )

    # Initialize W&B with safe wrapper
    wandb_logger = WandbLogger()
    wandb_init_success = wandb_logger.init(
        entity=pod_name,
        project=project_name,
        notes="Testing with additional parameters",
        name="test churn model run",
        job_type="training",
        tags=get_run_tags(env, run_config.run_kind, model_name),
    )

    if wandb_init_success:
        # Log standardized config fields
        wandb_logger.config_update(run_config.to_dict())

    # Helper functions to get hyperparameters and resource details
    hyperparameters = get_hyperparameters()
    get_resource_config()

    # Log hyperparameters to W&B config
    if wandb_init_success and hyperparameters:
        wandb_logger.config_update({"hyperparameters": hyperparameters})
        run_config.hyperparameters_logged = True

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

    # Log training metrics
    wandb_logger.log({"Training Score": train_score, "Test Score": test_score})

    # Calculate duration
    duration_ms = int((time.time() - start_time) * 1000)

    # Build summary
    run_summary = TrainingRunSummary(
        status=RunStatus.SUCCESS.value,
        duration_ms=duration_ms,
        train_score=train_score,
        test_score=test_score,
        best_params=grid_search.best_params_ if hasattr(grid_search, 'best_params_') else {},
        n_train_samples=len(x_train),
        n_test_samples=len(x_test),
        n_features=x_train.shape[1] if hasattr(x_train, 'shape') else 0,
    )
    wandb_logger.summary_update(run_summary.to_dict())

    # Model registry - pass wandb_logger for safe artifact logging
    register_model(
        rf_reg,
        model_name,
        pod_name,
        project_name,
        wandb_logger,
        run_config,
    )

    wandb_logger.finish()

    logger.info("Model saved to W&B Registry\n")


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


def register_model(
    model,
    model_name,
    pod_name,
    project_name,
    wandb_logger: WandbLogger,
    run_config: TrainingRunConfig,
):
    """
    Register model to W&B with linkage metadata.
    
    The metadata enables inference repos to link back to this training run.
    """
    joblib.dump(model, model_name)
    
    # Build artifact metadata for inference linkage
    artifact_metadata = get_model_artifact_metadata(
        run_config,
        run_id=wandb_logger.run_id or "",
    )
    
    # Create artifact with metadata
    artifact = wandb_logger.create_artifact(
        name=model_name,
        artifact_type="model",
        description=f"RandomForestClassifier model for {project_name}",
        metadata=artifact_metadata,
    )
    
    if artifact is None:
        logger.warning("Failed to create W&B artifact, skipping model registration")
        return
    
    logger.info("Created W&B artifact")
    
    # Add the model file to the artifact
    artifact.add_file(model_name)
    logger.info("Added model file to artifact")
    
    # Log the artifact to the W&B run
    wandb_logger.log_artifact(artifact)
    logger.info("Logged artifact to W&B")
    
    # Link the artifact to the model registry
    registry_path = f"{pod_name}/{project_name}/{model_name}"
    wandb_logger.link_artifact(artifact, registry_path)
    logger.info(f"Linked artifact to registry: {registry_path}")


def get_resource_config():
    try:
        with open("/opt/ml/input/config/resourceconfig.json", "r") as json_file:
            resourceconfig = json.load(json_file)
        print(resourceconfig)
    except FileNotFoundError:
        logger.warning("Resource config file not found, using defaults")
        return {}


def get_hyperparameters():
    try:
        with open("/opt/ml/input/config/hyperparameters.json", "r") as json_file:
            hyperparameters = json.load(json_file)
            print(hyperparameters)
        return hyperparameters
    except FileNotFoundError:
        logger.warning("Hyperparameters config file not found, using defaults")
        return {
            "n_estimators": [50, 100],
            "max_depth": [3, 6],
            "max_leaf_nodes": [3, 6],
        }


if __name__ == "__main__":
    env = os.getenv("ENV")
    train(env)
