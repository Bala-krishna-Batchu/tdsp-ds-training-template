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
from src.wandb_tracking import RunTrackerConfig, WandbLogger, build_registry_path

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


def _get_first_env(*names):
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def _safe_load_json(path: str):
    try:
        with open(path, "r") as json_file:
            return json.load(json_file)
    except FileNotFoundError:
        logger.info("Optional file not found: %s", path)
        return {}


def train(env):

    env = env or "dev"
    config = read_config(env)
    project_name = config.get("Model").get("ProjectName")
    model_name = config.get("Model").get("ModelName")
    model_version = config.get("Model").get("ModelVersion")
    model_alias = config.get("Model").get("ModelAlias")
    pod_name = config.get("TeamDetails").get("PodName")

    wandb_logger = WandbLogger()

    registry_path = build_registry_path(pod_name, project_name, model_name)
    code_git_sha = _get_first_env("GIT_SHA", "GITHUB_SHA", "CODEBUILD_RESOLVED_SOURCE_VERSION")
    image_uri = _get_first_env("IMAGE_URI", "CONTAINER_IMAGE_URI")
    image_digest = _get_first_env("IMAGE_DIGEST", "CONTAINER_IMAGE_DIGEST")

    # Initialize W&B tracking (best-effort, safe if disabled/unavailable).
    wandb_logger.init(
        entity=pod_name,
        project=project_name,
        job_type="training",
        notes="Training run (template)",
        name=f"{model_name}-{env}",
        tags=[
            f"env={env}",
            f"model_name={model_name}",
            f"model_version={model_version}",
            f"model_alias={model_alias}",
        ],
        group=f"{project_name}:{model_name}",
    )

    wandb_logger.config_update_run_tracker(
        RunTrackerConfig(
            run_kind="training",
            env=env,
            service_name="sagemaker-training",
            model_name=model_name,
            model_version=model_version,
            model_alias=model_alias,
            model_registry_path=registry_path,
            trained_from_run_id=None,
            dataset_name="sagemaker-feature-store",
            dataset_version=config.get("FeatureStore", {}).get("FeatureGroupName"),
            dataset_uri=config.get("FeatureStore", {}).get("FeatureStoreOutputPath"),
            code_git_sha=code_git_sha,
            image_uri=image_uri,
            image_digest=image_digest,
        )
    )
    try:
        # Helper functions to get hyperparameters and resource details
        hyperparameters = get_hyperparameters()
        resourceconfig = get_resource_config()
        if hyperparameters:
            wandb_logger.config_update({"hyperparameters": hyperparameters})
        if resourceconfig:
            wandb_logger.config_update({"resourceconfig": resourceconfig})

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

        wandb_logger.log(
            {
                # Keep existing names for backward compatibility
                "Training Score": train_score,
                "Test Score": test_score,
                # Standardized keys (recommended)
                "train_accuracy": train_score,
                "test_accuracy": test_score,
            }
        )
        wandb_logger.summary_update({"train_accuracy": train_score, "test_accuracy": test_score})

        # Model registry
        register_model(
            rf_reg,
            model_name=model_name,
            model_version=model_version,
            model_alias=model_alias,
            env=env,
            pod_name=pod_name,
            project_name=project_name,
            registry_path=registry_path,
            trained_from_run_id=wandb_logger.run_id,
            code_git_sha=code_git_sha,
            feature_store=config.get("FeatureStore", {}),
            wandb_logger=wandb_logger,
        )

        wandb_logger.finish(status="success")
    except Exception:
        wandb_logger.finish(status="failed")
        raise

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


def register_model(
    model,
    *,
    model_name,
    model_version,
    model_alias,
    env,
    pod_name,
    project_name,
    registry_path,
    trained_from_run_id,
    code_git_sha,
    feature_store,
    wandb_logger: WandbLogger,
):
    joblib.dump(model, model_name)

    artifact_metadata = {
        "run_tracker_schema_version": RunTrackerConfig().run_tracker_schema_version,
        "run_kind": "training",
        "env": env,
        "entity": pod_name,
        "project": project_name,
        "model_name": model_name,
        "model_version": model_version,
        "model_alias": model_alias,
        "model_registry_path": registry_path,
        "trained_from_run_id": trained_from_run_id,
        "code_git_sha": code_git_sha,
        "feature_store": feature_store,
    }

    wandb_logger.log_model_file_artifact(
        model_file_path=model_name,
        artifact_name=model_name,
        artifact_type="model",
        metadata=artifact_metadata,
        registry_path=registry_path,
    )


def get_resource_config():
    return _safe_load_json("/opt/ml/input/config/resourceconfig.json")


def get_hyperparameters():
    return _safe_load_json("/opt/ml/input/config/hyperparameters.json")


if __name__ == "__main__":
    env = os.getenv("ENV")
    train(env)
