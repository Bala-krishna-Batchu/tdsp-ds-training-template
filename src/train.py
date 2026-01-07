import ast
import json
import logging
import os

import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

from src.wandb_tracking.schema import RunKinds, build_common_config
from src.wandb_tracking.wandb_logger import WandbLogger

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

    config = read_config(env)
    model_cfg = config.get("Model", {}) or {}
    team_cfg = config.get("TeamDetails", {}) or {}
    fs_cfg = config.get("FeatureStore", {}) or {}

    project_name = model_cfg.get("ProjectName")
    model_name = model_cfg.get("ModelName")
    model_version = model_cfg.get("ModelVersion")
    model_alias = model_cfg.get("ModelAlias")
    pod_name = team_cfg.get("PodName")

    run_name = f"{model_name}-{env}".strip("-") if model_name and env else "training-run"
    registry_path = (
        f"{pod_name}/{project_name}/{model_name}" if pod_name and project_name and model_name else None
    )

    wb = WandbLogger.init(
        entity=pod_name,
        project=project_name,
        name=run_name,
        job_type=RunKinds.TRAINING,
        tags=[f"env:{env}", f"model:{model_name}", f"model_version:{model_version}", f"alias:{model_alias}"],
        config=build_common_config(
            run_kind=RunKinds.TRAINING,
            env=env,
            model_name=model_name,
            model_version=model_version,
            model_alias=model_alias,
            model_registry_path=registry_path,
            dataset_name="sagemaker_feature_store",
            dataset_version=None,
            dataset_uri=fs_cfg.get("FeatureGroupName") or fs_cfg.get("FeatureStoreOutputPath"),
            code_git_sha=os.getenv("GIT_SHA") or os.getenv("GIT_COMMIT") or os.getenv("CODEBUILD_RESOLVED_SOURCE_VERSION"),
            image_uri=os.getenv("IMAGE_URI"),
            image_digest=os.getenv("IMAGE_DIGEST"),
            extra={
                "feature_group_arn": fs_cfg.get("FeatureGroupName"),
                "feature_store_output_path": fs_cfg.get("FeatureStoreOutputPath"),
            },
        ),
        notes="Training run (safe W&B logging; schema-ready for inference linkage).",
    )

    # Helper functions to get hyperparameters and resource details
    hyperparameters = get_hyperparameters()
    get_resource_config()
    wb.config_update({"training_hyperparameters": hyperparameters})

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
    # Imported lazily to keep module import lightweight (unit tests don't need boto3/sagemaker).
    from src.utilities.fetch_features import fetch_features

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

    wb.log({"Training Score": train_score, "Test Score": test_score})
    wb.summary_update(
        {
            "training_score": train_score,
            "test_score": test_score,
            "status": "success",
        }
    )

    # Model registry
    register_model(
        rf_reg,
        model_name=model_name,
        model_version=model_version,
        model_alias=model_alias,
        pod_name=pod_name,
        project_name=project_name,
        wandb_logger=wb,
    )

    wb.finish()

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
    pod_name,
    project_name,
    wandb_logger: WandbLogger,
):
    joblib.dump(model, model_name)
    # Log model artifact (and attach training→inference linkage metadata).
    registry_path = (
        f"{pod_name}/{project_name}/{model_name}" if pod_name and project_name and model_name else None
    )
    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "model_alias": model_alias,
        "trained_from_run_id": wandb_logger.run_id,
        "model_registry_path": registry_path,
    }

    logged_artifact = wandb_logger.log_artifact_file(
        artifact_name=str(model_name),
        artifact_type="model",
        file_path=str(model_name),
        aliases=[a for a in [model_alias, model_version, "latest"] if a],
        metadata=metadata,
    )

    if logged_artifact is not None and registry_path:
        wandb_logger.link_artifact(logged_artifact, registry_path)
        logger.info("done with wandb model registry\n")


def get_resource_config():
    with open("/opt/ml/input/config/resourceconfig.json", "r") as json_file:
        resourceconfig = json.load(json_file)
    print(resourceconfig)


def get_hyperparameters():
    with open("/opt/ml/input/config/hyperparameters.json", "r") as json_file:
        hyperparameters = json.load(json_file)
        print(hyperparameters)
    return hyperparameters


if __name__ == "__main__":
    env = os.getenv("ENV")
    train(env)
