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
from src.wandb_tracking import WandbLogger, build_run_config, get_linkage_metadata, RunTrackerMetrics

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
    project_name = config.get("Model").get("ProjectName")
    model_name = config.get("Model").get("ModelName")
    model_version = config.get("Model", {}).get("ModelVersion")
    model_alias = config.get("Model", {}).get("ModelAlias")
    pod_name = config.get("TeamDetails").get("PodName")

    # Initialize W&B Logger
    wl = WandbLogger()

    # Build standardized config
    run_config = build_run_config(
        run_kind="training",
        env=env,
        service_name=project_name,
        model_name=model_name,
        model_version=model_version,
        model_alias=model_alias,
        pod_name=pod_name,
        project_name=project_name
    )

    wl.init(
        entity=pod_name,
        project=project_name,
        config=run_config,
        notes="Testing with additional parameters",
        name="test churn model run",
        job_type="training"
    )

    # Helper functions to get hyperparameters and resource details
    hyperparameters = get_hyperparameters()
    get_resource_config()

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

    wl.log({"Training Score": train_score, "Test Score": test_score})
    
    # Set summary status
    wl.set_summary(RunTrackerMetrics.STATUS, "success")

    # Model registry
    register_model(rf_reg, model_name, pod_name, project_name, wl, model_version)

    wl.finish()

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


def register_model(model, model_name, pod_name, project_name, wl, model_version=None):
    joblib.dump(model, model_name)
    
    # Prepare metadata for linkage
    registry_path = f"{pod_name}/{project_name}/{model_name}"
    metadata = get_linkage_metadata(
        run_id=wl.run_id,
        model_registry_path=registry_path,
        model_version=model_version
    )
    
    logger.info("done wandb artifact\n")
    
    # Log the artifact to the W&B run
    artifact = wl.log_artifact(
        artifact_name=model_name,
        artifact_type="model",
        file_path=model_name,
        metadata=metadata
    )
    
    if artifact:
        # Link the artifact to the model registry
        wl.link_artifact(artifact, registry_path)
        logger.info("done with wandb model registry\n")
    else:
        logger.warning("Artifact not logged, skipping linkage.")


def get_resource_config():
    try:
        with open("/opt/ml/input/config/resourceconfig.json", "r") as json_file:
            resourceconfig = json.load(json_file)
        print(resourceconfig)
    except FileNotFoundError:
        logger.warning("resourceconfig.json not found, skipping")


def get_hyperparameters():
    try:
        with open("/opt/ml/input/config/hyperparameters.json", "r") as json_file:
            hyperparameters = json.load(json_file)
            print(hyperparameters)
        return hyperparameters
    except FileNotFoundError:
        # Return default or empty if not found locally
        logger.warning("hyperparameters.json not found, using defaults")
        return {"n_estimators": 10}


if __name__ == "__main__":
    env = os.getenv("ENV", "dev") # Default to dev if not set
    train(env)
