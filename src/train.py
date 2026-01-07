import ast
import json
import logging
import os

import joblib
import wandb
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

from src.utilities.fetch_features import fetch_features

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
    pod_name = config.get("TeamDetails").get("PodName")

    wandb.init(
        entity=pod_name,
        project=project_name,
        notes="Testing with additional parameters",
        name="test churn model run",
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

    wandb.log({"Training Score": train_score, "Test Score": test_score})

    # Model registry
    register_model(rf_reg, model_name, pod_name, project_name)

    wandb.run.finish()

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


def register_model(model, model_name, pod_name, project_name):
    joblib.dump(model, model_name)
    # Create an artifact
    artifact = wandb.Artifact(model_name, type="model")
    logger.info("done wandb artifact\n")
    # Add the model file to the artifact
    artifact.add_file(model_name)
    logger.info("done wandb artifact add\n")
    # Log the artifact to the W&B run
    arti = wandb.log_artifact(artifact)
    arti.wait()
    logger.info("done wandb artifact log\n")
    # Link the artifact to the model registry
    wandb.run.link_artifact(artifact, f"{pod_name}/{project_name}/{model_name}")
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
