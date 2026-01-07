import os
import sys
import unittest
from unittest.mock import MagicMock

from sklearn.ensemble import RandomForestClassifier

# Get the absolute path of the parent directory
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(parent_directory)
# Add the parent directory to sys.path
sys.path.append(parent_directory)
# Import the functions to be tested from your script
from src.train import evaluate_model, train_model


class TestTrainingScript(unittest.TestCase):
    def test_evaluate_model(self):
        # Mock data
        rf_reg = MagicMock()
        x_train = [[1, 2], [3, 4]]
        x_test = [[5, 6], [7, 8]]
        y_train = [0, 1]
        y_test = [0, 1]

        # Mock the predict method to return a list/array-like object
        rf_reg.predict.return_value = [0, 1]  # Modify this based on your expected predictions

        # Call the function
        test_score, train_score = evaluate_model(rf_reg, x_test, x_train, y_test, y_train)

        # Check if scores are floats between 0 and 1
        self.assertTrue(isinstance(test_score, float))
        self.assertTrue(isinstance(train_score, float))
        self.assertTrue(0 <= test_score <= 1)
        self.assertTrue(0 <= train_score <= 1)

    def test_train_model(self):
        # Mock data
        grid_search = MagicMock()
        grid_search.best_params_ = {"n_estimators": 100, "max_depth": 10}
        x_train = [[1, 2], [3, 4]]
        x_test = [[5, 6], [7, 8]]
        y_train = [0, 1]

        # Call the function
        rf_reg = train_model(grid_search, x_test, x_train, y_train)

        # Check if rf_reg is an instance of RandomForestClassifier
        self.assertIsInstance(rf_reg, RandomForestClassifier)

    # You can write similar tests for other functions


if __name__ == "__main__":
    unittest.main()
