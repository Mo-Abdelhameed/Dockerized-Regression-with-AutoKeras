import os
import re
import warnings
from typing import List
import numpy as np
import keras
import pandas as pd
import autokeras as ak
from joblib import dump, load
from sklearn.exceptions import NotFittedError
from config import paths
from schema.data_schema import RegressionSchema
from utils import read_json_as_dict
from keras_tuner.engine.hyperparameters import Choice

warnings.filterwarnings("ignore")
PREDICTOR_FILE_NAME = "predictor.joblib"


def clean_and_ensure_unique(names: List[str]) -> List[str]:
    """
    Clean the provided column names by removing special characters and ensure their
    uniqueness.

    The function first removes any non-alphanumeric character (except underscores)
    from the names. Then, it ensures the uniqueness of the cleaned names by appending
    a counter to any duplicates.

    Args:
        names (List[str]): A list of column names to be cleaned.

    Returns:
        List[str]: A list of cleaned column names with ensured uniqueness.

    Example:
        >>> clean_and_ensure_unique(['3P%', '3P', 'Name', 'Age%', 'Age'])
        ['3P', '3P_1', 'Name', 'Age', 'Age_1']
    """

    # First, clean the names
    cleaned_names = [re.sub("[^A-Za-z0-9_]+", "", name) for name in names]

    # Now ensure uniqueness
    seen = {}
    for i, name in enumerate(cleaned_names):
        original_name = name
        counter = 1
        while name in seen:
            name = original_name + "_" + str(counter)
            counter += 1
        seen[name] = True
        cleaned_names[i] = name

    return cleaned_names


class Regressor:
    """A wrapper class for the AutoKeras Regressor.

    This class provides a consistent interface that can be used with other
    regressor models.
    """

    def __init__(self,
                 train_input: pd.DataFrame,
                 schema: RegressionSchema,
                 predictor_dir_path: str = paths.PREDICTOR_DIR_PATH
                 ):
        """Construct a new Regressor."""
        self._is_trained: bool = False
        self.x = train_input.drop(columns=[schema.target])
        self.y = train_input[schema.target]
        self.schema = schema
        self.model_name = "AutoKeras_regressor"
        self.model_config = read_json_as_dict(paths.MODEL_CONFIG_FILE_PATH)
        custom_search = self.model_config["custom_search"]

        if custom_search:
            # Define a customized search space
            input_node = ak.StructuredDataInput()
            # Specify the number of neurons and layers in the DenseBlock
            output_node = ak.DenseBlock(
                num_layers=Choice("num_layers", values=self.model_config["search_space"]["num_layers"]),
                num_units=Choice("num_units", values=self.model_config["search_space"]["num_units"]))(input_node)
            output_node = ak.RegressionHead()(output_node)
            self.predictor = ak.AutoModel(
                inputs=input_node,
                outputs=output_node,
                max_trials=self.model_config["max_trials"],
                overwrite=True
            )
        else:
            self.predictor = ak.StructuredDataRegressor(
                column_names=list(self.x.columns),
                output_dim=1,
                loss="mean_squared_error",
                max_trials=self.model_config["max_trials"],
                directory=predictor_dir_path,
                overwrite=True
            )

    def __str__(self):
        return f"Model name: {self.model_name}"

    def train(self) -> None:
        """Train the model on the provided data"""
        self.predictor.fit(
            x=self.x,
            y=self.y,
            epochs=self.model_config["epochs"],
            validation_split=self.model_config["validation_split"]
        )
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.

        Returns:
            np.ndarray: The output predictions.
        """
        return self.predictor.export_model().predict(inputs)

    def save(self, model_dir_path: str) -> None:
        """Save the regressor to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """

        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        dump(self.predictor.export_model(), os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> keras.Model:
        """Load the regressor from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Regressor: A new instance of the loaded regressor.
        """
        return load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    def get_column_type_dict(self) -> dict:
        """
        Constructs a dictionary of feature names as keys and feature types as values

        returns: dict
        """
        results = {}
        numeric_features = self.schema.numeric_features
        categorical_features = self.schema.categorical_features

        for f in numeric_features:
            results[f] = "numerical"
        for f in categorical_features:
            results[f] = "categorical"

        return results


def predict_with_model(model: keras.Model, data: pd.DataFrame) -> np.ndarray:
    """
    Predict labels for the given data.

    Args:
        model (keras.Models): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        np.ndarray: The predicted labels.
    """
    return model.predict(data)


def save_predictor_model(model: keras.models, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> keras.Model:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)
