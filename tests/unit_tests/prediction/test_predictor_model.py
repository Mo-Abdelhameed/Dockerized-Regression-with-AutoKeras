import os

import pytest
from keras import Model
from src.Regressor import Regressor, load_predictor_model, save_predictor_model
from src.preprocessing.pipeline import run_pipeline


@pytest.fixture
def regressor(sample_train_data, schema_provider):
    """Define the regressor fixture"""
    target = sample_train_data[schema_provider.target]
    sample_train_data = sample_train_data.drop(columns=[schema_provider.id, schema_provider.target])
    sample_train_data = run_pipeline(sample_train_data, schema_provider, training=True)
    sample_train_data[schema_provider.target] = target
    regressor = Regressor(sample_train_data, schema=schema_provider)
    return regressor


def test_fit_predict(regressor, sample_train_data, sample_test_data, schema_provider):
    """
    Test if the fit method trains the model correctly and if predict method work as expected.
    """
    regressor.train()
    sample_test_data = sample_test_data.drop(columns=[schema_provider.id, schema_provider.target])
    sample_test_data = run_pipeline(sample_test_data, schema_provider, training=False)
    predictions = regressor.predict(sample_test_data)
    assert predictions.shape[0] == sample_test_data.shape[0]


def test_regressor_str_representation(regressor):
    """
    Test the `__str__` method of the `Regressor` class.

    The test asserts that the string representation of a `Regressor` instance is
    correctly formatted and includes the model name and the correct hyperparameters.

    Args:
        regressor (Regressor): An instance of the `Regressor` class,
            created using the `hyperparameters` fixture.

    Raises:
        AssertionError: If the string representation of `regressor` does not
            match the expected format.
    """
    regressor_str = str(regressor)
    assert regressor.model_name in regressor_str


def test_save_predictor_model(tmpdir, regressor, sample_train_data, schema_provider):
    """
    Test that the 'save_predictor_model' function correctly saves a Regressor instance
    to disk.
    """
    regressor.train()
    model_dir_path = os.path.join(tmpdir, "model")
    save_predictor_model(regressor, model_dir_path)
    assert os.path.exists(model_dir_path)
    assert len(os.listdir(model_dir_path)) >= 1


def test_load_predictor_model(tmpdir, regressor, sample_train_data, schema_provider):
    """
    Test that the 'load_predictor_model' function correctly loads a Regressor
    instance from disk and that the loaded instance has the correct hyperparameters.
    """

    regressor.train()
    model_dir_path = os.path.join(tmpdir, "model")
    save_predictor_model(regressor, model_dir_path)

    loaded_clf = load_predictor_model(model_dir_path)
    assert isinstance(loaded_clf, Model)
