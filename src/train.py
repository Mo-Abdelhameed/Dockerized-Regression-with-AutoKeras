import os

from config import paths
from logger import get_logger, log_error
from Regressor import Regressor
from preprocessing.pipeline import run_pipeline
from schema.data_schema import load_json_data_schema, save_schema
from utils import read_csv_in_directory, set_seeds

logger = get_logger(task_name="train")


def run_training(
    input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
    saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
    train_dir: str = paths.TRAIN_DIR,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
) -> None:
    """
    Run the training process and saves model artifacts

    Args:
        input_schema_dir (str, optional): The directory path of the input schema.
        saved_schema_dir_path (str, optional): The path where to save the schema.
        train_dir (str, optional): The directory path of the train data.
        predictor_dir_path (str, optional): Dir path where to save the predictor model.
    Returns:
        None
    """
    try:
        logger.info("Starting training...")
        set_seeds(seed_value=123)

        logger.info("Loading and saving schema...")
        data_schema = load_json_data_schema(input_schema_dir)
        save_schema(schema=data_schema, save_dir_path=saved_schema_dir_path)
        logger.info("Loading training data...")
        x_train = read_csv_in_directory(train_dir)
        target = x_train[data_schema.target]
        x_train = x_train.drop(columns=[data_schema.id, data_schema.target])

        logger.info("Preprocessing training data...")

        for column in data_schema.categorical_features:
            x_train[column] = x_train[column].astype(str)

        x_train = run_pipeline(x_train, data_schema, training=True)
        x_train[data_schema.target] = target

        regressor = Regressor(x_train, data_schema)
        regressor.train()
        if not os.path.exists(predictor_dir_path):
            os.makedirs(predictor_dir_path)
        regressor.save(predictor_dir_path)
        logger.info("Model saved!")

    except Exception as exc:
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_training()
