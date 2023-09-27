import pandas as pd
from joblib import dump, load

from config import paths
from preprocessing.preprocess import (
    encode,
    impute_categorical,
    impute_numeric,
    normalize,
)
from schema.data_schema import RegressionSchema


def run_pipeline(
    input_data: pd.DataFrame,
    schema: RegressionSchema,
    training: bool = True,
    imputation_path: str = paths.IMPUTATION_FILE_PATH
) -> pd.DataFrame:
    """
    Apply transformations to the input data (Imputations, encoding and normalization).

    Args:
        input_data (pd.DataFrame): Data to be processed.
        schema (RegressionSchema): RegressionSchema object carrying data about the schema
        training (bool): Should be set to true if the data is for the training process.
        imputation_path (str): Path to the file containing values used for imputation.
    Returns:
        pd.DataFrame: The data after applying the transformations
    """
    if training:
        imputation_dict = {}
        for f in schema.categorical_features:
            input_data, value = impute_categorical(input_data, f)
            imputation_dict[f] = value

        for f in schema.numeric_features:
            input_data, value = impute_numeric(input_data, f)
            imputation_dict[f] = value

        input_data = normalize(input_data, schema)

        input_data = encode(input_data, schema)
        dump(imputation_dict, imputation_path)
    else:
        imputation_dict = load(imputation_path)

        for f in schema.features:
            input_data[f].fillna(
                imputation_dict.get(f, input_data[f].mode()[0]), inplace=True
            )
        input_data = normalize(input_data, schema, scaler="predict")
        input_data = encode(input_data, schema, encoder="predict")

    return input_data
