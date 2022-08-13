"""
This module defines the following routines used by the 'split' step of the regression pipeline:

- ``process_splits``: Defines customizable logic for processing & cleaning the training, validation,
  and test datasets produced by the data splitting procedure.
"""

from pandas import DataFrame


def process_splits(
    train_df: DataFrame, validation_df: DataFrame, test_df: DataFrame
) -> (DataFrame, DataFrame, DataFrame):
    """
    Perform additional processing on the split datasets.

    :param train_df: The training dataset produced by the data splitting procedure.
    :param validation_df: The validation dataset produced by the data splitting procedure.
    :param test_df: The test dataset produced by the data splitting procedure.
    :return: A tuple containing, in order: the processed training dataset, the processed
             validation dataset, and the processed test dataset.
    """

    def process(df: DataFrame):
        # Drop invalid data points
        cleaned = df.dropna()
        # Filter out invalid fare amounts and trip distance
        cleaned = cleaned[
            (cleaned["FARE_AMOUNT"] > 0)
            & (cleaned["TRIP_DISTANCE"] < 400)
            & (cleaned["TRIP_DISTANCE"] > 0)
            & (cleaned["FARE_AMOUNT"] < 1000)
        ]

        return cleaned

    return process(train_df), process(validation_df), process(test_df)
