"""
This module defines the following routines used by the 'transform' step of the regression pipeline:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""

import pandas as pd
from pandas import Timestamp

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer

#from databricks.automl_runtime.sklearn import DatetimeImputer
#from databricks.automl_runtime.sklearn import TimestampTransformer
#from databricks.automl_runtime.sklearn.column_selector import ColumnSelector


def transformer_fn():

  num_imputers = []
  num_imputers.append(("impute_mean", SimpleImputer(), ["DROPOFF_LATITUDE", "DROPOFF_LONGITUDE", "PICKUP_LATITUDE", "PICKUP_LONGITUDE",  "TRIP_DISTANCE", "PICKUP_DOW", "PICKUP_HOUR", "TRIP_DURATION"]))

  numerical_pipeline = Pipeline(steps=[
      ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
      ("imputers", ColumnTransformer(num_imputers)),
      ("standardizer", StandardScaler()),
  ])

  numerical_transformers = [("numerical", numerical_pipeline, ["TRIP_DISTANCE",  "PICKUP_LATITUDE", "PICKUP_LONGITUDE", "DROPOFF_LONGITUDE", "DROPOFF_LATITUDE",  "PICKUP_DOW", "PICKUP_HOUR", "TRIP_DURATION"])]
  
  one_hot_imputers = []

  one_hot_pipeline = Pipeline(steps=[
      ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
      ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
  ])

  categorical_one_hot_transformers = [("onehot", one_hot_pipeline, ["PAYMENT_TYPE"])]
  
  transformers = numerical_transformers + categorical_one_hot_transformers

  preprocessor = ColumnTransformer(transformers, remainder="passthrough", ) #sparse_threshold=1
  
  pipeline = Pipeline([
    ("preprocessor", preprocessor),
  ])
  return pipeline



