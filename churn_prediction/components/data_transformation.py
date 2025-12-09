from churn_prediction.exception.exception import TelecomChurnException
from churn_prediction.logging.logger import logging

import sys, os
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from churn_prediction.constant.training_pipeline import TARGET_COLUMN
from churn_prediction.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from churn_prediction.entity.artifact_entity import (DataValidationArtifact, DataTransformationArtifact)
from churn_prediction.entity.config_entity import DataTransformationConfig
from churn_prediction.utils.main_utils.utils import save_numpy_array, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise TelecomChurnException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise TelecomChurnException(e, sys)

    @staticmethod
    def get_transformer_object(numerical_columns: list, categorical_columns: list) -> ColumnTransformer:
        """
        Build a ColumnTransformer that:
          - For numerical columns: KNNImputer (params from config) -> StandardScaler
          - For categorical columns: SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore')

        Returns ColumnTransformer ready to fit/transform.
        """
        try:
            # numeric pipeline
            numeric_pipeline = Pipeline(steps=[
                ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),  # KNNImputer for numeric
                ("scaler", StandardScaler())
            ])

            # categorical pipeline
            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", numeric_pipeline, numerical_columns),
                ("cat", categorical_pipeline, categorical_columns)
            ], remainder="drop")  # drop any other columns

            logging.info(f"Built ColumnTransformer with {len(numerical_columns)} numeric and {len(categorical_columns)} categorical cols")
            return preprocessor

        except Exception as e:
            raise TelecomChurnException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered data transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Separate input and target
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Detect numeric and categorical columns from training data
            numerical_columns = input_feature_train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
            categorical_columns = input_feature_train_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Build transformer using column lists
            preprocessor = self.get_transformer_object(numerical_columns, categorical_columns)

            # Fit on training set only
            preprocessor_obj = preprocessor.fit(input_feature_train_df)

            # Transform both train and test (test uses fitted transformer)
            transformed_input_train_feature = preprocessor_obj.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_obj.transform(input_feature_test_df)

            # Combine features + target
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # Save arrays and transformer
            save_numpy_array(self.data_transformation_config.data_transformation_train_file_path, array=train_arr)
            save_numpy_array(self.data_transformation_config.data_transformation_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.data_transformation_object_file_path, preprocessor_obj)

            # Also save a copy for model serving if desired
            save_object("final_models/preprocessor.pkl", preprocessor_obj)

            # Prepare artifact (note: fixed the swapped path bug)
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.data_transformation_object_file_path,
                transformed_train_file_path=self.data_transformation_config.data_transformation_train_file_path,
                transformed_test_file_path=self.data_transformation_config.data_transformation_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise TelecomChurnException(e, sys)
