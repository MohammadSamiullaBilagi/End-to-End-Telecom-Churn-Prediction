



from churn_prediction.exception.exception import TelecomChurnException
from churn_prediction.logging.logger import logging
from churn_prediction.entity.artifact_entity import DataIngestionArtifact

## configuration of data ingestion config
from churn_prediction.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig

import os
import sys
import pandas as pd
import numpy as np
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise TelecomChurnException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Read data from MongoDB collection and return a DataFrame.
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            if not MONGO_DB_URL:
                raise ValueError("MONGO_DB_URL is not set in environment variables")

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
            df.replace({"na": np.nan}, inplace=True)

            logging.info(f"Read {len(df)} records from {database_name}.{collection_name}")
            return df
        except Exception as e:
            raise TelecomChurnException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
        # config provides both directory and file path
            feature_store_dir = self.data_ingestion_config.feature_store_dir
            feature_store_file = self.data_ingestion_config.feature_store_file

            # ensure the feature_store_dir exists (data_ingestion/feature_store)
            if feature_store_dir:
                os.makedirs(feature_store_dir, exist_ok=True)

            # ensure the parent dir for the file exists (defensive)
            parent_dir = os.path.dirname(feature_store_file)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            # write the file
            dataframe.to_csv(feature_store_file, index=False, header=True)
            logging.info(f"Feature store saved at: {feature_store_file}")
            return feature_store_file
        except Exception as e:
            raise TelecomChurnException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Split dataframe into train and test, ensure ingested directory exists,
        and write train/test files to configured paths.
        """
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performing Train/Test split of dataframe")
            logging.info("Exited split_data_as_train_test method of DataIngestion class")

            # Ensure ingested_dir (directory) exists
            ingested_dir = self.data_ingestion_config.ingested_dir
            if ingested_dir:
                os.makedirs(ingested_dir, exist_ok=True)

            # Ensure parent directories for train/test files exist (defensive)
            train_parent = os.path.dirname(self.data_ingestion_config.train_file_path)
            test_parent = os.path.dirname(self.data_ingestion_config.test_file_path)
            if train_parent:
                os.makedirs(train_parent, exist_ok=True)
            if test_parent:
                os.makedirs(test_parent, exist_ok=True)

            # Write train and test to the correct file paths
            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

            logging.info(f"Exported train file to: {self.data_ingestion_config.train_file_path}")
            logging.info(f"Exported test file to: {self.data_ingestion_config.test_file_path}")
        except Exception as e:
            raise TelecomChurnException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrate the data ingestion: read from MongoDB, save feature store,
        split and save train/test, then return artifact.
        """
        try:
            # read data from mongodb
            dataframe = self.export_collection_as_dataframe()

            # export to feature store (file)
            feature_store_file = self.export_data_into_feature_store(dataframe)

            # split and write train/test files inside ingested dir
            self.split_data_as_train_test(dataframe)

            # prepare artifact
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path,
            )
            logging.info(f"DataIngestionArtifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise TelecomChurnException(e, sys)

