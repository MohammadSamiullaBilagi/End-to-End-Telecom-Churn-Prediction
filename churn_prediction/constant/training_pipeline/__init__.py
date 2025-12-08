import os
import sys
import numpy as np
import pandas as pd


'''
Defining some common constant variable for training pipeline
'''
TARGET_COLUMN="Churn"
PIPELINE_NAME:str="TelecomChurn"
ARTIFACT_DIR:str="Artifacts"
FILENAME:str="Churn_prediction_cleaned.csv"

TRAIN_FILE_NAME="train.csv"
TEST_FILE_NAME="test.csv"

SCHEMA_FILE_PATH=os.path.join("data_schema","schema.yaml")

''' 
Data Ingestion related constant start with DATA_INGESTION VAR NAME
'''
DATA_INGESTION_DATABASE_NAME:str="Samiulla_Churn"
DATA_INGESTION_COLLECTION_NAME:str="TelecomChurn"
DATA_INGESTION_DIR_NAME:str="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str="feature_store"
DATA_INGESTION_INGESTED_DIR:str="ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float=0.2

''' 
Data Validation related constant start with DATA_VALIDATION VAR NAME
'''
DATA_VALIDATION_DIR_NAME:str="data_validation"
DATA_VALIDATION_VALID_DIR:str="validated"
DATA_VALIDATION_INVALID_DIR:str="invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str="drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str="report.yaml"



