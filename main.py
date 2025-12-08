from churn_prediction.components.data_ingestion import DataIngestion
from churn_prediction.exception.exception import TelecomChurnException
from churn_prediction.logging.logger import logging
from churn_prediction.entity.config_entity import DataIngestionConfig,DataValidationConfig
from churn_prediction.entity.config_entity import TrainingPipelineConfig
from churn_prediction.components.data_validation import DataValidation
import sys
import os

if __name__=="__main__":
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate DataIngestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        logging.info("Data Inititation Completed")
        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate Data validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Data validation completed")
        print(data_validation_artifact)
    except Exception as e:
        raise TelecomChurnException(e,sys)





