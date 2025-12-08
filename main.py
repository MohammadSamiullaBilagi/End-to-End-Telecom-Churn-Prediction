from churn_prediction.components.data_ingestion import DataIngestion
from churn_prediction.exception.exception import TelecomChurnException
from churn_prediction.logging.logger import logging
from churn_prediction.entity.config_entity import DataIngestionConfig
from churn_prediction.entity.config_entity import TrainingPipelineConfig
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
    except Exception as e:
        raise TelecomChurnException(e,sys)





