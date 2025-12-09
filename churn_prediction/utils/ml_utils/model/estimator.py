## to return TelecomChurn model last step

from churn_prediction.exception.exception import TelecomChurnException
from churn_prediction.logging.logger import logging
from churn_prediction.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
import os,sys


class TelecomChurnModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor=preprocessor
            self.model=model
        except Exception as e:
            raise TelecomChurnException(e,sys)

    def predict(self,x):
       try:
           x_transform=self.preprocessor.transform(x)
           y_hat=self.model.predict(x_transform)
           return y_hat
       except Exception as e:
           raise TelecomChurnException(e,sys)

