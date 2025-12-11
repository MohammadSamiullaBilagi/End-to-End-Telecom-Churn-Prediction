## to return TelecomChurn model last step

from churn_prediction.exception.exception import TelecomChurnException
from churn_prediction.logging.logger import logging
from churn_prediction.constant.training_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
import os,sys
import numpy as np


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

# class TelecomChurnModel:
#     def __init__(self, preprocessor, model, threshold: float = 0.5):
#         try:
#             self.preprocessor = preprocessor
#             self.model = model
#             self.threshold = threshold  # store decision threshold
#         except Exception as e:
#             raise TelecomChurnException(e, sys)

#     def predict_proba(self, x):
#         """
#         Return churn probability for the positive class.
#         """
#         try:
#             x_transform = self.preprocessor.transform(x)
#             proba = self.model.predict_proba(x_transform)[:, 1]
#             return proba
#         except Exception as e:
#             raise TelecomChurnException(e, sys)

#     def predict(self, x):
#         """
#         Use custom threshold on probability to produce class labels.
#         """
#         try:
#             proba = self.predict_proba(x)
#             y_hat = (proba >= self.threshold).astype(np.int64)
#             return y_hat
#         except Exception as e:
#             raise TelecomChurnException(e, sys)

