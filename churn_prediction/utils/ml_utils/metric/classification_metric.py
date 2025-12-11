from churn_prediction.entity.artifact_entity import ClassificationMetricArtifact
from churn_prediction.exception.exception import TelecomChurnException
import os, sys
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score



def get_classification_score(y_true,y_pred,y_proba=None)->ClassificationMetricArtifact:
    try:
        model_f1_score=f1_score(y_true,y_pred)
        model_recall_score=recall_score(y_true,y_pred)
        model_precision_score=precision_score(y_true,y_pred)
        model_auc_score = None

        if y_proba is not None:
            model_auc_score = roc_auc_score(y_true, y_proba)

        classification_metric=ClassificationMetricArtifact(
            f1_score=model_f1_score,
            recall_score=model_recall_score,
            precision_score=model_precision_score,
            auc_score=model_auc_score
        )
        return classification_metric
    except Exception as e:
        raise TelecomChurnException(e,sys)

