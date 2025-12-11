from churn_prediction.exception.exception import TelecomChurnException
from churn_prediction.logging.logger import logging
from churn_prediction.entity.config_entity import ModelTrainerConfig
from churn_prediction.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact

from churn_prediction.utils.main_utils.utils import save_object,load_object
from churn_prediction.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from churn_prediction.utils.ml_utils.metric.classification_metric import get_classification_score
from churn_prediction.utils.ml_utils.model.estimator import TelecomChurnModel
import os,sys
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

# got error here because of the package installed was of higher version
# hence fixed by using pip install "mlflow>=2.10,<3"
# also the token is in the dagshub token folder in e drive
import dagshub
dagshub.init(repo_owner='MohammadSamiullaBilagi', repo_name='End-to-End-Telecom-Churn-Prediction', mlflow=True)

mlflow.set_tracking_uri(
    "https://dagshub.com/MohammadSamiullaBilagi/End-to-End-Telecom-Churn-Prediction.mlflow"
)
class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise TelecomChurnException(e,sys)
    
    def track_mlflow(self,best_model,classificationmetric):
        with mlflow.start_run():
            f1_score=classificationmetric.f1_score
            precision_score=classificationmetric.precision_score
            recall_score=classificationmetric.recall_score
            train_auc=classificationmetric.auc_score
            test_auc=classificationmetric.auc_score

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.log_metric("train_auc", train_auc)
            mlflow.log_metric("test_auc", test_auc)
            mlflow.sklearn.log_model(best_model,"best_model")


    
    def train_model(self,x_train,y_train,x_test,y_test):
        models={
            # verbose helps to see what training is happening
            "Random Forest":RandomForestClassifier(verbose=1),
            "Decision Tree":DecisionTreeClassifier(),
            "Gradient Boosting":GradientBoostingClassifier(verbose=1),
            "Logistic Regression":LogisticRegression(verbose=1),
            "AdaBoost":AdaBoostClassifier(),
        }

    #     params={
    #         "Decision Tree": {
    #             'criterion':['gini', 'entropy', 'log_loss'],
    #             # 'splitter':['best','random'],
    #             # 'max_features':['sqrt','log2'],
    #         },
    #         "Random Forest":{
    #             # 'criterion':['gini', 'entropy', 'log_loss'],
                
    #             # 'max_features':['sqrt','log2',None],
    #             'n_estimators': [8,16,32,128,256]
    #         },
    #         "Gradient Boosting":{
    #             # 'loss':['log_loss', 'exponential'],
    #             'learning_rate':[.1,.01,.05,.001],
    #             'subsample':[0.6,0.7,0.75,0.85,0.9],
    #             # 'criterion':['squared_error', 'friedman_mse'],
    #             # 'max_features':['auto','sqrt','log2'],
    #             'n_estimators': [8,16,32,64,128,256]
    #         },
    #         "Logistic Regression":{},
    #         "AdaBoost":{
    #             'learning_rate':[.1,.01,.001],
    #             'n_estimators': [8,16,32,64,128,256]
    #         }
    #    }

        params = {
            "Decision Tree": {
                "max_depth": [3, 5, 7, 10],
                "min_samples_leaf": [2, 5, 10],
                "class_weight": [None, "balanced"]
            },
            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_leaf": [2, 5],
                "class_weight": [None, "balanced"]
            },
            "Gradient Boosting": {
                "learning_rate": [0.05, 0.1],
                "n_estimators": [100, 200, 300],
                "max_depth": [2, 3, 4],
                "subsample": [0.7, 0.9, 1.0]
            },
            "Logistic Regression": {
                "C": [0.1, 1, 10, 100],
                "penalty": ["l2"],
                "class_weight": [None, "balanced"]
            },
            "AdaBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.05, 0.1, 0.5]
            }
        }

        model_report, best_models_dict=evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,models=models,param=params)

        #To get best model score from dict
        #  Get BEST model (by test_auc)
        best_result = max(model_report.items(), key=lambda x: x[1]['test_auc'])
        best_model_name, best_model_info = best_result
        best_model = best_models_dict[best_model_name]  # Tune it

        print(f" Best Model: {best_model_name}")
        print(f"Test AUC: {best_model_info['test_auc']}")

        # best_model=models[best_model_name]
        # y_train_pred=best_model.predict(x_train)
        # y_test_pred=best_model.predict(x_test)

        best_model.fit(x_train, y_train)

        y_train_pred = best_model.predict(x_train)
        y_test_pred = best_model.predict(x_test)

        y_train_proba = best_model.predict_proba(x_train)[:, 1]
        y_test_proba = best_model.predict_proba(x_test)[:, 1]


        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred,y_proba=y_train_proba)
        # Track experiments mlflow
        self.track_mlflow(best_model,classification_train_metric)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred,y_proba=y_test_proba)
        self.track_mlflow(best_model,classification_test_metric)

        

        print(f"Train F1: {classification_train_metric.f1_score:.4f}")
        print(f"Test F1: {classification_test_metric.f1_score:.4f}")
        #track ml flow

        preprocessor=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Telecom_churn_model=TelecomChurnModel(preprocessor=preprocessor,model=best_model)

        save_object(self.model_trainer_config.trained_model_file_path,obj=Telecom_churn_model)

        # model pusher locally
        save_object("final_model/model.pkl",best_model)

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric,)
        
        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path

            train_arr=load_numpy_array_data(train_file_path)
            test_arr=load_numpy_array_data(test_file_path)

            # x_train,x_test,y_train,y_test={
            #     train_arr[:,:-1],
            #     test_arr[:,:-1],
            #     train_arr[:,-1],
            #     test_arr[:,-1],
            # }
            x_train = train_arr[:, :-1]
            x_test  = test_arr[:, :-1]
            y_train = train_arr[:, -1]
            y_test  = test_arr[:, -1]

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact
            
        except Exception as e:
            raise TelecomChurnException(e,sys)




