import yaml
import dill
from churn_prediction.exception.exception import TelecomChurnException
from churn_prediction.logging.logger import logging
import pickle
import numpy as np
import pandas as pd
import os,sys
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import r2_score

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix,f1_score
from sklearn.model_selection import cross_val_score

def read_yaml_file(file_path:str)->dict:
    try:
        with open(file_path,"rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise TelecomChurnException(e,sys)

def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise TelecomChurnException(e,sys)

## for data transformation
def save_numpy_array(file_path:str,array:np.array):
    '''
    save numpy array data to file
    file_path:str location of file to save
    array:np.array data to save
    '''
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise TelecomChurnException(e,sys) from e
    

def save_object(file_path:str,obj:object)->None:
    try:
        logging.info("Entered save object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise TelecomChurnException(e,sys) from e

def load_object(file_path:str,)->object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f" The file: {file_path} doesn't exist")
        with open(file_path,"rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise TelecomChurnException(e,sys) from e

def load_numpy_array_data(file_path: str) -> np.ndarray:
    '''
    Load numpy array data from .npy file
    '''
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = np.load(file_path)
        return data
        
    except Exception as e:
        raise TelecomChurnException(e, sys) from e

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate models with GridSearchCV. Returns metrics + PRE-TRAINED best models.
    """
    try:
        report = {}
        best_models_dict = {}  #  Using your preferred name
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # 1. GridSearchCV
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            gs = GridSearchCV(
                model, param[name], 
                cv=cv, 
                scoring='roc_auc_ovr',
                n_jobs=-1, 
                verbose=0
            )
            gs.fit(X_train, y_train)
            
            #  gs.best_estimator_ is ALREADY TRAINED
            best_model = gs.best_estimator_
            
            # 2. Predictions & metrics
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            y_train_proba = best_model.predict_proba(X_train)[:, 1]
            y_test_proba = best_model.predict_proba(X_test)[:, 1]
            
            train_auc = roc_auc_score(y_train, y_train_proba)
            test_auc = roc_auc_score(y_test, y_test_proba)
            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            
            # 3. Store results
            report[name] = {
                'best_params': gs.best_params_,
                'train_auc': round(train_auc, 4),
                'test_auc': round(test_auc, 4),
                'train_f1': round(train_f1, 4),
                'test_f1': round(test_f1, 4),
                'cv_best_score': round(gs.best_score_, 4)
            }
            
            # Store in best_models_dict
            best_models_dict[name] = gs.best_estimator_
            
        return report, best_models_dict  # Returns your exact variable name
        
    except Exception as e:
        raise TelecomChurnException(e, sys)



