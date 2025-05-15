# this file contains all the common functions that are used in the project

import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file:
            dill.dump(obj, file)
            logging.info(f"object saved as filepath: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(xtrain, ytrain, xtest, ytest, models, params = {}):
    try:
        report = {}
        
        for i in range(len(list(models))):

            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            logging.info(f"Training and evaluating {model}")


            if len(params) != 0 and model_name in params.keys():
                logging.info(f"Hyperparameter tuning for {model_name}")
                best_model, best_params, best_score = hyperparameter_tuning(model, xtrain, ytrain, params[model_name])
                report[model_name] = best_score
                logging.info(f"Tuned Score added")

            else:
                model.fit(xtrain, ytrain)

                y_train_pred = model.predict(xtrain)
                y_test_pred = model.predict(xtest)

                # train_model_score = r2_score(ytrain, y_train_pred)
                test_model_score = r2_score(ytest, y_test_pred)

                report[model_name] = test_model_score
            
            logging.info(f"Model: {model_name} trained")

        return report
    
    except Exception as e:
        raise CustomException(e,sys)


def hyperparameter_tuning(model, xtrain, ytrain, param_grid):
    try:
        logging.info(f"Hyperparameter tuning for model: {model}")
        grid_search = GridSearchCV(estimator = model, param_grid=param_grid, cv = 3, n_jobs = -1, verbose = 2)
        grid_search.fit(xtrain, ytrain)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        logging.info(f"Best Model: {best_model} , Best Params: {best_params}, Best Score: {best_score}")

        return best_model, best_params, best_score
    

    except Exception as e:
        raise CustomException(e, sys)
    

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e,sys)