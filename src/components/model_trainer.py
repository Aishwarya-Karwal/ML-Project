import os
import sys

from dataclasses import dataclass
import yaml

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig : 
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting into training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
                "AdaBoostRegressor": AdaBoostRegressor(),
            }

            with open("src\config\model_params.yaml", "r") as file:
                param_grid = yaml.safe_load(file)
                logging.info("Loading model parameters from config file")
                
            logging.info(f"Model parameters: {param_grid}")

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, param_grid)

            logging.info(f"Model report: {model_report}")
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.5:
                raise CustomException("No best model found", sys)

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # Save the best model as pkl file
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square, best_model_name

        except Exception as e:
            raise CustomException(e, sys)