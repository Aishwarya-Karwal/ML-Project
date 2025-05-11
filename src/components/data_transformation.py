import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
#from src.components.data_ingestion import DataIngestion

import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        ''' it will create all pickle file that r responsible to perfoem different transformations on data '''
        try:
            numerical_columns = ['writing score', 'reading score']
            categorical_columns = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course',
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )
            logging.info("Numerical columns pipelined")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical columns pipelined")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            logging.info("Categorical and numerical columns pipelines combined in a column transformer")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

        
    def initiate_data_transormation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")

            logging.info("Obtaining preprocessing object")

            preprocessor = self.get_data_transformer_object()

            target = "math score"
            numerical_columns = ['writing score', 'reading score']
            categorical_columns = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course',
            ]

            input_train_df = train_df.drop(target, axis = 1)
            target_train_df = train_df[[target]]

            input_test_df = test_df.drop(target, axis = 1)
            target_test_df = test_df[[target]]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            #print("input_train_df shape:", input_train_df.shape)  # Check before transformation

            input_feature_train_arr = preprocessor.fit_transform(input_train_df)
            input_feature_test_arr = preprocessor.transform(input_test_df)

            # print(input_feature_train_arr.toarray())

            #print(input_feature_train_arr.shape, target_train_df.shape)
            # np.c_ is used to concatenate the two arrays column wise - it is called column stacking
            #print(target_train_df.to_numpy().shape)

            # print("input_feature_train_arr shape:", input_feature_train_arr.shape)
            # print("target_train_df.to_numpy() shape:", target_train_df.to_numpy().reshape(-1,1).shape)
            # print("input_train_df shape:", input_train_df.shape)  # Check before transformation

            # print("input_feature_train_arr shape:", input_feature_train_arr.shape)
            # print("target_train_df shape:", target_train_df.shape)
            # print("target_train_df.to_numpy() shape:", target_train_df.to_numpy().reshape(-1, 1).shape)

            # Ensure the number of rows match
            if input_feature_train_arr.shape[0] != target_train_df.shape[0]:
                raise ValueError(
                    f"Mismatch in rows: input_feature_train_arr has {input_feature_train_arr.shape[0]} rows, "
                    f"but target_train_df has {target_train_df.shape[0]} rows."
                )

            train_arr = np.c_[
                input_feature_train_arr, target_train_df.to_numpy().reshape(-1, 1)
            ]

            test_arr = np.c_[
                input_feature_test_arr, target_test_df.to_numpy().reshape(-1, 1)
            ]
            # train_arr = np.column_stack(
            #     (input_feature_train_arr, target_train_df.to_numpy().reshape(-1, 1))
            # )
            
            # assert input_feature_train_arr.shape[0] == target_train_df.shape[0], "Row count mismatch"


            # train_arr = np.c_[
            #     input_feature_train_arr, target_train_df.to_numpy().reshape(-1,1)
            # ]

            # test_arr = np.c_[input_feature_test_arr, target_test_df.to_numpy().reshape(-1,1)]

            # test_arr = np.column_stack((input_feature_test_arr, target_test_df.to_numpy().reshape(-1, 1)))

            logging.info("Data Transformed !")
            

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )
            logging.info("saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
        

# if __name__ == "__main__":
    # get_data = DataIngestion()
    # train_data_path, test_data_path = get_data.initiate_data_ingestion()

    # obj = DataTransformation()
    # train_arr, test_arr, _ = obj.initiate_data_transormation(
    #     train_path=train_data_path,
    #     test_path=test_data_path
    # )

    # print(train_arr[:10,:])
    # pass

