import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
# it is used to create class variables
from dataclasses import dataclass

# when u just have to define the inputs to the class then 
# use dataclass otherwiae use class with init method

@dataclass 
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts",'train.csv')
    test_data_path: str = os.path.join("artifacts",'test.csv')
    raw_data_path: str = os.path.join("artifacts",'data.csv')

class DataIngestion:
    def __init__(self):
        # this ingestion config variable will have all the above created paths objects
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        # if data is stored in database then we rite about it here
        logging.info("Entered the data ingestion method")
        try :
            df = pd.read_csv('notebook\data\StudentsPerformance.csv')
            logging.info('Read the data as df')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            logging.info('Saved the raw data')

            logging.info('initiated train test split')
            train, test = train_test_split(df, random_state=42, test_size=0.2)
            train.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("Saved the data as train and test set")
            logging.info('ingestion completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    

    