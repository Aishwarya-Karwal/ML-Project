# this file contains all the common functions that are used in the project

import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file:
            dill.dump(obj, file)
            logging.info(f"object saved as filepath: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)
