# provides functions and variales that manipulate python runtime env
# so if any exception is getting controlled , it will have that info
#import logging 
import sys
from src.logger import logging

def error_message_detail(error, error_detail : sys):
    # error detail will come from sys module
    _,_,exc_tb = sys.exc_info()
    # exc_tb is the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(file_name, exc_tb.tb_lineno, str(error))
    return error_message

# Inherits from Python’s built-in Exception class, meaning it behaves like a normal exception but with custom behavior.
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    

# if __name__ == "__main__":
#     try:
#         a = 1/0
#     except Exception as e:
#         logging.info("Logging is working")
#         print(CustomException(e,sys))