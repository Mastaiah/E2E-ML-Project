import sys
import os
import inspect
from utils.logger import CustomLogger

"""def error_message_detail(message , filename , lineno):
    error_message = f"The exception has occured in python file {filename} line number {lineno} message {str(message)}"   
    return error_message"""


#Custom exception class
class CustomException(Exception):
    def __init__(self,message):
        super().__init__(message)
        self.message = message

    def __str__(self):
       return self.message
