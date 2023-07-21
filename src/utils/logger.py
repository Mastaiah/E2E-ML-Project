import logging 
import os
import inspect 
from datetime import datetime


def CustomLogger(loglevel = 'DEBUG'):
    #Set class / method name from where its called
    logger_name = inspect.stack()[1][3]

    #Create a custom logger
    logger = logging.getLogger(logger_name)

    """if logger 'name' already exists, return it to avoid logging duplicate
     messages by attaching multiple handlers of the same type.
    logger.handlers attribute allows us to access and manipulate the list of handlers attached to a specific logger instance."""
    if logger.handlers:
        return logger
    
    # if logger 'name' does not already exist, create it and attach handlers
    else:
        #create a log path
        log_dir = os.path.join(os.getcwd(),"logs",f"{datetime.now().strftime('%d_%m_%Y')}")
        os.makedirs(log_dir, exist_ok = True)

        #Create a log file name
        LOG_FILE_NAME = f"{datetime.now().strftime('%d_%m_%Y')}.log"
        LOG_FILE_PATH = os.path.join(log_dir,LOG_FILE_NAME)
        
        #getattr is important else will not write to log
        loglevel = getattr(logging, loglevel.upper())

        #Set a severity level of a logger
        #loglevel = logging.INFO
        logger.setLevel(loglevel)
        
        #Create a formater
        fmt = logging.Formatter("%(asctime)s - [%(filename)s: %(lineno)s]  - %(levelname)s : %(message)s", datefmt= ('%m-%d-%y::%H:%M:%S'))

        #Create an handler
        handler = logging.FileHandler(LOG_FILE_PATH)

        #Set formatter to handler
        handler.setFormatter(fmt)

        #Add handler to the logger
        logger.addHandler(handler)

        #Set it as False to enable the logger and True to disable it
        #logger.disabled = False

        return logger

