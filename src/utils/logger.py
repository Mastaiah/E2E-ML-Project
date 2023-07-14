import logging 
import os
from datetime import datetime

#create a log path
logs_path = os.path.join(os.getcwd(),"logs",f"{datetime.now().strftime('%d_%m_%Y')}")
os.makedirs(logs_path,exist_ok=True)

#Create a log file name
LOG_FILE_NAME = f"{datetime.now().strftime('%d_%m_%Y-%H_%M_%S')}.log"

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE_NAME)

#Create a custom logger
logger = logging.getLogger(__name__)

#Set a severity level of a logger
logger.setLevel(logging.INFO)

#Create a formater
fmt = logging.Formatter("%(asctime)s - [%(filename)s: %(lineno)s] - %(levelname)s : %(message)s", datefmt= ('%m-%d-%y::%H:%M:%S'))

#Create an handler
handler = logging.FileHandler(LOG_FILE_PATH)

#Set formatter to handler
handler.setFormatter(fmt)

#Add handler to the logger
logger.addHandler(handler)

#Set it as False to enable the logger and True to disable it
logger.disabled = False
logger.info("Logging Enabled . Enjoy !!!")