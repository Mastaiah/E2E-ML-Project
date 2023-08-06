#This module is for data ingestion 

import os
import pandas as pd

from dataclasses import dataclass
from utils.exception import CustomException
from utils.logger import CustomLogger

from components.data_transform import DataTransformation


#Decorator class
@dataclass
class DataIngestionConfig:
    raw_data_path:str   = os.path.join('artifacts','raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.data_config = DataIngestionConfig()

    #function to initiate a data ingestion
    def initiate(self):

        try:    
            CustomLogger().info('Inititated the data ingestion')

            student_df = pd.read_csv('data_repository\cleaned\StudentPerformance.csv')

            if student_df.empty:
                raise CustomException("Failed to read the data")          
            else:
                os.makedirs(os.path.dirname(self.data_config.raw_data_path), exist_ok = True)

                # saving the DataFrame as a CSV file
                student_df.to_csv(self.data_config.raw_data_path, index = False, header = True)

                CustomLogger().info('Data ingestion completed')

                return self.data_config.raw_data_path

        except CustomException as e:
                CustomLogger.error(e)

"""
#Below code is written for testing purpose only

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    raw_data = data_ingestion.initiate()

    data_transform = DataTransformation()
    data_transform.initiate(raw_data)
"""