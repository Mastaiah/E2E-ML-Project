import os
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from utils.exception import CustomException
from utils.logger import CustomLogger

#Decorator class
@dataclass
class DataIngestionConfig:
    raw_data_path:str   = os.path.join('artifacts','raw_data.csv')
    train_data_path:str = os.path.join('artifacts','train_data.csv')
    test_data_path:str  = os.path.join('artifacts','test_data.csv')

class DataIngestion:
    def __init__(self):
        self.data_config = DataIngestionConfig()

    #function to initiate a data ingestion
    def initiate_data_ingestion(self):

        try:    
            CustomLogger().info('Inititated the data ingestion')

            student_df = pd.read_csv('data_repository\cleaned\StudentPerformance.csv')

            if student_df.empty:
                raise CustomException("Failed to read the data")          
            else:
                os.makedirs(os.path.dirname(self.data_config.raw_data_path), exist_ok = True)
            
                CustomLogger().info('Train-Test split initated')
                train_data , test_data = train_test_split(student_df, test_size = 0.2 , random_state = 42)

                # saving the DataFrame as a CSV file
                train_data.to_csv(self.data_config.train_data_path, index = False, header = True) 
                test_data.to_csv(self.data_config.test_data_path,  index = False, header = True)
                student_df.to_csv(self.data_config.raw_data_path, index = False, header = True)

                CustomLogger().info('Train-Test split completed')

                return{
                    self.data_config.train_data_path,
                    self.data_config.test_data_path
                 }

        except CustomException as e:
                CustomLogger.error(e)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
     

                





