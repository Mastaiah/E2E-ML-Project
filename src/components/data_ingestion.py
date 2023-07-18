import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from utils.exception import CustomException
from utils.logger import CustomLogger



@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts''raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        CustomLogger().info ("Data ingestion started")
        
        try:
            #df = pd.read_csv('A:\ML-Projects\mlproject\data_repository\cleaned\StudentPerformance.csv')
            i_num = int(input("Enter a number: "))
            if i_num == 0:
                raise CustomException('i_num is zeor')
            #if df.empty:
             #raise CustomException('failed',sys)
        except CustomException as e:
            CustomLogger().info(e)
            print(e)
            _,_,exc_tb = sys.exc_info()
            filename = exc_tb.tb_frame.f_code.co_filename
            print(filename)

if __name__ == "__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()