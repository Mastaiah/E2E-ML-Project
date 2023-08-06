import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from utils.exception import CustomException
from utils.logger import CustomLogger
from components.data_splitter import DataSplitter
from utils.model_persistence  import Saver

@dataclass
class DataTransformationConfig:
    preprocessor_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.transformed = False
        self.raw_df = None
        self.data_splitter = None
    

    def data_transformer(self,data):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = data.select_dtypes(include=[float, int]).columns.tolist()
            categorical_columns = data.select_dtypes(include=[object]).columns.tolist()

            num_pipeline = Pipeline(
                steps = [
                        ("imputer",SimpleImputer(strategy="median")),
                        ("scaler",StandardScaler())
                        ]
            )

            cat_pipeline = Pipeline(
                steps = [
                        ("imputer",SimpleImputer(strategy="most_frequent")),
                        ("one_hot_encoder",OneHotEncoder()),
                        ("scaler",StandardScaler(with_mean=False))
                        ]
            )

            #logging.info(f"Categorical columns: {categorical_columns}")
            #logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                ("numerical_pipeline", num_pipeline, numerical_columns),
                ("categorical_pipeline", cat_pipeline, categorical_columns)

                ]

            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e)
        
    def initiate(self,data_path ,val_size = 0.20, test_size = 0.10 ,random_state= None):
        
        try:
            self.raw_df = pd.read_csv(data_path)
            self.data_splitter = DataSplitter()

            CustomLogger().info('Reading data completed....data preprocessing is about to start!!!')

            target_name = "math_score"

            features = self.raw_df.drop(columns=[target_name],axis=1)
            target  = self.raw_df[target_name]

            train_feat , val_feat , test_feat , train_target , val_target, test_target = self.data_splitter.split_data(features, target,val_size , test_size, random_state)

            preprocessing = self.data_transformer(train_feat)
            #Fit and transform on  training data 
            train_feature_processed = preprocessing.fit_transform(train_feat)

            #Transform on validation and test data.
            val_feature_processed = preprocessing.transform(val_feat)
            test_feature_processed = preprocessing.transform(test_feat)

            CustomLogger().info(f"Saving preprocessing details.")

            processed_data = Saver( preprocessing , self.data_transformation_config.preprocessor_file_path)

            CustomLogger().info("Saved preprocessing details")


            return ( train_feature_processed , 
                     val_feature_processed,
                     test_feature_processed,
                     train_target,
                     val_target,
                     test_target,
                 )
        
        except Exception as e:
            raise CustomException(e)
      

    def is_transformed(self):
        return self.transformed