
import os
from utils.model_persistence import Loader

#Data collector 
class DataToPredict:
    def __init__(self , gender:str , race_ethnicity:str , parental_level_of_education:str, 
                 test_preparation_course:str, reading_score:float , writing_score:float):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score


    def convert_to_data_frame (self):
        data_dict = {'gender' : [self.gender],
                           'race_ethnicity' : [self.race_ethnicity],
                           'parental_level_of_education' : [self.parental_level_of_education],
                           'test_preparation_cours' : [self.test_preparation_course],
                           'reading_score' : [self.reading_score],
                           'writing_score' : [self.writing_score],
                           }
        return pd.DataFrame(data_dict)


class Predicter:
    def __init__(self):
        self.preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
        self.model_path = os.path.join('artifacts','model.pkl')
    
    def predict(self,features):
        #Load the  preprocessor, process the new features.
        preprocessor_object = Loader()
        preprocessor = preprocessor_object.load(self.preprocessor_path)
        processed_data = preprocessor.transform(features)

        #Load the model , predict based on features provided. 
        model_object = Loader()
        model = model_object.load(self.model_path)
        predicted_value = model.predict(processed_data)

        return predicted_value
        


        
