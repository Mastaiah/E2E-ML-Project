from src.data_ingestion import DataIngestion
from src.data_selector import ModelSelector
from src.hyperparameter_tuner import HyperparameterTuner
from src.model_evaluator import ModelEvaluator , OverFittingDetector

class ModelTrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTranformation()
        self.model_selector = ModelSelector()
        self.model_manager = ModelManager()
        self.model_selected_flag = False
        self.models_list = None
        self.model = None

        
    def train(self):
        #Load the data
        train_data, test_data = self.data_ingestion.initiate()

        #Data transformation
        train_features , train_target , test_featues , test_target = self.data_transformation.inititate(train_data, test_data)

        if self.model_selected_flag == False:
            self.models_list = model_manager.get_models()
            self.model = ModelSelector(self.model_list)
            self.model_selected_flag = True

        