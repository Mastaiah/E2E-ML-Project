from components.data_ingestion import DataIngestion
from components.data_transform import DataTransformation
from components.model_selector import ModelSelector
from components.model_manager import ModelManager
from components.hyperparmeter_tuner import HyperparameterTuner
from components.model_evaluator import ModelEvaluator , OverFittingDetector
from sklearn.linear_model import Ridge
from utils.logger import CustomLogger
from utils.plotter import LearningCurvePlotter ,ValidationCurvePlotter
import numpy as np


class ModelTrainingPipeline:
    def __init__(self):
        self.model_mgr = ModelManager()
        self.model_evaluator= ModelEvaluator()
        self.overfitting = OverFittingDetector()
        self.hyper_obj = HyperparameterTuner()
        self.model_selected_flag = False
        self.models_list = None
        self.best_model = None
        self.best_model_hyper = None
        self.val_score = None
        self.best_model_name = None

        
    def train(self , features, target):
        #Model selection
        if self.model_selected_flag == False:
            self.models_list = self.model_mgr.get_models()
            model_obj = ModelSelector(self.models_list)
            model_list, self.val_score = model_obj.train_and_select_best_model(features, target)
            self.best_model = model_obj.get_best_model()
            self.best_model_name = model_obj.get_best_model_name()
            print(f"{model_list}\n")
            print(f"Model Selected : {self.best_model_name}\n")
            if self.best_model is not None:
                self.model_selected_flag = True
                print(f"**** Model Training Initated ****\n")
                self.best_model.fit(features,target)
                print(f"**** Model Training Finished ****\n")
            else:
                print("Failed to select the Model.")
                self.model_selected_flag = False

            return self.val_score


    def evaluate(self, features, target ):
        mse, r2, rmse = self.model_evaluator.evaluate(self.best_model,features, target)
        report = self.model_evaluator.generate_report()
        return mse, r2 , rmse , report

    def detect_overfitting(self, x_train , y_train , x_val , y_val):
        train_mse, val_mse = self.overfitting.detect(self.best_model, x_train , y_train , x_val , y_val)
        return train_mse , val_mse

    def hyperparameter_tune_and_fit(self , param_grid, features , target , scoring ='neg_mean_squared_error',cv  = 5 ):
        self.best_model_hyper = self.hyper_obj.tune_hyperparameters(self.best_model, param_grid, features , target ,scoring = scoring,cv = cv )
        print("Best Hyperparameters:",self.best_model.get_params())
        print("Best model with hyperparameter:",self.best_model_hyper)
        #Fitting the model with hyperparameters
        print(f"\n**** Model Training with Hyperparameter Initated ****\n")
        self.best_model_hyper.fit(features, target)
        print(f"**** Model Training  with Hyperparameter Finished ****\n")
        
        return self.best_model


if __name__ == "__main__":
    #Ingest the data 
    data_ingestion = DataIngestion()
    raw_data_path = data_ingestion.initiate()

    #Preprocess the data
    data_transformation = DataTransformation()
    train_feat , val_feat , test_feat , train_target , val_target, test_target = data_transformation.initiate(raw_data_path, random_state=42)

    model = ModelTrainingPipeline()
    
    model.train(train_feat, train_target)

    print("***** Pre-hyperparameter *****\n")
    mse, r2 , rmse , report = model.evaluate(train_feat, train_target)
    print(f"Training evaluation  {report}")

    train_mse, val_mse = model.detect_overfitting(train_feat, train_target, val_feat, val_target)
    print (f"Training-MSE   {train_mse:.2f}")
    print (f"Validation-MSE {val_mse:.2f}\n")


    params = {
                'alpha': [ 0.01, 0.1, 1.0, 10.0], 
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
             }

    model_with_hyperparameter = model.hyperparameter_tune_and_fit(params, train_feat, train_target, cv = 10)
    
    train_mse, train_r2, train_rmse, train_report = model.evaluate(train_feat , train_target)
    val_mse, val_r2 , val_rmse , val_report = model.evaluate(val_feat, val_target)
    print("***** Post-hyperparameter *****")
    print(f"Training evaluation : {train_report}")
    print(f"Validation evalutation:{val_report}")


    train_mse, val_mse = model.detect_overfitting(train_feat, train_target, val_feat , val_target)
    print (f"Training-MSE   {train_mse:.2f}")
    print (f"Validation-MSE  {val_mse:.2f}")


    if train_mse < val_mse:
        print("Warning: Model might be overfitting.\n")
    else:
        print("Model seems to generalize well.\n")


    # Set up Learning Plotter
    train_sizes = np.linspace(0.1, 1.0, 5)  # Fraction of the dataset used for training
    plotter = LearningCurvePlotter(model_with_hyperparameter, train_feat, train_target, train_sizes, cv = 10)

    # Plot the learning curve
    plotter.plot_learning_curve()   


    #Set up validation plotter
    val_plotter = ValidationCurvePlotter(model_with_hyperparameter, train_feat, 
                train_target, param_name="alpha", param_range=params.get('alpha'), cv = 10)
    
    val_plotter.plot_validation_curve()
