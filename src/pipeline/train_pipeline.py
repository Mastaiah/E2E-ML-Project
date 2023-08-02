from components.data_ingestion import DataIngestion
from components.data_transform import DataTransformation
from components.model_selector import ModelSelector
from components.model_manager import ModelManager
from components.hyperparmeter_tuner import HyperparameterTuner
from components.model_evaluator import ModelEvaluator , OverFittingDetector
from sklearn.linear_model import Ridge
from utils.logger import CustomLogger

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
        #Modle selection and performing the hyperparameter tunning
        if self.model_selected_flag == False:
            self.models_list = self.model_mgr.get_models()
            model_obj = ModelSelector(self.models_list)
            _ , self.val_score = model_obj.train_and_select_best_model(features, target)
            self.best_model = model_obj.get_best_model()
            self.best_model_name = model_obj.get_best_model_name()
            print(f"Model Selected : {self.best_model_name}\n")
            if self.best_model is not None:
                self.model_selected_flag = True
                print(f"**** Model Training Initated ****\n")
                self.best_model.fit(features,target)
                print(f"***  Model Training Finished ****\n")
            else:
                print("Failed to select the Model.")

            return self.val_score


    def evaluate(self, features, target ):
        mse, r2, rmse = self.model_evaluator.evaluate(self.best_model,features, target)
        report = self.model_evaluator.generate_report()
        return mse, r2 , rmse , report

    def detect_overfitting(self, x_train , y_train , x_val , y_val):
        train_mse, val_mse = self.overfitting.detect(self.best_model, x_train , y_train , x_val , y_val)
        return train_mse , val_mse

    def hyperparameter_tune_and_fit(self , param_grid, features , target ):
        self.best_model_hyper = self.hyper_obj.tune_hyperparameters(self.best_model, param_grid, features , target  )
        print("Best Hyperparameters:",self.best_model.get_params())
        #Fitting the model with hyperparameters
        print(f"**** Model Training with Hyperparameter Initated ****\n")
        self.best_model_hyper.fit(x_train, y_train)
        print(f"***  Model Training  with Hyperparameter Finished ****\n")
        

        return self.best_model


if __name__ == "__main__":
    #Ingest the data 
    data_ingestion = DataIngestion()
    train_data,test_data = data_ingestion.initiate()

    #Preprocess the data
    data_transformation=DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    print(type(train_arr))
    X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
    model = ModelTrainingPipeline()
    model.train(X_train, y_train)

    mse, r2 , rmse , report = model.evaluate(X_train, y_train)
    print("***** Pre-hyperparameter *****")
    print(f"Training evaluation  {report}")
    mse,r2 ,rmse, report = model.evaluate(X_test,y_test)
    print(f"Validation evaluation  {report}")

    train_r2_score, val_r2_score = model.detect_overfitting(X_train, y_train, X_test ,y_test)
    print (f"Training R2 Score   {train_r2_score:.2f}")
    print (f"Validation R2 Score  {val_r2_score:.2f}")

    params = {
                'alpha': [0.01, 0.1, 1.0, 10.0], 
                'selection': ['cyclic', 'random'],
             }

    result = model.hyperparameter_tune_and_fit(params,X_train, y_train)
    
    mse, r2 , rmse , report = model.evaluate(X_train, y_train)
    print("***** Post-hyperparameter *****")
    print(f"Training evaluation : {report}")

    mse,r2 ,rmse, report = model.evaluate(X_test,y_test)
    print(f"Validation evaluation : {report}")

    train_r2_score, val_r2_score = model.detect_overfitting(X_train, y_train, X_test ,y_test)
    print (f"Training R2 Score  : {train_r2_score:.2f}")
    print (f"Validation R2 Score : {val_r2_score:.2f}")

