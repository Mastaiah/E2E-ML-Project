from dataclasses import dataclass
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from typing import Dict
import pandas as pd
import numpy as np
from model_manager import ModelManager

"""
@dataclass
class ModelManager:
    models_list: Dict[str, object] 
"""
    

class ModelSelector:
    def __init__(self, models):
        self.models = models
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.df = None

    def train_and_select_best_model(self, X, y, cv=None, scoring='neg_mean_squared_error'):
        """
        Train different models on the data and select the best model based on cross-validation.

        Parameters:
            X (array-like): Features (input variables).
            y (array-like): Target variable (output variable).
            cv (int, optional): Number of cross-validation folds (if we use None it will set to default=5).
            scoring (str, optional): Scoring metric for evaluation (default='neg_mean_squared_error').

        Returns:
            None
        """
        best_score = float('-inf')

        #Replace self.models.items() with self.models.models_list.item() if you are using the ModelManager class in the same file.
        for model_name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

            if scoring == 'neg_mean_squared_error':
                rmse_scores = np.sqrt(-scores)
                mean_score = rmse_scores.mean()
            else:
                mean_score = scores.mean()

            self.results[model_name] = mean_score


            if mean_score > best_score:
                best_score = mean_score
                self.best_model = model
                self.best_model_name = model_name
        
        #Use pd.DataFrame.from_dict when there is a nested dictionary .                
        self.df = pd.DataFrame.from_dict(self.results ,orient='index',columns=['Value'])
        self.df.reset_index(inplace=True)
        self.df.rename(columns={'index': 'Model'}, inplace=True)
    
        return self.df

    def get_best_model(self):
        return self.best_model

    def get_best_model_name(self):
        return self.best_model_name
    
"""
Uncomment this block if you want to use the below code in the same file. 
# Create a dictionary of candidate models
models = {
    'Random Forest': RandomForestRegressor(),
    'Support Vector Machine': SVR(),
    'Decision Tree' : DecisionTreeRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost Regressor' : XGBRegressor(),
    'CatBoosting Regressor' :CatBoostRegressor(verbose=False),
    'AdaBoost Regressor': AdaBoostRegressor(),
    'Linear Regression': LinearRegression(),
}
"""

"""
#The below code is written just for testing purpose
import numpy as np
# Sample data
X = np.random.rand(100, 2)
y = np.random.rand(100)

if __name__ == "__main__":
    model_manager = ModelManager()
    model = model_manager.get_models()
    model_selector = ModelSelector(model)
    result = model_selector.train_and_select_best_model(X, y)


    print(result)
    print(f"\n Best Model ==> [ {model_selector.get_best_model_name()} ]")
"""
