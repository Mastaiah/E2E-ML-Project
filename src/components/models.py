
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

@dataclass
class ModelManager:
    models: Dict[str, object]

# Create a dictionary of candidate models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Machine': SVR(),
    'Decision Tree' : DecisionTreeRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost Regressor' : XGBRegressor(),
    'CatBoosting Regressor' :CatBoostRegressor(verbose=False),
    'AdaBoost Regressor': AdaBoostRegressor()
   

}

# Create a ModelManager instance with the models dictionary
model_manager = ModelManager(models)