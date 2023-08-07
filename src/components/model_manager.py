from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from typing import Dict
from dataclasses import dataclass, field

@dataclass
class ModelManager:
    models_list: Dict[str, object] = field( default_factory = lambda:{
                                        'Random Forest': RandomForestRegressor(),
                                        'Support Vector Machine': SVR(),
                                        'Decision Tree' : DecisionTreeRegressor(),
                                        'Gradient Boosting': GradientBoostingRegressor(),
                                        'XGBoost Regressor' : XGBRegressor(),
                                        'CatBoosting Regressor' :CatBoostRegressor(verbose=False),
                                        'AdaBoost Regressor': AdaBoostRegressor(),
                                        'Linear Regression': LinearRegression(),
                                        'Lasso Regression': Lasso(),
                                        'Ridge Regression': Ridge(),
                                    })

    def get_models(self)->dict:
        return self.models_list
