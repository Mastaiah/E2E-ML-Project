from sklearn.model_selection import GridSearchCV
from utils.model_persistence import Saver
import os

class HyperparameterTuner:
    def __init__(self):
        self.model = None
        self.param_grid = None
        self.scoring = None
        self.cv = None
        self.best_model = None
        self.best_params = None
        self.hyperparameter_path = os.path.join('artifacts','hyperparameter.pkl')
    
    def tune_hyperparameters(self, model, param_grid, x_train , y_train, scoring ='neg_mean_squared_error', cv=5 ):
        self.model = model
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        #Use GridSearchCV for hyperparameter tuning with cross-validation
        grid_search = GridSearchCV(self.model , param_grid=self.param_grid ,cv=self.cv ,scoring=self.scoring)
        grid_search.fit(x_train, y_train)
        self.best_model = grid_search.best_estimator_
        self.best_params = self.best_model.get_params()

        if self.best_model is None:
            raise CustomException("Hyper-parameters are not tunned , please tune it !!!.")
        else:
            hyperparameters = Saver(self.best_params, self.hyperparameter_path)
            hyperparameters.save()
            return self.best_model
