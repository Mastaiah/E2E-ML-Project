from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class OverFittingDetector:
    def __init__(self ):
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
    
    def detect(self, model, x_train , y_train , x_val , y_val):
        self.model= model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.model.fit(self.x_train, self.y_train)
        y_train_pred = self.model.predict(self.x_train)
        y_val_pred = self.model.predict(self.x_val)

        train_mse = mean_squared_error(self.y_train, y_train_pred)
        val_mse = mean_squared_error(self.y_val, y_val_pred)

        return train_mse, val_mse


class ModelEvaluator:
    def __init__(self):
        self.model = None
        self.features = None
        self.target = None

    def evaluate(self, model, features , target):
        self.model = model
        self.features = features
        self.target = target
        predicted_target = model.predict(features)
        mse = mean_squared_error(target, predicted_target)
        rmse = mean_squared_error(target, predicted_target, squared=False)
        r2 = r2_score(target, predicted_target)
        #rmse = np.sqrt(mse)

        return mse, r2, rmse


    def generate_report(self):
        mse, r2, rmse = self.evaluate(self.model, self.features, self.target)
        report = f"report :\n"
        report += f"-" * 35 
        report += f"\n\nMean Squared Error (MSE): {mse:.2f}\n"
        report += f"R-squared (R2): {r2:.2f}\n"
        report += f"Root Mean Squared Error (RMSE): {rmse:.2f}\n\n"
        report += f"-" * 35
        report += f"\n"
        return report
        