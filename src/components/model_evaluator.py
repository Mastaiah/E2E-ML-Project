from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class OverFittingDetector:
    def __init__(self, model, x_train , y_train , x_val , y_val):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
    
    def detect_overfitting(self):
        self.model.fit(self.X_train, self.y_train)
        y_train_pred = self.model.predict(self.X_train)
        y_val_pred = self.model.predict(self.X_val)

        train_error = mean_squared_error(self.y_train, y_train_pred)
        val_error = mean_squared_error(self.y_val, y_val_pred)

        return train_error, val_error


class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        return mse, r2, rmse

    def generate_report(self):
        mse, r2, rmse = self.evaluate()
        report = f"Model Evaluation Report:\n"
        report += f"Mean Squared Error (MSE): {mse:.2f}\n"
        report += f"R-squared (R2): {r2:.2f}\n"
        report += f"Root Mean Squared Error (RMSE): {rmse:.2f}\n"
        return report