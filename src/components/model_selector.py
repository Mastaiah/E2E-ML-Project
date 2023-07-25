from sklearn.model_selection import cross_val_score

class ModelSelector:
    def __init__(self, models):
        self.models = models
        self.best_model = None
        self.best_model_name = None

    def train_and_select_best_model(self, X, y, cv=5, scoring='neg_mean_squared_error'):
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

        for model_name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            mean_score = scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                self.best_model = model
                self.best_model_name = model_name

    def get_best_model(self):
        return self.best_model

    def get_best_model_name(self):
        return self.best_model_name