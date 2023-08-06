import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve , validation_curve

class LearningCurvePlotter:
    def __init__(self, model, feature, target, train_sizes, cv=None,  scoring='neg_mean_squared_error'):
        self.model = model
        self.feature = feature
        self.target = target
        self.train_sizes = train_sizes
        self.cv = cv
        self.scoring = scoring

    def plot_learning_curve(self):
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.feature, self.target, cv=self.cv, scoring=self.scoring,
            train_sizes=self.train_sizes)

        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)

        plt.figure()
        plt.title("Visualizing the effect of Training set size - Learning Curve")
        plt.xlabel("Training set size")
        plt.ylabel("Score")
        plt.grid()

        plt.plot(train_sizes, train_scores_mean, 'o-', color="red", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="green",label="Cross-validation score")

        plt.legend(loc="best")
        plt.show()

class ValidationCurvePlotter:
    def __init__(self, model, feature, target, param_name= None , param_range = None ,cv=None,  scoring='neg_mean_squared_error'):
        self.model = model
        self.feature = feature
        self.target = target
        self.cv = cv
        self.scoring = scoring
        self.param_name= param_name
        self.param_range = param_range

    def plot_validation_curve(self):
        train_score, val_score = validation_curve(
            self.model , self.feature , self.target , param_name= self.param_name , 
            param_range=self.param_range ,cv = self.cv , scoring=self.scoring)

        train_mean = -np.mean(train_score , axis=1)
        val_mean = -np.mean(val_score , axis=1)

        plt.figure()
        plt.title('Visualizing the effect of hyperparameter - Alpha')
        plt.xlabel('Alpha')
        plt.ylabel('Score')
        plt.grid()

        plt.plot(self.param_range, train_mean , 'o-', color ="red" , label="Training score")
        plt.plot(self.param_range, val_mean , 'o-', color='green', label = "Cross-validation score")

        plt.legend(loc="best")
        plt.show()

        


