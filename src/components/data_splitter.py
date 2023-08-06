from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self):
        self.val_size = None
        self.test_size = None
        self.random_state = None

    def split_data(self, features , target ,val_size = 0.20, test_size = 0.10 , random_state = None):
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        train_features, test_features , train_target, test_target = \
                train_test_split (features, target , test_size = self.test_size , random_state = self.random_state)
        
        #Splitting the training data into train and validation data.
        train_features, validation_features, train_target, validation_target = \
                train_test_split (train_features,train_target , test_size = self.val_size , random_state=self.random_state)

        return train_features, validation_features, test_features, train_target, validation_target, test_target

