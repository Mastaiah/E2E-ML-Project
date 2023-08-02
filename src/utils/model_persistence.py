import os
import joblib
import pickle
#import dill


class Saver:
    def __init__(self , model, filepath , format='pickle'):
        self.model = model
        self.filepath = filepath
        self.format = format

    def save(self):
        with open (self.filepath ,'wb') as file:
            if self.format =='pickle':
                pickle.dump(self.model,file)
                # Use dill - to serialize more complex Python objects, including certain classes or functions.
                #dill.dump(self.model, file)
            elif self.format == 'joblib':
                joblib.dump(self.model,file)
            else:
                print("Unknown format \n")


class Loader:
    def __init__(self):
        self.model = None

    def load(self,filepath):
         # Extract the file extension using os.path.splitext()
        _ , stored_file_ext = os.path.splitext(filepath)

        if stored_file_ext == ".pkl":
            self.model = pickle.load(filepath)
            #self.model = dill.load(filepath)
        elif stored_file_ext == ".joblib":
            self.model = joblib.dump(filepath)
        else:
            self.model = None
            print("Unknown file extension.")

        return self.model


"""
Below code is for testing purpose only"

if __name__ == "__main__":
     trained_model = 'abc'
     model_storage = ModelSaverLoader(trained_model)
     model_storage.save_model('xyz','pickle')
     # Example file path
     file_path = "nimavidu/model.joblib"
     model_storage.load_model(file_path)

"""