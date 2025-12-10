import joblib
import pandas as pd

class Utils:
    def load_csv(self, path):
        df = pd.read_csv(path)
        return df

    def variables(self, dataset, drop_columns, y):
        X = dataset.drop(drop_columns, axis=1)
        y = dataset[y]
        return X, y

    def model_export(self, model):
        joblib.dump(model, 'models/model.pkl')

