from models.model_factory import get_model_from_config
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path

class Trainer:
    def __init__(self, model_config, output_dir="./models/"):
        #model config like {'model_type': things, 'hyperparams': {}}
        self.model_config = model_config
        self.model = get_model_from_config(model_config)
        self.output_dir = Path(output_dir)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, name='model.pkl'):
        joblib.dump(self.model, self.output_dir / name)