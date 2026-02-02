from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def create_model(self):
        self.model = RandomForestRegressor(
            n_estimators=self.hyperparameters.get('n_estimators',100),
            max_depth=self.hyperparameters.get('max_depth', None),
            random_state=self.hyperparameters.get('random_state', 42)
        )
        return self

class GradientBoostingModel(BaseModel):
    def create_model(self):
        self.model = GradientBoostingRegressor(
            n_estimators=self.hyperparameters.get('n_estimators', 100),
            learning_rate=self.hyperparameters.get('learning_rate', 0.01),
            max_depth=self.hyperparameters.get('max_depth',None),
            random_state=self.hyperparameters.get('random_state', 42)
        )