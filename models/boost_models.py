from xgboost import XGBRegressor
from .base_model import BaseModel

class XgBoostModel(BaseModel):
    def create_model(self):
        self.model = XGBRegressor(
            n_estimators=self.hyperparameters.get('n_estimators', 300),
            learning_rate=self.hyperparameters.get('learning_rate', 0.05),
            max_depth=self.hyperparameters.get('max_depth', 6),

            subsample=self.hyperparameters.get('subsample', 0.8),
            colsample_bytree=self.hyperparameters.get('colsample_bytree', 0.8),

            min_child_weight=self.hyperparameters.get('min_child_weight', 1),
            gamma=self.hyperparameters.get('gamma', 0),

            reg_alpha=self.hyperparameters.get('reg_alpha', 0),
            reg_lambda=self.hyperparameters.get('reg_lambda', 1),

            random_state=self.hyperparameters.get('random_state', 42),
            n_jobs=self.hyperparameters.get('n_jobs', -1),

            objective=self.hyperparameters.get('objective', 'reg:squarederror'),

            tree_method=self.hyperparameters.get('tree_method', 'hist'),
            device=self.hyperparameters.get('device', 'cuda')
        )
