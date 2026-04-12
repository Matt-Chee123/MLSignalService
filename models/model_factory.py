from .sklearn_models import RandomForestModel, GradientBoostingModel
from .boost_models import XgBoostModel

def get_model_from_config(model_config):
    model_type = model_config.get('model_type').lower()
    hyperparams = model_config.get('hyperparams', {})

    if model_type == 'random_forest':
        return RandomForestModel("Random Forest", hyperparams)
    elif model_type == 'gradient_boosting':
        return GradientBoostingModel("Gradient Boosting", hyperparams)
    elif model_type == 'xgboost':
        return XgBoostModel("XGBoost", hyperparams)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
