from .sklearn_models import RandomForestModel, GradientBoostingModel

def get_model_from_config(model_config):
    model_type = model_config.get('type').lower()
    hyperparams = model_config.get('hyperparams', {})

    if model_type == 'random_forest':
        return RandomForestModel("Random Forest", hyperparams)
    elif model_type == 'grandient_boosting':
        return GradientBoostingModel("Gradient Boosting", hyperparams)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
