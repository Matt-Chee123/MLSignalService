from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, name, hyperparameters):
        self.name = name
        self.hyperparameters = hyperparameters
        self.model = None

    @abstractmethod
    def create_model(self):
        pass

    def get_model(self):
        if self.model is None:
            self.create_model()
        return self.model

    def fit(self, X, y):
        if self.model is None:
            self.create_model()
        self.model.fit(X, y)

    def predict(self, X):
        if self.model is None:
            self.create_model()
        return self.model.predict(X)