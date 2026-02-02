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