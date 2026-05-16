from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def predict_batch(self, images, **kwargs):
        """Perform inference on a list of images."""
        pass
