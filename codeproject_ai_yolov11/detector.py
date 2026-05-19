"""Lightweight adapter for YOLOv11 models.

This module intentionally avoids importing heavy ML libs so unit tests
can run without installing Torch during initial development.
"""
from typing import Callable, Any, Optional


class YOLOv11Detector:
    """Adapter for YOLOv11-style models.

    The detector accepts a `model_loader` callable that should return an
    object exposing a `predict(image)` method which returns a list of
    detection dicts: {'label': str, 'confidence': float, 'bbox': [x,y,w,h]}.
    """

    def __init__(self, model_loader: Optional[Callable[[Optional[str]], Any]] = None, device: str = "cpu"):
        self.device = device
        self.model = None
        self._model_loader = model_loader

    def load(self, model_path: Optional[str] = None):
        """Load a model using `model_loader` or raise if none provided."""
        if self._model_loader is None:
            raise RuntimeError("No model_loader provided. Provide a callable that returns a model with `predict(image)`.")
        self.model = self._model_loader(model_path)

    def detect(self, image: Any):
        """Run detection on `image` and return model output unchanged.

        `image` should be in the format accepted by the underlying model.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call `load(...)` first.")
        return self.model.predict(image)
