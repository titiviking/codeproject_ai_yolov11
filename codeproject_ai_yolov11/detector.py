"""Lightweight YOLOv11 detector wrapper.

Designed to be testable without heavy ML dependencies. Pass a `model_loader`
callable that returns any object with a ``predict(image)`` method. In
production the loader would return an ``ultralytics.YOLO`` instance.
"""
from typing import Any, Callable, List, Optional


class YOLOv11Detector:
    """Thin adapter around a YOLOv11 model.

    Expected ``predict`` return format (list of dicts):
        [{"label": str, "confidence": float, "x_min": int, "y_min": int,
          "x_max": int, "y_max": int}, ...]
    """

    def __init__(
        self,
        model_loader: Optional[Callable[[Optional[str]], Any]] = None,
        device: str = "cpu",
    ):
        self.device = device
        self.model: Optional[Any] = None
        self._model_loader = model_loader

    def load(self, model_path: Optional[str] = None) -> None:
        """Load a model via the provided loader callable."""
        if self._model_loader is None:
            raise RuntimeError(
                "No model_loader provided. Supply a callable that returns a "
                "model exposing a predict(image) method."
            )
        self.model = self._model_loader(model_path)

    def detect(self, image: Any) -> List[dict]:
        """Run inference on *image* and return the raw model output."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self.model.predict(image)
