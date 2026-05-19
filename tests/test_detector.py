import unittest

from codeproject_ai_yolov11.detector import YOLOv11Detector


class DummyModel:
    """Minimal stand-in for an ultralytics YOLO model."""

    def predict(self, image):
        return [
            {"label": "person", "confidence": 0.9,
             "x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200},
            {"label": "car",    "confidence": 0.75,
             "x_min": 50, "y_min": 60, "x_max": 300, "y_max": 250},
        ]


class TestYOLOv11Detector(unittest.TestCase):

    def _make_detector(self):
        det = YOLOv11Detector(model_loader=lambda path: DummyModel())
        det.load()
        return det

    def test_detect_returns_list(self):
        res = self._make_detector().detect(None)
        self.assertIsInstance(res, list)

    def test_detect_prediction_keys(self):
        pred = self._make_detector().detect(None)[0]
        for key in ("label", "confidence", "x_min", "y_min", "x_max", "y_max"):
            self.assertIn(key, pred)

    def test_detect_label_value(self):
        res = self._make_detector().detect(None)
        self.assertEqual(res[0]["label"], "person")

    def test_load_raises_without_loader(self):
        det = YOLOv11Detector()
        with self.assertRaises(RuntimeError):
            det.load()

    def test_detect_raises_before_load(self):
        det = YOLOv11Detector(model_loader=lambda path: DummyModel())
        with self.assertRaises(RuntimeError):
            det.detect(None)

    def test_device_stored(self):
        det = YOLOv11Detector(device="cuda")
        self.assertEqual(det.device, "cuda")


if __name__ == "__main__":
    unittest.main()
