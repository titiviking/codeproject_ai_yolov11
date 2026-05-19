import unittest

from codeproject_ai_yolov11.detector import YOLOv11Detector


class DummyModel:
    def predict(self, image):
        return [{"label": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]}]


class TestDetector(unittest.TestCase):
    def test_detect_returns_list(self):
        det = YOLOv11Detector(model_loader=lambda path: DummyModel())
        det.load()
        res = det.detect(None)
        self.assertIsInstance(res, list)
        self.assertEqual(res[0]["label"], "person")


if __name__ == "__main__":
    unittest.main()
