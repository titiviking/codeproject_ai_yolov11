import os
import sys
import time

# Required for PyTorch on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from codeproject_ai_sdk import RequestData, ModuleRunner, LogMethod, JSON

from PIL import Image
from options import Options
from detect import do_detection


class YOLOv11_adapter(ModuleRunner):

    def __init__(self):
        super().__init__()
        self.opts = Options()
        self.models_last_checked = None
        self.custom_model_names  = []

        self.use_CUDA     = self.opts.use_CUDA
        self.use_MPS      = self.opts.use_MPS
        self.use_DirectML = self.opts.use_DirectML

        if (self.use_CUDA and self.half_precision == "enable"
                and not self.system_info.hasTorchHalfPrecision):
            self.half_precision = "disable"

    def initialise(self):
        if not self.launched_by_server:
            self.queue_name = "objectdetection_queue"

        if self.use_CUDA:
            self.use_CUDA = self.system_info.hasTorchCuda
            try:
                import cudnn as cudnn
                if cudnn.is_available():
                    cudnn.benchmark = False
            except Exception:
                pass

        if self.use_CUDA:
            self.use_MPS      = False
            self.use_DirectML = False
        else:
            self.use_MPS = self.system_info.hasTorchMPS

        # DirectML is currently not reliable enough to enable by default
        self.use_DirectML = False

        self.can_use_GPU = self.system_info.hasTorchCuda or self.system_info.hasTorchMPS

        if self.use_CUDA:
            self.inference_device  = "GPU"
            self.inference_library = "CUDA"
        elif self.use_MPS:
            self.inference_device  = "GPU"
            self.inference_library = "MPS"
        elif self.use_DirectML:
            self.inference_device  = "GPU"
            self.inference_library = "DirectML"

        self._num_items_found = 0
        self._histogram       = {}

    def process(self, data: RequestData) -> JSON:

        if data.command == "list-custom":
            return self._list_custom_models()

        elif data.command == "segment":
            threshold: float = float(data.get_value("min_confidence", "0.4"))
            img: Image       = data.get_image(0)
            return do_detection(self, self.opts.models_dir,
                                self.opts.std_seg_model_name, self.opts.resolution_pixels,
                                self.use_CUDA, self.accel_device_name,
                                self.use_MPS, self.use_DirectML, self.half_precision,
                                img, threshold, do_segmentation=True)

        elif data.command == "detect":
            threshold: float = float(data.get_value("min_confidence", "0.4"))
            img: Image       = data.get_image(0)
            return do_detection(self, self.opts.models_dir,
                                self.opts.std_model_name, self.opts.resolution_pixels,
                                self.use_CUDA, self.accel_device_name,
                                self.use_MPS, self.use_DirectML, self.half_precision,
                                img, threshold, do_segmentation=False)

        elif data.command == "custom":
            self._list_custom_models()  # populate / refresh the 60-second cache
            if not self.custom_model_names:
                return {"success": False, "error": "No custom models found"}

            threshold: float = float(data.get_value("min_confidence", "0.4"))
            img: Image       = data.get_image(0)

            model_dir:  str = self.opts.custom_models_dir
            model_name: str = "general"
            if data.segments and data.segments[0]:
                model_name = data.segments[0]

            if model_name == "general":
                model_dir  = self.opts.custom_models_dir
                model_name = "ipcam-general"

            self.log(LogMethod.Info | LogMethod.Server, {
                "filename": __file__,
                "loglevel": "information",
                "method":   sys._getframe().f_code.co_name,
                "message":  f"Detecting using {model_name}",
            })

            return do_detection(self, model_dir, model_name,
                                self.opts.resolution_pixels, self.use_CUDA,
                                self.accel_device_name, use_MPS=False,
                                use_DirectML=self.use_DirectML,
                                half_precision=self.half_precision,
                                img=img, threshold=threshold)

        else:
            self.report_error(None, __file__, f"Unknown command {data.command}")
            return {"success": False, "error": f"Unknown command {data.command}"}

    def status(self) -> JSON:
        statusData = super().status()
        statusData["numItemsFound"] = self._num_items_found
        statusData["histogram"]     = self._histogram
        return statusData

    def update_statistics(self, response):
        super().update_statistics(response)
        if response.get("success") and "predictions" in response:
            predictions = response["predictions"]
            self._num_items_found += len(predictions)
            for prediction in predictions:
                label = prediction["label"]
                self._histogram[label] = self._histogram.get(label, 0) + 1

    def selftest(self) -> JSON:
        file_name = os.path.join("test", "home-office.jpg")

        request_data = RequestData()
        request_data.queue   = self.queue_name
        request_data.command = "detect"
        request_data.add_file(file_name)
        request_data.add_value("min_confidence", 0.4)

        result = self.process(request_data)
        print(f"Info: Self-test for {self.module_id}. Success: {result['success']}")
        return {"success": result["success"], "message": "Object detection test successful"}

    def _list_custom_models(self):
        if (self.models_last_checked is None
                or (time.time() - self.models_last_checked) >= 60):
            self.custom_model_names = []
            try:
                models_path = self.opts.custom_models_dir
                if os.path.exists(models_path):
                    self.custom_model_names = [
                        entry.name[:-3]
                        for entry in os.scandir(models_path)
                        if (entry.is_file()
                            and entry.name.endswith(".pt")
                            and not entry.name.startswith("yolo11"))
                    ]
            except Exception:
                pass
            self.models_last_checked = time.time()

        return {"success": True, "models": self.custom_model_names}


if __name__ == "__main__":
    YOLOv11_adapter().start_loop()
