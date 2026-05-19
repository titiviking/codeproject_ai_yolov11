import os
from os.path import exists
import sys
import time
from threading import Lock

import torch
import numpy as np
from PIL import UnidentifiedImageError

from ultralytics import YOLO

from codeproject_ai_sdk import LogMethod

# Per-model YOLO detector cache; keyed by model name.
detectors    = {}
models_lock  = Lock()
predict_lock = Lock()


def get_detector(module_runner, models_dir: str, model_name: str, resolution: int,
                 use_Cuda: bool, accel_device_name, use_MPS: bool,
                 use_DirectML: bool, half_precision: str):
    """Return a cached YOLO detector, creating and caching it on first use."""

    detector = detectors.get(model_name, None)
    if detector is None:
        with models_lock:
            detector = detectors.get(model_name, None)
            if detector is None:
                model_path = os.path.join(models_dir, model_name + ".pt")

                if use_Cuda:
                    device = torch.device(accel_device_name if accel_device_name else "cuda")
                    device_name = torch.cuda.get_device_name(device)
                    print(f"GPU compute capability is "
                          f"{torch.cuda.get_device_capability()[0]}."
                          f"{torch.cuda.get_device_capability()[1]}")
                    half = half_precision != "disable"
                    qualifier = "" if half else "not "
                    print(f"{qualifier.capitalize()}Using half-precision for '{device_name}'")
                elif use_MPS:
                    device      = torch.device("mps")
                    device_name = "Apple Silicon GPU"
                    half        = False
                elif use_DirectML:
                    import torch_directml
                    device      = torch_directml.device()
                    device_name = "DirectML"
                    half        = False
                else:
                    device      = torch.device("cpu")
                    device_name = "CPU"
                    half        = False

                print(f"Inference processing will occur on device '{device_name}'")

                if exists(model_path):
                    try:
                        detector = YOLO(model_path)
                        detectors[model_name] = detector
                        module_runner.log(LogMethod.Server, {
                            "filename": __file__,
                            "method":   sys._getframe().f_code.co_name,
                            "loglevel": "debug",
                            "message":  f"Model path is {model_path}",
                        })
                    except Exception as ex:
                        module_runner.report_error(ex, __file__,
                            f"Unable to load model at {model_path} ({ex})")
                        detector = None
                else:
                    module_runner.report_error(None, __file__,
                        f"{model_path} does not exist")

    return detector


def do_detection(module_runner, models_dir: str, model_name: str, resolution: int,
                 use_Cuda: bool, accel_device_name, use_MPS: bool,
                 use_DirectML: bool, half_precision: str,
                 img, threshold: float, do_segmentation: bool = False):

    start_process_time = time.perf_counter()
    create_err_msg = f"Unable to create YOLO detector for model {model_name}"

    detector = None
    try:
        detector = get_detector(module_runner, models_dir, model_name,
                                resolution, use_Cuda, accel_device_name,
                                use_MPS, use_DirectML, half_precision)
    except Exception as ex:
        create_err_msg = f"{create_err_msg} ({ex})"

    if detector is None:
        module_runner.report_error(None, __file__, create_err_msg)
        return {"success": False, "error": create_err_msg}

    try:
        start_inference_time = time.perf_counter()

        # Use half-precision during inference only on CUDA, and only when not disabled.
        use_half = use_Cuda and half_precision not in ("disable", "false", "")

        with predict_lock:
            results = detector.predict(img, imgsz=resolution,
                                       half=use_half,
                                       device=accel_device_name)

        inferenceMs = int((time.perf_counter() - start_inference_time) * 1000)

        outputs = []
        for result in results:
            boxes = result.boxes
            masks = result.masks if do_segmentation else None

            for i in range(len(boxes.conf)):
                score = boxes.conf[i].item()
                if score < threshold:
                    continue

                x_min = boxes.xyxy[i][0].item()
                y_min = boxes.xyxy[i][1].item()
                x_max = boxes.xyxy[i][2].item()
                y_max = boxes.xyxy[i][3].item()
                label = detector.names[int(boxes.cls[i].item())]

                detection = {
                    "confidence": score,
                    "label":      label,
                    "x_min":      int(x_min),
                    "y_min":      int(y_min),
                    "x_max":      int(x_max),
                    "y_max":      int(y_max),
                }

                if do_segmentation and masks is not None:
                    points = np.int32([masks[i].xy])
                    detection["mask"] = points.tolist()[0][0]

                outputs.append(detection)

        if len(outputs) > 3:
            message = "Found " + ", ".join(p["label"] for p in outputs[:3]) + "..."
        elif outputs:
            message = "Found " + ", ".join(p["label"] for p in outputs)
        else:
            message = "No objects found"

        return {
            "success":     True,
            "message":     message,
            "count":       len(outputs),
            "predictions": outputs,
            "processMs":   int((time.perf_counter() - start_process_time) * 1000),
            "inferenceMs": inferenceMs,
        }

    except UnidentifiedImageError:
        module_runner.report_error(None, __file__, "The image provided was of an unknown type")
        return {"success": False, "error": "invalid image file"}

    except Exception as ex:
        module_runner.report_error(ex, __file__)
        return {"success": False, "error": "Error occurred on the server"}
