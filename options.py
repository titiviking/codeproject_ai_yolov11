import os
from codeproject_ai_sdk import ModuleOptions


class Settings:
    def __init__(self, RESOLUTION, STD_MODEL_NAME, STD_SEG_MODEL_NAME):
        self.RESOLUTION         = RESOLUTION
        self.STD_MODEL_NAME     = STD_MODEL_NAME
        self.STD_SEG_MODEL_NAME = STD_SEG_MODEL_NAME


class Options:

    def __init__(self):
        self.MODEL_SETTINGS = {
            "tiny":   Settings(STD_MODEL_NAME="yolo11n", STD_SEG_MODEL_NAME="yolo11n-seg", RESOLUTION=640),
            "small":  Settings(STD_MODEL_NAME="yolo11s", STD_SEG_MODEL_NAME="yolo11s-seg", RESOLUTION=640),
            "medium": Settings(STD_MODEL_NAME="yolo11m", STD_SEG_MODEL_NAME="yolo11m-seg", RESOLUTION=640),
            "large":  Settings(STD_MODEL_NAME="yolo11l", STD_SEG_MODEL_NAME="yolo11l-seg", RESOLUTION=640),
            "huge":   Settings(STD_MODEL_NAME="yolo11x", STD_SEG_MODEL_NAME="yolo11x-seg", RESOLUTION=640),
        }

        self._show_env_variables = True

        self.app_dir           = os.path.normpath(ModuleOptions.getEnvVariable("APPDIR", os.getcwd()))
        self.models_dir        = os.path.normpath(ModuleOptions.getEnvVariable("MODELS_DIR",        f"{self.app_dir}/assets"))
        self.custom_models_dir = os.path.normpath(ModuleOptions.getEnvVariable("CUSTOM_MODELS_DIR", f"{self.app_dir}/custom-models"))

        self.sleep_time   = 0.01

        self.model_size   = ModuleOptions.getEnvVariable("MODEL_SIZE", "Medium")
        self.use_CUDA     = ModuleOptions.getEnvVariable("USE_CUDA",   "True")
        self.use_MPS      = True   # resolved against hardware availability at runtime
        self.use_DirectML = True   # resolved against hardware availability at runtime

        self.model_size = self.model_size.lower()
        self.use_CUDA   = ModuleOptions.enable_GPU and self.use_CUDA.lower() == "true"

        if self.model_size not in self.MODEL_SETTINGS:
            self.model_size = "medium"

        settings = self.MODEL_SETTINGS[self.model_size]
        self.resolution_pixels  = settings.RESOLUTION
        self.std_model_name     = settings.STD_MODEL_NAME
        self.std_seg_model_name = settings.STD_SEG_MODEL_NAME

        if self._show_env_variables:
            print(f"Debug: APPDIR:      {self.app_dir}")
            print(f"Debug: MODEL_SIZE:  {self.model_size}")
            print(f"Debug: MODELS_DIR:  {self.models_dir}")
