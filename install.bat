:: Development mode setup script ::::::::::::::::::::::::::::::::::::::::::::::
::
::                         Object Detection (YOLOv11)
::
:: This script is called from the ObjectDetectionYOLOv11 directory using:
::
::    ..\..\..\CodeProject.AI-Server\src\setup.bat
::
:: The setup.bat script will find this install.bat file and execute it.
::
:: For help with install scripts, notes on variables and methods available,
:: tips, and explanations, see /src/modules/install_script_help.md

@if "%1" NEQ "install" (
    timeout /t 3 /nobreak >NUL
    echo This script is only called from: ..\..\..\CodeProject.AI-Server\src\setup.bat
    echo.
    exit /b 1
)

:: Download YOLOv11 models
REM call "%utilsScript%" GetFromServer "models/" "objectdetection-coco-yolo11-pt-nsmlx.zip"    "assets"        "Downloading YOLOv11 object detection models..."
REM call "%utilsScript%" GetFromServer "models/" "objectsegmentation-coco-yolo11-pt-nsmlx.zip" "assets"        "Downloading YOLOv11 segmentation models..."
REM call "%utilsScript%" GetFromServer "models/" "objectdetection-custom-yolo11-pt-m.zip"      "custom-models" "Downloading custom YOLOv11 models..."

:: TODO: Validate assets created and has files
:: set moduleInstallErrors=...
