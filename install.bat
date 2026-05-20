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

:: Ensure model weights are present.
:: If the module was installed from the git repo (with LFS) the .pt files are
:: already in assets\. Otherwise download them from ultralytics.
set assetsDir=%moduleDirPath%\assets
if not exist "%assetsDir%" mkdir "%assetsDir%"

set ultralytics_base=https://github.com/ultralytics/assets/releases/download/v8.3.0

for %%M in (yolo11n yolo11s yolo11m yolo11l yolo11x yolo11n-seg yolo11m-seg yolo11x-seg) do (
    if not exist "%assetsDir%\%%M.pt" (
        echo Downloading %%M.pt from ultralytics...
        curl -fsSL "%ultralytics_base%/%%M.pt" -o "%assetsDir%\%%M.pt"
        if errorlevel 1 (
            echo Warning: could not download %%M.pt — place it manually in assets\
        )
    )
)
