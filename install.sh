#!/bin/bash

# Development mode setup script ::::::::::::::::::::::::::::::::::::::::::::::
#
#                        Object Detection (YOLOv11)
#
# This script is called from the ObjectDetectionYOLOv11 directory using:
#
#    bash ../../CodeProject.AI-Server/src/setup.sh
#
# The setup.sh script will find this install.sh file and execute it.
#
# For help with install scripts, notes on variables and methods available, tips,
# and explanations, see /src/modules/install_script_help.md

if [ "$1" != "install" ]; then
    read -t 3 -p "This script is only called from: bash ../../CodeProject.AI-Server/src/setup.sh"
    echo
    exit 1
fi

# OpenCV has a specific version requirement on macOS 11 (Big Sur)
if [ "$os_name" = "Big Sur" ]; then
    installPythonPackagesByName "opencv-python==4.6.0.66" "OpenCV 4.6.0.66 for macOS 11.x"
fi

# Download YOLOv11 models
if [ "$moduleInstallErrors" = "" ]; then
    getFromServer "models/" "objectdetection-coco-yolo11-pt-nsmlx.zip"    "assets"        "Downloading YOLOv11 object detection models..."
    getFromServer "models/" "objectsegmentation-coco-yolo11-pt-nsmlx.zip" "assets"        "Downloading YOLOv11 segmentation models..."
    getFromServer "models/" "objectdetection-custom-yolo11-pt-m.zip"      "custom-models" "Downloading custom YOLOv11 models..."
fi

# Fallback: download pre-trained models directly from ultralytics if the
# CPAI model server is unavailable (useful for offline/dev environments).
if [ "$moduleInstallErrors" = "" ]; then
    assetsDir="${moduleDirPath}/assets"
    mkdir -p "$assetsDir"

    ultralytics_base="https://github.com/ultralytics/assets/releases/download/v8.3.0"

    for model in yolo11n yolo11s yolo11m yolo11l yolo11x yolo11n-seg yolo11m-seg yolo11x-seg; do
        modelFile="${model}.pt"
        if [ ! -f "${assetsDir}/${modelFile}" ]; then
            writeLine "Downloading ${modelFile} from ultralytics..." $color_info
            wget -q "${ultralytics_base}/${modelFile}" -O "${assetsDir}/${modelFile}" 2>/dev/null || \
                curl -sL "${ultralytics_base}/${modelFile}" -o "${assetsDir}/${modelFile}" 2>/dev/null || \
                writeLine "Warning: could not download ${modelFile} — place it manually in assets/" $color_warn
        fi
    done
fi

# Download self-test image
if [ "$moduleInstallErrors" = "" ]; then
    testDir="${moduleDirPath}/test"
    mkdir -p "$testDir"
    if [ ! -f "${testDir}/home-office.jpg" ]; then
        wget -q "https://codeproject-ai.s3.ca-central-1.amazonaws.com/sense/installer/dev/home-office.jpg" \
             -O "${testDir}/home-office.jpg" 2>/dev/null || true
    fi
fi

# TODO: Validate that assets were created successfully
# moduleInstallErrors=...
