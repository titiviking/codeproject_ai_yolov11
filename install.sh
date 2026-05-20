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

# Ensure model weights are present in assets/.
# If the module was installed from the git repo (with LFS), the .pt files are
# already there. Otherwise download them directly from ultralytics.
if [ "$moduleInstallErrors" = "" ]; then
    assetsDir="${moduleDirPath}/assets"
    mkdir -p "$assetsDir"

    ultralytics_base="https://github.com/ultralytics/assets/releases/download/v8.3.0"
    missing=0

    for model in yolo11n yolo11s yolo11m yolo11l yolo11x yolo11n-seg yolo11m-seg yolo11x-seg; do
        modelFile="${model}.pt"
        modelPath="${assetsDir}/${modelFile}"

        # A git-lfs pointer file is ~130 bytes; real .pt files are megabytes.
        # Treat small files as missing so they get fetched properly.
        if [ ! -f "${modelPath}" ] || [ "$(wc -c < "${modelPath}")" -lt 1024 ]; then
            missing=$((missing + 1))
            writeLine "Downloading ${modelFile} from ultralytics..." $color_info
            wget -q "${ultralytics_base}/${modelFile}" -O "${modelPath}" 2>/dev/null || \
                curl -fsSL "${ultralytics_base}/${modelFile}" -o "${modelPath}" 2>/dev/null || \
                writeLine "Warning: could not download ${modelFile} — place it manually in assets/" $color_warn
        fi
    done

    if [ "$missing" -eq 0 ]; then
        writeLine "All YOLOv11 model weights already present." $color_info
    fi
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
