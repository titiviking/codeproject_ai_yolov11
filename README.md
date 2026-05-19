# Object Detection (YOLOv11)

**GitHub:** https://github.com/titiviking/codeproject_ai_yolov11

A drop-in module for [CodeProject.AI Server](https://github.com/codeproject/CodeProject.AI-Server) that
adds YOLOv11 object detection and segmentation using the
[ultralytics](https://github.com/ultralytics/ultralytics) library.

## Features

- Standard object detection via `POST /v1/vision/detection`
- Object segmentation via `POST /v1/vision/segmentation`
- Custom model inference via `POST /v1/vision/custom/<model-name>`
- Custom model listing via `POST /v1/vision/custom/list`
- NVIDIA CUDA, Apple MPS, and CPU backends
- Configurable model size (tiny → huge) through the dashboard
- Interactive browser-based explorer (`explore.html`)

## Model sizes

| Size   | Model file   | Notes                              |
|--------|--------------|------------------------------------|
| Tiny   | yolo11n.pt   | Fastest; suitable for edge devices |
| Small  | yolo11s.pt   |                                    |
| Medium | yolo11m.pt   | Default                            |
| Large  | yolo11l.pt   |                                    |
| Huge   | yolo11x.pt   | Most accurate                      |

Segmentation variants (`yolo11n-seg.pt`, `yolo11m-seg.pt`, `yolo11x-seg.pt`) are
included alongside the detection models.

## Installation (CodeProject.AI Server)

1. Copy this directory into your CodeProject.AI Server `modules/` folder:
   ```
   modules/ObjectDetectionYOLOv11/
   ```
2. Run the server setup script from the server `src/` directory:
   ```bash
   bash src/setup.sh
   ```
   The `install.sh` script will install Python dependencies and download
   YOLOv11 model weights.

3. Start (or restart) CodeProject.AI Server. The module appears in the
   dashboard and auto-starts when `AutoStart` is `true` in
   `modulesettings.json`.

## Manual / development setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install ultralytics>=8.3.0 Pillow CodeProject-AI-SDK

# Place model weights in assets/
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt \
     -O assets/yolo11m.pt

# Run the adapter directly (outside the server)
python detect_adapter.py
```

## Custom models

Place `.pt` model files in the `custom-models/` directory. They are
automatically listed under `/v1/vision/custom/list` and callable via
`/v1/vision/custom/<model-name>`.

## API reference

### `POST /v1/vision/detection`

| Field          | Type   | Description                              |
|----------------|--------|------------------------------------------|
| `image`        | file   | Image to analyse                         |
| `min_confidence` | float | Detection threshold (default 0.4)       |

Returns `predictions` array of `{label, confidence, x_min, y_min, x_max, y_max}`.

### `POST /v1/vision/segmentation`

Same inputs as detection. Each prediction additionally contains a `mask`
field with polygon coordinates.

### `POST /v1/vision/custom/<model-name>`

Same inputs as detection. Uses a custom `.pt` file from `custom-models/`.

### `POST /v1/vision/custom/list`

No inputs. Returns `{"models": ["name1", "name2", ...]}`.

## Repository layout

```
ObjectDetectionYOLOv11/
├── modulesettings.json      # CodeProject.AI module configuration
├── detect_adapter.py        # ModuleRunner entry point
├── detect.py                # YOLOv11 inference logic
├── options.py               # Environment-variable configuration
├── requirements.txt         # Python dependencies
├── install.sh               # Linux / macOS setup
├── install.bat              # Windows setup
├── explore.html             # Browser test UI
├── assets/                  # Model weights (.pt files)
├── custom-models/           # Custom model weights
├── test/                    # Test images
└── codeproject_ai_yolov11/  # Installable Python wrapper package
    └── detector.py
```

## License

AGPL-3.0 — see [LICENSE](LICENSE).
