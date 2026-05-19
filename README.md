# CodeProject AI YOLOv11 Adapter

This repository provides a lightweight adapter module to enable YOLOv11 object detection support for CodeProject AI 2.7.9.

Features

- `YOLOv11Detector` wrapper with pluggable model loader
- Unit-testable without heavy ML dependencies
- CI workflow to run unit tests

Quick start

- Install your YOLOv11 runtime (PyTorch, model code) separately.
- Use `YOLOv11Detector` with a custom `model_loader` that returns an object with a `predict(image)` method.
