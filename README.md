# AI Virtual Painter — OpenCV + MediaPipe

This repository contains a compact, modular implementation of an AI‑powered virtual painter.
The app uses a webcam feed, MediaPipe hand tracking, and OpenCV to let a user paint and erase
on a live video stream using hand gestures.

Key features
- Real‑time hand tracking (MediaPipe) with a reusable `HandDetector` class.
- Header UI overlays for selecting brushes and the eraser.
- Gesture controls: two‑finger selection (index + middle) and single‑finger drawing (index).
- Colorblind‑friendly default palette and configurable brush/eraser sizes.

Quick start
1. Create and activate a Python 3.7+ virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. (Linux only) If you plan to run the GUI and webcam locally, install system graphics dependencies
	 so OpenCV can load native display libraries (e.g., provide `libGL.so.1`):

```bash
sudo apt update
sudo apt install -y libgl1 libglvnd0 libegl1-mesa libx11-6 libxext6 libxrender1 libsm6 libglib2.0-0
```

If you are running in a headless container (CI, Codespaces) and do not need `cv2.imshow()` or
native GUI windows, replace `opencv-python` with `opencv-python-headless` in `requirements.txt`.

Run the app

```bash
python virtual_painter.py
```

Controls
- Selection mode: raise index + middle finger and move hand into the header area to select a tool/color.
- Drawing mode: raise only the index finger to draw.
- Press `c` to clear the canvas.
- Press `q` to quit.

Files and structure
- `virtual_painter.py` — main application, header overlay, drawing logic, and UI.
- `hand_tracking_module.py` — reusable MediaPipe wrapper (`HandDetector`).
- `header/` — folder for header images (`brush1.png`, `brush2.png`, `brush3.png`, `eraser.png`).
- `requirements.txt` — Python package dependencies.

Customization
- Update `COLOR_PALETTE`, `BRUSH_THICKNESS`, and `ERASER_THICKNESS` in `virtual_painter.py` to change
	default colors and sizes.
- Drop your own header PNGs into the `header/` folder (recommended size ~125x640) to replace placeholders.

Notes and troubleshooting
- The most common runtime error when importing OpenCV in minimal containers is missing system
	graphics libraries (error complaining about `libGL.so.1`). See the Linux install step above or use
	the headless OpenCV wheel for non‑GUI environments.
- Webcam access and GUI windows may not function from cloud containers without additional
	configuration (USB passthrough, X server / display forwarding).
