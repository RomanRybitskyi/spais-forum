# Real-Time CPU Computer Vision Safety Monitor

End-to-end implementation of the plan in [`plan.md`](plan.md).

Runs a multi-stage OpenCV pipeline on a CPU-only laptop with a Logitech 720p
webcam: person detection + tracking, ROI presence/dwell analytics, gender
classification, helmet detection — with red-circle-on-bare-head and
thumbs-up-on-helmet overlays.

---

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For an extra ~2× speedup on Intel CPUs, additionally install:
```bash
pip install onnxruntime-openvino openvino
```
…and set `USE_OPENVINO_EP = True` in `config.py`.

---

## 2. Generate the thumbs-up asset

```bash
python -m tools.make_thumbs_up
```

---

## 3. Train / obtain the models

You will train (or download) **three** model files and drop them at:

| Path | Format | Notes |
|------|--------|-------|
| `models/person_detector/yolov8n.onnx` | ONNX | YOLOv8n COCO-pretrained, no fine-tuning needed |
| `models/helmet_detector/yolov8n_helmet.onnx` | ONNX | YOLOv8n fine-tuned on helmet dataset (2 classes) |
| `models/gender_classifier/gender_mnv3.onnx` | ONNX | MobileNetV3-Small fine-tuned on CelebA / Adience |

> The pipeline runs even when these files are missing — each stage falls back
> to a stub. You can therefore validate the rest of the system first, then drop
> in models incrementally.

### 3.1 Person detector — no training needed
```bash
# Download the pre-trained weights (any COCO-trained YOLOv8n will do)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Export to ONNX at 416 px (FP32; add --int8 if you have a calibration set)
python -m tools.export_onnx --weights yolov8n.pt \
    --output models/person_detector/yolov8n.onnx --imgsz 416
```

### 3.2 Helmet detector — fine-tune YOLOv8n on 2 classes
Classes (in this order):
- `0` → `helmet`
- `1` → `no_helmet`

Recommended dataset: **Safety Helmet Detection** (Roboflow Universe), already
labelled in YOLO format.

```bash
# Train (GPU recommended for training; CPU for inference is fine).
yolo detect train model=yolov8n.pt data=helmet.yaml imgsz=256 epochs=60 batch=32

# Export the resulting best.pt
python -m tools.export_onnx \
    --weights runs/detect/train/weights/best.pt \
    --output models/helmet_detector/yolov8n_helmet.onnx \
    --imgsz 256 --int8 --data helmet.yaml
```

### 3.3 Gender classifier
Two options:

**Option A (zero training, quick demo prototype)** — drop the public Caffe
gender_net into `models/gender_classifier/`:
```
gender_deploy.prototxt
gender_net.caffemodel
```
Set `USE_CAFFE_GENDER = True` in `config.py`.

**Option B (preferred)** — fine-tune MobileNetV3-Small on CelebA `Male`
attribute and export to ONNX (64×64 input, output logits of shape `[1, 2]`,
order `[Male, Female]`). Save as
`models/gender_classifier/gender_mnv3.onnx`.

A minimal training recipe (PyTorch):
```python
import torch
from torchvision import models, transforms, datasets
m = models.mobilenet_v3_small(weights="DEFAULT")
m.classifier[3] = torch.nn.Linear(1024, 2)
# train 10 epochs frozen backbone, 10 epochs full
# then:
dummy = torch.randn(1, 3, 64, 64)
torch.onnx.export(m, dummy, "gender_mnv3.onnx", opset_version=12,
                  input_names=["input"], output_names=["logits"])
```

---

## 4. Define the ROI zone

```bash
python -m tools.define_roi
```
Click 3+ vertices on the live feed → press `s` to save into `config.py`.

---

## 5. Run

```bash
python main.py
```
Hotkeys:
- `q` quit
- `a` toggle gender label anonymisation (privacy)
- `d` toggle debug HUD with per-stage latencies

---

## 6. Benchmark

```bash
python -m tools.benchmark --source 0 --frames 200
# or:
python -m tools.benchmark --source tests/sample_clips/demo.mp4
```

Prints average / p50 / p95 latency per stage and end-to-end FPS.

---

## 7. Tests

```bash
pytest -q
```

---

## Project layout

```
met/
├── main.py                    # entry point
├── config.py                  # all tuneables
├── plan.md                    # the design document
├── modules/
│   ├── detector.py            # YOLOv8n ONNX person detector
│   ├── tracker.py             # SORT
│   ├── gender.py              # ONNX or Caffe gender classifier
│   ├── helmet.py              # YOLOv8n ONNX helmet detector (head crop)
│   ├── zone.py                # ROI polygon + entry / dwell analytics
│   └── renderer.py            # OpenCV overlay drawing
├── tools/
│   ├── export_onnx.py         # YOLO → ONNX export helper
│   ├── define_roi.py          # interactive ROI definer (writes config.py)
│   ├── benchmark.py           # per-module FPS profiler
│   └── make_thumbs_up.py      # generates assets/thumbs_up.png
├── tests/
│   ├── test_zone.py
│   └── test_tracker.py
├── assets/
│   └── thumbs_up.png
└── models/
    ├── person_detector/       # yolov8n.onnx
    ├── helmet_detector/       # yolov8n_helmet.onnx
    └── gender_classifier/     # gender_mnv3.onnx (or Caffe files)
```

---

## Performance targets

| Mode | Expected end-to-end FPS |
|------|------------------------|
| Pure CPU, ONNX FP32 | ~17–22 FPS |
| CPU + INT8 quantization | ~22–28 FPS |
| CPU + OpenVINO EP | ~25–35 FPS |

If FPS drops below 15:
1. lower `PERSON_INPUT_SIZE` to 320 in `config.py`
2. raise `CLASSIFIER_FRAME_STRIDE` to 3 or 4
3. set `USE_OPENVINO_EP = True`
4. drop capture to 640×480 (`CAPTURE_WIDTH/HEIGHT`)

