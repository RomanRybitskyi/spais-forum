
## Real-Time CPU Computer Vision Safety Monitor

A multi-stage OpenCV pipeline running entirely on CPU, ingesting a 720p webcam feed and performing: person detection + tracking, ROI zone analytics (entry count + dwell time), binary gender classification, and helmet detection — all rendered live with annotated overlays. Every model is selected and optimized specifically for CPU-only real-time inference.

---

### Project Folder Structure

```
met/
├── main.py                  # entry point & main loop
├── config.py                # all tuneable constants (ROI points, thresholds, paths)
├── requirements.txt
├── assets/
│   └── thumbs_up.png        # RGBA like-icon overlay image
├── models/
│   ├── person_detector/     # YOLOv8n ONNX weights
│   ├── gender_classifier/   # MobileNetV3-Small ONNX weights
│   └── helmet_detector/     # YOLOv8n-helmet ONNX weights
├── modules/
│   ├── detector.py          # person detection wrapper
│   ├── tracker.py           # SORT tracker wrapper
│   ├── gender.py            # gender classifier wrapper
│   ├── helmet.py            # helmet detector wrapper
│   ├── zone.py              # ROI polygon, entry counting, dwell time
│   └── renderer.py          # all OpenCV drawing / overlay logic
├── tools/
│   ├── export_onnx.py       # Ultralytics → ONNX export helper
│   ├── define_roi.py        # interactive ROI polygon tool
│   └── benchmark.py         # per-module FPS profiler
└── tests/
    ├── test_zone.py
    ├── test_tracker.py
    └── sample_clips/
```

---

### 1. Environment & Libraries

**`requirements.txt`:**
```
opencv-python==4.9.*
onnxruntime==1.18.*
numpy==1.26.*
scipy==1.13.*
filterpy==1.4.*
Pillow==10.*
ultralytics==8.2.*          # training/export only
openvino==2024.*            # optional, for OpenVINO EP speedup on Intel CPUs
```

---

### 2. Model Selection & Optimization

#### 2a. Person Detection — YOLOv8n (COCO pre-trained)
- Use `yolov8n.pt` (pre-trained, no fine-tuning needed — COCO covers people)
- Export to **ONNX INT8**: `ultralytics export format=onnx int8=True imgsz=416`
- Inference via `onnxruntime`, input resized to **416×416**
- Estimated speed: **~20–28 FPS on CPU**
- Fallback: drop to `imgsz=320` for ~30 FPS

#### 2b. Multi-Object Tracking — SORT
- Pure Python + Kalman filter, **< 1 ms overhead per frame**
- Why not DeepSORT/ByteTrack: both require a re-ID embedding net (too slow on CPU)
- Each person gets a persistent `track_id` used for dwell time & entry counting

#### 2c. Gender Classification — MobileNetV3-Small
- **Option A (quick)**: Use the pre-built Caffe `gender_net.caffemodel` loadable directly via `cv2.dnn` — no training, ~90% frontal accuracy, zero setup
- **Option B (better)**: Fine-tune `torchvision.models.mobilenet_v3_small` on **CelebA** (Male attribute, 200 k images) or **Adience Benchmark** (~26 k images), export to ONNX FP16 at 64×64 px
  - 10 epochs frozen backbone → 10 epochs full fine-tune
- Inference input: cropped person bounding box resized to 64×64
- Estimated cost: **< 5 ms per frame** for up to 10 persons

#### 2d. Helmet Detection — YOLOv8n fine-tuned
- Dataset: **Safety Helmet Detection** on [Roboflow Universe](https://universe.roboflow.com) (~5 k images, classes: `helmet`, `no_helmet`)
- Fine-tune YOLOv8n from COCO weights on this dataset
- Export to **ONNX INT8** at `imgsz=256`
- Input: **head crop** = top 25–30% of each person bounding box
- Estimated cost: **~4–6 ms per frame**

#### 2e. Global Optimizations
- All models via `onnxruntime.InferenceSession` with `inter_op_num_threads = os.cpu_count() // 2`
- Enable **OpenVINO Execution Provider** on Intel CPUs: `providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"]` → additional ~2× speedup
- **Frame skipping**: gender & helmet classifiers run every **2nd frame**, results cached per `track_id`
- Detection runs every frame; tracking runs every frame

---

### 3. ROI Zone Definition & Analytics — `modules/zone.py`

#### Zone Definition
- Convex polygon: list of `(x, y)` pixel points in `config.py`
- Interactive setup: `tools/define_roi.py` — click to define 4–6 vertices via `cv2.setMouseCallback`, auto-saves to `config.py`
- Per-frame test: `cv2.pointPolygonTest(zone_contour, centroid, False)` where centroid = bottom-center of bounding box

#### Entry Counting
```python
entered_ids = set()   # track IDs that have ever been inside zone

for track in tracks:
    if inside_zone(track.centroid) and track.id not in entered_ids:
        total_count += 1
        entered_ids.add(track.id)
```

#### Dwell Time
```python
entry_times = {}   # {track_id: time.time() at entry}
dwell_log = {}     # {track_id: total_seconds}

# on zone entry:
entry_times[track_id] = time.time()

# on zone exit or track lost:
dwell_log[track_id] = time.time() - entry_times.pop(track_id)

# display:
avg_dwell = mean(dwell_log.values())
```

---

### 4. Overlay Rendering — `modules/renderer.py`

| Condition | Visual |
|-----------|--------|
| No helmet | `cv2.circle(frame, head_center, radius, (0,0,255), 2)` — **red circle** around head |
| Helmet detected | Alpha-blend `assets/thumbs_up.png` (40×40 RGBA) above head center |
| In-zone person | Bounding box in **green** |
| Out-of-zone | Bounding box in **white** |
| HUD (top-left) | Semi-transparent panel: `Total entered: N` · `Avg dwell: X.Xs` · FPS counter |

---

### 5. Main Pipeline Loop (`main.py`)

```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_idx = 0
gender_cache, helmet_cache = {}, {}

while True:
    ret, frame = cap.read()
    frame_small = cv2.resize(frame, (416, 416))        # for detection
    detections = detector.run(frame_small)
    boxes = scale_boxes(detections, frame.shape)       # back to 720p coords
    tracks = tracker.update(boxes)                     # SORT

    if frame_idx % 2 == 0:                            # every 2nd frame
        for track in tracks:
            gender_cache[track.id] = gender.run(crop(frame, track.box))
            helmet_cache[track.id] = helmet.run(head_crop(frame, track.box))

    zone.update(tracks)
    renderer.draw(frame, tracks, gender_cache, helmet_cache, zone.stats())
    cv2.imshow("Safety Monitor", frame)
    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

### 6. Expected FPS Budget (Intel Core i5/i7, 8 logical cores, no GPU)

| Stage | Cost/frame | Notes |
|-------|-----------|-------|
| Capture + resize | ~1 ms | negligible |
| YOLOv8n ONNX INT8 @ 416px | ~35–50 ms | **bottleneck** |
| SORT tracking | < 1 ms | — |
| Gender every 2nd frame | ~3 ms amortised | 10 persons |
| Helmet every 2nd frame | ~5 ms amortised | head crops only |
| Rendering | ~2 ms | — |
| **End-to-end** | **~46–60 ms** | **~17–22 FPS** |

With OpenVINO EP: **~25–35 FPS**

---

### 7. Implementation Roadmap

| Milestone | Tasks | Est. Time |
|-----------|-------|-----------|
| M1 – Scaffold | Folders, `config.py`, webcam loop, ROI click-to-define | 1 day |
| M2 – Detect + Track | YOLOv8n ONNX + SORT, verify ≥ 20 FPS | 2 days |
| M3 – Zone Analytics | Polygon test, entry counter, dwell time, HUD | 1 day |
| M4 – Gender | Caffe model prototype → optionally fine-tune MobileNetV3 | 2 days |
| M5 – Helmet | Download dataset, fine-tune YOLOv8n, ONNX export, integrate | 3 days |
| M6 – Rendering | Red circle, thumbs-up alpha blend, polished HUD | 1 day |
| M7 – Optimization | OpenVINO EP, INT8 quant, frame-skip tuning, benchmark | 1 day |
| M8 – Testing | Unit tests, offline benchmark, conference dry-run | 1 day |
| **Total** | | **~12 days** |

---

### 8. Testing & Validation

- **Unit tests**: `test_zone.py` — synthetic centroids, assert entry count & dwell math; `test_tracker.py` — mock boxes, assert track ID continuity
- **Offline benchmark** (`tools/benchmark.py`): run on pre-recorded 720p clip, measure per-module latency with `time.perf_counter`
- **Accuracy spot-check**: record 50-frame clips, manually label gender/helmet, compute accuracy vs. model output
- **Conference dry-run**: test at venue 1 hour before presentation, verify FPS ≥ 15

---

### 9. Fallback Strategies if FPS < 15

1. Reduce detection input to **320×320** → ~+8 FPS
2. Run detection every **2nd frame**, propagate boxes via SORT predictions on skip frames
3. **Disable gender classification** (least critical for safety demo)
4. Lower capture to **640×480**
5. Enable **OpenVINO EP** (`pip install openvino`)

---

### 10. Additional Considerations

- **Gender model**: try the Caffe model first (zero training, `cv2.dnn` compatible) before committing to fine-tuning
- **Helmet dataset bias**: Roboflow dataset is construction-site-oriented — if conference helmets look different, add 200–300 in-domain samples and re-fine-tune
- **Privacy**: displaying live gender labels in public may require a visible consent/disclaimer notice at the booth; add a toggle in `config.py` to anonymise overlays if needed
