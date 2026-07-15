# Deep Learning for Microfluidic Droplet Characterization via Semi-Automatic Annotation

[![DOI](https://zenodo.org/badge/1284413061.svg)](https://doi.org/10.5281/zenodo.21049182)


Automated pipeline for **detection, tracking, counting, and sizing** of microfluidic droplets in video microscopy. A semi-automatic annotation stage based on OpenCV template matching generates a YOLO-format dataset, which is used to train a YOLOv8n model. At inference time, ByteTrack associates detections across frames while a virtual counting line and a pixel-to-µm calibration yield per-droplet diameter measurements and production-rate statistics.

---

## Repository structure

```
.
├── dataset_droplets_template.ipynb   # Stage 1 — Semi-automatic annotation & dataset creation
├── merge_and_export.ipynb            # Stage 2 — Merge multiple datasets & export CVAT archives
├── droplet_tracking_counting.ipynb   # Stage 3 — Inference, tracking, counting & sizing
├── batch_tracking.ipynb              # Stage 4 — Batch tracking/counting for all videos in a folder
├── droplet_annotation/               # Python package: Annotation utilities
│   └── pipeline.py
├── droplet_tracking/                 # Python package: YOLOv8 + ByteTrack pipeline
│   └── pipeline.py
├── requirements.txt
└── README.md
```

---

## Workflow overview

```
Video (.avi, .mp4, .mov, .mkv)
    │
    ▼
[1] Semi-automatic annotation          dataset_droplets_template.ipynb
    └─ User selects one reference droplet
    └─ matchTemplate scans sampled frames
    └─ NMS + train/val split
    └─ YOLO labels + dataset.yaml
    │
    ▼
[2] Merge datasets (optional)          merge_and_export.ipynb
    └─ Merge multiple YOLO datasets
    └─ Resolve filename conflicts
    └─ Generate merged dataset.yaml
    └─ Export CVAT YOLO 1.1 archives
    │
    ▼
[3] Model training                     Docker / ultralytics CLI
    └─ YOLOv8n fine-tuned on dataset
    └─ best.pt saved in runs/
    │
    ▼
[4] Tracking & characterization        droplet_tracking_counting.ipynb
    └─ YOLOv8 detection per frame
    └─ ByteTrack association
    └─ Pixel calibration (scale bar)
    └─ Virtual counting line
    └─ Per-droplet diameter + production rate
    │
    ▼
[5] Batch mode (optional)             batch_tracking.ipynb
    └─ Configure one model + one input folder
    └─ Calibrate once (um/px)
    └─ Define counting line once
    └─ Process all videos automatically
    └─ Export batch_summary.csv + all_droplets.csv
```

---

## Stage 1 — Semi-automatic annotation (`dataset_droplets_template.ipynb`)

### What it does

1. The user draws a rectangle around **one representative droplet** in a chosen frame.
2. `TemplateMatcher` runs `cv2.matchTemplate` on every sampled frame (`frame_step` interval) and applies non-maximum suppression (NMS).
3. Detections are converted to YOLO bounding-box format and split into train / val sets.
4. A `dataset.yaml` and optional CVAT-compatible ZIP exports are written to disk.

### Output structure

```
dataset/
├── images/train/
├── images/val/
├── labels/train/
├── labels/val/
├── debug_preview/    ← visually verify detections here
└── dataset.yaml
```

### Key configuration (`PipelineConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `video_path` | — | Path to the input video |
| `output_dir` | — | Destination folder for the dataset |
| `class_name` | `"droplet"` | YOLO class label |
| `val_split` | `0.2` | Fraction of frames reserved for validation |
| `frame_step` | `30` | Sample every N frames |
| `template_frame` | `10` | Frame used for template selection |
| `scale` | `2` | Display zoom factor for the selection UI |
| `threshold` | `0.75` | Minimum `matchTemplate` correlation score |
| `nms_iou` | `0.3` | IoU threshold for NMS |
| `skip_empty` | `True` | Discard frames with no detections |
| `max_frames` | `0` | `0` = use all frames |

### CVAT export

The pipeline also exports `cvat_train.zip` and `cvat_val.zip` in **YOLO 1.1** format, ready for import into CVAT for manual review and annotation refinement.

---
## Stage 2 — Dataset Merge and CVAT Export (`merge_and_export.ipynb`)

When multiple videos are annotated independently, this notebook combines the resulting YOLO datasets into a single dataset while preserving the original **train/validation** split.

The merged dataset is fully compatible with **Ultralytics YOLO** and can also be exported as **CVAT YOLO 1.1** archives for manual inspection, correction, or additional annotation.

### What it does

1. Merges images and labels from multiple YOLO datasets.
2. Automatically resolves filename conflicts by renaming duplicate files.
3. Preserves the original **train** and **validation** directory structure.
4. Generates a new `dataset.yaml`.
5. Exports separate CVAT annotation archives for the training and validation splits.

### Output structure

```text
dataset_merged/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── dataset.yaml
├── cvat_train.zip
└── cvat_val.zip
```

### Configuration

| Parameter      | Description                                                   |
| -------------- | ------------------------------------------------------------- |
| `DATASET_DIRS` | List of YOLO datasets to merge                                |
| `OUTPUT_DIR`   | Destination folder for the merged dataset                     |
| `CLASS_NAME`   | Class name written to `dataset.yaml` and used for CVAT export |

### CVAT export

The pipeline also exports `cvat_train.zip` and `cvat_val.zip` in **YOLO 1.1** format, ready for import into CVAT for manual review and annotation refinement.

---

## Stage 3 — Model training

Training is executed inside the official Ultralytics Docker image to ensure a reproducible environment.

```powershell
docker run --rm --gpus all `
    --shm-size=16g `
    -v "<PATH_TO_DATASET>:/dataset" `
    -v "<PATH_TO_RUNS>:/runs" `
    ultralytics/ultralytics:8.4.82 `
    yolo train `
        model=yolov8n.pt `
        data=/dataset/dataset.yaml `
        epochs=100 `
        imgsz=640 `
        project=/runs `
        name=droplets_v1 `
        workers=4 `
        batch=8 `
        patience=10
```

**Environment:** Ultralytics v8.4.82 · PyTorch 2.11.0 · CUDA · Ubuntu 24.04  
**Best weights** are saved to `runs/droplets_v1/weights/best.pt`.

---

## Stage 4 — Tracking & characterization (`droplet_tracking_counting.ipynb`)

### What it does

1. **Detection** — YOLOv8 runs inference on every frame (`best.pt`).
2. **Tracking** — ByteTrack links detections into persistent trajectories with unique IDs.
3. **Calibration** — the user clicks two points on a known scale bar; the notebook derives `µm/px`.
4. **Counting line** — the user draws a virtual line; any track that crosses it is counted and its bounding-box diameter is recorded in µm.
5. **Output** — annotated video (IDs, trails, real-time counter) and a summary DataFrame with per-droplet diameter and production-rate statistics (droplets / 10 s window).

### Key configuration (`TrackingConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conf` | `0.1` | Minimum detection confidence |
| `imgsz` | `640` | Inference image size (px) |
| `track_thresh` | `0.25` | ByteTrack activation threshold |
| `track_buffer` | `30` | Frames to keep a lost track alive |
| `match_thresh` | `0.80` | IoU threshold for track–detection matching |
| `frame_rate` | `25` | Video frame rate (fps) |
| `min_frames` | `2` | Minimum consecutive detections to confirm a track |
| `line_margin_px` | `2.0` | Tolerance around the counting line (px) |

### Outputs

| File / object | Description |
|---------------|-------------|
| `*_tracked.avi` | Annotated video with IDs, trails, and counter overlay |
| `df_line` | DataFrame: timestamp, track ID, diameter (µm), production rate |

---

## Stage 5 — Batch Tracking & Counting (`batch_tracking.ipynb`)

Use this notebook when you want to process **all videos in a folder** using the same calibration and counting line.

### Batch workflow

1. Configure model path, input/output folders, and tracking parameters.
2. Calibrate once from a reference image/video to get `um_per_px`.
3. Define the counting line once on a representative frame.
4. Run batch processing over all compatible videos.
5. Aggregate results into summary tables and CSV files.

### Main settings

| Parameter | Description |
|-----------|-------------|
| `MODEL_PATH` | Path to trained YOLO weights (`best.pt`) |
| `INPUT_FOLDER` | Folder containing videos to process |
| `OUTPUT_FOLDER` | Folder where tracked videos and CSVs are saved |
| `CAL_VIDEO_PATH` | Reference file used for calibration UI |
| `VIDEO_EXTS` | Accepted input extensions (`.avi`, `.mp4`, `.mov`, `.mkv`) |
| `NAME_FILTER` | Optional filename substring filter (`""` = process all) |
| `TrackingConfig(...)` | Detection/ByteTrack/visual style parameters |

### Expected outputs

| File / object | Description |
|---------------|-------------|
| `*_tracked.avi` | One annotated output video per processed input |
| `df_batch` | Per-video batch summary shown in notebook |
| `df_all_drops` | Combined droplet table across all videos |
| `batch_summary.csv` | Exported per-video summary |
| `all_droplets.csv` | Exported merged per-droplet table |

### Notes

- The model is loaded once and reused during the loop.
- Errors are handled per video so a single failure does not stop the full batch.
- If no counting line is confirmed, batch execution is interrupted with validation error.

---

## Requirements

Install dependencies (inside each notebook or manually):

```bash
pip install -r requirements.txt
```

Core dependencies: `ultralytics`, `supervision`, `opencv-python`, `numpy`, `matplotlib`, `ipywidgets`.

---

## Citation

If you use this pipeline in your work, please cite:


```bibtex
@software{maranho2026code,
  author    = {Maranho, J{\'u}lio Gallinaro and {Silva Jr.}, Jo{\~a}o Lameu},
  title     = {{Droplet Semi-Automatic Annotation}},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.21049183},
  url       = {https://doi.org/10.5281/zenodo.21049183}
}
```

