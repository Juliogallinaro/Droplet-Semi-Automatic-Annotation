from __future__ import annotations

import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import ipywidgets as widgets
import numpy as np
import yaml
from IPython.display import display
from matplotlib import pyplot as plt


@dataclass
class PipelineConfig:
    video_path: str
    output_dir: str
    class_name: str = "droplet"
    val_split: float = 0.2
    frame_step: int = 30
    skip_empty: bool = True
    max_frames: int = 0
    template_frame: int = 10
    scale: int = 2
    threshold: float = 0.75
    nms_iou: float = 0.3
    seed: int = 42


@dataclass
class TemplateMatcher:
    template: np.ndarray
    template_radius: int
    scale: int
    threshold: float
    nms_iou: float


@dataclass
class ExtractionState:
    out_dir: Path
    tmp_dir: Path
    candidates: list
    saved: int
    skipped: int
    total_frames: int
    frame_width: int
    frame_height: int
    fps: float


@dataclass
class DatasetSplitResult:
    n_train: int
    n_val: int
    debug_samples: int


@dataclass
class MergeResult:
    copied: int
    conflicts: int
    train_images: int
    val_images: int


def select_template(video_path: str, template_frame: int, scale: int) -> tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, template_frame)
    ret, ref_frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {template_frame} from {video_path}")

    small = cv2.resize(ref_frame, (0, 0), fx=scale, fy=scale)

    print(f"[INFO] Window opened at frame {template_frame}")
    print("       Draw a rectangle around ONE droplet and press ENTER.")

    roi = cv2.selectROI("Select one droplet - press ENTER to confirm", small, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, tw, th = roi
    if tw == 0 or th == 0:
        raise ValueError("No region selected.")

    template = cv2.cvtColor(small[y : y + th, x : x + tw], cv2.COLOR_BGR2GRAY)
    template_radius = int(min(tw, th) / 2 / scale)

    plt.figure(figsize=(3, 3))
    plt.imshow(template, cmap="gray")
    plt.title(f"Template: {tw}x{th} px  |  original radius ~ {template_radius} px")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"Template ready - original radius ~ {template_radius} px")
    return template, template_radius


def detect_droplets(frame: np.ndarray, matcher: TemplateMatcher) -> list[tuple[int, int, int]]:
    small = cv2.resize(frame, (0, 0), fx=matcher.scale, fy=matcher.scale)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray, matcher.template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= matcher.threshold)

    tw, th = matcher.template.shape[1], matcher.template.shape[0]
    boxes = [[pt[0], pt[1], pt[0] + tw, pt[1] + th] for pt in zip(*locations[::-1])]
    scores = result[locations].tolist()

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, matcher.threshold, matcher.nms_iou)
    if len(indices) == 0:
        return []

    droplets = []
    for idx in np.array(indices).flatten():
        x1, y1, x2, y2 = boxes[int(idx)]
        cx = int((x1 + x2) / 2 / matcher.scale)
        cy = int((y1 + y2) / 2 / matcher.scale)
        droplets.append((cx, cy, matcher.template_radius))

    return droplets


def preview_frame(video_path: str, frame_number: int, matcher: TemplateMatcher) -> None:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read frame {frame_number}")
        return

    droplets = detect_droplets(frame, matcher)
    ann = frame.copy()
    for x, y, r in droplets:
        cv2.circle(ann, (x, y), r, (0, 255, 0), 2)
        cv2.circle(ann, (x, y), 3, (0, 0, 255), -1)

    plt.figure(figsize=(14, 4))
    plt.imshow(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
    plt.title(
        f"Frame {frame_number} - {len(droplets)} droplet(s) | threshold={matcher.threshold}",
        fontsize=13,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    print(f"Detected droplets: {len(droplets)}")


def circle_to_yolo(x: int, y: int, r: int, img_w: int, img_h: int, class_id: int = 0) -> str:
    x1, y1 = max(0, x - r), max(0, y - r)
    x2, y2 = min(img_w, x + r), min(img_h, y + r)
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def create_folder_structure(base: Path) -> None:
    for split in ("train", "val"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)
    (base / "debug_preview").mkdir(exist_ok=True)


def extract_candidates(config: PipelineConfig, matcher: TemplateMatcher) -> ExtractionState:
    random.seed(config.seed)

    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {config.video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video      : {config.video_path}")
    print(f"Frames     : {total}  |  FPS: {fps:.1f}")
    print(f"Resolution : {width}x{height}")
    print(f"Step       : {config.frame_step}  ->  ~{total // config.frame_step} candidate frames")
    print()

    out_dir = Path(config.output_dir)
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    create_folder_structure(out_dir)

    candidates = []
    frame_idx = 0
    saved = 0
    skipped = 0

    progress = widgets.IntProgress(
        value=0,
        min=0,
        max=total,
        description="Processing:",
        bar_style="info",
        layout=widgets.Layout(width="600px"),
    )
    label_w = widgets.Label(value="")
    display(widgets.HBox([progress, label_w]))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        progress.value = frame_idx

        if frame_idx % config.frame_step == 0:
            droplets = detect_droplets(frame, matcher)

            if config.skip_empty and not droplets:
                skipped += 1
            else:
                stem = f"frame_{frame_idx:07d}"
                tmp = tmp_dir / f"{stem}.jpg"
                cv2.imwrite(str(tmp), frame)
                lines = [circle_to_yolo(x, y, r, width, height) for x, y, r in droplets]
                candidates.append((tmp, stem, lines, droplets))
                saved += 1
                label_w.value = f"frame {frame_idx} | {saved} saved"

            if config.max_frames and saved >= config.max_frames:
                print(f"Limit of {config.max_frames} frames reached.")
                break

        frame_idx += 1

    cap.release()
    progress.value = total
    progress.bar_style = "success"

    print(f"\nExtracted frames  : {saved}")
    print(f"Skipped frames    : {skipped}  (no detections)")
    print(f"Total droplets    : {sum(len(c[3]) for c in candidates)}")

    return ExtractionState(
        out_dir=out_dir,
        tmp_dir=tmp_dir,
        candidates=candidates,
        saved=saved,
        skipped=skipped,
        total_frames=total,
        frame_width=width,
        frame_height=height,
        fps=fps,
    )


def write_dataset(config: PipelineConfig, state: ExtractionState) -> DatasetSplitResult:
    if not state.candidates:
        raise RuntimeError("No candidate frames were generated. Check threshold and frame step.")

    random.seed(config.seed)
    random.shuffle(state.candidates)

    n_val = max(1, int(len(state.candidates) * config.val_split))
    n_train = len(state.candidates) - n_val

    splits = (
        [("train", c) for c in state.candidates[:n_train]]
        + [("val", c) for c in state.candidates[n_train:]]
    )

    for split in ["train", "val"]:
        (state.out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (state.out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    debug_indices = set(random.sample(range(len(splits)), min(10, len(splits))))

    for i, (split, (tmp_img, stem, yolo_lines, droplets)) in enumerate(splits):
        dst_img = state.out_dir / "images" / split / f"{stem}.jpg"
        shutil.move(str(tmp_img), str(dst_img))
        (state.out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(yolo_lines))

        if i in debug_indices:
            ann = cv2.imread(str(dst_img))
            for x, y, r in droplets:
                cv2.circle(ann, (x, y), r, (0, 255, 0), 2)
                cv2.circle(ann, (x, y), 2, (0, 0, 255), -1)
            cv2.putText(
                ann,
                f"{split} | {len(droplets)} droplets",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(state.out_dir / "debug_preview" / f"{stem}.jpg"), ann)

    if state.tmp_dir.exists():
        shutil.rmtree(state.tmp_dir)

    yaml_content = {
        "path": "/dataset",
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": [config.class_name],
    }
    with open(state.out_dir / "dataset.yaml", "w", encoding="utf-8") as file_obj:
        yaml.dump(yaml_content, file_obj, default_flow_style=False, allow_unicode=True)

    debug_samples = min(10, len(splits))
    print(f"Split  ->  train: {n_train}  |  val: {n_val}")
    print(f"Debug: {debug_samples} samples at {(state.out_dir / 'debug_preview').resolve()}")
    print(f"Dataset saved at: {state.out_dir.resolve()}")

    return DatasetSplitResult(n_train=n_train, n_val=n_val, debug_samples=debug_samples)


def show_debug_samples(out_dir: Path, max_samples: int = 6) -> None:
    debug_imgs = sorted((out_dir / "debug_preview").glob("*.jpg"))

    if not debug_imgs:
        print("No debug images found.")
        return

    sample = debug_imgs[:max_samples]
    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 3))
    if len(sample) == 1:
        axes = [axes]

    for ax, path in zip(axes, sample):
        img = cv2.imread(str(path))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(path.stem, fontsize=8)
        ax.axis("off")

    plt.suptitle("Dataset sample (debug_preview)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.show()
    print(f"Total debug images: {len(debug_imgs)}")


def dataset_summary(config: PipelineConfig, out_dir: Path) -> str:
    train_imgs = list((out_dir / "images" / "train").glob("*.jpg"))
    val_imgs = list((out_dir / "images" / "val").glob("*.jpg"))

    lines = [
        "=" * 50,
        "  DATASET SUMMARY",
        "=" * 50,
        f"  Class        : {config.class_name}",
        f"  Train        : {len(train_imgs)} images",
        f"  Val          : {len(val_imgs)} images",
        f"  Total        : {len(train_imgs) + len(val_imgs)} images",
        f"  Template     : frame {config.template_frame}  |  threshold={config.threshold}",
        f"  dataset.yaml : {(out_dir / 'dataset.yaml').resolve()}",
        "=" * 50,
        "",
        "To train with YOLOv8 (recommended):",
        "  pip install ultralytics",
        f"  yolo train model=yolov8n.pt data={out_dir}/dataset.yaml epochs=100 imgsz=640",
        "",
        "To train with YOLOv5:",
        f"  python train.py --data {out_dir}/dataset.yaml --weights yolov5s.pt --epochs 100",
    ]
    return "\n".join(lines)


def make_cvat_zip(out_dir: Path, split: str, class_name: str, zip_path: Path) -> int:
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    pairs = [(path.stem, lbl_dir / (path.stem + ".txt")) for path in sorted(img_dir.glob("*.jpg"))]

    folder = "obj_train_data" if split == "train" else "obj_valid_data"
    list_file = "train.txt" if split == "train" else "valid.txt"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("obj.names", class_name + "\n")
        zip_file.writestr(
            "obj.data",
            "classes = 1\n"
            f"train   = data/{list_file}\n"
            f"valid   = data/{list_file}\n"
            "names   = data/obj.names\n"
            "backup  = backup/\n",
        )
        file_list = "\n".join(f"data/{folder}/{stem}.jpg" for stem, _ in pairs)
        zip_file.writestr(list_file, file_list + "\n")
        for stem, label_path in pairs:
            content = label_path.read_text() if label_path.exists() else ""
            zip_file.writestr(f"{folder}/{stem}.txt", content)

    return len(pairs)


def merge_datasets(dataset_dirs: list[str], output_dir: str) -> MergeResult:
    out = Path(output_dir)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    copied = 0
    conflicts = 0

    for run_dir in dataset_dirs:
        run = Path(run_dir)
        if not run.exists():
            print(f"Dataset folder not found: {run_dir}")
            continue

        prefix = run.name
        for split in ("train", "val"):
            for img in sorted((run / "images" / split).glob("*.jpg")):
                lbl = run / "labels" / split / f"{img.stem}.txt"

                dst_img = out / "images" / split / img.name
                dst_lbl = out / "labels" / split / f"{img.stem}.txt"

                if dst_img.exists():
                    dst_img = out / "images" / split / f"{prefix}_{img.name}"
                    dst_lbl = out / "labels" / split / f"{prefix}_{img.stem}.txt"
                    conflicts += 1

                shutil.copy2(img, dst_img)
                if lbl.exists():
                    shutil.copy2(lbl, dst_lbl)
                else:
                    dst_lbl.write_text("")
                copied += 1

    train_images = len(list((out / "images" / "train").glob("*.jpg")))
    val_images = len(list((out / "images" / "val").glob("*.jpg")))

    return MergeResult(
        copied=copied,
        conflicts=conflicts,
        train_images=train_images,
        val_images=val_images,
    )


def write_dataset_yaml(out_dir: str, class_name: str) -> Path:
    out = Path(out_dir)
    yaml_content = {
        "path": "/dataset",
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": [class_name],
    }

    yaml_path = out / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as file_obj:
        yaml.dump(yaml_content, file_obj, default_flow_style=False, allow_unicode=True)

    return yaml_path


def export_cvat_archives(out_dir: str, class_name: str) -> tuple[int, int]:
    base = Path(out_dir)
    train_count = make_cvat_zip(base, "train", class_name, base / "cvat_train.zip")
    val_count = make_cvat_zip(base, "val", class_name, base / "cvat_val.zip")
    return train_count, val_count
