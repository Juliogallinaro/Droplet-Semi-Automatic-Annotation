from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import supervision as sv
from IPython.display import clear_output, display
from ultralytics import YOLO


@dataclass
class TrackingConfig:
    conf: float = 0.1
    imgsz: int = 640
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.80
    frame_rate: int = 25
    min_frames: int = 2
    line_margin_px: float = 2.0


def load_model_and_video_info(model_path: str, video_path: str) -> tuple[YOLO, sv.VideoInfo]:
    model = YOLO(model_path)
    video_info = sv.VideoInfo.from_video_path(video_path)

    print(f"Model      : {model_path}")
    print(f"Video      : {video_path}")
    print(f"Resolution : {video_info.width}x{video_info.height}")
    print(f"FPS        : {video_info.fps}")
    print(f"Frames     : {video_info.total_frames}")
    print(f"Duration   : {video_info.total_frames / video_info.fps:.1f}s")

    return model, video_info


def create_calibration_ui(video_path: str, default_reference_um: float = 100.0) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read first frame from: {video_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    state: dict[str, Any] = {
        "points": [],
        "um_per_px": None,
        "figure": None,
        "axis": None,
    }

    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.canvas.header_visible = False
    ax.imshow(frame_rgb)
    ax.set_title("Click the 2 endpoints of the scale bar", fontsize=11)
    ax.axis("off")
    fig.tight_layout(pad=0)

    ref_input = widgets.BoundedFloatText(
        value=default_reference_um,
        min=0.1,
        max=10000.0,
        step=1.0,
        description="Reference:",
        layout=widgets.Layout(width="180px"),
        style={"description_width": "70px"},
    )
    btn_reset = widgets.Button(
        description="Reset",
        button_style="warning",
        layout=widgets.Layout(width="100px"),
    )
    btn_ok = widgets.Button(
        description="Confirm",
        button_style="success",
        layout=widgets.Layout(width="100px"),
    )
    result_out = widgets.Output()

    def redraw() -> None:
        ax.cla()
        ax.imshow(frame_rgb)
        ax.set_title("Click the 2 endpoints of the scale bar", fontsize=11)
        ax.axis("off")

        for idx, (x, y) in enumerate(state["points"], start=1):
            ax.plot(x, y, "+", color="yellow", markersize=16, markeredgewidth=2)
            ax.annotate(
                str(idx),
                (x, y),
                color="yellow",
                fontsize=10,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        if len(state["points"]) == 2:
            x1, y1 = state["points"][0]
            x2, y2 = state["points"][1]
            d_px = float(np.hypot(x2 - x1, y2 - y1))
            ref = ref_input.value
            ax.plot([x1, x2], [y1, y2], "-", color="cyan", linewidth=2)
            ax.set_title(
                f"{d_px:.1f} px = {ref:.0f} um  ->  {ref / d_px:.4f} um/px  |  click Confirm",
                fontsize=11,
                color="black",
            )

        fig.canvas.draw_idle()

    def onclick(event: Any) -> None:
        if event.inaxes != ax or len(state["points"]) >= 2:
            return
        x, y = int(event.xdata), int(event.ydata)
        state["points"].append((x, y))
        redraw()

    def on_reset(_: Any) -> None:
        state["points"].clear()
        state["um_per_px"] = None
        redraw()
        with result_out:
            clear_output()

    def on_confirm(_: Any) -> None:
        with result_out:
            clear_output()
            if len(state["points"]) < 2:
                print("Select 2 points first.")
                return
            x1, y1 = state["points"][0]
            x2, y2 = state["points"][1]
            d_px = float(np.hypot(x2 - x1, y2 - y1))
            state["um_per_px"] = ref_input.value / d_px
            print(f"Distance    : {d_px:.2f} px")
            print(f"Reference   : {ref_input.value:.1f} um")
            print(f"Scale       : {state['um_per_px']:.4f} um/px")

    fig.canvas.mpl_connect("button_press_event", onclick)
    btn_reset.on_click(on_reset)
    btn_ok.on_click(on_confirm)

    controls = widgets.VBox(
        [
            widgets.HBox([ref_input, widgets.Label("um"), btn_reset, btn_ok]),
            result_out,
        ],
        layout=widgets.Layout(padding="5px"),
    )

    state["figure"] = fig
    state["axis"] = ax
    display(controls)
    plt.show()

    return state


def create_line_selector_ui(video_path: str) -> dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read first frame from: {video_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]

    state: dict[str, Any] = {
        "click": None,
        "line_start": None,
        "line_end": None,
        "orientation": "vertical",
        "length_percent": 80,
    }

    plt.close("all")
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.canvas.header_visible = False
    ax.imshow(frame_rgb)
    ax.set_title("Click to position the line", fontsize=10)
    ax.axis("off")
    plt.tight_layout()

    line_artist = [None]

    def draw_line() -> tuple[int, int, int, int] | None:
        if state["click"] is None:
            return None
        cx, cy = state["click"]
        size = state["length_percent"] / 100
        orientation = state["orientation"]

        if orientation == "vertical":
            half = int(height * size / 2)
            x1, y1 = cx, max(0, cy - half)
            x2, y2 = cx, min(height, cy + half)
        else:
            half = int(width * size / 2)
            x1, y1 = max(0, cx - half), cy
            x2, y2 = min(width, cx + half), cy

        if line_artist[0] is not None:
            line_artist[0].remove()
        line_artist[0], = ax.plot([x1, x2], [y1, y2], "-", color="lime", linewidth=2)
        ax.set_title(f"({x1},{y1}) -> ({x2},{y2})  |  click to move", fontsize=9, color="lime")
        fig.canvas.draw_idle()
        return x1, y1, x2, y2

    def onclick(event: Any) -> None:
        if event.inaxes != ax:
            return
        state["click"] = (int(event.xdata), int(event.ydata))
        draw_line()

    toggle_ori = widgets.ToggleButtons(
        options=["vertical", "horizontal"],
        value="vertical",
        description="",
        button_style="info",
        layout=widgets.Layout(width="220px"),
    )
    slider_size = widgets.IntSlider(
        value=80,
        min=5,
        max=100,
        step=1,
        description="Length %:",
        continuous_update=True,
        layout=widgets.Layout(width="400px"),
        style={"description_width": "90px"},
    )
    btn_ok = widgets.Button(
        description="Confirm",
        button_style="success",
        layout=widgets.Layout(width="150px"),
    )
    result_out = widgets.Output()

    def on_ori_change(_: Any) -> None:
        state["orientation"] = toggle_ori.value
        draw_line()

    def on_size_change(_: Any) -> None:
        state["length_percent"] = slider_size.value
        draw_line()

    def on_confirm(_: Any) -> None:
        with result_out:
            clear_output()
            if state["click"] is None:
                print("Click on the image first.")
                return
            coords = draw_line()
            if coords is None:
                return
            x1, y1, x2, y2 = coords
            state["line_start"] = sv.Point(x1, y1)
            state["line_end"] = sv.Point(x2, y2)
            print(f"Orientation : {state['orientation']}")
            print(f"Point 1     : ({x1}, {y1}) px")
            print(f"Point 2     : ({x2}, {y2}) px")
            print("Line confirmed")

    fig.canvas.mpl_connect("button_press_event", onclick)
    toggle_ori.observe(on_ori_change, names="value")
    slider_size.observe(on_size_change, names="value")
    btn_ok.on_click(on_confirm)

    controls = widgets.VBox(
        [
            widgets.HBox([toggle_ori, btn_ok]),
            slider_size,
            result_out,
        ],
        layout=widgets.Layout(padding="5px"),
    )

    display(controls)
    plt.show()

    return state


def process_tracking_video(
    model: YOLO,
    video_info: sv.VideoInfo,
    video_path: str,
    output_path: str,
    line_start: sv.Point,
    line_end: sv.Point,
    config: TrackingConfig,
) -> dict[str, Any]:
    byte_tracker = sv.ByteTrack(
        track_activation_threshold=config.track_thresh,
        lost_track_buffer=config.track_buffer,
        minimum_matching_threshold=config.match_thresh,
        frame_rate=config.frame_rate,
        minimum_consecutive_frames=config.min_frames,
    )
    byte_tracker.reset()

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_thickness=1,
        text_scale=0.5,
        text_color=sv.Color.BLACK,
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=40,
        position=sv.Position.CENTER,
    )
    line_zone = sv.LineZone(start=line_start, end=line_end)
    line_annotator = sv.LineZoneAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=0.8,
        display_in_count=True,
        display_out_count=False,
    )

    line_touch_records: list[dict[str, Any]] = []
    prev_side_by_id: dict[int, float] = {}
    crossed_ids: set[int] = set()

    x1, y1 = line_start.x, line_start.y
    x2, y2 = line_end.x, line_end.y
    line_len = float(np.hypot(x2 - x1, y2 - y1))

    def signed_distance_to_line(cx: float, cy: float) -> float:
        return ((x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)) / (line_len + 1e-9)

    progress = widgets.IntProgress(
        value=0,
        min=0,
        max=video_info.total_frames,
        description="Processing:",
        bar_style="info",
        layout=widgets.Layout(width="600px"),
    )
    label_w = widgets.Label(value="")
    display(widgets.HBox([progress, label_w]))

    frame_count = [0]

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        results = model(frame, conf=config.conf, imgsz=config.imgsz, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = byte_tracker.update_with_detections(detections)

        labels = [f"#{tid} {conf:.2f}" for conf, tid in zip(detections.confidence, detections.tracker_id)]

        if len(detections) > 0 and detections.tracker_id is not None:
            for tid, bbox in zip(detections.tracker_id, detections.xyxy):
                if tid is None:
                    continue

                tid_int = int(tid)
                cx = float((bbox[0] + bbox[2]) / 2)
                cy = float((bbox[1] + bbox[3]) / 2)
                side = signed_distance_to_line(cx, cy)
                prev_side = prev_side_by_id.get(tid_int)

                touched = abs(side) <= config.line_margin_px
                crossed = prev_side is not None and (prev_side * side < 0)

                if tid_int not in crossed_ids and (touched or crossed):
                    line_touch_records.append(
                        {
                            "frame": int(index),
                            "track_id": tid_int,
                            "bbox": (
                                float(bbox[0]),
                                float(bbox[1]),
                                float(bbox[2]),
                                float(bbox[3]),
                            ),
                        }
                    )
                    crossed_ids.add(tid_int)

                prev_side_by_id[tid_int] = side

        ann = frame.copy()
        ann = trace_annotator.annotate(scene=ann, detections=detections)
        ann = box_annotator.annotate(scene=ann, detections=detections)
        ann = label_annotator.annotate(scene=ann, detections=detections, labels=labels)

        line_zone.trigger(detections)
        ann = line_annotator.annotate(ann, line_counter=line_zone)

        frame_count[0] += 1
        if frame_count[0] % 10 == 0:
            progress.value = frame_count[0]
            label_w.value = f"frame {frame_count[0]} | droplets: {line_zone.in_count}"

        return ann

    sv.process_video(source_path=video_path, target_path=output_path, callback=callback)

    progress.value = video_info.total_frames
    progress.bar_style = "success"

    print(f"\nVideo saved to: {output_path}")
    print(f"Total droplets : {line_zone.in_count}")
    print(f"Diameters measured on the line : {len(line_touch_records)}")

    return {
        "line_touch_records": line_touch_records,
        "in_count": line_zone.in_count,
        "output_path": output_path,
    }


def print_tracking_summary(
    line_touch_records: list[dict[str, Any]],
    um_per_px: float | None,
    video_info: sv.VideoInfo,
    total_count: int,
    video_path: str,
    output_path: str,
    window_s: float = 10.0,
) -> pd.DataFrame | None:
    records = [dict(record) for record in line_touch_records]

    for record in records:
        x1b, y1b, x2b, y2b = record.pop("bbox")
        w_px = x2b - x1b
        h_px = y2b - y1b
        d_px = (w_px + h_px) / 2.0
        sph = min(w_px, h_px) / max(w_px, h_px) if max(w_px, h_px) > 0 else 0.0

        record["dx_px"] = round(w_px, 2)
        record["dy_px"] = round(h_px, 2)
        record["d_px"] = round(d_px, 2)
        record["sphericity"] = round(sph, 4)

        if um_per_px:
            record["dx_um"] = round(w_px * um_per_px, 2)
            record["dy_um"] = round(h_px * um_per_px, 2)
            record["d_um"] = round(d_px * um_per_px, 2)
        else:
            record["dx_um"] = None
            record["dy_um"] = None
            record["d_um"] = None

    fps_video = video_info.fps
    duration = video_info.total_frames / fps_video
    rate = total_count / duration if duration > 0 else 0.0

    print("=" * 55)
    print("  EXPERIMENT SUMMARY")
    print("=" * 55)
    print(f"  Video          : {Path(video_path).name}")
    print(f"  Duration       : {duration:.1f} s")
    print(f"  Droplets IN    : {total_count}")
    print(f"  Rate           : {rate:.2f} droplets/s")
    print(f"  Frequency      : {rate * 60:.1f} droplets/min")

    if not records:
        print()
        print("  No droplet touched/crossed the line during processing.")
        print("=" * 55)
        print(f"\n  Output: {output_path}")
        return None

    df_line = pd.DataFrame(records)
    df_line["time_s"] = df_line["frame"] / fps_video

    def stats(values: np.ndarray) -> tuple[float, float, float]:
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        cv = (sd / mean * 100.0) if mean > 0 else 0.0
        return mean, sd, cv

    df_line["window"] = (df_line["time_s"] // window_s).astype(int)

    window_counts = (
        df_line.groupby("window")
        .size()
        .rename("count")
        .reset_index()
    )
    window_counts["freq_hz"] = window_counts["count"] / window_s
    window_counts["freq_min"] = window_counts["freq_hz"] * 60.0
    window_counts["t_start"] = window_counts["window"] * window_s
    window_counts["t_end"] = window_counts["t_start"] + window_s

    freq_arr = window_counts["freq_hz"].to_numpy(dtype=float)
    freq_mean, freq_std, freq_cv = stats(freq_arr)

    print()
    print(f"  Generation frequency  (Dt = {window_s:.0f} s windows)")
    print(f"  {'Window':>8}  {'t_start':>8}  {'t_end':>8}  {'count':>6}  {'freq (Hz)':>10}  {'freq (min^-1)':>12}")
    print(f"  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 6}  {'-' * 10}  {'-' * 12}")
    for _, row in window_counts.iterrows():
        print(
            f"  {int(row['window']):>8}  "
            f"{row['t_start']:>7.0f}s  "
            f"{row['t_end']:>7.0f}s  "
            f"{int(row['count']):>6}  "
            f"{row['freq_hz']:>10.3f}  "
            f"{row['freq_min']:>12.2f}"
        )
    print()
    print(f"  Freq mean      : {freq_mean:.3f} Hz  ({freq_mean * 60.0:.2f} min^-1)")
    print(f"  Freq SD        : {freq_std:.3f} Hz")
    print(f"  Freq CV        : {freq_cv:.1f} %")

    col_x = "dx_um" if um_per_px else "dx_px"
    col_y = "dy_um" if um_per_px else "dy_px"
    unit = "um" if um_per_px else "px"

    arr_x = df_line[col_x].to_numpy(dtype=float)
    arr_y = df_line[col_y].to_numpy(dtype=float)
    arr_d = (arr_x + arr_y) / 2.0
    arr_sph = df_line["sphericity"].to_numpy(dtype=float)

    print()
    print(f"  Diameters & sphericity at line crossing  [{unit}]")
    print(f"  {'frame':>7}  {'id':>5}  {'dx':>8}  {'dy':>8}  {'d_mean':>8}  {'aspect':>7}  {'Psi':>7}")
    print(f"  {'-' * 7}  {'-' * 5}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 7}  {'-' * 7}")
    for _, row in df_line.iterrows():
        dx = row[col_x]
        dy = row[col_y]
        d_mean = (dx + dy) / 2.0
        aspect = dx / dy if dy > 0 else float("nan")
        print(
            f"  {int(row['frame']):>7}  "
            f"#{int(row['track_id']):<4}  "
            f"{dx:>8.1f}  "
            f"{dy:>8.1f}  "
            f"{d_mean:>8.1f}  "
            f"{aspect:>7.3f}  "
            f"{row['sphericity']:>7.3f}"
        )

    mx, sdx, cvx = stats(arr_x)
    my, sdy, cvy = stats(arr_y)
    md, sdd, cvd = stats(arr_d)
    ms, sds, cvs = stats(arr_sph)

    print()
    print(f"  {'':14}  {'dx':>8}  {'dy':>8}  {'d_mean':>8}  {'Psi':>7}  [unit: {unit}]")
    print(f"  {'Mean':14}  {mx:>8.1f}  {my:>8.1f}  {md:>8.1f}  {ms:>7.3f}")
    print(f"  {'SD':14}  {sdx:>8.1f}  {sdy:>8.1f}  {sdd:>8.1f}  {sds:>7.3f}")
    print(f"  {'CV (%)':14}  {cvx:>8.1f}  {cvy:>8.1f}  {cvd:>8.1f}  {cvs:>7.1f}")
    print(f"  {'Aspect dx/dy':14}  {mx / my:>8.3f}")

    print("=" * 55)
    print(f"\n  Output: {output_path}")

    return df_line


def build_droplet_dataframe(
    line_touch_records: list[dict[str, Any]],
    um_per_px: float | None = None,
) -> pd.DataFrame:
    columns = [
        "frame",
        "track_id",
        "dx_px",
        "dy_px",
        "d_px",
        "sphericity",
        "dx_um",
        "dy_um",
        "d_um",
    ]

    if not line_touch_records:
        return pd.DataFrame(columns=columns)

    records: list[dict[str, Any]] = []
    for raw in line_touch_records:
        record: dict[str, Any] = {
            "frame": raw.get("frame"),
            "track_id": raw.get("track_id"),
        }

        bbox = raw.get("bbox")
        if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            x1b, y1b, x2b, y2b = [float(v) for v in bbox]
            w_px = x2b - x1b
            h_px = y2b - y1b
            d_px = (w_px + h_px) / 2.0
            sph = min(w_px, h_px) / max(w_px, h_px) if max(w_px, h_px) > 0 else 0.0

            record["dx_px"] = round(w_px, 2)
            record["dy_px"] = round(h_px, 2)
            record["d_px"] = round(d_px, 2)
            record["sphericity"] = round(sph, 4)
        else:
            dx_px = raw.get("dx_px")
            dy_px = raw.get("dy_px")
            d_px = raw.get("d_px")
            sphericity = raw.get("sphericity")

            record["dx_px"] = dx_px
            record["dy_px"] = dy_px
            record["d_px"] = d_px
            record["sphericity"] = sphericity

        if um_per_px is not None and pd.notna(record["dx_px"]) and pd.notna(record["dy_px"]):
            record["dx_um"] = round(float(record["dx_px"]) * um_per_px, 2)
            record["dy_um"] = round(float(record["dy_px"]) * um_per_px, 2)
            record["d_um"] = round(float(record["d_px"]) * um_per_px, 2)
        else:
            record["dx_um"] = raw.get("dx_um")
            record["dy_um"] = raw.get("dy_um")
            record["d_um"] = raw.get("d_um")

        records.append(record)

    df = pd.DataFrame(records)
    return df[columns].sort_values(["frame", "track_id"]).reset_index(drop=True)


def build_drops_dataframe(
    line_touch_records: list[dict[str, Any]],
    um_per_px: float | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for build_droplet_dataframe."""
    return build_droplet_dataframe(line_touch_records, um_per_px=um_per_px)
