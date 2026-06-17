#!/usr/bin/env python
import argparse
import json
import sys
import types
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.modules.setdefault("cpp", types.ModuleType("cpp"))
sys.modules.setdefault("cpp.alignment", types.ModuleType("cpp.alignment"))

from alignment import Trajectory, euroc_csv_to_trajectory  # noqa: E402
from utils import error, info, load_csv_safer, warn  # noqa: E402


DISPLAY_TS_COLS = [
    "display_time",
    "wait_frame",
    "begin_frame",
    "locate_views",
    "predict_filter",
    "present",
]
CAMERA_TS_COLS = ["exposure", "usb_transfer_done", "sent_to_basalt", "flushed"]


def load_first_ts_from_dataset(dataset_dir: Path) -> int:
    cam_path = dataset_dir / "mav0" / "cam0" / "data.csv"
    if not cam_path.exists():
        error(f"Dataset cam CSV not found: {cam_path}")
    df = pd.read_csv(cam_path, comment="#", header=None, names=["timestamp", "filename"])
    return int(df["timestamp"].iloc[0])


def consistency_check(
    camera_data: np.ndarray,
    camera_cols: List[str],
    tracking: Trajectory,
    display_data: np.ndarray,
    display_cols: List[str],
    filtering: Trajectory,
) -> List[str]:
    warnings = []

    exposure_idx = camera_cols.index("exposure")
    cam_first = int(camera_data[0, exposure_idx])
    track_first = int(tracking.ts[0, 0])
    if cam_first != track_first:
        msg = (
            f"tracking/camera first-ts mismatch: camera.exposure[0]={cam_first}, "
            f"tracking.timestamp[0]={track_first}, diff={track_first - cam_first} ns"
        )
        warn(msg)
        warnings.append(msg)

    display_time_idx = display_cols.index("display_time")
    disp_first = int(display_data[0, display_time_idx])
    filt_first = int(filtering.ts[0, 0])
    if abs(disp_first - filt_first) > 1_000_000_000:
        msg = (
            f"display/filtering first-ts disagree by >1s — likely different runs: "
            f"display.display_time[0]={disp_first}, filtering.timestamp[0]={filt_first}, "
            f"diff={filt_first - disp_first} ns"
        )
        warn(msg)
        warnings.append(msg)

    return warnings


def find_t(camera_data: np.ndarray, camera_cols: List[str], cam0_first_ts: int) -> Tuple[int, str]:
    exposure_idx = camera_cols.index("exposure")
    cam_first = int(camera_data[0, exposure_idx])
    t = cam0_first_ts - cam_first
    method = "first-row anchor: cam0[0] - camera.exposure[0]"
    if t == 0:
        info("Camera timeline already in dataset clock (EUROC_USE_SOURCE_TS likely active)")
        method += " (t==0 detected)"
    return t, method


def find_s(
    display_data: np.ndarray,
    display_cols: List[str],
    filtering: Trajectory,
) -> Tuple[int, str, dict]:
    s = 0
    display_time_idx = display_cols.index("display_time")
    display_ts = np.sort(display_data[:, display_time_idx])
    filt_ts = filtering.ts.flatten()

    idx = np.searchsorted(display_ts, filt_ts)
    idx = np.clip(idx, 1, len(display_ts) - 1)
    left = display_ts[idx - 1]
    right = display_ts[idx]
    residuals = np.minimum(np.abs(filt_ts - left), np.abs(filt_ts - right))
    stats = {
        "max_ns": int(residuals.max()),
        "p99_ns": int(np.percentile(residuals, 99)),
        "median_ns": int(np.median(residuals)),
        "count_over_1ms": int((residuals > 1_000_000).sum()),
        "total": int(len(residuals)),
    }

    return s, stats


def filter_init_poses(traj: Trajectory) -> int:
    nonzero_mask = ~np.all(traj.xyz == 0, axis=0)
    n_dropped = int((~nonzero_mask).sum())
    if n_dropped == 0:
        return 0
    traj.ts = traj.ts[nonzero_mask.reshape(-1, 1)].reshape(-1, 1)
    traj.xyz = np.ascontiguousarray(traj.xyz[:, nonzero_mask])
    if traj.quat is not None:
        traj.quat = np.ascontiguousarray(traj.quat[:, nonzero_mask])
    return n_dropped


def find_preanchor_cutoff(ts: np.ndarray) -> int:
    ts = np.asarray(ts).flatten()
    cutoff = 0
    for i in range(1, len(ts)):
        if ts[i] + 10**12 < ts[i - 1]:
            cutoff = i
    return cutoff


def load_event_csv_drop_preanchor(path: Path, ts_col: str) -> Tuple[List[str], np.ndarray]:
    with open(path, "r", encoding="utf8") as f:
        first_line = next(f)
    assert first_line[0] == "#" and first_line[-1] == "\n", "first csv line should be a comment with column names"
    cols = first_line[1:-1].split(",")
    data = np.genfromtxt(path, delimiter=",", comments="#", dtype=np.int64, invalid_raise=True)
    assert len(cols) == data.shape[1], "number of column names differ from data columns"
    cutoff = find_preanchor_cutoff(data[:, cols.index(ts_col)])
    if cutoff:
        info(f"Dropped {cutoff} pre-anchor rows from {path.name}")
        data = data[cutoff:]
    return cols, data


def drop_preanchor_traj(traj: Trajectory) -> int:
    ts = traj.ts.flatten()
    cutoff = find_preanchor_cutoff(ts)
    if cutoff == 0:
        return 0
    keep = np.zeros(len(ts), dtype=bool)
    keep[cutoff:] = True
    traj.ts = traj.ts[keep.reshape(-1, 1)].reshape(-1, 1)
    traj.xyz = np.ascontiguousarray(traj.xyz[:, keep])
    if traj.quat is not None:
        traj.quat = np.ascontiguousarray(traj.quat[:, keep])
    return cutoff


def shift_event_csv(
    data: np.ndarray, cols: List[str], ts_cols: List[str], offset: int
) -> np.ndarray:
    out = data.copy()
    for name in ts_cols:
        idx = cols.index(name)
        out[:, idx] = out[:, idx] + offset
    return out


def shift_trajectory(traj: Trajectory, offset: int) -> Trajectory:
    traj.ts = traj.ts + offset
    return traj


def write_event_csv(path: Path, cols: List[str], data: np.ndarray) -> None:
    header = "#" + ",".join(cols)
    np.savetxt(path, data, fmt="%d", delimiter=",", header=header, comments="")


def write_trajectory_csv(path: Path, traj: Trajectory) -> None:
    header = (
        "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],"
        "q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []"
    )
    ts = traj.ts.flatten()
    xyz = traj.xyz
    quat = traj.quat  # stored xyzw, EuRoC writes wxyz
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i in range(ts.shape[0]):
            f.write(
                f"{int(ts[i])},"
                f"{xyz[0, i]:.10f},{xyz[1, i]:.10f},{xyz[2, i]:.10f},"
                f"{quat[3, i]:.10f},{quat[0, i]:.10f},{quat[1, i]:.10f},{quat[2, i]:.10f}\n"
            )


def post_validate(out_dir: Path, dataset_dir: Path) -> List[str]:
    warnings: List[str] = []
    cam0_first = load_first_ts_from_dataset(dataset_dir)

    aligned_camera_cols, aligned_camera = load_csv_safer(out_dir / "camera.csv", dtype=np.int64)
    aligned_track = euroc_csv_to_trajectory(out_dir / "tracking.csv")
    exposure_idx = aligned_camera_cols.index("exposure")

    if int(aligned_camera[0, exposure_idx]) != cam0_first:
        error(
            f"Post-validation: aligned camera.exposure[0]={aligned_camera[0, exposure_idx]} "
            f"!= cam0[0]={cam0_first}"
        )

    if int(aligned_track.ts[0, 0]) != int(aligned_camera[0, exposure_idx]):
        msg = (
            f"Post-validation: aligned tracking.timestamp[0]={int(aligned_track.ts[0, 0])} "
            f"!= aligned camera.exposure[0]={int(aligned_camera[0, exposure_idx])} "
        )
        warn(msg)
        warnings.append(msg)

    gt_path = dataset_dir / "mav0" / "gt" / "data.csv"
    gt_traj = euroc_csv_to_trajectory(gt_path)
    cam_min = int(aligned_camera[:, exposure_idx].min())
    cam_max = int(aligned_camera[:, exposure_idx].max())
    gt_min = int(gt_traj.ts.min())
    gt_max = int(gt_traj.ts.max())
    if gt_min > cam_min or gt_max < cam_max:
        msg = (
            f"GT does not fully cover camera range: gt=[{gt_min}, {gt_max}], "
            f"cam=[{cam_min}, {cam_max}]"
        )
        warn(msg)
        warnings.append(msg)

    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align Monado timing CSVs into the dataset clock domain."
    )
    parser.add_argument("--timing-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    timing_dir: Path = args.timing_dir
    dataset_dir: Path = args.dataset
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    info(f"Loading inputs from {timing_dir} and {dataset_dir}")
    display_cols, display_data = load_event_csv_drop_preanchor(timing_dir / "display.csv", "display_time")
    camera_cols, camera_data = load_csv_safer(timing_dir / "camera.csv", dtype=np.int64)
    tracking = euroc_csv_to_trajectory(timing_dir / "tracking.csv")
    filtering = euroc_csv_to_trajectory(timing_dir / "filtering.csv")
    prediction = euroc_csv_to_trajectory(timing_dir / "prediction.csv")
    cam0_first = load_first_ts_from_dataset(dataset_dir)

    n_pre_filt = drop_preanchor_traj(filtering)
    n_pre_pred = drop_preanchor_traj(prediction)
    if n_pre_filt or n_pre_pred:
        info(f"Dropped pre-anchor rows: filtering={n_pre_filt}, prediction={n_pre_pred}")

    consistency_warnings = consistency_check(
        camera_data, camera_cols, tracking, display_data, display_cols, filtering
    )

    t, t_method = find_t(camera_data, camera_cols, cam0_first)
    s, stats = find_s(display_data, display_cols, filtering)
    info(f"t (camera→dataset) = {t} ns ({t / 1e6:.3f} ms)")
    info(
        f"s (display→camera) = {s} ns ({s / 1e6:.3f} ms), bridge residual "
        f"median={stats['median_ns']} p99={stats['p99_ns']} "
        f"max={stats['max_ns']} ns "
        f"({stats['count_over_1ms']}/{stats['total']} > 1ms)"
    )

    n_init = filter_init_poses(tracking)
    if n_init:
        info(f"Filtered {n_init} init poses from tracking.csv (all-zero xyz)")

    display_aligned = shift_event_csv(display_data, display_cols, DISPLAY_TS_COLS, s + t)
    camera_aligned = shift_event_csv(camera_data, camera_cols, CAMERA_TS_COLS, t)
    tracking = shift_trajectory(tracking, t)
    filtering = shift_trajectory(filtering, s + t)
    prediction = shift_trajectory(prediction, s + t)

    write_event_csv(out_dir / "display.csv", display_cols, display_aligned)
    write_event_csv(out_dir / "camera.csv", camera_cols, camera_aligned)
    write_trajectory_csv(out_dir / "tracking.csv", tracking)
    write_trajectory_csv(out_dir / "filtering.csv", filtering)
    write_trajectory_csv(out_dir / "prediction.csv", prediction)

    post_warnings = post_validate(out_dir, dataset_dir)

    offsets = {
        "dataset_cam0_ts0_ns": cam0_first,
        "camera_exposure_ts0_ns": int(camera_data[0, camera_cols.index("exposure")]),
        "display_time_ts0_ns": int(display_data[0, display_cols.index("display_time")]),
        "camera_to_dataset_offset_ns": int(t),
        "display_to_camera_offset_ns": int(s),
        "display_to_dataset_offset_ns": int(s + t),
        "residual_ns": stats,
        "init_poses_dropped": int(n_init),
        "method_t": t_method,
        "warnings": consistency_warnings + post_warnings,
    }
    with open(out_dir / "offsets.json", "w", encoding="utf-8") as f:
        json.dump(offsets, f, indent=2)

    info(f"Wrote aligned CSVs + offsets.json to {out_dir}")
    gt = dataset_dir / "mav0" / "gt" / "data.csv"
    info("Next steps:")
    info(f"  python xrtslam-metrics/tracking.py ate {gt} {out_dir}/tracking.csv --plot")
    info(f"  python xrtslam-metrics/tracking.py ate {gt} {out_dir}/filtering.csv --plot")
    info(f"  python xrtslam-metrics/tracking.py ate {gt} {out_dir}/prediction.csv --plot")


if __name__ == "__main__":
    main()
