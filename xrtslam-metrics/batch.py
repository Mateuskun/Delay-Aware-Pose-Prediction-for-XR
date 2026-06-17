#!/usr/bin/env python

import json
from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
import math
from math import inf
from evo.core.geometry import GeometryException

import numpy as np
import pandas as pd
from tabulate import tabulate

from completion import load_completion_stats
from features import FeaturesStats
from timing import TimingStats
from tracking import get_tracking_stats
from utils import (
    COMPLETION_FULL_SINCE,
    DEFAULT_SEGMENT_DRIFT_TOLERANCE_M,
    DEFAULT_TIMING_COLS,
    Vector2,
    isnan,
    color_string,
    warn,
)


@dataclass
class Batch:
    evaluation_path: Path
    targets_path: Path
    timing_columns: Dict[str, Tuple[str, str]]
    metrics: List[str] = None
    verbose: bool = False
    save_file: Optional[Path] = None
    load_files: Optional[List[Path]] = None
    allow_ds_prefixes: List[str] = None
    deny_ds_prefixes: List[str] = None
    allow_sys_prefixes: List[str] = None
    deny_sys_prefixes: List[str] = None
    highlights: List[str] = None
    names: List[str] = None
    upper_bounds: Dict[str, float] = field(default_factory=dict)

    data: Dict[str, pd.DataFrame] = field(init=False)

    def __post_init__(self):
        self.data = {}
        if self.load_files is None or self.load_files == []:
            return
        storage = {}
        for load_file in self.load_files:
            with open(load_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                # Merge loaded files
                for metric, runs in loaded.items():
                    if metric not in storage:
                        storage[metric] = {}
                    for run, results in runs.items():
                        if run not in storage[metric]:
                            storage[metric][run] = {}
                        for ds, val in results.items():
                            if ds in storage[metric][run]:
                                warn(f"Overwriting {metric} for {run}/{ds} with {load_file}")
                            storage[metric][run][ds] = val

            for metric, json_dict in storage.items():
                df = pd.read_json(StringIO(json.dumps(json_dict)))
                df = df.map(lambda x: np.nan if x is None else np.array(x))
                self.data[metric] = df
                if self.verbose:
                    print(f"Loaded {metric} data from {load_file}")

    @property
    def highlight_markdown(self) -> bool:
        keywords = {"md", "markdown", "bold"}  # NOTE: bold is for backwards compatibility
        return any(k in self.highlights for k in keywords)

    @property
    def highlight_ansi(self) -> bool:
        keywords = {"cli", "ansi", "color"}  # NOTE: color is for backwards compatibility
        return any(k in self.highlights for k in keywords)


SimpleMeasureF = Callable[[Path, str], Any]
TargetMeasureF = Callable[[Path, Path], Any]
MeasureFunction = Union[SimpleMeasureF, TargetMeasureF]
MeasureStringFunction = Callable[[Any], str]


def foreach_dataset(
    batch: Batch,
    result_fn: str,
    target_fn: Optional[str],
    measure,  #: MeasureFunction
):
    sys_dirs = [r for r in batch.evaluation_path.iterdir() if r.is_dir()]
    sys_dirs = [n for n in sys_dirs if not n.name.startswith("_")]
    sys_names = sorted([d.name for d in sys_dirs])
    ordered_set = {d.name: 0 for r in sys_dirs for d in r.iterdir() if d.is_dir()}
    ds_names = sorted(ordered_set.keys())
    ds_names = [n for n in ds_names if not n.startswith("_")]

    # Filter datasets based on allow/deny prefixes
    ds_allow = lambda n: any(n.startswith(p) for p in (batch.allow_ds_prefixes or [""]))
    ds_deny = lambda n: any(n.startswith(p) for p in (batch.deny_ds_prefixes or ["_"]))
    ds_names = [n for n in ds_names if ds_allow(n)]
    ds_names = [n for n in ds_names if not ds_deny(n)]

    # Filter systems based on allow/deny prefixes
    sys_allow = lambda n: any(n.startswith(p) for p in (batch.allow_sys_prefixes or [""]))
    sys_deny = lambda n: any(n.startswith(p) for p in (batch.deny_sys_prefixes or ["_"]))
    sys_names = [n for n in sys_names if sys_allow(n)]
    sys_names = [n for n in sys_names if not sys_deny(n)]
    sys_dirs = [d for d in sys_dirs if d.name in sys_names]

    df = pd.DataFrame(None, columns=sys_names, index=ds_names)

    sys_count = len(sys_dirs)
    ds_count = len(ds_names)
    total_count = sys_count * ds_count
    current_count = 0
    for sys_dir in sys_dirs:
        for ds_name in ds_names:
            ds_dir = sys_dir / ds_name
            if ds_dir.is_dir():
                sys_name = sys_dir.name
                result_csv = ds_dir / result_fn
                if result_csv.exists() and result_csv.stat().st_size != 0:
                    if target_fn is None:
                        df.loc[ds_name, sys_name] = measure(result_csv, sys_name)
                    else:
                        target_csv = batch.targets_path / ds_name / target_fn
                        if target_csv.exists():
                            current_count += 1
                            val = measure(result_csv, target_csv)
                            df.loc[ds_name, sys_name] = val
                            if batch.verbose:
                                print(f"[{(current_count/total_count) * 100:.2f}%] " f"{sys_name}/{ds_name}: {val}")
    return df


def print_dataframe(
    df: pd.DataFrame,
    measure_str: MeasureStringFunction,
    key: Optional[Callable[[Any], Any]],
    batch: Batch,
    upper_bound: Optional[float] = None,
) -> None:
    """Format the DataFrame by adding average and median rows."""
    measure_str_none = lambda m: "—" if isnan(m) else measure_str(m)

    # Add average and median rows
    df = df.copy()

    # Filter datasets based on allow/deny prefixes
    mask = np.array([False] * len(df.index), dtype=bool)
    for allow_prefix in batch.allow_ds_prefixes or [""]:
        mask |= df.index.str.startswith(allow_prefix)
    for deny_prefix in batch.deny_ds_prefixes:
        mask &= ~df.index.str.startswith(deny_prefix)
    df = df[mask]

    # Filter systems based on allow/deny prefixes
    mask = np.array([False] * len(df.columns), dtype=bool)
    for allow_prefix in batch.allow_sys_prefixes or [""]:
        mask |= df.columns.str.startswith(allow_prefix)
    for deny_prefix in batch.deny_sys_prefixes:
        mask &= ~df.columns.str.startswith(deny_prefix)
    df = df.loc[:, mask]

    # Build a NaN cell, changes based on shape of the data
    nan_cell = np.array([np.nan])
    valid_row = df.first_valid_index()
    if valid_row is not None:
        valid_col = df.loc[valid_row].first_valid_index()
        if valid_col is not None:
            non_nan_cell = df.loc[valid_row].loc[valid_col]
            if isinstance(non_nan_cell, float):
                non_nan_cell = np.array([non_nan_cell])
            nan_cell = np.zeros_like(non_nan_cell)
            nan_cell.fill(np.nan)

    # List of ds (rows) in to ignore from the average computation
    ds_wo_avg = df.index[df.isna().any(axis=1)]  # skip nans

    # Skip outliers
    if upper_bound is not None and key is not None:
        above_ub_ds = df.index[
            df.apply(lambda row: row.apply(lambda v: key(v) if not isnan(v) else inf).max() > upper_bound, axis=1)
        ]
        ds_wo_avg = ds_wo_avg.union(above_ub_ds)

    def _avg(col):
        "Average only over non-nan values below upper bound"
        vals = col.dropna()
        vals = vals[~vals.index.isin(ds_wo_avg)]
        if vals.size == 0:
            return nan_cell
        stacked = np.vstack(vals)
        if upper_bound is not None:
            masked = np.ma.masked_where(stacked > upper_bound, stacked)
            return np.ma.mean(masked, axis=0).filled(np.nan)
        return np.mean(stacked, axis=0)

    def _med(col):
        "Median over all values, nans are replaced by infinity"
        vals = col.apply(lambda v: np.inf if isnan(v) else v)
        if vals.size == 0:
            return nan_cell
        stacked = np.vstack(vals)
        return np.median(stacked, axis=0)

    avg_row = df.apply(_avg)
    med_row = df.apply(_med)
    sanitize = lambda v: (
        np.nan
        if isinstance(v, float) and np.isnan(v)
        else np.array(list(v.values())) if len(v) > 1 else list(v.values())[0]
    )
    avg_row = {k: sanitize(v) for k, v in avg_row.to_dict().items()}
    med_row = {k: sanitize(v) for k, v in med_row.to_dict().items()}
    df.loc["[AVG]"] = avg_row
    df.loc["[MED]"] = med_row

    df_as_string = df.map(measure_str_none)

    # Highlight the best elements in each row
    if len(batch.highlights) != 0 and key is not None:
        df_keys = df.map(key)
        best_elems = df_keys.idxmin(axis=1)
        for ds, sys in best_elems.items():
            v = df.loc[ds, sys]
            h = df_as_string.loc[ds, sys]
            if upper_bound is not None and key(v) >= upper_bound or isnan(v):
                continue
            if batch.highlight_ansi:
                h = f"{color_string(color_string(h, fg='cyan'), fg='bold')}"
            if batch.highlight_markdown:
                h = f"**{h}**"
            df_as_string.loc[ds, sys] = h

    # Make bad numbers (above the upper bound) strikethrough
    if upper_bound is not None and key is not None:
        for ds, sys in df_keys.stack().index:
            v = df.loc[ds, sys]
            h = df_as_string.loc[ds, sys]
            if key(v) <= upper_bound and not isnan(v):
                continue
            if batch.highlight_ansi:
                h = f"{color_string(color_string(h, fg='faint'), fg='strikethrough')}"
            if batch.highlight_markdown:
                h = f"~~{h}~~"
            df_as_string.loc[ds, sys] = h

    # Add names to the columns/systems if provided
    if batch.names is not None and len(batch.names) > 0:
        if len(batch.names) != len(df.columns):
            warn(f"Not all columns specified {len(batch.names)=} != {len(df.columns)=}")
        df_as_string.rename(
            columns={o: f"{n}\n{o}" for o, n in zip(df.columns, batch.names)},
            inplace=True,
        )

    print(tabulate(df_as_string, headers="keys", tablefmt="pipe"))  # type: ignore


def timing_main(batch: Batch) -> pd.DataFrame:
    print("\nAverage (± stdev) pose estimation time [ms]\n")

    def measure_timing(result_csv: Path, sys_name: str) -> Vector2:
        cols = batch.timing_columns.get(sys_name, DEFAULT_TIMING_COLS)
        s = TimingStats(csv_fn=result_csv, cols=cols)
        return np.array([s.mean, s.std])

    def measure_timing_str(measure: Vector2) -> str:
        mean, std = measure
        return f"{mean:.2f} ± {std:.2f}"

    if "timing" in batch.data:
        df = batch.data["timing"]
    else:
        df = foreach_dataset(batch, "timing.csv", None, measure_timing)

    key = lambda x: x[0] if not isnan(x) else inf

    print_dataframe(df, measure_timing_str, key, batch, batch.upper_bounds.get("timing"))
    return df


def memory_main(batch: Batch) -> pd.DataFrame:
    print("\nPeak memory usage [MB]\n")

    def measure_memory(result_csv: Path, _: Path) -> float:
        mb = json.load(open(result_csv, encoding="utf-8"))["resident_memory_peak"][0] / 2**20
        return mb

    def measure_memory_str(memory: float) -> str:
        return f"{memory:.1f}"

    if "memory" in batch.data:
        df = batch.data["memory"]
    else:
        df = foreach_dataset(batch, "stats_vio.json", None, measure_memory)
    key = lambda x: x

    print_dataframe(df, measure_memory_str, key, batch, batch.upper_bounds.get("memory"))
    return df


def features_main(batch: Batch) -> pd.DataFrame:
    print("\nAverage feature count for each camera\n")

    def measure_features(result_csv: Path, _: str) -> Vector2:
        s = FeaturesStats(csv_fn=result_csv)
        return np.array(np.concatenate([s.mean, s.std]))

    def measure_features_str(measure: Vector2) -> str:
        mid = measure.shape[0] // 2
        mean, std = measure[0:mid], measure[mid:4]
        return f"{mean.astype(int)} ± {std.astype(int)}"

    if "features" in batch.data:
        df = batch.data["features"]
    else:
        df = foreach_dataset(batch, "features.csv", None, measure_features)

    key = lambda x: -sum(x[::2]) if not isnan(x) else inf

    print_dataframe(df, measure_features_str, key, batch, batch.upper_bounds.get("features"))
    return df


def completion_main(batch: Batch) -> pd.DataFrame:
    print("\nAverage completion percentage [%]\n")

    def measure_completion(result_csv: Path, target_csv: Path) -> float:
        s = load_completion_stats(result_csv, target_csv)
        return s.tracking_completion

    def measure_completion_str(completion: float) -> str:
        return f"{completion * 100:.2f}%" if completion < COMPLETION_FULL_SINCE else "✓"

    if "completion" in batch.data:
        df = batch.data["completion"]
    else:
        df = foreach_dataset(batch, "tracking.csv", "cam0.csv", measure_completion)

    key = lambda x: -x if not isnan(x) else inf

    print_dataframe(df, measure_completion_str, key, batch, batch.upper_bounds.get("completion"))
    return df


def success_main(batch: Batch) -> pd.DataFrame:
    print("\nAverage success percentage [%]\n")

    def measure_success(result_csv: Path, target_csv: Path) -> float:
        s = load_completion_stats(result_csv, target_csv)
        return s.tracking_success

    def measure_success_str(success: float) -> str:
        return f"{success * 100:.2f}%" if success < COMPLETION_FULL_SINCE else "✓"

    if "success" in batch.data:
        df = batch.data["success"]
    else:
        df = foreach_dataset(batch, "tracking.csv", "cam0.csv", measure_success)

    key = lambda x: -x if not isnan(x) else inf

    print_dataframe(df, measure_success_str, key, batch, batch.upper_bounds.get("success"))
    return df


def ate_main(batch: Batch) -> pd.DataFrame:
    print("\nAbsolute trajectory error (ATE) [m]\n")

    def measure_ape(result_csv: Path, target_csv: Path) -> Vector2:
        try:
            results = get_tracking_stats("ate", [result_csv], target_csv, silence=True)
        except Exception as e:  # Can happen if the trajectory has only one pose
            print(f"Error: {result_csv=}, {e=}")
            return math.nan
        s = results[result_csv].stats
        # Notice that std runs over APE while rmse over APE²
        return np.array([s["rmse"], s["std"]])

    def measure_ape_str(measure: Vector2) -> str:
        rmse, std = measure
        return f"{rmse:.3f} ± {std:.3f}"

    if "ate" in batch.data:
        df = batch.data["ate"]
    else:
        df = foreach_dataset(batch, "tracking.csv", "gt.csv", measure_ape)

    key = lambda x: x[0] if not isnan(x) else inf

    print_dataframe(df, measure_ape_str, key, batch, batch.upper_bounds.get("ate"))
    return df


def rte_main(batch: Batch) -> pd.DataFrame:
    print("\nRelative trajectory error (RTE) [m]\n")

    def measure_rpe(result_csv: Path, target_csv: Path) -> Vector2:
        try:
            results = get_tracking_stats("rte", [result_csv], target_csv, silence=True)
        except Exception as e:  # Can happen if the trajectory has only one pose
            print(f"Error: {result_csv=}, {e=}")
            return math.nan
        s = results[result_csv].stats
        # Notice that std runs over RPE while rmse over RPE²
        return np.array([s["rmse"], s["std"]])

    def measure_rpe_str(measure: Vector2) -> str:
        rmse, std = measure
        return f"{rmse:.6f} ± {std:.6f}"

    if "rte" in batch.data:
        df = batch.data["rte"]
    else:
        df = foreach_dataset(batch, "tracking.csv", "gt.csv", measure_rpe)

    key = lambda x: x[0] if not isnan(x) else inf

    print_dataframe(df, measure_rpe_str, key, batch, batch.upper_bounds.get("rte"))
    return df


def atec_main(batch: Batch) -> pd.DataFrame:
    print("\nAbsolute trajectory error (ATE) [m]\n")

    def measure_atec(result_csv: Path, target_csv: Path) -> float:
        try:
            results = get_tracking_stats("atec", [result_csv], target_csv, silence=True)
        except Exception as e:  # Can happen if the trajectory has only one pose
            print(f"Error: {result_csv=}, {e=}")
            return math.nan
        s = results[result_csv]
        return s

    def measure_atec_str(measure: float) -> str:
        return f"{measure:.3f}"

    if "atec" in batch.data:
        df = batch.data["atec"]
    else:
        df = foreach_dataset(batch, "tracking.csv", "gt.csv", measure_atec)

    key = lambda x: x if not isnan(x) else inf

    print_dataframe(df, measure_atec_str, key, batch, batch.upper_bounds.get("atec"))
    return df


def rtec_main(batch: Batch) -> pd.DataFrame:
    print("\nRelative trajectory error (RTE) [m]\n")

    def measure_rtec(result_csv: Path, target_csv: Path) -> float:
        try:
            results = get_tracking_stats("rtec", [result_csv], target_csv, silence=True)
        except Exception as e:  # Can happen if the trajectory has only one pose
            print(f"Error: {result_csv=}, {e=}")
            return math.nan
        s = results[result_csv]
        return s

    def measure_rtec_str(measure: float) -> str:
        return f"{measure:.6f}"

    if "rtec" in batch.data:
        df = batch.data["rtec"]
    else:
        df = foreach_dataset(batch, "tracking.csv", "gt.csv", measure_rtec)

    key = lambda x: x if not isnan(x) else inf

    print_dataframe(df, measure_rtec_str, key, batch, batch.upper_bounds.get("rtec"))
    return df


def seg_main(batch: Batch) -> pd.DataFrame:
    tol = DEFAULT_SEGMENT_DRIFT_TOLERANCE_M
    print(f"\n Segment drift per meter error (SDM {tol}m) [m/m]\n")

    def measure_seg(result_csv: Path, target_csv: Path) -> Vector2:
        results = get_tracking_stats("sdm", [result_csv], target_csv, silence=True)
        s = results[result_csv].stats
        return np.array([s["SDM"], s["SDM std"]])

    def measure_seg_str(measure: Vector2) -> str:
        drift, std = measure
        return f"{drift:.4f} ± {std:.4f}"

    if "seg" in batch.data:
        df = batch.data["seg"]
    else:
        df = foreach_dataset(batch, "tracking.csv", "gt.csv", measure_seg)

    key = lambda x: x[0] if not isnan(x) else inf

    print_dataframe(df, measure_seg_str, key, batch, batch.upper_bounds.get("seg"))
    return df


METRICS = {
    "timing": timing_main,
    "memory": memory_main,
    "features": features_main,
    "completion": completion_main,
    "success": success_main,
    "ate": ate_main,
    "rte": rte_main,
    "seg": seg_main,
    "atec": atec_main,
    "rtec": rtec_main,
}


def parse_args():
    parser = ArgumentParser(
        description="Batch evaluation of Monado visual-inertial runs of datasets\n\n"
        "Example execution: \n"
        "python batch.py test/data/runs/ test/data/targets/ \\ \n"
        "\t--timing Basalt opticalflow_received vio_produced \\\n"
        "\t--timing Kimera tracker_pushed processed \\\n"
        "\t--timing ORB-SLAM3 about_to_process processed",
        formatter_class=RawTextHelpFormatter,
    )
    # TODO: Make runs_dir optional (since load_files could make it unnecessary)
    # TODO: Update calls to batch.py in Basalt and readme
    parser.add_argument(
        "runs_dir",
        type=Path,
        help="Directory with runs subdirectories, each with datasets subdirectories."
        "The structure of runs_dir is like: <runs_dir>/<run>/<dataset>/{tracking, timing}.csv",
    )
    parser.add_argument(
        "targets_dir",
        type=Path,
        help="Directory with dataset groundtruth and camera timestamps."
        "The structure of targets_dir is like: <targets_dir>/<dataset>/{gt, cam0}.csv",
    )
    parser.add_argument(
        "--timing",
        action="append",
        nargs=3,
        default=[],
        help="For each <run> directory in <runs_dir> specify the first and last"
        "timing column names to use as --timing <run> <first_col> <last_col>."
        "If a <run> is not specified assuming"
        f"<first_col> = {DEFAULT_TIMING_COLS[0]} and <last_col> = {DEFAULT_TIMING_COLS[1]}",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["atec"],
        help="Metrics to evaluate",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress information")
    parser.add_argument("--save_file", type=Path, default=None, help="Save to file")
    parser.add_argument("--load_files", type=Path, nargs="+", default=[], help="Load from file")
    parser.add_argument(
        "--allow_ds_prefixes",
        type=str,
        nargs="*",
        default=[],
        help="Only consider datasets with this prefix",
    )
    parser.add_argument(
        "--deny_ds_prefixes",
        type=str,
        nargs="*",
        default=[],
        help="After allow_ds_prefixes, ignore datasets with this prefix",
    )
    parser.add_argument(
        "--allow_sys_prefixes",
        type=str,
        nargs="*",
        default=[],
        help="Only consider systems with this prefix",
    )
    parser.add_argument(
        "--deny_sys_prefixes",
        type=str,
        nargs="*",
        default=[],
        help="After allow_sys_prefixes, ignore systems with this prefix",
    )
    parser.add_argument(
        "--highlights",
        type=str,
        nargs="*",
        default=["cli"],
        help="Add score highlights for markdown (md) and/or ANSI (cli)",
    )
    parser.add_argument("--names", type=str, nargs="+", default=None, help="Column names")
    parser.add_argument(
        "--names_from_files",
        "-nf",
        action="store_true",
        help="Set --names property to the stems of the files in --load_files. Make sure to have a single run per file",
    )
    parser.add_argument(
        "--upper_bounds",
        type=str,
        nargs="+",
        default=[],
        help="Upper bounds per metric for avg/median computation, e.g. --upper_bounds atec=0.5 rtec=0.01",
    )
    return parser.parse_args()


def batch_from_args(args) -> Batch:
    timing_columns = {}
    for run, first_col, last_col in args.timing:
        timing_columns[run] = (first_col, last_col)
    upper_bounds = {}
    for ub in args.upper_bounds:
        metric, value = ub.split("=")
        upper_bounds[metric] = float(value)
    if args.names_from_files:
        args.names = [f.stem for f in args.load_files]
        print(f"Using --names {args.names} from --load_files")
    batch = Batch(
        args.runs_dir,
        args.targets_dir,
        timing_columns,
        args.metrics,
        args.verbose,
        args.save_file,
        args.load_files,
        args.allow_ds_prefixes,
        args.deny_ds_prefixes,
        args.allow_sys_prefixes,
        args.deny_sys_prefixes,
        args.highlights,
        args.names,
        upper_bounds,
    )
    return batch


def main():
    batch = batch_from_args(parse_args())
    print(f"Evaluating metrics: {', '.join(batch.metrics)}")

    # TODO@mateosss: modify batch.data en each computation and then do a try
    # catch to at least save the partial results
    data = {}
    for metric, compute_metric in METRICS.items():
        if metric not in batch.metrics:
            continue
        data[metric] = compute_metric(batch)

    if batch.save_file:
        storage = {}
        for metric, df in data.items():
            json_str = df.to_json()
            json_dict = json.loads(json_str)
            storage[metric] = json_dict

        with open(batch.save_file, "w", encoding="utf-8") as f:
            json.dump(storage, f, indent=4)

        print(f"\nSaved results to {batch.save_file}")


if __name__ == "__main__":
    main()
