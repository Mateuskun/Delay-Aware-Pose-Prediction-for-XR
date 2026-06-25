import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIO = ROOT / "basalt/build/basalt_vio"
EVAL = ROOT / "runs/baseline_eval"
TGT = ROOT / "runs/baseline_targets"
SLAM = ROOT / "runs/slam"
TABLE_TXT = ROOT / "runs/baseline_table.txt"
TABLE_JSON = ROOT / "runs/baseline_table.json"

DATASETS = [
    ("MOO02", "MOO02_hand_puncher_2",  "mav0/gt/data.csv",     "msd/msdmo_calib.json", "msd/msdmo_config.json"),
    ("MGO02", "MGO02_hand_puncher",    "mav0/gt/data.csv",     "msd/msdmg_calib.json", "msd/msdmg_config.json"),
    ("MIO02", "MIO02_hand_puncher_2",  "mav0/gt/data.csv",     "msd/msdmi_calib.json", "msd/msdmi_config.json"),
    ("MOO13", "MOO13_sudden_movements","mav0/gt/data.csv",     "msd/msdmo_calib.json", "msd/msdmo_config.json"),
    ("TR2",   "dataset-room2_512_16",  "mav0/mocap0/data.csv", "tum/tumvi_512_ds_calib.json", "tum/tumvi_512_config.json"),
    ("MH02",  "MH_02_easy",            "mav0/state_groundtruth_estimate0/data.csv", "euroc/euroc_ds_calib.json", "euroc/euroc_config.json"),
]

TIMINGS = [
    ("runs/01_wmr_battery",     "TBP0", "TBP0F0"),
    ("runs/02_wmr_performance", "TPP0", "TPP0F0"),
]

def run_basalt(name, ds_rel, calib, config, force):
    out = SLAM / name
    traj, rels = out / "trajectory.csv", out / "slam_relations.csv"
    if not force and traj.is_file() and rels.is_file():
        nrows = sum(1 for l in traj.open() if not l.startswith("#"))
        print(f"=== {name}: basalt already done ({nrows} rows), skip ===", flush=True)
        return traj.is_file()

    out.mkdir(parents=True, exist_ok=True)
    log = out / "basalt_run.log"
    print(f"=== {name}: running basalt_vio ({ds_rel}) ===", flush=True)
    with log.open("w") as lf:
        proc = subprocess.Popen(
            [str(VIO),
             "--dataset-path", str(ROOT / ds_rel), "--dataset-type", "euroc",
             "--cam-calib", str(ROOT / "basalt/data" / calib),
             "--config-path", str(ROOT / "basalt/data" / config),
             "--show-gui", "0",
             "--save-trajectory", "euroc", "--save-trajectory-fn", "trajectory.csv",
             "--save-relations-fn", "slam_relations.csv"],
            cwd=str(out), stdout=lf, stderr=subprocess.STDOUT,
        )
        for _ in range(240):
            if proc.poll() is not None:
                break
            if log.read_text().find("Saved SLAM relations") != -1:
                break
            time.sleep(5)
        time.sleep(2)
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    nrows = sum(1 for l in traj.open() if not l.startswith("#")) if traj.is_file() else 0
    print(f"=== {name}: basalt done, trajectory rows={nrows} ===", flush=True)
    return traj.is_file()


def run_replays(name, ds_rel, gt_rel, force):
    from csvio import basalt_slam_source
    from replay import replay_run
    from filter import FilterConfig
    from predict import PredictionType

    ds_dir = ROOT / ds_rel
    slam_dir = SLAM / name
    traj = slam_dir / "trajectory.csv"
    if not traj.is_file():
        print(f"[{name}] SKIP replays: no trajectory.csv (basalt not done)", flush=True)
        return
    if not force and (EVAL / "TBP0" / name / "tracking.csv").is_file():
        print(f"[{name}] SKIP replays: already done", flush=True)
        return

    (EVAL / "main" / name).mkdir(parents=True, exist_ok=True)
    shutil.copy(traj, EVAL / "main" / name / "tracking.csv")
    (TGT / name).mkdir(parents=True, exist_ok=True)
    shutil.copy(ds_dir / gt_rel, TGT / name / "gt.csv")

    src = basalt_slam_source(slam_dir, ds_dir)
    print(f"[{name}] relations={len(src.load_relations())} imu={len(src.load_imu().ts)}", flush=True)

    for timing_rel, pcol, fcol in TIMINGS:
        timing_dir = ROOT / timing_rel
        res = replay_run(
            timing_dir=timing_dir,
            out_dir=slam_dir / f"_replay_{timing_dir.name}",
            pred_type=PredictionType.DEAD_RECKONING,
            filter_config=FilterConfig(use_one_euro_filter=True),
            dataset_dir=ds_dir,
            slam_source=src,
        )
        for col, srcfile in ((pcol, res.prediction_path), (fcol, res.filtering_path)):
            dst = EVAL / col / name / "tracking.csv"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(srcfile, dst)
        print(f"[{name}] {timing_dir.name}: predicted={res.n_predicted} skipped={res.n_skipped}", flush=True)
    print(f"[{name}] DONE", flush=True)


def build_table():
    print("######## ATE/RTE table ########", flush=True)
    proc = subprocess.run(
        [sys.executable, "batch.py", str(EVAL), str(TGT),
         "--metrics", "ate", "rte", "--save_file", str(TABLE_JSON)],
        cwd=str(ROOT / "xrtslam-metrics"),
        capture_output=True, text=True,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        sys.exit(f"batch.py failed (exit {proc.returncode})")
    TABLE_TXT.write_text(proc.stdout)
    print(f"\nTable written to {TABLE_TXT.relative_to(ROOT)} and {TABLE_JSON.relative_to(ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    names = [d[0] for d in DATASETS]
    ap.add_argument("--datasets", nargs="+", choices=names, default=names,
                    help="Subset of datasets to (re)build. Default: all.")
    ap.add_argument("--skip-basalt", action="store_true", help="Skip basalt_vio; use existing trajectories.")
    ap.add_argument("--skip-replays", action="store_true", help="Skip replays; only (re)build the table.")
    ap.add_argument("--force", action="store_true", help="Redo basalt + replays even if outputs exist.")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "xrtslam-metrics"))
    selected = [d for d in DATASETS if d[0] in args.datasets]

    if not args.skip_basalt:
        print("######## STEP 1: basalt_vio per dataset ########", flush=True)
        for name, ds_rel, _gt, calib, config in selected:
            run_basalt(name, ds_rel, calib, config, args.force)

    if not args.skip_replays:
        print("######## STEP 2: assemble main + WMR-timing replays ########", flush=True)
        for name, ds_rel, gt_rel, _c, _cf in selected:
            run_replays(name, ds_rel, gt_rel, args.force)

    build_table()


if __name__ == "__main__":
    main()
