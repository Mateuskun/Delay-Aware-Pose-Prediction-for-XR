# runs/ — recorded timing runs for the offline harness

Each subfolder here is **one monado-service recording** under a specific
condition (machine / app / power profile / stress on-off / display rate). The
offline harness (`xrtslam-metrics/`) reads a run's CSVs, moves them into the
EuRoC dataset clock, regenerates poses per `when_ns`, and scores ATE/RTE against
the dataset ground truth.

This folder holds the **timing**; the **ground truth** comes from the EuRoC/MSD
dataset (e.g. `../MOO01_hand_puncher_1`). The harness combines the two.

## Reproduce the baseline matrix (one command)

**Prerequisite — patch Basalt.** `build_baseline.py` runs `basalt_vio` with
`--save-relations-fn`, which needs the velocity-bearing-relations patch (it is
not in upstream Basalt). Apply it to the `basalt/` submodule and rebuild once:

```bash
git -C basalt apply ../patches/basalt-save-relations.patch
cmake --build basalt/build --target basalt_vio    # or however you build Basalt
```

(The patch adds a `--save-relations-fn` option emitting a 14-column EuRoC-state
CSV: ts, p_xyz, q_wxyz, v_xyz, ω_xyz. Dead-reckoning needs the velocity that a
plain `--save-trajectory` does not carry.)

`build_baseline.py` is the single driver for the baseline ATE/RTE matrix ("the
metric to beat"). It runs the whole Flavor-B pipeline and prints the table:

```bash
python3 runs/build_baseline.py            # full matrix, resumes finished stages
python3 runs/build_baseline.py --datasets MOO02 TR2   # subset
python3 runs/build_baseline.py --skip-basalt          # only replays + table
python3 runs/build_baseline.py --force                # rebuild everything
```

Stages (all idempotent): **(1)** `basalt_vio` per dataset → `slam/<name>/`;
**(2)** inject the two recorded WMR timing runs (`01_wmr_battery`,
`02_wmr_performance`) on top → estimate trajectories in `baseline_eval/<col>/`;
**(3)** `batch.py` → the table to stdout **and** `baseline_table.{txt,json}`.

Columns (T=timing, B/P=battery/performance power profile, P0=base
dead_reckoning predict, F0=base One-Euro filter): `main` (raw Basalt, the
reference), `TBP0`, `TBP0F0`, `TPP0`, `TPP0F0`. The dataset list, calibs and
WMR timing runs are the `DATASETS` / `TIMINGS` config blocks at the top of the
script — add `EMH02` (the missing 6th row) there once the dataset is present.

### What is in git vs. regenerable

Tracked (the reproducible skeleton): `build_baseline.py`, this README,
`_template/`, every run's `meta.yaml`, the final `baseline_table.{txt,json}`,
and each WMR run's recorded **`display.csv` + `camera.csv`** (the actual timing
inputs — not regenerable). Everything else is large and rebuilt by the script:
`slam/` (basalt intermediates), `baseline_targets/` (GT copies),
`baseline_eval/` (estimates), and each run's own discarded SLAM outputs
(`filtering/prediction/imu/...`). Clone → `build_baseline.py` → same table.

## Layout

```
runs/
  README.md
  _template/meta.yaml        # copy this into each new run folder
  01_<label>/                # one recording (e.g. 01_desktop_idle_90hz)
    display.csv              # REQUIRED — display-path timing (when_ns / locate_views / present)
    camera.csv               # REQUIRED — camera-path timing (provides the clock anchor)
    slam_relations.csv       # REQUIRED (flavor A) — SLAM pose + velocity per frame
    imu.csv                  # REQUIRED (flavor A) — gyro + accel samples
    tracking.csv             # optional — raw SLAM poses (for the validation branch)
    prediction.csv           # optional — recorded C++ predicted poses
    filtering.csv            # optional — recorded C++ filtered poses
    meta.yaml                # the condition this run was recorded under
    out/                     # harness output (gitignored)
  02_<label>/
  ...
```

Naming: `NN_<short-label>`, two-digit index + a label that hints at the
condition, e.g. `03_laptop_powersave_stress`.

## Two flavors

The thesis methodology is **Flavor B**: record realistic *timing* on the real
WMR headset, then apply it on top of a dataset that has ground truth.

- **Flavor B — WMR timing injection (the real goal).** Wear the headset, run a
  real app/game under a given machine / power profile / stress; record
  `display.csv` / `camera.csv` (the realistic pacing). **No GT** — the run's own
  poses are discarded. The SLAM / IMU / GT come from the MSD dataset; the WMR
  timing is moved onto the dataset clock and drives `predict_pose`. Needs a
  small harness change (read display/camera from the run, slam/imu from the
  dataset) — today `replay.py` reads all CSVs from one folder.
- **Flavor A — full EuRoC replay under a condition (works today).** Record the
  EuRoC dataset replay (StereoKit) on a given machine/power/stress: the full CSV
  set lands in the folder, GT is the dataset's, score directly. A simpler
  variant to exercise the pipeline before Flavor B is wired.

## Record a run

Point every CSV output at this run's folder (never at `timing/`, so the
canonical baseline is not clobbered), then fill in `$RUN/meta.yaml` (copy from
`_template/meta.yaml`).

**Flavor B — WMR headset (no `EUROC_*` vars; headset plugged in):**

```bash
RUN=runs/01_workstation_perf_beatsaber
mkdir -p "$RUN"; cp _template/meta.yaml "$RUN/meta.yaml"
SDL_VIDEODRIVER=x11 XRT_DEBUG_GUI=1 SLAM_UI=1 \
SLAM_CONFIG=basalt/build/data/vit/<wmr_calib>.toml \
VIT_SYSTEM_LIBRARY_PATH=basalt/build/libbasalt.so \
SLAM_WRITE_CSVS=true SLAM_CSV_PATH=$PWD/$RUN/ \
MONADO_DISPLAY_TIMING_CSV=$PWD/$RUN/display.csv \
MONADO_CAMERA_TIMING_CSV=$PWD/$RUN/camera.csv \
MONADO_PACER_CSV=$PWD/$RUN/pacer.csv \
monado/build/src/xrt/targets/service/monado-service
# then start a real OpenXR app / game and move for ~1-2 min
```

**Flavor A — EuRoC replay under a condition:** the full `monado-service`
recording command from `CLAUDE.md` (with the `EUROC_*` vars), but with
`SLAM_CSV_PATH` + every `MONADO_*_CSV` pointed at `$RUN/`.

To vary the condition: change machine, power profile (`performance`/`powersave`),
run a CPU/GPU stress program in the background, or change the display rate
(`XRT_COMPOSITOR_DEFAULT_FRAMERATE=60/90/120`).

## Run the harness on a run

```bash
# sweep the prediction methods, ATE/RTE vs GT
python xrtslam-metrics/ops/latency_ops.py experiment \
    --timing-dir runs/01_desktop_idle_90hz \
    --dataset MOO01_hand_puncher_1 \
    --out runs/01_desktop_idle_90hz/out

# or a single method
python xrtslam-metrics/ops/latency_ops.py replay \
    --timing-dir runs/01_desktop_idle_90hz \
    --dataset MOO01_hand_puncher_1 \
    --out runs/01_desktop_idle_90hz/out --method dead_reckoning
```
