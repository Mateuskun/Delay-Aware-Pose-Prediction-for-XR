# Delay-Aware Pose Prediction for XR

> Bachelor's thesis — *Delay-Aware SLAM: Modeling and Mitigating Prediction Latency.*

In XR, the pose used to render a frame is always computed from sensor data from
*the past*, then extrapolated to *the future* moment the frame will be shown.
This project measures that prediction latency end-to-end inside a real OpenXR
runtime, and builds an offline harness to evaluate and improve the
predict/filter step that compensates for it — under **realistic** timing, but
against **ground-truth** trajectories.

Built on top of [Monado](https://monado.dev) (open-source OpenXR runtime) and
[Basalt](https://gitlab.freedesktop.org/mateosss/basalt) (visual-inertial SLAM).

## The idea

A clean dataset replay has ground truth but unrealistically perfect timing; a
real headset has realistic timing but no ground truth. This project bridges them
with a **hybrid harness**:

1. **Record** the real timing characteristics on a WMR headset under varied
   conditions (machine, power profile, background load) — timing only.
2. **Inject** that timing onto an existing SLAM dataset that *has* ground truth.
3. **Regenerate** one predicted/filtered pose per rendered frame and score
   ATE/RTE against the dataset ground truth.
4. **Sweep** prediction methods and filter parameters to beat the baseline.

## Repository layout

| Path | What it is |
|---|---|
| `monado/` | The OpenXR runtime, with the timing instrumentation (the C/C++ work). |
| `basalt/` | Visual-inertial SLAM backend, loaded by Monado at runtime. |
| `StereoKit/` | OpenXR test application used to drive the runtime. |
| `xrtslam-metrics/` | Python evaluation toolchain — the offline harness + trajectory metrics. |
| `runs/` | Recorded timing runs (one folder per condition). |
| `timing/` | The canonical baseline recording. |

## The instrumentation

`monado-service` emits, per run, five CSVs sharing one host-monotonic clock:

- **Event tables** — `camera.csv` (sensor → SLAM input timing) and `display.csv`
  (compositor + OpenXR timing). Together they reconstruct the full
  sensor-to-photon delay per frame, broken down by stage.
- **Pose tables** — `tracking.csv` (raw SLAM), `prediction.csv` (predicted to the
  display time), `filtering.csv` (predicted + filtered). Scored separately, they
  decompose pose-at-use-time error into estimation vs. prediction vs. filter.

## The offline harness (`xrtslam-metrics/`)

Python ports of Monado's predict/filter math (golden-tested against the C++
runtime), plus a replay engine that runs them offline at notebook speed.

```bash
# move a recorded run into the dataset clock domain
python xrtslam-metrics/ops/latency_ops.py move-timeline \
    --timing-dir runs/01_example --dataset MOO01_hand_puncher_1 --out runs/01_example/aligned

# sweep the prediction methods, scored ATE/RTE vs ground truth
python xrtslam-metrics/ops/latency_ops.py experiment \
    --timing-dir runs/01_example --dataset MOO01_hand_puncher_1 --out runs/01_example/out
```

Core modules:

- `csvio.py` — CSV loaders + the dataset-clock offset.
- `replay.py` — the replay engine (one pose per `when_ns`, causal).
- `experiment.py` — sweeps the prediction methods and scores ATE/RTE.
- `ops/latency_ops.py` — CLI: `move-timeline` / `replay` / `experiment`.
- `predict.py`, `filter.py`, `math3d.py` — the 1:1 ports of the runtime math.
- `tracking.py` — ATE/RTE metrics (via [evo](https://github.com/MichaelGrupp/evo)).

## Building

```bash
cmake -B monado/build -S monado -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build monado/build
```

`basalt/` and `StereoKit/` are submodules — clone with `--recurse-submodules`.

## Status

- ✅ Timing instrumentation in Monado; five-CSV pipeline.
- ✅ Python ports of predict/filter, golden-validated against C++.
- ✅ Offline replay harness; reproduces the documented baseline.
- ⏳ Recording WMR runs under varied conditions; sweeping predict/filter to
  improve ATE/RTE; porting the improvement back into Monado.

## Acknowledgements

The `xrtslam-metrics` trajectory-metrics tooling is by the thesis supervisor
([Mateo de Mayo](https://gitlab.freedesktop.org/mateosss)); this project extends
it with the timing instrumentation and the offline replay harness.
