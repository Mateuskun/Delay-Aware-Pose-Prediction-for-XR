# Monado visual-inertial tracking measurement tools

This project contains multiple scripts for working around SLAM datasets,
systems, and evaluations. For the evaluation we use EVO as a base but
reimplement in C++ some computations for more efficient evaluation.

There is an [old blog post](https://mateosss.github.io/blog/xrtslam-metrics) for
an overview on what you could expect in the initial versions of the project. Now
it contains much more functionality available.

## Installation and dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install poetry
poetry update
```

## Download groundtruth for usual datasets

```bash
cd test/data/
wget https://gitlab.freedesktop.org/mateosss/basalt-lfs/-/raw/main/targets.tar.xz
tar -xvf targets.tar.xz
```

## Usage examples

### Timing

See a plot of the times each pose from the dataset took to compute. If no
start/end CSV column names are specified the script assumes defaults.

```bash
./timing.py test/data/runs/Basalt/TR5/timing.csv --plot
```

### Features

See the number of features each pose was computed with.
It supports comparing multiple files as well.

```bash
./features.py --plot test/data/runs/Basalt/EMH02/features.csv
```

### Completion

See what percentage of the dataset the run was able to complete without crashing.

```bash
./completion.py test/data/runs/Kimera/TR5/tracking.csv test/data/targets/TR5/cam0.csv
```

### Tracking error

See ATE, RTE, or SDM stats for a particular run.

```bash
./tracking.py ate test/data/targets/EV202/gt.csv test/data/runs/Basalt/EV202/tracking.csv --plot --plot_mode xyz

# Or if you want to compare multiple trajectories
./tracking.py ate test/data/targets/EMH02/gt.csv test/data/runs/Basalt/EMH02/tracking.csv test/data/runs/ORB-SLAM3/EMH02/tracking.csv --plot --plot_mode xy
```

### Batch comparison

Generate tables comparing averages of multiple runs on multiple datasets. Take
notice of the `runs` and `targets` directory structures and of the fact that you
need to specify the start/end timing column for each run name. The latter
will be fixed once standard start/end column names are in place.

```bash
./batch.py test/data/runs/ test/data/targets/
```

`batch.py` expects the `targets` directory to have camera timestamps `cam0.csv`
and optionally groundtruth `gt.csv` files that you can get from the datasets
themselves. To ease things a bit, you can uncompress
`tar -xvf test/data/targets.tar.xz -C test/data/` to get those files for all EuRoC,
TUM-VI room, and [our custom (without
groundtruth)](https://bit.ly/monado-datasets) inside `test/data/targets`.

### Help for each script

- <details><summary>batch.py</summary>

  ```bash
  $ ./batch.py --help
  usage: batch.py [-h] [--timing TIMING TIMING TIMING]
                  [--metrics METRICS [METRICS ...]] [--verbose]
                  [--save_file SAVE_FILE]
                  [--load_files LOAD_FILES [LOAD_FILES ...]]
                  [--allow_ds_prefixes [ALLOW_DS_PREFIXES ...]]
                  [--deny_ds_prefixes [DENY_DS_PREFIXES ...]]
                  [--allow_sys_prefixes [ALLOW_SYS_PREFIXES ...]]
                  [--deny_sys_prefixes [DENY_SYS_PREFIXES ...]]
                  [--highlights [HIGHLIGHTS ...]] [--names NAMES [NAMES ...]]
                  runs_dir targets_dir

  Batch evaluation of Monado visual-inertial runs of datasets

  Example execution:
  python batch.py test/data/runs/ test/data/targets/ \
    --timing Basalt opticalflow_received vio_produced \
    --timing Kimera tracker_pushed processed \
    --timing ORB-SLAM3 about_to_process processed

  positional arguments:
    runs_dir              Directory with runs subdirectories, each with datasets subdirectories.The structure of runs_dir is like: <runs_dir>/<run>/<dataset>/{tracking, timing}.csv
    targets_dir           Directory with dataset groundtruth and camera timestamps.The structure of targets_dir is like: <targets_dir>/<dataset>/{gt, cam0}.csv

  options:
    -h, --help            show this help message and exit
    --timing TIMING TIMING TIMING
                          For each <run> directory in <runs_dir> specify the first and lasttiming column names to use as --timing <run> <first_col> <last_col>.If a <run> is not specified assuming<first_col> = 3 and <last_col> = -1
    --metrics METRICS [METRICS ...]
                          Metrics to evaluate
    --verbose, -v         Print progress information
    --save_file SAVE_FILE
                          Save to file
    --load_files LOAD_FILES [LOAD_FILES ...]
                          Load from file
    --allow_ds_prefixes [ALLOW_DS_PREFIXES ...]
                          Only consider datasets with this prefix
    --deny_ds_prefixes [DENY_DS_PREFIXES ...]
                          After allow_ds_prefixes, ignore datasets with this prefix
    --allow_sys_prefixes [ALLOW_SYS_PREFIXES ...]
                          Only consider systems with this prefix
    --deny_sys_prefixes [DENY_SYS_PREFIXES ...]
                          After allow_sys_prefixes, ignore systems with this prefix
    --highlights [HIGHLIGHTS ...]
                          Highlight best score per row (dataset)
    --names NAMES [NAMES ...]
                          Column names

  ```
  </details>

- <details><summary>completion.py</summary>

    ```bash
    $ ./completion.py --help
    usage: completion.py [-h] [--units {ns,us,ms,s}]
                        tracking_csv cam_timestamps_csv

    Determine information about tracking completion based on groundtruth duration

    positional arguments:
      tracking_csv          File generated from Monado (either tracking.csv or
                            timing.csv)
      cam_timestamps_csv    Dataset cam0.csv (groundtruth file also works but is
                            less precise)

    options:
      -h, --help            show this help message and exit
      --units {ns,us,ms,s}  Time units to show things on

    ```
    </details>

- <details><summary>features.py</summary>

  ```bash
  $ ./features.py --help
  usage: features.py [-h] [-p] features_csvs [features_csvs ...]

  Measure visual features metrics for Monado visual-inertial tracking

  positional arguments:
    features_csvs  Features file generated from Monado

  options:
    -h, --help     show this help message and exit
    -p, --plot     Whether to plot a stacked feature count graph

  ```
  </details>

- <details><summary>timing.py</summary>

  ```bash
  $ ./timing.py --help
  usage: timing.py [-h] [-fc FIRST_COLUMN] [-lc LAST_COLUMN] [-p]
                  [--units {ns,us,ms,s}]
                  timing_csvs [timing_csvs ...]

  Evaluate timing data for Monado visual-inertial tracking

  positional arguments:
    timing_csvs           Timing file generated from Monado

  options:
    -h, --help            show this help message and exit
    -fc FIRST_COLUMN, --first_column FIRST_COLUMN
                          Column name of timing_csvs to use as first timestamp
                          (default: frames_received)
    -lc LAST_COLUMN, --last_column LAST_COLUMN
                          Column name of timing_csvs to use as last timestamp
                          (default: pose_produced)
    -p, --plot            Whether to plot a stacked timing graph
    --units {ns,us,ms,s}  Time units to show things on

  ```
  </details>

- <details><summary>tracking.py</summary>

  ```bash
  $ ./tracking.py --help
  usage: tracking.py [-h] [-p] [--plot_mode {xy,xz,yx,yz,zx,zy,xyz}]
                    [--color_map] [--no_color_map] [--segment_color_map]
                    [--sd_tolerance SD_TOLERANCE] [--start_s START_S]
                    [--end_s END_S] [--names [NAMES ...]]
                    [--sd_error_components {xy,xz,yx,yz,zx,zy,xyz}]
                    {ate,rte,sdm,atec} groundtruth_csv tracking_csvs
                    [tracking_csvs ...]

  Determine absolute pose error for a trajectory and its groundtruth

  positional arguments:
    {ate,rte,sdm,atec}    What tracking metric to compute # ATE Usual absolute
                          trajectory error as described in EVO (see code for
                          specifics) # RTE Usual relative trajectory error as
                          described in EVO (see code for specifics) # SDM The
                          idea of the Segment Drift per Meter metric (SDM) or
                          equivalently the Segment Drift per Second (SDS) is as
                          follows: You have estimated trajectory est[i] and
                          reference trajectory (groundtruth) ref[i]. Both are a
                          sequential list of timestamped poses (ordered
                          chornologically). Let's assume both est and ref have
                          the same number of poses/timestamps, if they don't we
                          can make them match with some postprocessing. 1. You
                          start with pose index i=0, you _align_ est and ref in
                          such a way so that est[i] and ref[i] have the same
                          position 2. You advance i until you get that the error
                          est[i] - ref[i] is greater than a certan threshold E,
                          by default E=1cm (0.01m). 3. Once the threshold is
                          reached you separate all the previous poses until this
                          i (i included) as a segment 4. You now remove all
                          poses from est and ref up to i included and set i=0
                          again. Then repeat from point 1 This will gives us a
                          list of trajectory "segments" in which the error has
                          been lower than E except for the last pose of the
                          segment. Some insights we get from this are: 1. Less
                          pieces mean more accurate trajectory 2. The end of the
                          pieces, with E sufficiently small, will represent
                          moments in the dataset that make accuracy degrade
                          significantly enough. This specific moments need to be
                          studied and improved upon. 3. We can focus on
                          improving just one of these pieces at a time, and so
                          evaluation gets significantly sped up. 4. To find the
                          "types of pieces" we want to focus on running multiple
                          datasets, print each piece score, sort by worse
                          scores, and analyze the worst scoring pieces. I think
                          some "clusters" of type of movements will arise, you
                          can now focus on fixing those types of movement. Now,
                          the number we report, is the following 1. For SDM: The
                          average segment length 2. For SDS: The average segment
                          duration And that's it; it is worth noting that the
                          number is not as important as having the list of
                          segments. In fact the number has some correlation with
                          the RTE metric (although it is not super linear). To
                          use this metric in xrtslam-metrics/tracking.py you can
                          1. Pass `sdm` argument as the `metric` argument
                          (instead of e.g. `ate` or `rte`) 2. By default metrics
                          have diferent random colors just for differentiation
                          but you can use a color map per segment by mixing
                          `--color_map` and `--segument_color_map` 3. With
                          `--sd_tolerance` you can set the segment drift
                          tolerance (i.e. "E") 4. By default
                          `--sd_error_components` is `xyz` meaning that all
                          three axes are used for the est[i] - ref[i] distance
                          but you can use other axes if you want
    groundtruth_csv       Dataset groundtruth file
    tracking_csvs         Tracking files generated from Monado to compare

  options:
    -h, --help            show this help message and exit
    -p, --plot            Enable to show trajectory plot
    --plot_mode {xy,xz,yx,yz,zx,zy,xyz}, -pm {xy,xz,yx,yz,zx,zy,xyz}
                          Axes of the trajectory to plot
    --color_map, -cm      Use color map for trajectory color based on error from
                          groundtruth
    --no_color_map, -nocm
                          Do not use color map for trajectory color based on
                          error from groundtruth
    --segment_color_map, -scm
                          Use color map to paint entire segments in the segment-
                          drift plot
    --sd_tolerance SD_TOLERANCE, -sdtol SD_TOLERANCE
                          Segment error tolerance for the SD metric
    --start_s START_S     Trim start second
    --end_s END_S         Trim end second
    --names [NAMES ...]   Names of each estimate
    --sd_error_components {xy,xz,yx,yz,zx,zy,xyz}, -sdec {xy,xz,yx,yz,zx,zy,xyz}
                          Which axes to use for error computation in the SD
                          metric

  ```
  </details>

- <details><summary>ops/euroc_ops.py</summary>

  ```bash
  $ ./ops/euroc_ops.py --help
  usage: euroc_ops.py [-h]
                      {imu2cam_ts,get_duration,verify,get_max_sensor_dt,cam_offset_ts,trim,apply_imu_calib,apply_transform,get_csv_duration,preview_video}
                      ... dataset_path

  Sanitize EuRoC datasets

  positional arguments:
    {imu2cam_ts,get_duration,verify,get_max_sensor_dt,cam_offset_ts,trim,apply_imu_calib,apply_transform,get_csv_duration,preview_video}
                          What operation to perform
      imu2cam_ts          Create a new IMU csv with timestamps modified (read
                          code)
      get_duration        Get duration of dataset
      verify              Perform many asserts on the dataset to check its
                          integrity
      get_max_sensor_dt   Get maximum sensor delta time
      cam_offset_ts       Create a new camera csv (and its extra file) with
                          timestamps modified by an offset
      trim                Trim dataset
      apply_imu_calib     Apply IMU calibration to dataset samples
      apply_transform     Get trajectory with SE3 transform applied
      get_csv_duration    Get duration of a sensor csv file
      preview_video       Generate a preview video of the entire dataset.
                          Dataset path should be an absolute path
    dataset_path          Dataset path (the path that contains the mav0
                          directory)

  options:
    -h, --help            show this help message and exit

  ```
  </details>

- <details><summary>ops/rosbag_ops.py</summary>

  ```bash
  $ ./ops/rosbag_ops.py --help
  usage: rosbag_ops.py [-h] {euroc2ros} ...

  Helper commands to convert datasets between rosbags and EuRoC formats

  positional arguments:
    {euroc2ros}  What operation to perform
      euroc2ros  Convert an dataset in EuRoC ASL format into a ROS 1 or 2 bag

  options:
    -h, --help   show this help message and exit

  ```
  </details>

- <details><summary>ops/opencv_ops.py</summary>
  TODO: Fix dependy conflict of opencv-python, different numpy required from scipy

  ```bash
  $ ./ops/opencv_ops.py --help
  usage: opencv_ops.py [-h] {undistort} ...

  Apply OpenCV operations

  positional arguments:
    {undistort}  What operation to perform
      undistort  Undistort all images in a directory

  options:
    -h, --help   show this help message and exit

  ```
  </details>

- <details><summary>ops/dpvo_ops.py</summary>

  ```bash
  $ ./ops/dpvo_ops.py --help
  usage: dpvo_ops.py [-h] {euroc2tumrgbd,traj_dpvo2euroc} ...

  Helper commands to convert between DROID SLAM and EuRoC formats

  positional arguments:
    {euroc2tumrgbd,traj_dpvo2euroc}
                          What operation to perform
      euroc2tumrgbd       Convert an EuRoC dataset into TUM RGBD format used by
                          DPVO
      traj_dpvo2euroc     Convert a DPVO trajectory to euroc format

  options:
    -h, --help            show this help message and exit

  ```
  </details>

- <details><summary>ops/okvis2_ops.py</summary>

  ```bash
  $ ./ops/okvis2_ops.py --help
  usage: okvis2_ops.py [-h] {bslt2okvis2_calib,traj_okvis2euroc} ...

  Helper commands to convert data between okvis2 and EuRoC formats

  positional arguments:
    {bslt2okvis2_calib,traj_okvis2euroc}
                          What operation to perform
      bslt2okvis2_calib   Convert a Basalt calibration file into OKVIS2
                          calibration file
      traj_okvis2euroc    Convert a OKVIS2 trajectory file into EuRoC format

  options:
    -h, --help            show this help message and exit

  ```
  </details>

- <details><summary>ops/dmvio_ops.py</summary>

  ```bash
  $ ./ops/dmvio_ops.py --help
  usage: dmvio_ops.py [-h] {traj_dm2euroc,euroc2dm_files} ...

  Helper commands to convert between DM-VIO and EuRoC formats

  positional arguments:
    {traj_dm2euroc,euroc2dm_files}
                          What operation to perform
      traj_dm2euroc       Convert a dm-vio output trajectory into euroc
                          trajectory format
      euroc2dm_files      Export interpolated imu.txt and times.txt files as
                          required by dm-vio from an euroc dataset

  options:
    -h, --help            show this help message and exit

  ```
  </details>

- <details><summary>ops/colmap_ops.py</summary>

  ```bash
  $ ./ops/colmap_ops.py --help
  usage: colmap_ops.py [-h] {gen_cameras,gen_images} ...

  Helper commands to convert data between colmap and EuRoC formats: See
  https://colmap.github.io/format.html

  positional arguments:
    {gen_cameras,gen_images}
                          What operation to perform
      gen_cameras         Generate cameras.txt from calibration json
      gen_images          Generate images.txt from EuRoC dataset

  options:
    -h, --help            show this help message and exit

  ```
  </details>

- <details><summary>ops/hilti_rosbag_to_euroc.py</summary>

  ```bash
  TODO: Requires a ROS1 environment setup so that `import rosbag` works, also requires opencv-python which conflicts with numpy
  $ ./ops/hilti_rosbag_to_euroc.py --help
  usage: hilti_rosbag_to_euroc.py [-h] {ds,gt} ...

  Make EuRoC dataset from Hilti ROS bag. Usage example for dataset exp14 (get it
  from https://hilti-challenge.com/dataset-2022.html): $
  ./utils/hilti_rosbag_to_euroc.py ds bags/exp14_basement_2.bag exp14 $
  ./utils/hilti_rosbag_to_euroc.py gt ~/Downloads/exp14_basement_2_imu.txt
  exp14/mav0/state_groundtruth_estimate0/data.csv

  positional arguments:
    {ds,gt}     Convert hilti rosbag or groundtruth file to euroc format
      ds        Convert hilti rosbag to euroc dataset
      gt        Convert hilti groundtruth file to euroc groundtruth csv

  options:
    -h, --help  show this help message and exit

  ```
  </details>

- <details><summary>ops/openvins_ops.py</summary>

  ```bash
  $ ./ops/openvins_ops.py --help
  usage: openvins_ops.py [-h] {traj_ov2euroc} ...

  Helper commands to convert between OpenVins and EuRoC formats

  positional arguments:
    {traj_ov2euroc}  What operation to perform
      traj_ov2euroc  Convert a OpenVINS output trajectory into euroc trajectory
                    format

  options:
    -h, --help       show this help message and exit

  ```
  </details>

- <details><summary>ops/jsondiff.py</summary>

  ```bash
  $ ./jsondiff.py --help
  usage: jsondiff.py [-h] [--use_relative] input_file1 input_file2 output_file

  Compute numerical differences between two JSON files with the same structure

  positional arguments:
    input_file1         Path to the first input JSON file
    input_file2         Path to the second input JSON file
    output_file         Path to the output JSON file

  options:
    -h, --help          show this help message and exit
    --use_relative, -r  Whether to show relative percentage diff instead of
                        absolute diff

  ```
  </details>

- <details><summary>ops/executor.py</summary>

  It is a Python script to execute multiple shell commands in parallel up to
  some given number of concurrent jobs. Requires hardcodding commands.

  </details>

- <details><summary>cpp/alignment_all_over_time.py</summary>

  ```bash
  $ ./cpp/alignment_all_over_time.py --help
  usage: alignment_all_over_time.py [-h] [--trim TRIM TRIM]
                                    [--names [NAMES ...]] [--title TITLE]
                                    [--plot_mode {xy,xz,yx,yz,zx,zy,xyz}]
                                    dataset ests [ests ...]

  ATE evolution

  positional arguments:
    dataset               EuRoC dataset path
    ests                  Estimated trajectory CSV file

  options:
    -h, --help            show this help message and exit
    --trim TRIM TRIM      Trim 0.0 to 1.0 of the reference (default: (0.0, 1.0))
    --names [NAMES ...]   Names of each estimate (default: None)
    --title TITLE         Title of this plot (default: PLOT)
    --plot_mode {xy,xz,yx,yz,zx,zy,xyz}
                          Axes of the trajectory to plot (default: xy)

  ```
  </details>

  </details>

- <details><summary>cpp/alignment_ate_over_time.py</summary>

  ```bash
  $ ./cpp/alignment_ate_over_time.py --help
  usage: alignment_ate_over_time.py [-h] [--trim TRIM TRIM]
                                    [--names [NAMES ...]] [--title TITLE]
                                    [--save SAVE] [--no-plot]
                                    ref ests [ests ...]

  ATE evolution

  positional arguments:
    ref                  Reference trajectory CSV file
    ests                 Estimated trajectory CSV file

  options:
    -h, --help           show this help message and exit
    --trim TRIM TRIM     Trim 0.0 to 1.0 of the reference (default: (0.0, 1.0))
    --names [NAMES ...]  Names of each estimate (default: None)
    --title TITLE        Title of this plot (default: PLOT)
    --save SAVE          Save plot as (default: None)
    --no-plot            Disable plot (default: False)

  ```
  </details>

- <details><summary>cpp/alignment_ate_over_time.py</summary>

  ```bash
  $ ./cpp/alignment_ate_over_time.py --help
  usage: alignment_ate_test.py [-h] ref est

  ATE evolution

  positional arguments:
    ref         Reference trajectory CSV file
    est         Estimated trajectory CSV file

  options:
    -h, --help  show this help message and exit

  ```
  </details>

- <details><summary>cpp/alignment_ate_test.py</summary>

  ```bash
  $ ./cpp/alignment_ate_test.py --help
  usage: alignment_ate_test.py [-h] ref est

  ATE evolution

  positional arguments:
    ref         Reference trajectory CSV file
    est         Estimated trajectory CSV file

  options:
    -h, --help  show this help message and exit

  ```
  </details>

- <details><summary>cpp/alignment_rte_over_time.py</summary>

  ```bash
  $ ./cpp/alignment_rte_over_time.py --help
  usage: alignment_rte_over_time.py [-h] [--trim TRIM TRIM]
                                    [--names [NAMES ...]] [--title TITLE]
                                    [--save SAVE] [--no-plot]
                                    ref ests [ests ...]

  ATE evolution

  positional arguments:
    ref                  Reference trajectory CSV file
    ests                 Estimated trajectory CSV file

  options:
    -h, --help           show this help message and exit
    --trim TRIM TRIM     Trim 0.0 to 1.0 of the reference (default: (0.0, 1.0))
    --names [NAMES ...]  Names of each estimate (default: None)
    --title TITLE        Title of this plot (default: PLOT)
    --save SAVE          Save plot as (default: None)
    --no-plot            Disable plot (default: False)

  ```
  </details>

- <details><summary>cpp/alignment_rte_test.py</summary>

  ```bash
  $ ./cpp/alignment_rte_test.py --help
  usage: alignment_rte_test.py [-h] ref est

  ATE evolution

  positional arguments:
    ref         Reference trajectory CSV file
    est         Estimated trajectory CSV file

  options:
    -h, --help  show this help message and exit

  ```
  </details>


# Roadmap

<details><summary>Various TODOs</summary>

## Installation:

- specific python version
- create .venv
- source it
- pip install poetry
- poetry update

## Getting targets:

- Get targets tar.xz (maybe a download script?)
- uncompress into test/data/...etc
- OPTIONAL: Would be nice for the targets directory to be default and not need to pass it

## cpp:

- Instructions for python requirements (use either cpp/requirements.txt or poetry deps)
- Unify Makefile and CMakeLists.txt
- ./alignment_ate_over_time.py $xrtmet/test/data/targets/MOO06/gt.csv ~/Desktop/christoph/trajectories/compressedCRF35bitrate0/MOO06/tracking.csv ~/Desktop/christoph/trajectories/compressedCRF15bitrate0/MOO06/tracking.csv --title MOO06 --names CRF35 CRF15
- ./alignment_rte_over_time.py $xrtmet/test/data/targets/MOO06/gt.csv ~/Desktop/christoph/trajectories/compressedCRF35bitrate0/MOO06/tracking.csv ~/Desktop/christoph/trajectories/compressedCRF15bitrate0/MOO06/tracking.csv --title MOO06 --names CRF35 CRF15
- ./alignment_all_over_time.py --trim 0.5 0.6 $msdmo/MOO_others/MOO06_inspect_hard ~/Desktop/christoph/trajectories/compressedCRF15bitrate0/MOO06/tracking.csv ~/Desktop/christoph/trajectories/compressedCRF35bitrate0/MOO06/tracking.csv --title MOO06 --names CRF15 CRF35
- TEST RTE: should be deleted? ./alignment_rte_test.py $xrtmet/test/data/targets/MOO02/gt.csv ~/Desktop/delete/basalt/results/BasaltDET/MOO02/tracking.csv
- TEST ATE: should be deleted? ./alignment_ate_test.py $xrtmet/test/data/targets/MOO02/gt.csv ~/Desktop/delete/basalt/results/BasaltDET/MOO02/tracking.csv

## batch.py:

- Usage of --save_file/load_files
- Usage of --metric
- Usage of --allow_prefix, --deny_prefix
- Usage of --highlight
- Add --names parameter to not need the \n trick
- Things stopped working unless you use the --names param
- Use cpp alignment (atec/rtec) when available

## euroc_ops

- Rename directory from euroc_ops to something like "data_ops"
- Document usage of every script (at least copy the --help page) (see video ref)

## Misc

- Address remaining `TODO@mateosss`s

</details>

# License

The license of contributions in this repository is BSL, if you contribute to it
you accept so. Dependencies can, and often have, different licenses.
