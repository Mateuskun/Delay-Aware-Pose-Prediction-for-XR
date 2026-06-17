#!/usr/bin/env python

import alignment as al
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
import time
import argparse
from pathlib import Path
import mplcursors
from scipy.spatial.transform import Rotation as R
from math import sqrt
from typing import List, Tuple
from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator, MultipleLocator
import matplotlib.lines as mlines

from evo.core.result import Result
from evo.core.trajectory import PoseTrajectory3D
from evo.tools.plot import PlotMode, PlotException, _get_length_formatter
from evo.tools import file_interface, plot
from evo.tools.settings import SETTINGS
from evo.core.units import LENGTH_UNITS, Unit
from evo.core import sync, lie_algebra as lie

rcParams["font.family"] = "CMU Serif"
rcParams["font.family"] = "CMU Serif"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["axes.linewidth"] = 1

DEFAULT_PLOT_MODE = "xy"  # This also works for everything (not sure why it does for MI?)

SCALAR = np.float32
DELTA = 6
FRAME_HELP_TEXT = (
    "Click on the curves to select a frame.\n"
    "Right click on yellow box to deselect.\n"
    "Move frame with Shift+{Left,Right} arrows.\n"
)

# TODO@mateosss: duped
COLORS = [  # Regular cycle
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FFC107",  # amber
    "#E91E63",  # pink
    "#673AB7",  # deeppurple
    "#00BCD4",  # cyan
    "#CDDC39",  # lime
    "#FF5722",  # deeporange
    "#9C27B0",  # purple
    "#03A9F4",  # lightblue
    "#8BC34A",  # lightgreen
    "#FF9800",  # orange
    "#3F51B5",  # indigo
    "#009688",  # teal
    "#FFEB3B",  # yellow
    "#F44336",  # red
    "#795548",  # brown
    "#607D8B",  # bluegrey
]

DARK_BLUEGREY = "#263238"
RED = "#F44336"
BLUEGREY = "#607D8B"

# TODO@mateosss: duped
make_color_iterator = lambda: cycle(COLORS)

previous_lines = {
    "ATE": None,
    "dATE": None,
    "RTE": None,
    "dRTE": None,
    "ATE norm": None,
    "dATE norm": None,
    "RTE norm": None,
    "dRTE norm": None,
    "ATE log": None,
    "dATE log": None,
    "RTE log": None,
    "dRTE log": None,
}
frame_text = None
frame_img = None

ds_path: Path
cam_ts: np.ndarray
cam_pngs: List[str]

global first_i
def enable_mouse_interactions(fig, ax):
    global previous_lines

    # Show vertical line on mouse action
    cursor = mplcursors.cursor(ax.values(), hover=False)

    def on_select(sel):
        global previous_lines, frame_text, frame_img

        # Remove previous lines
        for k, line in previous_lines.items():
            if line is not None:
                line.remove()
                previous_lines[k] = None

        # Draw new vertical lines
        for ax_name, a in ax.items():
            if ax_name not in previous_lines:
                continue
            line = a.axvline(x=sel.target[0], color=RED, linestyle="--", linewidth=2, alpha=1.0)
            previous_lines[ax_name] = line

        # Show image
        if "PLOT" in sel.artist.axes.title.get_text(): # When clicking trajectory
            # Here sel.index goes from 0 to count of poses, but we want to match it to the frame timestamps, which are independent of the number of poses (e.g. Basalt has a pose for each frame, but other systems don't necessarily have that)
            global first_i
            match_idx = first_i + int(sel.index)
        else: # When clicking line plots
            match_idx = np.searchsorted(cam_ts, cam_ts[0] + np.int64(sel.target[0] * 1e9))
        frame_png = cam_pngs[match_idx]
        # print(f"Selected frame {match_idx} with timestamp {cam_ts[match_idx]} and png {frame_png}")
        img = mpimg.imread(ds_path / "mav0/cam0/data" / frame_png)
        if frame_img is None:
            frame_img = ax["frame"].imshow(img, cmap="gray")
        else:
            frame_img.set_data(img)
        frame_text.set_visible(False)
        frame_img.set_visible(True)

        # Set text in bubble
        sel.annotation.set_text(f"run:{sel.artist.get_label()}\nframe: {match_idx}\ntime: {sel.target[0]:.2f}s")

        # Finish draw
        # fig.canvas.draw() # For some reason this crashes
        plt.draw()

    cursor.connect("add", on_select)

    # This works a bit weird when clicking multiple times for different frames
    # def on_deselect(sel):
    #     global previous_lines
    #     for k, line in previous_lines.items():
    #         if line is not None:
    #             line.remove()
    #             previous_lines[k] = None
    #     frame_img.set_visible(False)
    #     frame_text.set_visible(True)
    #     fig.canvas.draw()

    # cursor.connect("remove", on_deselect)


# TODO@mateosss: duped
class TrajectoryErrorPlot:
    fig: plt.Figure = None
    ax: plt.Axes = None
    show_plot: bool = False
    plot_mode: PlotMode = PlotMode.xyz
    use_color_map: bool = False
    traj_ref: PoseTrajectory3D

    def __init__(self, fig, ax, show_plot: bool, plot_mode: str, use_color_map: bool) -> None:
        self.fig = fig
        self.ax = ax
        self.show_plot = show_plot
        if not self.show_plot:
            return

        self.plot_mode = PlotMode(plot_mode)
        self.use_color_map = use_color_map
        self.ax["traj"] = self.prepare_axis()
        self.colors = make_color_iterator()

    def prepare_axis(self, subplot_arg: int = 111, length_unit: Unit = Unit.meters) -> Axes:
        """
        prepares an axis according to the plot mode (for trajectory plotting)
        :param subplot_arg: optional if using subplots - the subplot id (e.g. '122')
        :param length_unit: Set to another length unit than meters to scale plots.
                            Note that trajectory data is still expected in meters.
        :return: the matplotlib axis
        """
        if length_unit not in LENGTH_UNITS:
            raise PlotException(f"{length_unit} is not a length unit")

        pos = self.ax["traj"].get_position()
        subplot_pos = self.ax["traj"].get_subplotspec()
        self.fig.delaxes(self.ax["traj"])

        if self.plot_mode == PlotMode.xyz:
            ax = self.fig.add_subplot(subplot_pos, projection="3d")
            # ax.set_zlim(-5, 5)
            ax.set_zlim(-4, 4)
        else:
            ax = self.fig.add_axes(pos)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-2, 2)

        if self.plot_mode in {PlotMode.xy, PlotMode.xz, PlotMode.xyz}:
            xlabel = f"$x$ ({length_unit.value})"
        elif self.plot_mode in {PlotMode.yz, PlotMode.yx}:
            xlabel = f"$y$ ({length_unit.value})"
        else:
            xlabel = f"$z$ ({length_unit.value})"
        if self.plot_mode in {PlotMode.xy, PlotMode.zy, PlotMode.xyz}:
            ylabel = f"$y$ ({length_unit.value})"
        elif self.plot_mode in {PlotMode.zx, PlotMode.yx}:
            ylabel = f"$x$ ({length_unit.value})"
        else:
            ylabel = f"$z$ ({length_unit.value})"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if self.plot_mode == PlotMode.xyz and isinstance(ax, Axes3D):
            ax.set_zlabel(f"$z$ ({length_unit.value})")
        if SETTINGS.plot_invert_xaxis:
            self.fig.gca().invert_xaxis()
        if SETTINGS.plot_invert_yaxis:
            self.fig.gca().invert_yaxis()
        if not SETTINGS.plot_show_axis:
            ax.set_axis_off()

        if length_unit is not Unit.meters:
            formatter = _get_length_formatter(length_unit)
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
            if self.plot_mode == PlotMode.xyz and isinstance(ax, Axes3D):
                ax.zaxis.set_major_formatter(formatter)

        return ax

    def plot_reference_trajectory(self, traj_ref: PoseTrajectory3D, ref_name: str = "Ground-truth"):
        if not self.show_plot:
            return
        self.traj_ref = traj_ref
        plot.traj(
            self.ax["traj"],
            self.plot_mode,
            traj_ref,
            style="--",
            color="gray",
            label=ref_name,
            plot_start_end_markers=True,
        )

    def plot_estimate_trajectory(  # pylint: disable=arguments-differ
        self,
        traj_est: PoseTrajectory3D,
        est_name: str = "estimate",
        T_ref_est: np.ndarray = None,
    ):
        if not self.show_plot:
            return

        if T_ref_est is not None:
            t_a = T_ref_est[:3, 3]
            r_a = T_ref_est[:3, :3]
            traj_est.transform(lie.se3(r_a, t_a))

        plot.traj(
            self.ax["traj"],
            self.plot_mode,
            traj_est,
            color=next(self.colors),
            label=est_name,
            alpha=0.75,
            plot_start_end_markers=True,
        )

    def show(self):
        plt.show()


def compute_rte(ts, est_xyz, ref_xyz, est_quat, ref_quat, i, j):
    assert j > i

    def se3(xyz, quat):
        mat = np.eye(4, dtype=SCALAR)
        mat[:3, :3] = R.from_quat(quat).as_matrix()
        mat[:3, 3] = xyz
        return mat

    def se3_inv(mat):
        inv_mat = np.eye(4, dtype=SCALAR)
        inv_mat[:3, :3] = mat[:3, :3].T
        inv_mat[:3, 3] = -inv_mat[:3, :3] @ mat[:3, 3]
        return inv_mat

    # number of pose pairs + 1 for the initial pose/timestamp
    rel_count = (j - i) // DELTA
    rel_count += 1 if (j - i) % DELTA != 0 else 0  # Edge case when j-i is mult. of DELTA
    # Example:
    # - If DELTA=3, i=0, j=13
    # - ts=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # - est_xyz=[p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12]
    # Then:
    # timestamps = [0, 3, 6, 9, 12]
    # residuals = [0, rte(p0, p3), rte(p3, p6), rte(p6, p9), rte(p9, p12)]

    # - If DELTA=3, i=0, j=12
    # - ts=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # - est_xyz=[p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11]
    # Then:
    # timestamps = [0, 3, 6, 9, 12]
    # residuals = [0, rte(p0, p3), rte(p3, p6), rte(p6, p9)]
    timestamps = np.zeros(rel_count, dtype=np.int64)
    residuals = np.zeros(rel_count, dtype=SCALAR)
    timestamps[0] = ts[i, 0]
    residuals[0] = 0

    for k in range(i + DELTA, j, DELTA):
        k0 = k - DELTA
        est0 = se3(est_xyz[:, k0], est_quat[:, k0])
        ref0 = se3(ref_xyz[:, k0], ref_quat[:, k0])

        k1 = k
        est1 = se3(est_xyz[:, k1], est_quat[:, k1])
        ref1 = se3(ref_xyz[:, k1], ref_quat[:, k1])

        est_delta = se3_inv(est0) @ est1
        ref_delta = se3_inv(ref0) @ ref1
        estref_delta = se3_inv(est_delta) @ ref_delta
        rel_err = norm(estref_delta[:3, 3])

        timestamps[k1 // DELTA] = ts[k1, 0]
        residuals[k1 // DELTA] = rel_err

    return timestamps, residuals


def get_sanitized_trajectories(
    tracking_csv: Path,
    groundtruth_csv: Path,
    silence=False,
    start_s: float = None,
    end_s: float = None,
) -> Tuple[PoseTrajectory3D, PoseTrajectory3D]:
    """Trim and synchronizes trajectories so that they have the same amount of poses"""
    global first_i
    # NOTE: Evo uses doubles for its timestamps and thus looses a bit of
    # precision, but even in the worst case, the precision is about ~1usec
    traj_ref = file_interface.read_euroc_csv_trajectory(groundtruth_csv)
    traj_est = file_interface.read_euroc_csv_trajectory(tracking_csv)

    # Trim both trajectories so that only overlapping timestamps are kept
    e0, e1 = traj_est.timestamps[0], traj_est.timestamps[-1]
    r0, r1 = traj_ref.timestamps[0], traj_ref.timestamps[-1]
    first_ts = max(e0, r0)
    last_ts = min(e1, r1)

    if start_s is not None:
        first_ts = r0 + start_s
        i = np.searchsorted(traj_est.timestamps, first_ts)
        j = np.searchsorted(traj_ref.timestamps, first_ts)

        if i == len(traj_est.timestamps):
            i = -1
        if j == len(traj_ref.timestamps):
            j = -1

        e0 = traj_est.timestamps[i]
        r0 = traj_ref.timestamps[j]
        first_ts = max(e0, r0)
        first_i = i
    if end_s is not None:
        last_ts = r0 + end_s - start_s
        i = np.searchsorted(traj_est.timestamps, last_ts)
        j = np.searchsorted(traj_ref.timestamps, last_ts)

        if i == len(traj_est.timestamps):
            i = -1
        if j == len(traj_ref.timestamps):
            j = -1

        e1 = traj_est.timestamps[i]
        r1 = traj_ref.timestamps[j]
        last_ts = min(e1, r1)

    # Return empty trajectory if no overlapping timestamps
    if first_ts > last_ts:
        id_pos = np.array([[0, 0, 0]])
        id_quat = np.array([[1, 0, 0, 0]])
        id_ts = np.array([0])
        MakeEmptyTrajectory = lambda: PoseTrajectory3D(id_pos, id_quat, id_ts)
        return MakeEmptyTrajectory(), MakeEmptyTrajectory()

    traj_ref.reduce_to_time_range(first_ts, last_ts)
    traj_est.reduce_to_time_range(first_ts, last_ts)

    # TODO: PR with a more realtime-appropriate trajectory alignment.
    # `associate_trajectories`` synchronizes the two trajectories as follows:
    # 1. The trajectory with less poses is kept
    # 2. In the second trajectory only the poses with closest timestamps to the
    #    first trajectory are kept.
    # A way of syncing trajectories a tad more meaningful for VR would be to
    # always use the previously tracked pose for each groundtruth pose.
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    return traj_est, traj_ref


def main():
    parser = argparse.ArgumentParser(
        description="ATE evolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trim",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        help="Trim 0.0 to 1.0 of the reference",
    )
    parser.add_argument("dataset", type=Path, help="EuRoC dataset path")
    parser.add_argument("ests", type=Path, nargs="+", help="Estimated trajectory CSV file")
    parser.add_argument("--names", type=str, nargs="*", help="Names of each estimate")
    parser.add_argument("--title", type=str, default="PLOT", help="Title of this plot")
    parser.add_argument(
        "--plot_mode",
        default=DEFAULT_PLOT_MODE,
        help="Axes of the trajectory to plot",
        choices=["xy", "xz", "yx", "yz", "zx", "zy", "xyz"],
    )
    args = parser.parse_args()
    trim_start, trim_end = args.trim
    dataset_path = args.dataset
    est_csvs = args.ests
    names = args.names
    title = args.title
    plot_mode = args.plot_mode

    ref_csv = dataset_path / "mav0/state_groundtruth_estimate0/data.csv"
    if not ref_csv.exists():
        ref_csv = dataset_path / "mav0/gt/data.csv"
    if not ref_csv.exists():
        ref_csv = dataset_path / "mav0/mocap0/data.csv"
    assert ref_csv.exists(), f"Reference CSV not found at {ref_csv}"
    frames_csv = dataset_path / "mav0/cam0/data.csv"

    if frames_csv is not None:
        with open(frames_csv, "r", encoding="utf-8") as f:
            cam_lines = f.readlines()

        cam_lines = [line.strip().split(",") for line in cam_lines]

        if cam_lines[0][0].startswith("#"):
            cam_lines = cam_lines[1:]

        global cam_ts, cam_pngs, ds_path
        cam_ts = np.array([int(line[0]) for line in cam_lines], dtype=np.int64)
        cam_pngs = [line[1] for line in cam_lines]
        ds_path = dataset_path

    assert len(names) == len(est_csvs)

    mosaic = [
        ["ATE", "ATE norm", "ATE norm", "ATE norm", "ATE norm", "ATE norm", "ATE log"],
        ["dATE", "dATE norm", "dATE norm", "dATE norm", "dATE norm", "dATE norm", "dATE log"],
        ["RTE", "RTE norm", "RTE norm", "RTE norm", "RTE norm", "RTE norm", "RTE log"],
        ["dRTE", "dRTE norm", "dRTE norm", "dRTE norm", "dRTE norm", "dRTE norm", "dRTE log"],
        ["traj", "traj", "traj", "traj", "frame", "frame", "frame"],
        ["traj", "traj", "traj", "traj", "frame", "frame", "frame"],
        ["traj", "traj", "traj", "traj", "frame", "frame", "frame"],
        ["traj", "traj", "traj", "traj", "frame", "frame", "frame"],
        ["traj", "traj", "traj", "traj", "frame", "frame", "frame"],
    ]

    mosaic = [
        ["traj", "traj", "traj", "traj", "traj", "traj", "traj", "traj", "traj", "frame", "frame", "frame", "frame", "frame", "frame", "frame"],
        ["traj", "traj", "traj", "traj", "traj", "traj", "traj", "traj", "traj", "frame", "frame", "frame", "frame", "frame", "frame", "frame"],
        ["traj", "traj", "traj", "traj", "traj", "traj", "traj", "traj", "traj", "frame", "frame", "frame", "frame", "frame", "frame", "frame"],
        ["traj", "traj", "traj", "traj", "traj", "traj", "traj", "traj", "traj", "frame", "frame", "frame", "frame", "frame", "frame", "frame"],
        ["traj", "traj", "traj", "traj", "traj", "traj", "traj", "traj", "traj", "frame", "frame", "frame", "frame", "frame", "frame", "frame"],
        ["traj", "traj", "traj", "traj", "traj", "traj", "traj", "traj", "traj", "frame", "frame", "frame", "frame", "frame", "frame", "frame"],
        ["RTE norm", "RTE norm", "RTE norm", "RTE norm", "RTE norm", "RTE norm", "RTE norm", "RTE norm", "RTE norm", "RTE", "RTE", "RTE", "RTE", "RTE log", "RTE log", "RTE log"],
        ["dRTE norm", "dRTE norm", "dRTE norm", "dRTE norm", "dRTE norm", "dRTE norm", "dRTE norm", "dRTE norm", "dRTE norm", "dRTE", "dRTE", "dRTE", "dRTE", "dRTE log", "dRTE log", "dRTE log"],
        ["ATE norm", "ATE norm", "ATE norm", "ATE norm", "ATE norm", "ATE norm", "ATE norm", "ATE norm", "ATE norm", "ATE", "ATE", "ATE", "ATE", "ATE log", "ATE log", "ATE log"],
        ["dATE norm", "dATE norm", "dATE norm", "dATE norm", "dATE norm", "dATE norm", "dATE norm", "dATE norm", "dATE norm", "dATE", "dATE", "dATE", "dATE", "dATE log", "dATE log", "dATE log"],
    ] # fmt: skip

    curves = list(set(m for row in mosaic for m in row) - {"frame", "traj"})
    fig, ax = plt.subplot_mosaic(mosaic)

    # Make all x axes shared
    for c in curves:
        ax[c].sharex(ax[curves[0]])
        ax[c].spines["top"].set_visible(False)
        ax[c].spines["right"].set_visible(False)
        ax[c].spines["left"].set_color(DARK_BLUEGREY)
        ax[c].spines["bottom"].set_color(DARK_BLUEGREY)
        if "norm" in c:
            ax[c].set_ylim(0, 1.05)
            ax[c].set_yticks([0, 0.5, 1.0])
            ax[c].set_yticklabels(["0", None, "1.0"])
        ax[c].grid(alpha=0.3)
        ax[c].tick_params(axis="both", which="both", reset=True, direction="out", length=4, color=DARK_BLUEGREY)
        ax[c].tick_params(bottom=True, left=True, top=False, right=False)
        metric = "RTE" if "RTE" in c else "ATE"
        continuation = " inc." if c.startswith("d") else ""
        ax[c].set_ylabel(f"Normalized\n{metric}{continuation}")
        if c.startswith("d"):
            ax[c].set_xlabel("Time [s]")
            ax[c].xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            plt.setp(ax[c].get_xticklabels(), visible=False)
            plt.setp(ax[c].get_xticklines(), visible=False)

    # ax["dRTE norm"].set_xlabel("Time [s]")
    # ax["dRTE norm"].xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.setp(ax["RTE norm"].get_xticklabels(), visible=False)
    # plt.setp(ax["RTE norm"].get_xticklines(), visible=False)

    for c in curves:
        if "norm" in c:
            continue
        # ax[c].set_visible(False)

    fig.tight_layout()
    # plt.subplots_adjust(hspace=0.05, wspace=0.05)

    # plt.subplots_adjust(wspace=0.5)
    plt.subplots_adjust(top=0.98, bottom=0.02)

    title = f"{title} ({trim_start*100:.0f}-{trim_end*100:.0f}%)"
    # fig.suptitle(title)

    global frame_text
    frame_text = ax["frame"].text(0.5, 0.5, FRAME_HELP_TEXT, ha="center", va="center", fontsize=10)

    ax["ATE"].set_ylabel("ATE [cm]")
    ax["dATE"].set_ylabel("dATE [mm]")
    ax["RTE"].set_ylabel("RTE [cm]")
    ax["dRTE"].set_ylabel("dRTE [mm]")

    # TODO@mateosss: xlabel not showing
    ax["dRTE"].set_xlabel("Time [s]")

    # ax["ATE"].set_title("Error")
    # ax["ATE norm"].set_title("Errors [0, 1] normalized")
    # ax["ATE log"].set_title("Errors in log scale")

    ax["ATE log"].set_yscale("log", base=10)
    ax["dATE log"].set_yscale("log", base=10)
    ax["RTE log"].set_yscale("log", base=10)
    ax["dRTE log"].set_yscale("log", base=10)

    # SETTINGS["plot_seaborn_style"] = "whitegrid"
    # SETTINGS["plot_fontfamily"] = "serif"
    # SETTINGS["plot_fontscale"] = 1.2
    # SETTINGS["plot_linewidth"] = 1.0
    # SETTINGS["plot_reference_linestyle"] = "-"
    # SETTINGS["plot_figsize"] = [5,4.5]
    # SETTINGS["plot_usetex"] = True
    # SETTINGS["plot_start_end_markers"] = True

    with open(ref_csv, "r", encoding="utf-8") as f:
        ref_lines = f.readlines()
    ref_lines = [line.strip().split(",") for line in ref_lines]
    if ref_lines[0][0].startswith("#"):
        ref_lines = ref_lines[1:]
    ref_ts = np.array([int(line[0]) for line in ref_lines], dtype=np.int64)

    ref_start_idx = int(trim_start * len(ref_ts))
    ref_end_idx = int(trim_end * len(ref_ts))

    start_s = (ref_ts[ref_start_idx] - ref_ts[0]) / 1e9
    end_s = (ref_ts[ref_end_idx - 1] - ref_ts[0]) / 1e9

    show_plot = True
    use_color_map = False
    tracking_plot = TrajectoryErrorPlot(fig, ax, show_plot, plot_mode, use_color_map)

    ax["traj"].grid(linewidth=1, alpha=0.3)
    for spine in ax["traj"].spines.values():
        spine.set_color(DARK_BLUEGREY)
    ax["traj"].set_xlabel(f"{DEFAULT_PLOT_MODE[0]} [m]")
    ax["traj"].set_ylabel(f"{DEFAULT_PLOT_MODE[1]} [m]")
    ax["traj"].set_title(title, y=1.0, pad=-21)

    # pos = ax["traj"].get_position()
    # pos.y0 += 0.1
    # ax["traj"].set_position(pos)

    # ax["traj"].xaxis.set_major_locator(MultipleLocator(0.05))
    # ax["traj"].yaxis.set_major_locator(MultipleLocator(0.05))
    ax["frame"].grid(visible=False)
    # ax["frame"].axis("off")
    ax["frame"].set_xticks([])
    ax["frame"].set_yticks([])
    for spine in ax["frame"].spines.values():
        spine.set_linestyle("dotted")
        spine.set_color(RED)
        spine.set_linewidth(6)

    start_marker = mlines.Line2D(
        [], [], color=DARK_BLUEGREY, marker="o", linestyle="None", markersize=10, label="Start"
    )
    end_marker = mlines.Line2D([], [], color=DARK_BLUEGREY, marker="x", linestyle="None", markersize=10, label="End")
    frame_marker = mlines.Line2D([], [], color=RED, linestyle="dotted", linewidth=2, label="Frame")
    legend2 = ax["traj"].legend(handles=[start_marker, end_marker, frame_marker], loc="upper left")
    ax["traj"].add_artist(legend2)

    # Add a separate legend in ax["traj"] marking circle as "start" and cross as "end"

    # TODO: In reality this should use cam0/data.csv instead of est_csvs[0], so
    # that returning gt trajectory has exactly one pose for each frame
    # for now, try to always have in the first trajectory a system that has a
    # successful estimation 100% of the frames like (usually) Basalt
    _, gt = get_sanitized_trajectories(est_csvs[0], ref_csv, start_s=start_s, end_s=end_s)
    tracking_plot.plot_reference_trajectory(gt)

    colors = make_color_iterator()
    for est_csv, name in zip(est_csvs, names):
        with open(est_csv, "r", encoding="utf-8") as f:
            est_lines = f.readlines()
        with open(ref_csv, "r", encoding="utf-8") as f:
            ref_lines = f.readlines()

        est_lines = [line.strip().split(",") for line in est_lines]
        ref_lines = [line.strip().split(",") for line in ref_lines]

        if est_lines[0][0].startswith("#"):
            est_lines = est_lines[1:]
        if ref_lines[0][0].startswith("#"):
            ref_lines = ref_lines[1:]

        # Trim
        ref_lines = ref_lines[ref_start_idx:ref_end_idx]
        ref_start_ts = int(ref_lines[0][0])
        ref_end_ts = int(ref_lines[-1][0])

        est_ts = np.array([int(line[0]) for line in est_lines], dtype=np.int64).reshape((-1, 1))
        est_start = np.searchsorted(est_ts[:, 0], ref_start_ts)
        est_end = np.searchsorted(est_ts[:, 0], ref_end_ts)
        est_ts = est_ts[est_start:est_end]
        est_lines = est_lines[est_start:est_end]

        ref_ts = np.array([int(line[0]) for line in ref_lines], dtype=np.int64).reshape((-1, 1))
        est_xyz = np.array(
            [(float(line[1]), float(line[2]), float(line[3])) for line in est_lines],
            dtype=SCALAR,
        ).T
        ref_xyz = np.array(
            [(float(line[1]), float(line[2]), float(line[3])) for line in ref_lines],
            dtype=SCALAR,
        ).T
        est_quat = np.array(
            [(float(line[5]), float(line[6]), float(line[7]), float(line[4])) for line in est_lines],
            dtype=SCALAR,
        ).T
        ref_quat = np.array(
            [(float(line[5]), float(line[6]), float(line[7]), float(line[4])) for line in ref_lines],
            dtype=SCALAR,
        ).T

        # Associate, align, and compute, all at once
        # joint_rmse = al.compute_ate_and_align_ref(est_ts, ref_ts, est_xyz, ref_xyz)
        # print(f"{joint_rmse=}")

        # Associate, align, and compute, in separate steps
        if est_ts.size == 0 or ref_ts.size == 0:
            print(f"No poses for this start-end for {name}")
            continue

        pose_count = al.associate_full(est_ts, ref_ts, est_xyz, ref_xyz, est_quat, ref_quat)

        if pose_count == 0:
            print(f"Skipping {name}")
            continue

        # Trim numpy arrays
        ref_ts = ref_ts[:pose_count]
        est_ts = est_ts[:pose_count]
        ref_xyz = ref_xyz[:, :pose_count]
        est_xyz = est_xyz[:, :pose_count]
        est_quat = est_quat[:, :pose_count]
        ref_quat = ref_quat[:, :pose_count]

        T_ref_est = al.align_ref(est_xyz, ref_xyz, 0, pose_count)
        T_ref_est_full = T_ref_est.copy()
        ate = al.compute_ate(est_xyz, ref_xyz, 0, pose_count, T_ref_est)
        print(f"[{name}] ATE={ate:.3f} m")

        divisions = min(pose_count, 10000)
        assert divisions <= pose_count, f"{divisions=} {pose_count=}"
        # xs = np.linspace(start, pose_count, divisions, endpoint=True).astype(int)
        ts = (est_ts[:, 0] - cam_ts[0]) / 1e9  # conver to seconds
        ys = [0]
        for i, t in enumerate(ts[:-1]):
            T_ref_est = al.align_ref(est_xyz, ref_xyz, 0, i + 1)
            err = al.compute_ate(est_xyz, ref_xyz, 0, i + 1, T_ref_est)
            ys += [err]
        ys = np.array(ys)
        dys = np.diff(ys)

        # Apply moving average to dys
        # window = 100
        # dys = np.convolve(dys, np.ones(window) / window, mode="same")

        # Filter first points
        # ts = ts[1000:]
        # dys = dys[1000:]

        # Normalize
        # dys /= dys.max()
        # ys /= ys.max()

        # marker = "-" if divisions > 300 else "-." if divisions > 100 else "o-"
        # marker = ".-"
        # marker = "-"
        color = next(colors)
        estyle = {"color": color, "label": name}
        destyle = {"color": color, "marker": "o", "label": name, "alpha": 0.7}

        UNIT_ATE = "cm"
        UNIT_D_ATE = "mm"
        UNITS = {"mm": 1000, "cm": 100, "m": 1}
        mult_ate = UNITS[UNIT_ATE]
        mult_d_ate = UNITS[UNIT_D_ATE]

        # ATE absolute
        ax["ATE"].plot(ts, ys * mult_ate, **estyle)
        ax["dATE"].plot(ts[1:], dys * mult_d_ate, **destyle)

        # ATE normalized
        ys_norm = ys / ys.max()
        dys_norm = dys / dys.max()
        ax["ATE norm"].plot(ts, ys_norm, **estyle)
        ax["dATE norm"].plot(ts[1:], dys_norm, **destyle)

        # ATE logscale
        ax["ATE log"].plot(ts, ys * mult_ate, **estyle)
        ax["dATE log"].plot(ts[1:], dys * mult_d_ate, **destyle)

        timestamps, residuals = compute_rte(est_ts, est_xyz, ref_xyz, est_quat, ref_quat, 0, pose_count)

        # TODO@mateosss: document this behavor:
        # timestamps = [ts0, ts1, ts2, ...]
        # residuals = [0, diff0_1, diff1_2, diff2_3, ...]
        ts_s = (timestamps[:] - cam_ts[0]) / 1e9  # convert to seconds
        # ts_s = timestamps[:]
        # ts_s = np.arange(0, len(residuals)) * DELTA
        n = len(ts_s)
        ys = np.zeros(n, dtype=SCALAR)
        res2_sum = 0
        for i, _ in enumerate(ts_s):
            res2_sum += residuals[i] ** 2
            ys[i] = sqrt(res2_sum / (i + 1))
        assert ys[0] == 0

        dys = np.diff(ys)

        rte = sqrt(sum(r**2 for r in residuals) / n)
        print(f"[{name}] RTE={rte:.6f} m")

        # marker = "-" if divisions > 300 else "-." if divisions > 100 else "o-"
        marker = "-o"
        # marker = "-"
        # marker = "."

        UNIT_RTE = "cm"
        UNIT_D_RTE = "mm"
        UNITS = {"mm": 1000, "cm": 100, "m": 1}
        mult_rte = UNITS[UNIT_RTE]
        mult_d_rte = UNITS[UNIT_D_RTE]

        # RTE
        ax["RTE"].plot(ts_s, ys * mult_rte, **estyle)
        ax["dRTE"].plot(ts_s[1:], dys * mult_d_rte, **destyle)

        # RTE norm
        ys_norm = ys / ys.max()
        dys_norm = dys / dys.max()
        ax["RTE norm"].plot(ts_s, ys_norm, **estyle)
        ax["dRTE norm"].plot(ts_s[1:], dys_norm, **destyle)

        # RTE logscale
        ax["RTE log"].plot(ts_s, ys * mult_rte, **estyle)
        ax["dRTE log"].plot(ts_s[1:], dys * mult_d_rte, **destyle)

        # Plot trajectory
        traj_est, traj_ref = get_sanitized_trajectories(est_csv, ref_csv, silence=True, start_s=start_s, end_s=end_s)
        tracking_plot.plot_estimate_trajectory(traj_est, est_name=name, T_ref_est=T_ref_est_full)

    # ax["ATE"].legend()
    xlim_l = min(l.get_xdata()[0] for l in ax["ATE norm"].lines)
    xlim_r = max(l.get_xdata()[-1] for l in ax["ATE norm"].lines)
    pad = 0.01 * (xlim_r - xlim_l)
    for c in curves:
        ax[c].set_xlim(xlim_l - pad, xlim_r + pad)
    enable_mouse_interactions(fig, ax)
    plt.show()
    # fig.savefig(f"{title}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
