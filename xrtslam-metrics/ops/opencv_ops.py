#!/usr/bin/env python

from pathlib import Path
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Callable
import cv2
from multiprocessing import Pool, current_process

import json
import shutil
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    @dataclass
    class Command:
        name: str
        desc: str
        func: Callable[[Namespace], None]

    # fmt: off
    cmd_undistort = Command("undistort", "Undistort all images in a directory", undistort)
    # fmt: on

    parser = ArgumentParser(
        description="Apply OpenCV operations",
    )
    subparsers = parser.add_subparsers(help="What operation to perform")

    undistort_parser = subparsers.add_parser(cmd_undistort.name, help=cmd_undistort.desc)
    undistort_parser.set_defaults(func=cmd_undistort.func)
    undistort_parser.add_argument("calibration_file", type=Path, help="Calibration json file")
    undistort_parser.add_argument("input_dir", type=Path, help="Directory with input images")
    undistort_parser.add_argument("output_dir", type=Path, help="Directory to output undistorted images")
    undistort_parser.add_argument(
        "--cam_idx",
        "-i",
        type=int,
        default=0,
        help="Index of the intrinsics to use from the calibration file",
    )
    undistort_parser.add_argument(
        "--show_images",
        action="store_true",
        help="Show images while undistorting (press q to exit, any other key for next), if enabled jobs are set to 1",
    )
    undistort_parser.add_argument("--dont_save_images", action="store_false", help="Do not save the images")
    undistort_parser.add_argument(
        "--num_jobs",
        "-j",
        type=int,
        default=os.cpu_count() // 2,
        help="Amount of jobs to use, number of physical (not-logical) cores tends to yield best perf",
    )

    return parser.parse_args()


def undistort_worker(
    image: Path,
    i: int,
    num_images: int,
    opencv_model: str,
    intr: np.ndarray,
    coeffs: np.ndarray,
    new_intr: np.ndarray,
    save_images: bool,
    show_images: bool,
    output_dir: Path,
):
    if not image.is_file():
        return

    dimg = cv2.imread(str(image))
    if opencv_model == "regular":
        uimg = cv2.undistort(dimg, new_intr, coeffs)
    elif opencv_model == "fisheye":
        uimg = cv2.fisheye.undistortImage(dimg, intr, coeffs, None, new_intr)
    else:
        raise ValueError(f"Undistort unavailable for {opencv_model=}")

    if save_images:
        output_fn = output_dir / image.name
        cv2.imwrite(str(output_fn), uimg)

    if show_images:
        print(f"[{i}/{num_images}] ({i / num_images * 100:.2f}%): {image.name}")
        cv2.imshow("Distorted image", dimg)
        cv2.imshow("Undistorted image", uimg)
        if cv2.waitKey(0) == ord("q"):
            exit(0)
    elif i % 200 == 0:
        print(f"[{i}/{num_images}] ({i / num_images * 100:.2f}%): {image.name}")


def undistort(args: Namespace):
    calibration_fn = args.calibration_file
    input_dir = args.input_dir
    output_dir = args.output_dir
    cam_idx = args.cam_idx
    show_images = args.show_images
    save_images = args.dont_save_images  # Looks weird but is correct
    num_jobs = args.num_jobs if not show_images else 1

    if output_dir.exists():
        print(f"Output directory {output_dir} already exists. Deleting it.")
        shutil.rmtree(output_dir)

    calibration = json.load(open(calibration_fn, "r"))
    cam_model = calibration["value0"]["intrinsics"][cam_idx]["camera_type"]
    intrinsics = calibration["value0"]["intrinsics"][cam_idx]["intrinsics"]

    if cam_model == "pinhole-radtan8":
        keys = "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = [intrinsics[k] for k in keys]
        coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
        opencv_model = "regular"
    elif cam_model == "kb4":
        keys = "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"
        fx, fy, cx, cy, k1, k2, k3, k4 = [intrinsics[k] for k in keys]
        coeffs = np.array([k1, k2, k3, k4])
        opencv_model = "fisheye"
    else:
        raise ValueError(f"Undistort not yet implemented for {cam_model=}")

    if save_images:
        output_dir.mkdir(parents=True)

    intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    image_paths = sorted(input_dir.iterdir())

    first_image = cv2.imread(str(image_paths[0]))
    h, w = first_image.shape[:2]
    wh = (w, h)
    alpha = 0  # 0: crop to hide black regions, 1: show full FoV but with black regions
    print(f"[{opencv_model}]: {intr=}")
    if opencv_model == "regular":
        # NOTE:
        # 1. If we set new_intr=intr and not use getOptimalNewCameraMatrix, we
        #    retain the same intrinsics but the image might show black regions
        # 2. If we use getOptimalNewCameraMatrix with alpha=0, we have a new
        #    intr matrix cropped to remove black regions
        # 3. If we use getOptimalNewCameraMatrix with alpha=1, we retain all the
        #    FoV but have black regions
        new_intr, roi = cv2.getOptimalNewCameraMatrix(intr, coeffs, wh, alpha, wh)
        print(f"[{opencv_model}]: {roi=}")
    elif opencv_model == "fisheye":
        new_intr = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(intr, coeffs, wh, np.eye(3), balance=alpha)
    else:
        raise ValueError(f"Invalid {opencv_model=}")
    print(f"[{opencv_model}]: {new_intr=}")

    print(f"Using {num_jobs} jobs")

    if num_jobs == 1:  # Sequential run in the same process
        print(f"Using {num_jobs} jobs")
        for i, image in enumerate(image_paths):
            undistort_worker(
                image,
                i,
                len(image_paths),
                opencv_model,
                intr,
                coeffs,
                new_intr,
                save_images,
                show_images,
                output_dir,
            )
    elif num_jobs > 1:
        with Pool(num_jobs) as p:
            args = (
                (
                    image,
                    i,
                    len(image_paths),
                    opencv_model,
                    intr,
                    coeffs,
                    new_intr,
                    save_images,
                    show_images,
                    output_dir,
                )
                for i, image in enumerate(image_paths)
            )
            p.starmap(undistort_worker, args)


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
