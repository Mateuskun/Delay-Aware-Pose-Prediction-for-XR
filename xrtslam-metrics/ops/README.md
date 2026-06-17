# Quick docs of other scripts

## `opencv_ops.py`

```bash
# Requirements:
pip install opencv-python opencv-contrib-python

# Test visualization run
$xrtmet/ops/opencv_ops.py undistort $msdmo/extras/calibration.json $msdmo/MOC_calibration/MOC01_camcalib_1/mav0/cam0/data/ --show_images --dont_save_images MOC01_cam0_undistorted

# Full run
$xrtmet/ops/opencv_ops.py undistort $msdmo/extras/calibration.json $msdmo/MOC_calibration/MOC01_camcalib_1/mav0/cam0/data/ MOC01_cam0_undistorted
```
