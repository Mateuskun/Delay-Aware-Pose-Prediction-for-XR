# Instructions

## Compile with Linux

1. Modify Makefile SOPHUS_DIR and EIGEN_DIR variables
2. Have pybind11 installed (`pip install pybind11`) so that PYBIND_FLAGS gets set correctly
3. Compile alignment.cpp with `make`

## Compile with Windows

```powershell
cd xrtslam-metrics/cpp
.\vcpkg install fmt python3 # for cmake
.\vcpkg list python3 # Check python3:x64-windows version, in my case 3.11.10
# Install a compatible executable version from https://www.python.org/downloads/, in my case 3.11.9
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pybind11 numpy matplotlib mplcursors scipy

# Configure (update all paths)
cmake -B build `
   -DCMAKE_TOOLCHAIN_FILE="C:/Users/mateo/Documents/apps/vcpkg/scripts/buildsystems/vcpkg.cmake" `
   -DVENV_PATH=".venv" `
   -DEIGEN_DIR="C:/Users/mateo/Documents/apps/bsltdeps/basalt/thirdparty/basalt-headers/thirdparty/eigen" `
   -DSOPHUS_DIR="C:/Users/mateo/Documents/apps/bsltdeps/basalt/thirdparty/basalt-headers/thirdparty/Sophus" `
   -DCMAKE_INSTALL_PREFIX="."

cmake --build build --config Release --target install

# After this this should not throw an error:
python -c "import alignment"
```

## Usage

1. To get ATE you run for example:

   ```bash
   python ./alignment_ate_over_time.py ../test/data/targets/EV202/gt.csv ../test/data/runs/Basalt/EV202/tracking.csv ../test/data/runs/ORB-SLAM3/EV202/tracking.csv --names basalt orbslam3
   ```

2. For RTE is the same command but using `alignment_rte_over_time.py`
3. You get the final ATE/RTE numbers printed on the terminal
