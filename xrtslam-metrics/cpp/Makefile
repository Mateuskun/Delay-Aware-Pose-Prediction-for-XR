SOPHUS_DIR=/home/mateo/Documents/apps/bsltdeps/basalt/thirdparty/basalt-headers/thirdparty/Sophus
EIGEN_DIR=/home/mateo/Documents/apps/bsltdeps/basalt/thirdparty/basalt-headers/thirdparty/eigen
PYBIND_FLAGS=$(shell python -m pybind11 --includes)
PYTHON_EXT_SUFFIX=$(shell python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
all:
	eval c++ -O3 -Wall -shared -std=c++17 -fPIC -DSOPHUS_USE_BASIC_LOGGING -isystem $(SOPHUS_DIR) -isystem $(EIGEN_DIR) $(PYBIND_FLAGS) alignment.cpp -o alignment$(PYTHON_EXT_SUFFIX)
	stubgen -m alignment -o .

clean:
	rm alignment.cpython-*.so
	rm alignment.pyi
