cmake_minimum_required(VERSION 3.15)

# CMAKE_TOOLCHAIN_FILE needs to be set before project()
# See https://learn.microsoft.com/en-us/vcpkg/users/buildsystems/cmake-integration#cmake_toolchain_file

# set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
# set(CMAKE_TOOLCHAIN_FILE "C:/Users/mateo/Documents/apps/vcpkg/scripts/buildsystems/vcpkg.cmake")
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR "CMAKE_TOOLCHAIN_FILE not set")
endif()

# set(VENV_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.venv")
if (NOT DEFINED VENV_PATH)
    message(FATAL_ERROR "VENV_PATH not set")
endif()

# set(EIGEN_DIR "C:/Users/mateo/Documents/apps/bsltdeps/basalt/thirdparty/basalt-headers/thirdparty/eigen")
if (NOT DEFINED EIGEN_DIR)
    message(FATAL_ERROR "EIGEN_DIR not set")
endif()

# set(SOPHUS_DIR "C:/Users/mateo/Documents/apps/bsltdeps/basalt/thirdparty/basalt-headers/thirdparty/Sophus")
if (NOT DEFINED SOPHUS_DIR)
    message(FATAL_ERROR "SOPHUS_DIR not set")
endif()

project(alignment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "${VENV_PATH}/Lib/site-packages/pybind11/share/cmake/pybind11")

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(fmt REQUIRED)

pybind11_add_module(alignment alignment.cpp)
target_include_directories(alignment SYSTEM PRIVATE ${EIGEN_DIR} ${SOPHUS_DIR})
target_link_libraries(alignment PRIVATE fmt::fmt)

install(TARGETS alignment DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/Release/fmt.dll" DESTINATION ${CMAKE_INSTALL_PREFIX})
