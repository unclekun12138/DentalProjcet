# ============================================================================
# CMakeLists.txt — dental_inference
#
# Build a C++ ONNX Runtime inference executable for dental 3D segmentation.
#
# Dependencies:
#   - ONNX Runtime  (https://github.com/microsoft/onnxruntime)
#   - nifticlib     (NIfTI I/O; apt install libnifti-dev)
#   - A C++17 compiler (g++ >= 9 or clang >= 10)
#
# Typical usage:
#   mkdir build && cd build
#   cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime
#   make -j
# ============================================================================

cmake_minimum_required(VERSION 3.16)
project(dental_inference LANGUAGES CXX)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Build type ----
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Compile options ----
add_compile_options(-Wall -Wextra -O3)

# ---------------------------------------------------------------------------
# ONNX Runtime
#   Either pass -DONNXRUNTIME_ROOT=/path/to/onnxruntime, or let find_package
#   locate it. The root dir should contain:
#     include/onnxruntime_cxx_api.h
#     lib/libonnxruntime.so  (or .dylib / .dll)
# ---------------------------------------------------------------------------
if(NOT DEFINED ONNXRUNTIME_ROOT)
    find_package(onnxruntime QUIET)
    if(NOT onnxruntime_FOUND)
        message(FATAL_ERROR
            "ONNX Runtime not found. Please pass -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
    endif()
else()
    message(STATUS "Using ONNX Runtime at: ${ONNXRUNTIME_ROOT}")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include
)
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime libonnxruntime
    HINTS ${ONNXRUNTIME_ROOT}/lib
)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "Could not locate ONNX Runtime headers or library.")
endif()
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")

# ---------------------------------------------------------------------------
# nifticlib (NIfTI I/O)
#   On Debian/Ubuntu: sudo apt install libnifti-dev
# ---------------------------------------------------------------------------
find_path(NIFTI_INCLUDE_DIR NAMES nifti1_io.h)
find_library(NIFTI_LIB NAMES niftiio nifti)
if(NOT NIFTI_INCLUDE_DIR OR NOT NIFTI_LIB)
    message(FATAL_ERROR
        "nifticlib not found. Install with: sudo apt install libnifti-dev")
endif()
message(STATUS "nifticlib include: ${NIFTI_INCLUDE_DIR}")
message(STATUS "nifticlib library: ${NIFTI_LIB}")

# ---------------------------------------------------------------------------
# Optional: CUDA execution provider
#   Pass -DUSE_CUDA=ON to enable
# ---------------------------------------------------------------------------
option(USE_CUDA "Enable ONNX Runtime CUDA execution provider" OFF)
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
    message(STATUS "CUDA execution provider: ON")
endif()

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(dental_inference inference.cpp)

target_include_directories(dental_inference PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${NIFTI_INCLUDE_DIR}
)

target_link_libraries(dental_inference PRIVATE
    ${ONNXRUNTIME_LIB}
    ${NIFTI_LIB}
    z       # nifticlib depends on zlib
)

# ---------------------------------------------------------------------------
# Install target
# ---------------------------------------------------------------------------
install(TARGETS dental_inference RUNTIME DESTINATION bin)

# ---- Print summary ----
message(STATUS "")
message(STATUS "=== dental_inference configuration ===")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ std    : C++${CMAKE_CXX_STANDARD}")
message(STATUS "  ONNX RT    : ${ONNXRUNTIME_LIB}")
message(STATUS "  nifticlib  : ${NIFTI_LIB}")
message(STATUS "  CUDA EP    : ${USE_CUDA}")
message(STATUS "=======================================")
message(STATUS "")
