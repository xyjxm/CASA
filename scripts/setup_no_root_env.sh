#!/usr/bin/env bash
set -euo pipefail

USER_ROOT="${GR00T_USER_ROOT:-/mnt/data/students/lph}"

export TensorRT_ROOT="${GR00T_TENSORRT_ROOT:-$USER_ROOT/TensorRT}"
export onnxruntime_ROOT="${GR00T_ONNXRUNTIME_ROOT:-$USER_ROOT/onnxruntime}"
export onnxruntime_DIR="$onnxruntime_ROOT"
export GTest_DIR="${GR00T_GTEST_DIR:-$USER_ROOT/googletest-install/lib/cmake/GTest}"

if [ -d "$USER_ROOT/googletest-install" ]; then
    export CMAKE_PREFIX_PATH="$USER_ROOT/googletest-install:${CMAKE_PREFIX_PATH:-}"
fi

if [ -d "$onnxruntime_ROOT" ]; then
    export CMAKE_PREFIX_PATH="$onnxruntime_ROOT:${CMAKE_PREFIX_PATH:-}"
    export LD_LIBRARY_PATH="$onnxruntime_ROOT/lib:${LD_LIBRARY_PATH:-}"
fi

if [ -d "$TensorRT_ROOT" ]; then
    export LD_LIBRARY_PATH="$TensorRT_ROOT/lib:$TensorRT_ROOT/lib64:${LD_LIBRARY_PATH:-}"
fi

if [ -z "${CUDAToolkit_ROOT:-}" ]; then
    for cuda_root in /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-12; do
        if [ -f "$cuda_root/include/cuda_runtime.h" ] && [ -d "$cuda_root/lib64" ]; then
            export CUDAToolkit_ROOT="$cuda_root"
            break
        fi
    done
fi

if [ -n "${CUDAToolkit_ROOT:-}" ]; then
    export CUDA_HOME="$CUDAToolkit_ROOT"
    export PATH="$CUDAToolkit_ROOT/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDAToolkit_ROOT/lib64:$CUDAToolkit_ROOT/lib:${LD_LIBRARY_PATH:-}"
fi

export HAS_ROS2="${HAS_ROS2:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "No-root GR00T environment:"
echo "  TensorRT_ROOT=$TensorRT_ROOT"
echo "  onnxruntime_ROOT=$onnxruntime_ROOT"
echo "  GTest_DIR=$GTest_DIR"
echo "  CUDAToolkit_ROOT=${CUDAToolkit_ROOT:-<not found>}"
echo "  HAS_ROS2=$HAS_ROS2"
echo "  MUJOCO_GL=$MUJOCO_GL"
