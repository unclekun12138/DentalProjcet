# dental_inference — C++ ONNX Runtime Deployment

Standalone C++ inference skeleton for dental 3D medical image segmentation,
using **ONNX Runtime** for model execution and **nifticlib** for NIfTI I/O.

## Dependencies

| Dependency       | Version | Install (Ubuntu/Debian)              |
|------------------|---------|---------------------------------------|
| CMake            | ≥ 3.16  | `sudo apt install cmake`             |
| C++ compiler     | ≥ C++17 | `sudo apt install g++`               |
| ONNX Runtime     | ≥ 1.15  | Download prebuilt release from [GitHub](https://github.com/microsoft/onnxruntime/releases) |
| nifticlib        | latest  | `sudo apt install libnifti-dev zlib1g-dev` |
| zlib             | any     | (usually preinstalled)               |

### Download ONNX Runtime

```bash
# Example: Linux x64 GPU release
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-gpu-1.17.1.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.17.1.tgz
export ONNXRUNTIME_ROOT=$(pwd)/onnxruntime-linux-x64-gpu-1.17.1
```

## Build

```bash
cd deployment/
mkdir build && cd build
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

To enable CUDA execution provider:

```bash
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime -DUSE_CUDA=ON
```

The resulting binary is `build/dental_inference`.

## Run

```bash
./dental_inference \
  --model  /path/to/model.onnx \
  --input  /path/to/input.nii.gz \
  --output /path/to/segmentation.nii.gz \
  --input-shape 1,1,128,128,128 \
  --normalize zscore
```

### Command-line arguments

| Argument          | Description                                      | Default            |
|-------------------|--------------------------------------------------|--------------------|
| `--model`         | Path to exported ONNX model (**required**)       | —                  |
| `--input`         | Input NIfTI image (`.nii` or `.nii.gz`)           | —                  |
| `--output`        | Output segmentation label map (`.nii.gz`)         | —                  |
| `--input-shape`   | Model input shape `N,C,D,H,W`                     | `1,1,128,128,128`  |
| `--normalize`      | Preprocessing: `zscore` or `minmax`              | `zscore`           |

## Pipeline Overview

```
input.nii.gz ──► Load (nifticlib) ──► Normalize ──► Resize (trilinear)
                                                          │
                                                          ▼
                                            ONNX Runtime Session.Run()
                                                          │
                                                          ▼
                                          Argmax over channel axis
                                                          │
                                                          ▼
                              segmentation.nii.gz (int32 label map)
```

## Supported Formats

- **Input**: NIfTI-1 / NIfTI-2 (`.nii`, `.nii.gz`), 3D volume
  - Supported voxel types: float32, float64, int16, uint16, int32, uint8
- **Output**: NIfTI-1 (`.nii.gz`), int32 label map
- **Model**: ONNX (opset ≥ 11), input tensor shape `[N, C, D, H, W]`

## Notes

- The preprocessing (normalization + trilinear resize) is a reasonable default
  for nnU-Net–exported models. Adjust to match your training-time preprocessing.
- Multi-channel input (e.g. CT + MR) is supported by repeating the resized
  volume across channels; extend `model_input` construction as needed.
- The output label map uses the same spatial dimensions as the model output.
  If you need to upsample back to original resolution, add a resize step
  before `saveNifti`.
