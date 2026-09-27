// ============================================================================
// dental_inference.cpp
// C++ ONNX Runtime inference skeleton for dental 3D segmentation
//
// Pipeline:
//   1. Load ONNX model via ONNX Runtime C++ API
//   2. Read input NIfTI (.nii.gz) volume
//   3. Preprocess: z-score / [0,1] normalization + resize to model input shape
//   4. Run inference
//   5. Post-process: argmax over channel axis -> label map
//   6. Save label map as NIfTI (.nii.gz)
//
// Usage:
//   dental_inference --model model.onnx --input image.nii.gz --output seg.nii.gz \
//                    [--input-shape 1,1,128,128,128] [--normalize zscore|minmax]
// ============================================================================

#include <onnxruntime_cxx_api.h>

#include <nifti1_io.h>   // nifticlib: NIfTI read/write

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Small utility: parse command-line flags of form --key value
// ---------------------------------------------------------------------------
struct Args {
    std::string model_path;
    std::string input_path;
    std::string output_path;
    std::vector<int64_t> input_shape;  // NCDHWD expected by model
    std::string normalize_mode = "zscore";  // zscore | minmax
};

static std::vector<int64_t> parseIntList(const std::string& s) {
    std::vector<int64_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // trim spaces
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) out.push_back(std::stoll(item));
    }
    return out;
}

static Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
        std::string val = argv[++i];
        if (key == "--model")            a.model_path = val;
        else if (key == "--input")       a.input_path = val;
        else if (key == "--output")      a.output_path = val;
        else if (key == "--input-shape") a.input_shape = parseIntList(val);
        else if (key == "--normalize")   a.normalize_mode = val;
        else throw std::runtime_error("Unknown argument: " + key);
    }
    if (a.model_path.empty() || a.input_path.empty() || a.output_path.empty()) {
        throw std::runtime_error(
            "Usage: dental_inference --model <onnx> --input <in.nii.gz> "
            "--output <out.nii.gz> [--input-shape 1,1,D,H,W] [--normalize zscore|minmax]");
    }
    return a;
}

// ---------------------------------------------------------------------------
// NIfTI I/O wrappers around nifticlib
// ---------------------------------------------------------------------------
struct NiftiVolume {
    std::vector<float> data;   // voxel data, converted to float32
    int dim[8] = {0};          // NIfTI dim[1..3] = spatial size
    float pixdim[8] = {0};     // voxel spacing
    nifti_image* raw = nullptr; // keep pointer for writing header back
};

static NiftiVolume loadNifti(const std::string& path) {
    nifti_image* img = nifti_image_read(path.c_str(), 1);  // 1 = read pixel data
    if (!img) throw std::runtime_error("Failed to read NIfTI: " + path);

    NiftiVolume vol;
    vol.raw = img;
    std::memcpy(vol.dim, img->dim, sizeof(img->dim));
    std::memcpy(vol.pixdim, img->pixdim, sizeof(img->pixdim));

    int64_t nvox = 1;
    for (int i = 1; i <= 3; ++i) nvox *= img->dim[i];
    vol.data.resize(nvox);

    // Convert raw voxel data to float32 regardless of source datatype
    // nifticlib provides nifti_image::data as void*; use nifti_convert_ndata_float if available,
    // otherwise manual conversion.
    float* fdata = static_cast<float*>(img->data);
    // nifticlib can convert in-place; simplest robust path: cast per dtype
    switch (img->datatype) {
        case NIFTI_TYPE_FLOAT32:
            for (int64_t i = 0; i < nvox; ++i) vol.data[i] = fdata[i];
            break;
        case NIFTI_TYPE_INT16: {
            int16_t* p = static_cast<int16_t*>(img->data);
            for (int64_t i = 0; i < nvox; ++i) vol.data[i] = static_cast<float>(p[i]);
            break;
        }
        case NIFTI_TYPE_UINT16: {
            uint16_t* p = static_cast<uint16_t*>(img->data);
            for (int64_t i = 0; i < nvox; ++i) vol.data[i] = static_cast<float>(p[i]);
            break;
        }
        case NIFTI_TYPE_INT32: {
            int32_t* p = static_cast<int32_t*>(img->data);
            for (int64_t i = 0; i < nvox; ++i) vol.data[i] = static_cast<float>(p[i]);
            break;
        }
        case NIFTI_TYPE_UINT8: {
            uint8_t* p = static_cast<uint8_t*>(img->data);
            for (int64_t i = 0; i < nvox; ++i) vol.data[i] = static_cast<float>(p[i]);
            break;
        }
        case NIFTI_TYPE_FLOAT64: {
            double* p = static_cast<double*>(img->data);
            for (int64_t i = 0; i < nvox; ++i) vol.data[i] = static_cast<float>(p[i]);
            break;
        }
        default:
            throw std::runtime_error("Unsupported NIfTI datatype: " +
                                     std::to_string(img->datatype));
    }
    return vol;
}

static void saveNifti(const std::string& path,
                      const std::vector<int32_t>& labels,
                      const int dim[8],
                      const float pixdim[8],
                      nifti_image* template_img) {
    // Create a new NIfTI image reusing geometry from template
    nifti_image* out = nifti_copy_nim_info(template_img);
    out->dim[0] = 3;
    out->dim[1] = dim[1];
    out->dim[2] = dim[2];
    out->dim[3] = dim[3];
    out->dim[4] = 1;
    out->datatype = NIFTI_TYPE_INT32;
    out->bitpix = 32;
    out->nvox = static_cast<int64_t>(dim[1]) * dim[2] * dim[3];
    out->nbyper = 4;
    out->data = malloc(static_cast<size_t>(out->nvox) * sizeof(int32_t));
    std::memcpy(out->data, labels.data(),
                static_cast<size_t>(out->nvox) * sizeof(int32_t));
    out->iname_offset = 0;

    int success = nifti_image_write(out);
    if (success < 0) throw std::runtime_error("Failed to write NIfTI: " + path);
    nifti_image_free(out);
}

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

/// Z-score normalization: (x - mean) / std
static void normalizeZScore(std::vector<float>& data) {
    double sum = 0.0;
    for (float v : data) sum += v;
    double mean = sum / static_cast<double>(data.size());
    double sq = 0.0;
    for (float v : data) sq += (v - mean) * (v - mean);
    double stdv = std::sqrt(sq / static_cast<double>(data.size()));
    if (stdv < 1e-8) stdv = 1.0;
    for (float& v : data) v = static_cast<float>((v - mean) / stdv);
}

/// Min-max normalization to [0, 1]
static void normalizeMinMax(std::vector<float>& data) {
    auto [mn, mx] = std::minmax_element(data.begin(), data.end());
    float range = *mx - *mn;
    if (range < 1e-8) range = 1.0f;
    for (float& v : data) v = (v - *mn) / range;
}

/// Trilinear resize of a 3D volume to (Dst, Hst, Wst).
/// Input layout: data[z][y][x] (NIfTI RAS-ish, dim[1]=x fastest in C-order).
/// We treat data as indexed [z * H * W + y * W + x].
static std::vector<float> resize3D(const std::vector<float>& src,
                                   int sx, int sy, int sz,
                                   int dx, int dy, int dz) {
    std::vector<float> dst(static_cast<size_t>(dx) * dy * dz, 0.0f);
    float xratio = static_cast<float>(sx - 1) / std::max(1, dx - 1);
    float yratio = static_cast<float>(sy - 1) / std::max(1, dy - 1);
    float zratio = static_cast<float>(sz - 1) / std::max(1, dz - 1);

    auto at = [&](int x, int y, int z) -> float {
        x = std::clamp(x, 0, sx - 1);
        y = std::clamp(y, 0, sy - 1);
        z = std::clamp(z, 0, sz - 1);
        return src[static_cast<size_t>(z) * sy * sx + y * sx + x];
    };

    for (int z = 0; z < dz; ++z) {
        float fz = z * zratio;
        int z0 = static_cast<int>(std::floor(fz));
        int z1 = std::min(z0 + 1, sz - 1);
        float dzf = fz - z0;
        for (int y = 0; y < dy; ++y) {
            float fy = y * yratio;
            int y0 = static_cast<int>(std::floor(fy));
            int y1 = std::min(y0 + 1, sy - 1);
            float dyf = fy - y0;
            for (int x = 0; x < dx; ++x) {
                float fx = x * xratio;
                int x0 = static_cast<int>(std::floor(fx));
                int x1 = std::min(x0 + 1, sx - 1);
                float dxf = fx - x0;
                // trilinear
                float c00 = at(x0, y0, z0) * (1 - dxf) + at(x1, y0, z0) * dxf;
                float c01 = at(x0, y1, z0) * (1 - dxf) + at(x1, y1, z0) * dxf;
                float c10 = at(x0, y0, z1) * (1 - dxf) + at(x1, y0, z1) * dxf;
                float c11 = at(x0, y1, z1) * (1 - dxf) + at(x1, y1, z1) * dxf;
                float c0 = c00 * (1 - dyf) + c01 * dyf;
                float c1 = c10 * (1 - dyf) + c11 * dyf;
                dst[static_cast<size_t>(z) * dy * dx + y * dx + x] =
                    c0 * (1 - dzf) + c1 * dzf;
            }
        }
    }
    return dst;
}

// ---------------------------------------------------------------------------
// ONNX Runtime helpers
// ---------------------------------------------------------------------------
static Ort::Session createSession(Ort::Env& env, const std::string& model_path) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef USE_CUDA
    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id = 0;
    opts.AppendExecutionProvider_CUDA(cuda_opts);
#endif
    return Ort::Session(env, model_path.c_str(), opts);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    try {
        Args args = parseArgs(argc, argv);

        // ---- 1. Load NIfTI ----
        std::cout << "[1/6] Loading NIfTI: " << args.input_path << std::endl;
        NiftiVolume vol = loadNifti(args.input_path);
        int sx = vol.dim[1], sy = vol.dim[2], sz = vol.dim[3];
        std::cout << "      Volume size: " << sx << " x " << sy << " x " << sz << std::endl;

        // ---- 2. Preprocess ----
        std::cout << "[2/6] Preprocessing (" << args.normalize_mode << ")..." << std::endl;
        if (args.normalize_mode == "zscore")
            normalizeZScore(vol.data);
        else
            normalizeMinMax(vol.data);

        // Determine target spatial size from input_shape (N,C,D,H,W)
        // Default: 1,1,128,128,128 if not provided
        int64_t modelD = 128, modelH = 128, modelW = 128, modelC = 1;
        if (args.input_shape.size() == 5) {
            modelC = args.input_shape[1];
            modelD = args.input_shape[2];
            modelH = args.input_shape[3];
            modelW = args.input_shape[4];
        } else if (args.input_shape.size() == 4) {
            // CDHW without batch
            modelC = args.input_shape[0];
            modelD = args.input_shape[1];
            modelH = args.input_shape[2];
            modelW = args.input_shape[3];
        }

        std::vector<float> input = resize3D(vol.data, sx, sy, sz,
                                            static_cast<int>(modelW),
                                            static_cast<int>(modelH),
                                            static_cast<int>(modelD));
        // Repeat channels if model expects >1 channel
        std::vector<float> model_input;
        model_input.reserve(input.size() * modelC);
        for (int64_t c = 0; c < modelC; ++c) model_input.insert(model_input.end(), input.begin(), input.end());

        // ---- 3. Load ONNX model ----
        std::cout << "[3/6] Loading ONNX model: " << args.model_path << std::endl;
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "dental_inference");
        Ort::Session session = createSession(env, args.model_path);

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name  = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        const char* input_name_ptr  = input_name.get();
        const char* output_name_ptr = output_name.get();

        std::vector<int64_t> input_shape_vec = {1, modelC, modelD, modelH, modelW};

        // ---- 4. Run inference ----
        std::cout << "[4/6] Running inference..." << std::endl;
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, model_input.data(), model_input.size(),
            input_shape_vec.data(), input_shape_vec.size());

        std::vector<const char*> input_names  = {input_name_ptr};
        std::vector<const char*> output_names = {output_name_ptr};
        auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                          input_names.data(), &input_tensor, 1,
                                          output_names.data(), 1);

        // ---- 5. Post-process: argmax over channels ----
        std::cout << "[5/6] Post-processing (argmax)..." << std::endl;
        float* out_data = output_tensors[0].GetTensorMutableData<float>();
        auto out_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        // Expect N, C, D, H, W
        int64_t outC = out_shape[1];
        int64_t outD = out_shape[2];
        int64_t outH = out_shape[3];
        int64_t outW = out_shape[4];
        int64_t nvox = outD * outH * outW;

        std::vector<int32_t> labels(static_cast<size_t>(nvox), 0);
        for (int64_t v = 0; v < nvox; ++v) {
            int best_c = 0;
            float best_val = out_data[0 * nvox + v];
            for (int64_t c = 1; c < outC; ++c) {
                float val = out_data[c * nvox + v];
                if (val > best_val) { best_val = val; best_c = static_cast<int>(c); }
            }
            labels[v] = static_cast<int32_t>(best_c);
        }

        // ---- 6. Save as NIfTI ----
        std::cout << "[6/6] Saving segmentation: " << args.output_path << std::endl;
        // Build output dims matching model output spatial size
        int out_dim[8] = {3, static_cast<int>(outW), static_cast<int>(outH),
                          static_cast<int>(outD), 1, 1, 1, 1};
        saveNifti(args.output_path, labels, out_dim, vol.pixdim, vol.raw);

        // Free original nifti
        nifti_image_free(vol.raw);

        std::cout << "Done. Output: " << args.output_path << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
