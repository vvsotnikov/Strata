#include "strata/hy4/vision.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>

namespace strata::hy4::vision_detail {
namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("Hy4 vision CUDA: ") + cudaGetErrorString(e));
}
void blas_check(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Hy4 vision cuBLAS error " + std::to_string(e));
}

__global__ void convert_bf16(const __nv_bfloat16 *in, __half *out, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half_rn(__bfloat162float(in[i]));
}
__global__ void cast_half(const float *in, __half *out, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = __float2half_rn(in[i]);
}
__global__ void add_bias(float *out, const float *bias, int dim, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] += bias[i % dim];
}
__global__ void add(float *out, const float *other, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] += other[i];
}
__global__ void activate(float *x, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        float z = x[i];
        x[i] = .5f * z * (1.f + erff(z * .7071067811865475244f));
    }
}

__device__ float block_sum(float value, float *sh) {
    sh[threadIdx.x] = value;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s >>= 1) {
        if (threadIdx.x < s)
            sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float value, float *sh) {
    sh[threadIdx.x] = value;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s >>= 1) {
        if (threadIdx.x < s)
            sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    float r = sh[0];
    __syncthreads();
    return r;
}
__global__ void normalize(const float *in, float *out, const float *w, const float *bias, int dim, float eps,
                          bool rms) {
    __shared__ float sh[256];
    size_t start = size_t(blockIdx.x) * dim;
    float sum = 0;
    if (!rms)
        for (int i = threadIdx.x; i < dim; i += blockDim.x)
            sum += in[start + i];
    float mean = rms ? 0 : block_sum(sum, sh) / dim;
    sum = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = in[start + i] - mean;
        sum += v * v;
    }
    float scale = rsqrtf(block_sum(sum, sh) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        out[start + i] = (in[start + i] - mean) * scale * w[i] + (bias ? bias[i] : 0);
}
__global__ void pack_heads(const float *in, float *out, int count, int dim, int hd) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(count) * dim)
        return;
    int token = i / dim, col = i % dim, head = col / hd, channel = col % hd;
    out[(size_t(head) * count + token) * hd + channel] = in[i];
}
__global__ void unpack_heads(const float *in, float *out, int count, int dim, int hd) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(count) * dim)
        return;
    int token = i / dim, col = i % dim, head = col / hd, channel = col % hd;
    out[i] = in[(size_t(head) * count + token) * hd + channel];
}
__global__ void softmax(float *scores, int count) {
    __shared__ float sh[256];
    float *row = scores + size_t(blockIdx.x) * count;
    float maximum = -CUDART_INF_F;
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        maximum = fmaxf(maximum, row[i]);
    maximum = block_max(maximum, sh);
    float sum = 0;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        row[i] = expf(row[i] - maximum);
        sum += row[i];
    }
    float den = block_sum(sum, sh);
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        row[i] /= den;
}
__global__ void pack_merge(const float *x, float *out, int pw, int ow, int oh, int dim, int merge) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x, inner = dim * merge * merge;
    if (i >= size_t(ow) * oh * inner)
        return;
    const int token = i / inner, rem = i % inner, c = rem / (merge * merge), dy = (rem / merge) % merge,
              dx = rem % merge;
    out[i] = x[(size_t(token / ow * merge + dy) * pw + token % ow * merge + dx) * dim + c];
}
__global__ void insert_newline(const float *x, float *out, const float *newline, int ow, int oh, int dim) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(ow + 1) * oh * dim)
        return;
    int token = i / dim, c = i % dim, y = token / (ow + 1), col = token % (ow + 1);
    out[i] = col == ow ? newline[c] : x[(size_t(y) * ow + col) * dim + c];
}

class Cuda final : public Backend {
    struct Matrix {
        void *data = nullptr;
        int type = 0, inner = 0, outer = 0;
    };

  public:
    Cuda(const Geometry &g, Lookup lookup, const VisionOptions &options)
        : g_(g), lookup_(std::move(lookup)), o_(options) {
        check(cudaSetDevice(o_.cuda_device));
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        try {
            blas_check(cublasCreate(&blas_));
            blas_check(cublasSetStream(blas_, stream_));
            blas_check(cublasSetMathMode(blas_, o_.fp32 ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH));
        } catch (...) {
            if (blas_)
                cublasDestroy(blas_);
            cudaStreamDestroy(stream_);
            throw;
        }
    }
    ~Cuda() {
        cudaSetDevice(o_.cuda_device);
        cudaStreamSynchronize(stream_);
        for (void *p : weights_)
            cudaFree(p);
        for (void *p : work_)
            cudaFree(p);
        if (blas_)
            cublasDestroy(blas_);
        if (stream_)
            cudaStreamDestroy(stream_);
    }
    std::string admission_error(int pw, int ph) override {
        check(cudaSetDevice(o_.cuda_device));
        check(cudaStreamSynchronize(stream_));
        for (void *p : work_)
            check(cudaFree(p));
        work_.clear();
        size_t weight_bytes = 0, temporary = 0;
        const auto matrix_bytes = [&](const std::string &name) {
            const auto &tensor = lookup_(name);
            const bool fp32 = o_.fp32 || tensor.info->type == 0;
            const size_t bytes = tensor.info->elements() * (fp32 ? 4 : 2);
            weight_bytes += bytes;
            if (!fp32 && tensor.info->type == 30 && !matrices_.contains(name))
                temporary = std::max(temporary, bytes);
        };
        const auto vector_bytes = [&](const std::string &name) {
            weight_bytes += lookup_(name).info->elements() * sizeof(float);
        };
        matrix_bytes("v.patch_embd.weight");
        vector_bytes("v.patch_embd.bias");
        for (int layer = 0; layer < g_.layers; ++layer) {
            const auto n = [&](const char *s) { return "v.blk." + std::to_string(layer) + "." + s; };
            for (const char *s : {"ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"})
                vector_bytes(n(s));
            for (const char *s : {"attn_q", "attn_k", "attn_v", "attn_out", "ffn_up", "ffn_down"}) {
                matrix_bytes(n((std::string(s) + ".weight").c_str()));
                vector_bytes(n((std::string(s) + ".bias").c_str()));
            }
        }
        vector_bytes("mm.pre_norm.weight");
        vector_bytes("v.image_newline");
        for (const char *s : {"mm.0", "mm.2"}) {
            matrix_bytes(std::string(s) + ".weight");
            vector_bytes(std::string(s) + ".bias");
        }
        size_t available = 0, total = 0;
        check(cudaMemGetInfo(&available, &total));
        const size_t workspace = cuda_workspace_bytes(g_, pw, ph);
        const size_t missing_weights = weight_bytes > weight_bytes_ ? weight_bytes - weight_bytes_ : 0;
        // cuBLAS internals/allocator granularity and concurrent serving retain
        // headroom beyond the exact arrays. Do not evict text experts implicitly.
        constexpr size_t reserve = size_t(256) << 20;
        if (cuda_memory_fits(available, missing_weights, workspace, temporary, reserve))
            return {};
        return "CUDA admission rejected " + std::to_string(pw * ph) + " image patches: free=" +
               std::to_string(available) + " new_weights=" + std::to_string(missing_weights) +
               " workspace=" + std::to_string(workspace) + " conversion=" + std::to_string(temporary) +
               " reserve=" + std::to_string(reserve);
    }
    Vec encode(const Vec &patches, const Vec &positions, int pw, int ph, const VisionTrace &trace) override {
        check(cudaSetDevice(o_.cuda_device));
        check(cudaStreamSynchronize(stream_));
        for (void *p : work_)
            cudaFree(p);
        work_.clear();
        const int count = pw * ph, d = g_.dim, hd = d / g_.heads, merged = count / (g_.merge * g_.merge),
                  inner = d * g_.merge * g_.merge;
        const size_t nd = size_t(count) * d;
        float *input = alloc<float>(patches.size());
        check(cudaMemcpyAsync(input, patches.data(), patches.size() * sizeof(float), cudaMemcpyHostToDevice,
                              stream_));
        float *pos = alloc<float>(positions.size());
        check(cudaMemcpyAsync(pos, positions.data(), positions.size() * sizeof(float), cudaMemcpyHostToDevice,
                              stream_));
        float *x = alloc<float>(nd);
        float *norm = alloc<float>(nd);
        float *q = alloc<float>(nd);
        float *k = alloc<float>(nd);
        float *v = alloc<float>(nd);
        float *ctx = alloc<float>(nd);
        float *tmp = alloc<float>(nd);
        float *qh = alloc<float>(nd);
        float *kh = alloc<float>(nd);
        float *vh = alloc<float>(nd);
        float *ch = alloc<float>(nd);
        float *up = alloc<float>(size_t(count) * g_.ff);
        float *scores = alloc<float>(size_t(count) * std::min(count, 128) * g_.heads);
        __half *half =
            alloc<__half>(size_t(count) * std::max({3 * g_.patch * g_.patch, d, g_.ff, g_.hidden, inner}));
        mm("v.patch_embd.weight", "v.patch_embd.bias", input, x, count, 3 * g_.patch * g_.patch, d, half);
        add<<<blocks(nd), 256, 0, stream_>>>(x, pos, nd);
        dump(trace, "vision_patch", -1, x, nd, d);
        for (int layer = 0; layer < g_.layers; ++layer) {
            auto n = [&](const char *s) { return "v.blk." + std::to_string(layer) + "." + s; };
            normalize<<<count, 256, 0, stream_>>>(x, norm, vector(n("ln1.weight")), vector(n("ln1.bias")), d,
                                                  g_.eps, false);
            mm(n("attn_q.weight"), n("attn_q.bias"), norm, q, count, d, d, half);
            mm(n("attn_k.weight"), n("attn_k.bias"), norm, k, count, d, d, half);
            mm(n("attn_v.weight"), n("attn_v.bias"), norm, v, count, d, d, half);
            pack_heads<<<blocks(nd), 256, 0, stream_>>>(q, qh, count, d, hd);
            pack_heads<<<blocks(nd), 256, 0, stream_>>>(k, kh, count, d, hd);
            pack_heads<<<blocks(nd), 256, 0, stream_>>>(v, vh, count, d, hd);
            for (int q0 = 0; q0 < count; q0 += 128) {
                const int chunk = std::min(128, count - q0);
                const float scale = 1.f / std::sqrt(float(hd)), one = 1, zero = 0;
                blas_check(cublasSgemmStridedBatched(blas_, CUBLAS_OP_T, CUBLAS_OP_N, count, chunk, hd,
                                                     &scale, kh, hd, (long long)count * hd,
                                                     qh + (size_t)q0 * hd, hd, (long long)count * hd, &zero,
                                                     scores, count, (long long)count * chunk, g_.heads));
                softmax<<<chunk * g_.heads, 256, 0, stream_>>>(scores, count);
                blas_check(cublasSgemmStridedBatched(blas_, CUBLAS_OP_N, CUBLAS_OP_N, hd, chunk, count, &one,
                                                     vh, hd, (long long)count * hd, scores, count,
                                                     (long long)count * chunk, &zero, ch + (size_t)q0 * hd,
                                                     hd, (long long)count * hd, g_.heads));
            }
            unpack_heads<<<blocks(nd), 256, 0, stream_>>>(ch, ctx, count, d, hd);
            mm(n("attn_out.weight"), n("attn_out.bias"), ctx, tmp, count, d, d, half);
            add<<<blocks(nd), 256, 0, stream_>>>(x, tmp, nd);
            normalize<<<count, 256, 0, stream_>>>(x, norm, vector(n("ln2.weight")), vector(n("ln2.bias")), d,
                                                  g_.eps, false);
            mm(n("ffn_up.weight"), n("ffn_up.bias"), norm, up, count, d, g_.ff, half);
            activate<<<blocks(size_t(count) * g_.ff), 256, 0, stream_>>>(up, size_t(count) * g_.ff);
            mm(n("ffn_down.weight"), n("ffn_down.bias"), up, tmp, count, g_.ff, d, half);
            add<<<blocks(nd), 256, 0, stream_>>>(x, tmp, nd);
            dump(trace, "vision_layer", layer, x, nd, d);
        }
        normalize<<<count, 256, 0, stream_>>>(x, norm, vector("mm.pre_norm.weight"), nullptr, d, g_.eps,
                                              true);
        float *packed = alloc<float>(size_t(merged) * inner);
        float *mid = alloc<float>(size_t(merged) * g_.hidden);
        float *projected = alloc<float>(size_t(merged) * g_.out);
        const int ow = pw / g_.merge, oh = ph / g_.merge;
        pack_merge<<<blocks(size_t(merged) * inner), 256, 0, stream_>>>(norm, packed, pw, ow, oh, d,
                                                                        g_.merge);
        mm("mm.0.weight", "mm.0.bias", packed, mid, merged, inner, g_.hidden, half);
        activate<<<blocks(size_t(merged) * g_.hidden), 256, 0, stream_>>>(mid, size_t(merged) * g_.hidden);
        mm("mm.2.weight", "mm.2.bias", mid, projected, merged, g_.hidden, g_.out, half);
        dump(trace, "vision_projected", -1, projected, size_t(merged) * g_.out, g_.out);
        const size_t out_count = size_t(ow + 1) * oh * g_.out;
        float *out = alloc<float>(out_count);
        insert_newline<<<blocks(out_count), 256, 0, stream_>>>(projected, out, vector("v.image_newline"), ow,
                                                               oh, g_.out);
        check(cudaGetLastError());
        Vec result(out_count);
        check(
            cudaMemcpyAsync(result.data(), out, out_count * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        check(cudaStreamSynchronize(stream_));
        return result;
    }

  private:
    static unsigned blocks(size_t n) { return unsigned((n + 255) / 256); }
    template <class T> T *alloc(size_t count) {
        void *p = nullptr;
        check(cudaMalloc(&p, count * sizeof(T)));
        work_.push_back(p);
        return static_cast<T *>(p);
    }
    void *weight_alloc(size_t bytes) {
        void *p = nullptr;
        check(cudaMalloc(&p, bytes));
        weights_.push_back(p);
        weight_bytes_ += bytes;
        return p;
    }
    const float *vector(const std::string &name) {
        auto it = vectors_.find(name);
        if (it != vectors_.end())
            return it->second;
        Vec host = lookup_(name).values();
        float *p = static_cast<float *>(weight_alloc(host.size() * sizeof(float)));
        check(cudaMemcpy(p, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
        vectors_.emplace(name, p);
        return p;
    }
    Matrix &matrix(const std::string &name, int inner, int outer) {
        auto found = matrices_.find(name);
        if (found != matrices_.end())
            return found->second;
        const Tensor &tensor = lookup_(name);
        if (tensor.info->elements() != size_t(inner) * outer)
            throw std::runtime_error("Hy4 vision CUDA matrix dimensions: " + name);
        Matrix m;
        m.inner = inner;
        m.outer = outer;
        if (o_.fp32 || tensor.info->type == 0) {
            Vec host = tensor.values();
            m.data = weight_alloc(host.size() * sizeof(float));
            check(cudaMemcpy(m.data, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
            m.type = 0;
        } else if (tensor.info->type == 1) {
            m.data = weight_alloc(size_t(inner) * outer * 2);
            check(cudaMemcpy(m.data, tensor.data, size_t(inner) * outer * 2, cudaMemcpyHostToDevice));
            m.type = 1;
        } else if (tensor.info->type == 30) {
            size_t n = size_t(inner) * outer;
            void *raw = nullptr;
            check(cudaMalloc(&raw, n * 2));
            try {
                check(cudaMemcpy(raw, tensor.data, n * 2, cudaMemcpyHostToDevice));
                m.data = weight_alloc(n * 2);
                convert_bf16<<<blocks(n), 256, 0, stream_>>>(static_cast<const __nv_bfloat16 *>(raw),
                                                             static_cast<__half *>(m.data), n);
                check(cudaStreamSynchronize(stream_));
                check(cudaFree(raw));
            } catch (...) {
                cudaFree(raw);
                throw;
            }
            m.type = 1;
        } else
            throw std::runtime_error("Hy4 vision CUDA unsupported matrix type");
        return matrices_.emplace(name, m).first->second;
    }
    void mm(const std::string &name, const std::string &bias, const float *input, float *output, int count,
            int inner, int outer, __half *half) {
        Matrix &w = matrix(name, inner, outer);
        const float one = 1, zero = 0;
        if (w.type == 0)
            blas_check(cublasSgemm(blas_, CUBLAS_OP_T, CUBLAS_OP_N, outer, count, inner, &one,
                                   static_cast<const float *>(w.data), inner, input, inner, &zero, output,
                                   outer));
        else {
            cast_half<<<blocks(size_t(count) * inner), 256, 0, stream_>>>(input, half, size_t(count) * inner);
            blas_check(cublasGemmEx(blas_, CUBLAS_OP_T, CUBLAS_OP_N, outer, count, inner, &one, w.data,
                                    CUDA_R_16F, inner, half, CUDA_R_16F, inner, &zero, output, CUDA_R_32F,
                                    outer, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
        add_bias<<<blocks(size_t(count) * outer), 256, 0, stream_>>>(output, vector(bias), outer,
                                                                     size_t(count) * outer);
    }
    void dump(const VisionTrace &trace, const std::string &stage, int layer, const float *p, size_t count,
              int width) {
        if (trace) {
            Vec host(count);
            check(cudaMemcpyAsync(host.data(), p, count * sizeof(float), cudaMemcpyDeviceToHost, stream_));
            check(cudaStreamSynchronize(stream_));
            trace(stage, layer, host, width);
        }
    }
    Geometry g_;
    Lookup lookup_;
    VisionOptions o_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    std::vector<void *> weights_, work_;
    size_t weight_bytes_ = 0;
    std::map<std::string, Matrix> matrices_;
    std::map<std::string, const float *> vectors_;
};
} // namespace
std::unique_ptr<Backend> cuda_backend(const Geometry &g, Lookup lookup, const VisionOptions &options) {
    return std::make_unique<Cuda>(g, std::move(lookup), options);
}
} // namespace strata::hy4::vision_detail
