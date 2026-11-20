// Hy4 latent experts: native CUDA execution, bounded streaming and residency.
// The Q4_K/Q6_K routed dot expressions below are adapted unchanged from
// Strata native_mmvq.cu, itself pinned to llama.cpp 3cf03257.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/hy4/device_experts.hpp"
#include "strata/hy4/profile.hpp"
#include "strata/hy4/offset_arena.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "prefill_gemm.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <list>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace strata::hy4 {
namespace {
using kernels::native_mmvq;
using kernels::native_mmvq_supported;
using kernels::native_q8_1_bytes;
using kernels::native_quantize_q8_1;

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
struct Q4KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
static_assert(sizeof(Q81Block) == 36 && sizeof(Q4KBlock) == 144 && sizeof(Q6KBlock) == 210);
__device__ __forceinline__ int load_int_b2(const void *p, int i) {
    const auto *u = static_cast<const uint16_t *>(p) + 2 * i;
    return int(u[0]) | int(u[1]) << 16;
}
__device__ __forceinline__ float q4_q8_dot_impl(const int *__restrict__ v, const int *__restrict__ u,
                                                const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
                                                const half2 &dm4, const float *__restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = STRATA_DP4A(v1i, u[2 * i + 1], STRATA_DP4A(v0i, u[2 * i], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i + 1], STRATA_DP4A(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float q4_q8_dot(const Q4KBlock *__restrict__ bq4, const Q81Block *__restrict__ bq8,
                                           int iqs) {
    int v[2];
    int u[4];
    float d8[2];
    const int bq8_offset = 2 * ((iqs / 2) / 4);
    const int *ql = reinterpret_cast<const int *>(bq4->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = ql[0];
    v[1] = ql[4];

    const uint16_t *scales = reinterpret_cast<const uint16_t *>(bq4->scales);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    uint16_t aux[2];
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t *sc = reinterpret_cast<const uint8_t *>(aux);
    const uint8_t *m = sc + 2;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const Q81Block *bq8i = bq8 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int *q8 = reinterpret_cast<const int *>(bq8i->qs) + ((iqs / 2) % 4);
        u[2 * i] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return q4_q8_dot_impl(v, u, sc, m, bq4->dm, d8);
}

__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int *__restrict__ u,
                                                const int8_t *__restrict__ scales, float d,
                                                const float *__restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}
__device__ __forceinline__ float q6_q8_dot(const Q6KBlock *__restrict__ w, const Q81Block *__restrict__ x,
                                           int iqs) {
    const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
    const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
    const int vh_shift = 2 * ((iqs % 16) / 8);
    const int vl = load_int_b2(w->ql, iqs);
    const int vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
    int u[2];
    float d8[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        u[i] = reinterpret_cast<const int *>(x[bq8_offset + 2 * i].qs)[iqs % 8];
        d8[i] = __low2float(x[bq8_offset + 2 * i].ds);
    }
    return q6_q8_dot_impl(vl, vh, u, w->scales + scale_offset, w->d, d8);
}

__device__ __forceinline__ float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}
__device__ __forceinline__ float warp_max(float x) {
    for (int offset = 16; offset > 0; offset >>= 1)
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, offset));
    return x;
}

// Identical arithmetic/reduction to the native single-column Q4_K/Q6_K adapters;
// the weight pointer and activation row are selected from routing on the device.
template <int Type, bool SmallK>
__global__ void routed_mmvq(const unsigned long long *table, const int32_t *ids, const Q81Block *x, float *y,
                            float *y_up, int n_in, int n_out, int used, int experts, size_t offset,
                            size_t up_offset, bool per_route) {
    constexpr int Rows = SmallK ? 4 : 1, T = Type == 12 ? 16 : 32, Bpi = 128 / T;
    const int tid = threadIdx.x + 32 * threadIdx.y;
    const int route = blockIdx.y, expert = ids[route], row0 = Rows * blockIdx.x;
    float *dst = blockIdx.z ? y_up : y;
    if (expert < 0 || expert >= experts || !table[expert]) {
        if (threadIdx.y == 0 && threadIdx.x < Rows && row0 + threadIdx.x < n_out)
            dst[size_t(route) * n_out + row0 + threadIdx.x] = nanf("");
        return;
    }
    const auto *weights =
        reinterpret_cast<const uint8_t *>(table[expert]) + (blockIdx.z ? up_offset : offset);
    const auto *input = x + size_t(per_route ? route : route / used) * (n_in / 32);
    const int nb = n_in / 256, part = Type == 12 ? 2 * (tid % 16) : tid % 32;
    float tmp[Rows]{};
    for (int block = tid / T; block < nb; block += Bpi) {
        for (int i = 0; i < Rows; ++i)
            if (row0 + i < n_out) {
                const size_t w = size_t(row0 + i) * nb + block;
                if constexpr (Type == 12)
                    tmp[i] +=
                        q4_q8_dot(reinterpret_cast<const Q4KBlock *>(weights) + w, input + block * 8, part);
                else
                    tmp[i] +=
                        q6_q8_dot(reinterpret_cast<const Q6KBlock *>(weights) + w, input + block * 8, part);
            }
    }
    __shared__ float partial[3][Rows][32];
    if (threadIdx.y > 0)
        for (int i = 0; i < Rows; ++i)
            partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    __syncthreads();
    if (threadIdx.y > 0)
        return;
    for (int i = 0; i < Rows; ++i) {
        for (int w = 0; w < 3; ++w)
            tmp[i] += partial[w][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out)
            dst[size_t(route) * n_out + row0 + i] = tmp[i];
    }
}

// The source model clips the gate before SiLU, not the SiLU output.
// Every whole warp is one q8_1 block, with the same stored product and quantizer
// as native_swiglu_quantize_q8_1, after the Hy4-specific clipping.
__global__ void clamped_swiglu_quant(const float *gate, const float *up, Q81Block *h, size_t n, float limit) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float g = limit > 0 ? fminf(gate[i], limit) : gate[i];
    const float u = limit > 0 ? fminf(fmaxf(up[i], -limit), limit) : up[i];
    const float value = __fmul_rn(__fdividef(g, __fadd_rn(1.0f, __expf(-g))), u);
    const float amax = warp_max(fabsf(value)), sum = warp_sum(value);
    const float d = kernels::q8_1_finite(amax / 127.0f);
    h[i / 32].qs[i % 32] = kernels::q8_1_quant(value, d, amax);
    if (i % 32 == 0)
        h[i / 32].ds = kernels::q8_1_ds(d, sum);
}

__global__ void clamped_swiglu_float(const float *gate, const float *up, float *h, size_t n, float limit) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float g = limit > 0 ? fminf(gate[i], limit) : gate[i];
    const float u = limit > 0 ? fminf(fmaxf(up[i], -limit), limit) : up[i];
    h[i] = __fmul_rn(__fdividef(g, __fadd_rn(1.0f, __expf(-g))), u);
}

// A prompt routes on the GPU once. Host offload decisions consume only one
// count per expert; route indices, activation gather and output scatter stay GPU.
__global__ void count_routes(const int32_t* ids,int32_t* counts,int entries,int used,int experts){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=entries)return;const int e=ids[i];
    if(e<0||e>=experts){atomicExch(counts+experts,1);return;}
    for(int k=0;k<i%used;++k)if(ids[i-i%used+k]==e)atomicExch(counts+experts,1);
    atomicAdd(counts+e,1);
}
__global__ void prefix_routes(const int32_t* counts,int32_t* offsets,int32_t* cursor,int32_t* task_expert,int32_t* task_start,int32_t* tasks,int experts){
    if(threadIdx.x)return;int offset=0,n=0;
    for(int e=0;e<experts;++e){offsets[e]=cursor[e]=offset;offset+=counts[e];
        for(int at=0;at<counts[e];at+=16){task_expert[n]=e;task_start[n++]=at;}}
    offsets[experts]=offset;*tasks=n;
}
__global__ void compact_routes(const int32_t* ids,int32_t* cursor,int32_t* indices,int entries,int experts){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<entries){const int e=ids[i];if(e>=0&&e<experts)indices[atomicAdd(cursor+e,1)]=i;}
}
__global__ void gather_routes(const float* input,float* output,const int32_t* indices,int count,int width,int used){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<size_t(count)*width)output[i]=input[size_t(indices[i/width]/used)*width+i%width];
}
__global__ void scatter_routes(const float* input,float* output,const int32_t* indices,int count,int width){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<size_t(count)*width)output[size_t(indices[i/width])*width+i%width]=input[i];
}
__global__ void poison_routes(float* output,const int32_t* counts,int experts,size_t n){
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i<n&&counts[experts])output[i]=nanf("");
}

__global__ void weighted_reduce(const float *routes, const float *weights, float *out, int latent, int used,
                                int rows) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * latent)
        return;
    const size_t row = i / latent, col = i % latent;
    float sum = 0;
    for (int k = 0; k < used; ++k)
        sum = __fadd_rn(sum, __fmul_rn(routes[(row * used + k) * latent + col], weights[row * used + k]));
    out[i] = sum;
}

__global__ void floating_mmv(const uint8_t *weights, const float *x, float *y, int type, int n_in,
                             int n_out) {
    const int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32, token = blockIdx.y;
    if (row >= n_out)
        return;
    float sum = 0;
    for (int i = lane; i < n_in; i += 32) {
        const size_t at = size_t(row) * n_in + i;
        const float w =
            type == 0   ? reinterpret_cast<const float *>(weights)[at]
            : type == 1 ? __half2float(reinterpret_cast<const half *>(weights)[at])
                        : __uint_as_float(unsigned(reinterpret_cast<const uint16_t *>(weights)[at]) << 16);
        sum += w * x[size_t(token) * n_in + i];
    }
    sum = warp_sum(sum);
    if (lane == 0)
        y[size_t(token) * n_out + row] = sum;
}

// Floating MTP experts use their native BF16/F16/F32 weights directly. The
// warp dot and reduction match floating_mmv; route selection stays on the GPU.
__global__ void routed_floating_mmv(const unsigned long long *table, const int32_t *ids, const float *x,
                                    float *y, float *y_up, int type, int up_type, int n_in, int n_out,
                                    int used, int experts, size_t offset, size_t up_offset, bool per_route) {
    const int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32, route = blockIdx.y;
    if (row >= n_out)
        return;
    float *dst = blockIdx.z ? y_up : y;
    const int expert = ids[route];
    if (expert < 0 || expert >= experts || !table[expert]) {
        if (lane == 0)
            dst[size_t(route) * n_out + row] = nanf("");
        return;
    }
    const auto *weights =
        reinterpret_cast<const uint8_t *>(table[expert]) + (blockIdx.z ? up_offset : offset);
    const int format = blockIdx.z ? up_type : type, token = per_route ? route : route / used;
    float sum = 0;
    for (int i = lane; i < n_in; i += 32) {
        const size_t at = size_t(row) * n_in + i;
        const float w =
            format == 0   ? reinterpret_cast<const float *>(weights)[at]
            : format == 1 ? __half2float(reinterpret_cast<const half *>(weights)[at])
                          : __uint_as_float(unsigned(reinterpret_cast<const uint16_t *>(weights)[at]) << 16);
        sum += w * x[size_t(token) * n_in + i];
    }
    sum = warp_sum(sum);
    if (lane == 0)
        dst[size_t(route) * n_out + row] = sum;
}

void ck(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("Hy4 experts: ") + cudaGetErrorString(error));
}
struct DeviceGuard {
    int previous;
    explicit DeviceGuard(int device) {
        ck(cudaGetDevice(&previous));
        ck(cudaSetDevice(device));
    }
    ~DeviceGuard() { cudaSetDevice(previous); }
};
struct Buffer {
    int device;
    void *ptr = nullptr;
    size_t bytes;
    Buffer(int device, size_t bytes) : device(device), bytes(bytes) {
        DeviceGuard guard(device);
        if (bytes) {
            size_t available=0,total=0;ck(cudaMemGetInfo(&available,&total));
            const auto error=cudaMalloc(&ptr,bytes);
            if(error!=cudaSuccess)throw std::runtime_error("Hy4 experts device="+std::to_string(device)+" cudaMalloc requested="+std::to_string(bytes)+" free="+std::to_string(available)+" total="+std::to_string(total)+": "+cudaGetErrorString(error));
        }
    }
    ~Buffer() {
        cudaSetDevice(device);
        if (ptr)
            cudaFree(ptr);
    }
    template <class T> T *as() const { return static_cast<T *>(ptr); }
};
struct HostBuffer {
    void *ptr = nullptr;
    size_t bytes;
    explicit HostBuffer(size_t bytes) : bytes(bytes) {
        if (bytes)
            ck(cudaMallocHost(&ptr, bytes));
    }
    ~HostBuffer() {
        if (ptr)
            cudaFreeHost(ptr);
    }
    template <class T> T *as() const { return static_cast<T *>(ptr); }
};
struct Stream {
    int device;
    cudaStream_t value = nullptr;
    explicit Stream(int device) : device(device) {
        DeviceGuard guard(device);
        ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
    }
    ~Stream() {
        cudaSetDevice(device);
        if (value) {
            cudaStreamSynchronize(value);
            cudaStreamDestroy(value);
        }
    }
};
struct Event {
    int device;
    cudaEvent_t value = nullptr;
    bool recorded = false;
    explicit Event(int device) : device(device) {
        DeviceGuard guard(device);
        ck(cudaEventCreateWithFlags(&value, cudaEventDisableTiming));
    }
    ~Event() {
        cudaSetDevice(device);
        if (value)
            cudaEventDestroy(value);
    }
    void record(cudaStream_t stream) {
        ck(cudaEventRecord(value, stream));
        recorded = true;
    }
    void wait(cudaStream_t stream) const {
        if (recorded)
            ck(cudaStreamWaitEvent(stream, value, 0));
    }
    void sync() const {
        if (recorded)
            ck(cudaEventSynchronize(value));
    }
};

struct Layer {
    Tensor gate, up, down;
    size_t gate_bytes = 0, up_bytes = 0, down_bytes = 0, bytes = 0;
    bool fast = false,draft=false;
};
Layer describe_layer(const Geometry &g, const ExpertExecutor::Lookup &lookup, int layer) {
    const auto name = [&](const char *suffix) { return "blk." + std::to_string(layer) + "." + suffix; };
    Layer l{lookup(name("ffn_gate_exps.weight")), lookup(name("ffn_up_exps.weight")),
            lookup(name("ffn_down_exps.weight"))};
    for (const auto *t : {&l.gate, &l.up, &l.down}) {
        const uint32_t type = t->info->type;
        if (!native_mmvq_supported(type) && type != 0 && type != 1 && type != 30)
            throw std::runtime_error("Hy4 experts: unsupported format for " + t->info->name);
        const bool down = t == &l.down;
        if (t->info->shape != std::vector<uint64_t>{uint64_t(down ? g.ff : g.latent),
                                                    uint64_t(down ? g.latent : g.ff), uint64_t(g.experts)})
            throw std::runtime_error("Hy4 experts: invalid shape for " + t->info->name);
        if (!t->data)
            throw std::runtime_error("Hy4 experts: null tensor data");
    }
    l.gate_bytes = l.gate.row_bytes() * g.ff;
    l.up_bytes = l.up.row_bytes() * g.ff;
    l.down_bytes = l.down.row_bytes() * g.latent;
    l.bytes = l.gate_bytes + l.up_bytes + l.down_bytes;
    if (l.gate_bytes % 4 || l.up_bytes % 4)
        throw std::runtime_error("Hy4 experts: unaligned expert matrices");
    const auto floating = [](uint32_t type) { return type == 0 || type == 1 || type == 30; };
    l.fast = (l.gate.info->type == 12 && l.up.info->type == 12 &&
              (l.down.info->type == 12 || l.down.info->type == 14)) ||
             (floating(l.gate.info->type) && floating(l.up.info->type) && floating(l.down.info->type));
    l.draft=layer>=g.layers;
    return l;
}

size_t max_blob(const Geometry &g, const ExpertExecutor::Lookup &lookup) {
    if (g.latent <= 0 || g.ff <= 0 || g.latent % 32 || g.ff % 32 || g.used <= 0 || g.experts < g.used ||
        g.layers <= 0)
        throw std::runtime_error("Hy4 experts: invalid geometry");
    size_t bytes = 0;
    for (int l = g.dense_layers; l < g.layers + g.draft_layers; ++l)
        bytes = std::max(bytes, describe_layer(g, lookup, l).bytes);
    if (!bytes)
        throw std::runtime_error("Hy4 experts: no expert layers");
    return bytes;
}

void copy_device(void *dst, int destination, const void *src, int source, size_t bytes, cudaStream_t stream) {
    if (destination == source)
        ck(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
    else
        ck(cudaMemcpyPeerAsync(dst, destination, src, source, bytes, stream));
}
void linear_device(uint32_t type, const uint8_t *weights, const float *x, const void *q, float *y, int n_in,
                   int n_out, int rows, cudaStream_t stream) {
    if (native_mmvq_supported(type)) {
        for(int at=0;at<rows;at+=8)native_mmvq(type,weights,static_cast<const uint8_t*>(q)+size_t(at)*(n_in/32)*sizeof(Q81Block),y+size_t(at)*n_out,n_in,n_out,std::min(8,rows-at),stream);
    }
    else
        floating_mmv<<<dim3((n_out + 3) / 4, rows), 128, 0, stream>>>(weights, x, y, type, n_in, n_out);
    ck(cudaGetLastError());
}

template <int Type>
void launch_routed_type(const unsigned long long *table, const int32_t *ids, const void *x, float *out,
                        float *up, int n_in, int n_out, int used, int experts, int entries, size_t offset,
                        size_t up_offset, bool per_route, cudaStream_t stream) {
    const bool small = n_in / 256 < (Type == 12 ? 8 : 4);
    const int rows = small ? 4 : 1;
    const dim3 blocks((n_out + rows - 1) / rows, entries, up ? 2 : 1), threads(32, 4);
    if (small)
        routed_mmvq<Type, true><<<blocks, threads, 0, stream>>>(table, ids, static_cast<const Q81Block *>(x),
                                                                out, up, n_in, n_out, used, experts, offset,
                                                                up_offset, per_route);
    else
        routed_mmvq<Type, false><<<blocks, threads, 0, stream>>>(table, ids, static_cast<const Q81Block *>(x),
                                                                 out, up, n_in, n_out, used, experts, offset,
                                                                 up_offset, per_route);
    ck(cudaGetLastError());
}
} // namespace

ExpertMemory ExpertExecutor::memory_requirements(const Geometry &g, const Lookup &lookup, int max_rows,
                                                 int lanes) {
    if (max_rows < 1 || max_rows > 512 || lanes < 1 || lanes > 16)
        throw std::runtime_error("Hy4 experts: rows must be 1..512 and lanes 1..16");
    const size_t blob = max_blob(g, lookup), rows = max_rows, entries = rows * g.used;
    const size_t latent = rows * g.latent * sizeof(float), ff = rows * g.ff * sizeof(float);
    const size_t xq = rows * (g.latent / 32) * sizeof(Q81Block), hq = rows * (g.ff / 32) * sizeof(Q81Block);
    const size_t lane = blob + 2 * latent + 3 * ff + xq + hq;
    const size_t routes = entries * g.latent * sizeof(float), full_ff = entries * g.ff * sizeof(float);
    const size_t full_hq = entries * (g.ff / 32) * sizeof(Q81Block);
    const size_t ids = entries * sizeof(int32_t), weights = entries * sizeof(float);
    const size_t tables = size_t(g.layers + g.draft_layers) * g.experts * sizeof(unsigned long long);
    const size_t tasks=(entries+15*size_t(g.experts))/16;
    const size_t metadata=max_rows>8?(3*size_t(g.experts)+entries+2*tasks+3)*sizeof(int32_t):0;
    const size_t host_metadata=max_rows>8?size_t(g.experts+1)*sizeof(int32_t):0;
    return {size_t(lanes) * lane + routes + 3 * full_ff + full_hq + xq + 2 * latent + ids + weights + tables+metadata,
            size_t(lanes) * blob + ids+host_metadata};
}

struct ExpertExecutor::Impl {
    using Key = uint64_t;
    struct View {
        void* ptr;size_t bytes,offset;
        template<class T>T* as()const{return static_cast<T*>(ptr);}
    };
    struct Entry {
        View weights;size_t charge;
        Event* last_use=nullptr; // fixed Lane retirement event, never owns CUDA memory
        bool pinned = false;
        std::list<Key>::iterator lru;
        Entry(void* ptr,size_t bytes,size_t offset,size_t charge):weights{ptr,bytes,offset},charge(charge){}
    };
    struct Lane {
        Stream copy, compute;
        HostBuffer staging;
        Buffer transient, input, xq, gate, up, hidden, hq, down;
        Event uploaded, weights_used, finished;
        bool touched = false;
        Lane(int device, size_t blob, const Geometry &g, int rows)
            : copy(device), compute(device), staging(blob), transient(device, blob),
              input(device, size_t(rows) * g.latent * 4), xq(device, size_t(rows)*(g.latent/32)*sizeof(Q81Block)),
              gate(device, size_t(rows) * g.ff * 4), up(device, size_t(rows) * g.ff * 4),
              hidden(device, size_t(rows) * g.ff * 4),
              hq(device, size_t(rows) * (g.ff / 32) * sizeof(Q81Block)),
              down(device, size_t(rows) * g.latent * 4), uploaded(device), weights_used(device),
              finished(device) {}
    };
    struct Card {
        int device;
        size_t capacity, used = 0, pinned = 0, fixed,observed_driver_bytes=0;
        Stream fast;
        Event input_ready, fast_done, final_done;
        HostBuffer host_ids,host_counts;
        Buffer routes, fast_gate, fast_up, fast_hidden, fast_hq, fast_xq, fast_input, fast_ids, fast_weights,
            fast_out, tables,route_counts,route_offsets,route_cursor,route_indices,task_expert,task_start,task_count;
        std::unique_ptr<Buffer> arena;
        std::unique_ptr<OffsetArena> arena_map;
        std::vector<std::unique_ptr<Lane>> lanes;
        std::list<Key> lru;
        std::unordered_map<Key, std::unique_ptr<Entry>> cache;
        size_t next_lane = 0;
        Card(const ExpertDevice &config, const Geometry &g, size_t blob, int rows, int n_lanes,
             ExpertMemory memory)
            : device(config.device), capacity(config.resident_bytes), fixed(memory.device_workspace),
              fast(device), input_ready(device), fast_done(device), final_done(device),
              host_ids(size_t(rows) * g.used * sizeof(int32_t)),host_counts(rows>8?size_t(g.experts+1)*4:0),
              routes(device, size_t(rows) * g.used * g.latent * 4),
              fast_gate(device, size_t(rows) * g.used * g.ff * 4),
              fast_up(device, size_t(rows) * g.used * g.ff * 4),
              fast_hidden(device, size_t(rows) * g.used * g.ff * 4),
              fast_hq(device, size_t(rows) * g.used * (g.ff / 32) * sizeof(Q81Block)),
              fast_xq(device, size_t(rows)*(g.latent/32)*sizeof(Q81Block)),
              fast_input(device, size_t(rows) * g.latent * 4),
              fast_ids(device, size_t(rows) * g.used * sizeof(int32_t)),
              fast_weights(device, size_t(rows) * g.used * 4), fast_out(device, size_t(rows) * g.latent * 4),
              tables(device, size_t(g.layers + g.draft_layers) * g.experts * sizeof(unsigned long long)),
              route_counts(device,rows>8?size_t(g.experts+1)*4:0),route_offsets(device,rows>8?size_t(g.experts+1)*4:0),
              route_cursor(device,rows>8?size_t(g.experts)*4:0),route_indices(device,rows>8?size_t(rows)*g.used*4:0),
              task_expert(device,rows>8?((size_t(rows)*g.used+15*size_t(g.experts))/16)*4:0),
              task_start(device,rows>8?((size_t(rows)*g.used+15*size_t(g.experts))/16)*4:0),task_count(device,rows>8?4:0) {
            DeviceGuard guard(device);
            for (int i = 0; i < n_lanes; ++i)
                lanes.push_back(std::make_unique<Lane>(device, blob, g, rows));
            ck(cudaMemsetAsync(tables.ptr, 0, tables.bytes, fast.value));
            ck(cudaStreamSynchronize(fast.value));
            size_t available=0,total=0;ck(cudaMemGetInfo(&available,&total));
            if(config.elastic)capacity=std::min(capacity,available)&~((size_t(2)<<20)-1);
            if(capacity>available)throw std::runtime_error("Hy4 expert arena device="+std::to_string(device)+" requested="+std::to_string(capacity)+" free_after_workspace="+std::to_string(available));
            arena=std::make_unique<Buffer>(device,capacity);
            arena_map=std::make_unique<OffsetArena>(capacity);
        }
        ~Card() {
            cudaSetDevice(device);
            if (final_done.recorded)
                cudaEventSynchronize(final_done.value);
            if (fast.value)
                cudaStreamSynchronize(fast.value);
            for (auto &lane : lanes) {
                cudaStreamSynchronize(lane->compute.value);
                cudaStreamSynchronize(lane->copy.value);
            }
        }
    };

    Geometry g;
    Lookup lookup;
    int max_rows, n_lanes;
    size_t max_bytes;
    ExpertMemory memory;
    std::vector<Layer> layers;
    std::vector<std::unique_ptr<Card>> cards;
    std::map<int, size_t> card_index, full_layers;
    ExpertCacheStats counters;
    mutable std::mutex mutex;
    int previous_output_card = -1;
    bool failed = false;

    static Key key(int layer, int expert) { return (uint64_t(uint32_t(layer)) << 32) | uint32_t(expert); }
    Impl(const Geometry &geometry, Lookup lookup, std::vector<ExpertDevice> devices, int rows, int lanes)
        : g(geometry), lookup(std::move(lookup)), max_rows(rows), n_lanes(lanes),
          max_bytes(max_blob(g, this->lookup)),
          memory(ExpertExecutor::memory_requirements(g, this->lookup, rows, lanes)),
          layers(g.layers + g.draft_layers) {
        if (devices.empty())
            throw std::runtime_error("Hy4 experts: at least one CUDA device is required");
        for (int l = g.dense_layers; l < int(layers.size()); ++l)
            layers[l] = describe_layer(g, this->lookup, l);
        int device_count = 0;
        ck(cudaGetDeviceCount(&device_count));
        for (const auto &config : devices) {
            if (config.device < 0 || config.device >= device_count || card_index.contains(config.device))
                throw std::runtime_error("Hy4 experts: invalid or duplicate CUDA device");
            DeviceGuard guard(config.device);
            size_t available, total;
            ck(cudaMemGetInfo(&available, &total));
            if (memory.device_workspace > available ||
                (!config.elastic&&config.resident_bytes > available - memory.device_workspace))
                throw std::runtime_error("Hy4 experts device="+std::to_string(config.device)+" requested_cache="+std::to_string(config.resident_bytes)+" workspace="+std::to_string(memory.device_workspace)+" free="+std::to_string(available));
            card_index[config.device] = cards.size();
            cards.push_back(std::make_unique<Card>(config, g, max_bytes, rows, lanes, memory));
            size_t after=0;ck(cudaMemGetInfo(&after,&total));cards.back()->observed_driver_bytes=available>=after?available-after:0;
        }
        for (auto &dst : cards)
            for (auto &src : cards)
                if (dst->device != src->device) {
                    int peer = 0;
                    ck(cudaDeviceCanAccessPeer(&peer, dst->device, src->device));
                    if (!peer)
                        throw std::runtime_error("Hy4 experts: selected CUDA devices require peer access; "
                                                 "use one device otherwise");
                    DeviceGuard guard(dst->device);
                    const auto error = cudaDeviceEnablePeerAccess(src->device, 0);
                    if (error == cudaErrorPeerAccessAlreadyEnabled)
                        cudaGetLastError();
                    else
                        ck(error);
                }
    }
    ~Impl() {
        try {
            synchronize_all();
        } catch (...) {
        }
    }

    void validate_layer(int layer) const {
        if (layer < g.dense_layers || layer >= int(layers.size()))
            throw std::runtime_error("Hy4 experts: layer has no latent experts");
    }
    void synchronize_all() {
        for (auto &c : cards) {
            DeviceGuard guard(c->device);
            c->final_done.sync();
        }
        for (auto &c : cards) {
            DeviceGuard guard(c->device);
            ck(cudaStreamSynchronize(c->fast.value));
            for (auto &lane : c->lanes) {
                ck(cudaStreamSynchronize(lane->compute.value));
                ck(cudaStreamSynchronize(lane->copy.value));
            }
        }
    }
    Entry *find(Card &c, Key k) {
        auto at = c.cache.find(k);
        if (at == c.cache.end())
            return nullptr;
        c.lru.splice(c.lru.begin(), c.lru, at->second->lru);
        return at->second.get();
    }
    void evict(Card& c,std::list<Key>::iterator victim){
        auto& entry=*c.cache.at(*victim);if(entry.pinned)throw std::runtime_error("Hy4 arena attempted pinned eviction");
        if(entry.last_use)entry.last_use->sync();
        c.arena_map->release(entry.weights.offset,entry.weights.bytes);c.used-=entry.charge;
        const Key old=*victim;c.lru.erase(victim);c.cache.erase(old);++counters.evictions;
    }
    Entry *admit(Card &c, Key k, size_t bytes, bool pin) {
        if(auto* existing=find(c,k)){if(pin&&!existing->pinned){existing->pinned=true;c.pinned+=existing->charge;}return existing;}
        const size_t charge=c.arena_map->allocation_bytes(bytes);
        if(charge>c.capacity-c.pinned)return nullptr;
        auto offset=c.arena_map->allocate(bytes);
        while(!offset){auto victim=c.lru.end();while(victim!=c.lru.begin()){--victim;if(!c.cache.at(*victim)->pinned)break;}
            if(victim==c.lru.end()||c.cache.at(*victim)->pinned)return nullptr;
            evict(c,victim);offset=c.arena_map->allocate(bytes);}
        auto entry=std::make_unique<Entry>(c.arena->as<uint8_t>()+*offset,bytes,*offset,charge);
        entry->pinned=pin;c.lru.push_front(k);entry->lru=c.lru.begin();c.used+=charge;if(pin)c.pinned+=charge;
        auto* result=entry.get();c.cache.emplace(k,std::move(entry));return result;
    }
    void upload(Card &c, Lane &lane, const Layer &l, int expert, void *destination, Entry *entry) {
        ProfileScope profile("hy4.experts.upload");
        // Reuse neither a host staging buffer still read by DMA nor a transient slot
        // still consumed by a previous expert. Copies overlap the other lane's compute.
        { ProfileScope phase("hy4.experts.upload.prior_event"); lane.uploaded.sync(); }
        { ProfileScope phase("hy4.experts.upload.weights_dependency");
          if (!entry) lane.weights_used.wait(lane.copy.value); }
        auto *host = lane.staging.as<uint8_t>();
        { ProfileScope phase("hy4.experts.upload.host_copy");
          std::memcpy(host, l.gate.data + size_t(expert) * l.gate_bytes, l.gate_bytes);
          std::memcpy(host + l.gate_bytes, l.up.data + size_t(expert) * l.up_bytes, l.up_bytes);
          std::memcpy(host + l.gate_bytes + l.up_bytes, l.down.data + size_t(expert) * l.down_bytes,
                      l.down_bytes); }
        { ProfileScope phase("hy4.experts.upload.h2d_enqueue");
          ck(cudaMemcpyAsync(destination, host, l.bytes, cudaMemcpyHostToDevice, lane.copy.value)); }
        { ProfileScope phase("hy4.experts.upload.record"); lane.uploaded.record(lane.copy.value); }
        ++counters.uploads;
        counters.uploaded_bytes += l.bytes;
        if(l.draft)counters.draft_uploaded_bytes+=l.bytes;else counters.target_uploaded_bytes+=l.bytes;
    }
    Card &select_card(int layer, int expert) {
        if (auto at = full_layers.find(layer); at != full_layers.end())
            return *cards[at->second];
        const Key k = key(layer, expert);
        for (auto &card : cards)
            if (card->cache.contains(k))
                return *card;
        return *cards[(size_t(layer) * 131u + size_t(expert)) % cards.size()];
    }

    void prepare_routes(Card& primary,const int32_t* ids,int rows,cudaStream_t stream){
        ck(cudaMemsetAsync(primary.route_counts.ptr,0,primary.route_counts.bytes,stream));
        const int entries=rows*g.used;
        count_routes<<<(entries+255)/256,256,0,stream>>>(ids,primary.route_counts.as<int32_t>(),entries,g.used,g.experts);
        prefix_routes<<<1,32,0,stream>>>(primary.route_counts.as<int32_t>(),primary.route_offsets.as<int32_t>(),primary.route_cursor.as<int32_t>(),primary.task_expert.as<int32_t>(),primary.task_start.as<int32_t>(),primary.task_count.as<int32_t>(),g.experts);
        compact_routes<<<(entries+255)/256,256,0,stream>>>(ids,primary.route_cursor.as<int32_t>(),primary.route_indices.as<int32_t>(),entries,g.experts);ck(cudaGetLastError());
    }
    bool tc_layer(const Layer& layer)const{return prefill_detail::supported(layer.gate.info->type)&&prefill_detail::supported(layer.up.info->type)&&prefill_detail::supported(layer.down.info->type);}
    void run_wide_resident(int layer,Card& primary,Card& owner,const float* latent,const int32_t* ids,const float* weights,float* out,int rows,cudaStream_t caller){
        ProfileScope profile("hy4.experts.prefill_resident");
        const auto& l=layers[layer];primary.input_ready.record(caller);DeviceGuard guard(owner.device);const auto stream=owner.fast.value;primary.input_ready.wait(stream);
        const float* x=latent;const float* scale=weights;const int32_t* selected=ids;
        if(owner.device!=primary.device){
            copy_device(owner.fast_input.ptr,owner.device,latent,primary.device,size_t(rows)*g.latent*4,stream);
            copy_device(owner.fast_ids.ptr,owner.device,ids,primary.device,size_t(rows)*g.used*4,stream);
            copy_device(owner.fast_weights.ptr,owner.device,weights,primary.device,size_t(rows)*g.used*4,stream);
            x=owner.fast_input.as<float>();selected=owner.fast_ids.as<int32_t>();scale=owner.fast_weights.as<float>();
        }
        prepare_routes(owner,selected,rows,stream);
        const auto* table=owner.tables.as<unsigned long long>()+size_t(layer)*g.experts;
        const int tasks=int(owner.task_expert.bytes/4),entries=rows*g.used;
        const auto* offsets=owner.route_offsets.as<int32_t>();const auto* counts=owner.route_counts.as<int32_t>();const auto* indices=owner.route_indices.as<int32_t>();
        prefill_detail::grouped(l.gate.info->type,l.up.info->type,nullptr,table,x,owner.fast_gate.as<float>(),owner.fast_up.as<float>(),g.latent,g.ff,rows,offsets,counts,indices,owner.task_expert.as<int32_t>(),owner.task_start.as<int32_t>(),owner.task_count.as<int32_t>(),tasks,0,0,g.used,false,false,0,l.gate_bytes,stream);
        const size_t n=size_t(entries)*g.ff;
        clamped_swiglu_float<<<unsigned((n+255)/256),256,0,stream>>>(owner.fast_gate.as<float>(),owner.fast_up.as<float>(),owner.fast_hidden.as<float>(),n,g.clamp.at(layer));
        prefill_detail::grouped(l.down.info->type,l.down.info->type,nullptr,table,owner.fast_hidden.as<float>(),owner.routes.as<float>(),nullptr,g.ff,g.latent,rows,offsets,counts,indices,owner.task_expert.as<int32_t>(),owner.task_start.as<int32_t>(),owner.task_count.as<int32_t>(),tasks,0,0,g.used,true,false,l.gate_bytes+l.up_bytes,0,stream);
        float* destination=owner.device==primary.device?out:owner.fast_out.as<float>();
        weighted_reduce<<<(rows*g.latent+255)/256,256,0,stream>>>(owner.routes.as<float>(),scale,destination,g.latent,g.used,rows);
        poison_routes<<<(rows*g.latent+255)/256,256,0,stream>>>(destination,counts,g.experts,size_t(rows)*g.latent);ck(cudaGetLastError());
        if(owner.device!=primary.device)copy_device(out,primary.device,destination,owner.device,size_t(rows)*g.latent*4,stream);
        owner.fast_done.record(stream);
        {DeviceGuard primary_guard(primary.device);owner.fast_done.wait(caller);primary.final_done.record(caller);}
        counters.routed_entries+=entries;counters.hits+=entries;++counters.resident_calls;
    }
    void run_wide_streamed(int layer,Card& primary,const float* latent,const float* weights,float* out,int rows,cudaStream_t caller){
        ProfileScope profile("hy4.experts.prefill_streamed");
        ck(cudaMemcpyAsync(primary.host_counts.ptr,primary.route_counts.ptr,primary.route_counts.bytes,cudaMemcpyDeviceToHost,caller));ck(cudaStreamSynchronize(caller));++counters.metadata_readbacks;
        const auto* counts=primary.host_counts.as<int32_t>();if(counts[g.experts])throw std::runtime_error("Hy4 prefill route IDs invalid or duplicated");
        const auto& l=layers[layer];counters.routed_entries+=uint64_t(rows)*g.used;
        for(auto& card:cards)for(auto& lane:card->lanes)lane->touched=false;
        int offset=0;
        for(int expert=0;expert<g.experts;++expert){const int n=counts[expert];if(!n)continue;++counters.unique_experts;
            Card& card=select_card(layer,expert);DeviceGuard guard(card.device);Lane& lane=*card.lanes[card.next_lane++%card.lanes.size()];lane.touched=true;
            Entry* entry=find(card,key(layer,expert));const bool hit=entry!=nullptr;if(entry)counters.hits+=n;else{counters.misses+=n;entry=admit(card,key(layer,expert),l.bytes,false);upload(card,lane,l,expert,entry?entry->weights.ptr:lane.transient.ptr,entry);}
            if(!hit)lane.uploaded.wait(lane.compute.value);const auto stream=lane.compute.value;
            const auto* blob=entry?entry->weights.as<uint8_t>():lane.transient.as<uint8_t>();
            const auto* indices=primary.route_indices.as<int32_t>()+offset;
            if(tc_layer(l)){
                prefill_detail::grouped(l.gate.info->type,l.up.info->type,blob,nullptr,latent,lane.gate.as<float>(),lane.up.as<float>(),g.latent,g.ff,n,nullptr,nullptr,indices,nullptr,nullptr,nullptr,0,0,n,g.used,false,true,0,l.gate_bytes,stream);
                clamped_swiglu_float<<<(size_t(n)*g.ff+255)/256,256,0,stream>>>(lane.gate.as<float>(),lane.up.as<float>(),lane.hidden.as<float>(),size_t(n)*g.ff,g.clamp.at(layer));
                prefill_detail::grouped(l.down.info->type,l.down.info->type,blob,nullptr,lane.hidden.as<float>(),lane.down.as<float>(),nullptr,g.ff,g.latent,n,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,0,n,1,false,true,l.gate_bytes+l.up_bytes,0,stream);
            }else{
                gather_routes<<<(size_t(n)*g.latent+255)/256,256,0,stream>>>(latent,lane.input.as<float>(),indices,n,g.latent,g.used);
                for(int at=0;at<n;at+=8)native_quantize_q8_1(lane.input.as<float>()+size_t(at)*g.latent,lane.xq.as<Q81Block>()+size_t(at)*(g.latent/32),g.latent,std::min(8,n-at),stream);
                linear_device(l.gate.info->type,blob,lane.input.as<float>(),lane.xq.ptr,lane.gate.as<float>(),g.latent,g.ff,n,stream);
                linear_device(l.up.info->type,blob+l.gate_bytes,lane.input.as<float>(),lane.xq.ptr,lane.up.as<float>(),g.latent,g.ff,n,stream);
                if(native_mmvq_supported(l.down.info->type))clamped_swiglu_quant<<<(size_t(n)*g.ff+255)/256,256,0,stream>>>(lane.gate.as<float>(),lane.up.as<float>(),lane.hq.as<Q81Block>(),size_t(n)*g.ff,g.clamp.at(layer));
                else clamped_swiglu_float<<<(size_t(n)*g.ff+255)/256,256,0,stream>>>(lane.gate.as<float>(),lane.up.as<float>(),lane.hidden.as<float>(),size_t(n)*g.ff,g.clamp.at(layer));
                linear_device(l.down.info->type,blob+l.gate_bytes+l.up_bytes,lane.hidden.as<float>(),lane.hq.ptr,lane.down.as<float>(),g.ff,g.latent,n,stream);
            }
            lane.weights_used.record(stream);if(entry)entry->last_use=&lane.weights_used;
            scatter_routes<<<(size_t(n)*g.latent+255)/256,256,0,stream>>>(lane.down.as<float>(),primary.routes.as<float>(),indices,n,g.latent);ck(cudaGetLastError());lane.finished.record(stream);offset+=n;
        }
        DeviceGuard guard(primary.device);for(auto& card:cards)for(auto& lane:card->lanes)if(lane->touched)lane->finished.wait(caller);
        weighted_reduce<<<(rows*g.latent+255)/256,256,0,caller>>>(primary.routes.as<float>(),weights,out,g.latent,g.used,rows);ck(cudaGetLastError());primary.final_done.record(caller);
    }

    void run_resident(int layer, Card &primary, Card &owner, const float *latent, const int32_t *ids,
                      const float *weights, float *out, int rows, cudaStream_t caller) {
        const auto &l = layers[layer];
        primary.input_ready.record(caller);
        DeviceGuard guard(owner.device);
        const auto stream = owner.fast.value;
        primary.input_ready.wait(stream);
        const float *x = latent;
        const int32_t *selected = ids;
        const float *scale = weights;
        if (owner.device != primary.device) {
            copy_device(owner.fast_input.ptr, owner.device, latent, primary.device,
                        size_t(rows) * g.latent * 4, stream);
            copy_device(owner.fast_ids.ptr, owner.device, ids, primary.device, size_t(rows) * g.used * 4,
                        stream);
            copy_device(owner.fast_weights.ptr, owner.device, weights, primary.device,
                        size_t(rows) * g.used * 4, stream);
            x = owner.fast_input.as<float>();
            selected = owner.fast_ids.as<int32_t>();
            scale = owner.fast_weights.as<float>();
        }
        const auto *table = owner.tables.as<unsigned long long>() + size_t(layer) * g.experts;
        const int entries = rows * g.used;
        const size_t n = size_t(entries) * g.ff;
        if (l.gate.info->type == 12) {
            native_quantize_q8_1(x, owner.fast_xq.ptr, g.latent, rows, stream);
            launch_routed_type<12>(table, selected, owner.fast_xq.ptr, owner.fast_gate.as<float>(),
                                   owner.fast_up.as<float>(), g.latent, g.ff, g.used, g.experts, entries, 0,
                                   l.gate_bytes, false, stream);
            clamped_swiglu_quant<<<unsigned((n + 255) / 256), 256, 0, stream>>>(
                owner.fast_gate.as<float>(), owner.fast_up.as<float>(), owner.fast_hq.as<Q81Block>(), n,
                g.clamp.at(layer));
            ck(cudaGetLastError());
            if (l.down.info->type == 12)
                launch_routed_type<12>(table, selected, owner.fast_hq.ptr, owner.routes.as<float>(), nullptr,
                                       g.ff, g.latent, g.used, g.experts, entries, l.gate_bytes + l.up_bytes,
                                       0, true, stream);
            else
                launch_routed_type<14>(table, selected, owner.fast_hq.ptr, owner.routes.as<float>(), nullptr,
                                       g.ff, g.latent, g.used, g.experts, entries, l.gate_bytes + l.up_bytes,
                                       0, true, stream);
        } else {
            routed_floating_mmv<<<dim3((g.ff + 3) / 4, entries, 2), 128, 0, stream>>>(
                table, selected, x, owner.fast_gate.as<float>(), owner.fast_up.as<float>(), l.gate.info->type,
                l.up.info->type, g.latent, g.ff, g.used, g.experts, 0, l.gate_bytes, false);
            clamped_swiglu_float<<<unsigned((n + 255) / 256), 256, 0, stream>>>(
                owner.fast_gate.as<float>(), owner.fast_up.as<float>(), owner.fast_hidden.as<float>(), n,
                g.clamp.at(layer));
            routed_floating_mmv<<<dim3((g.latent + 3) / 4, entries), 128, 0, stream>>>(
                table, selected, owner.fast_hidden.as<float>(), owner.routes.as<float>(), nullptr,
                l.down.info->type, l.down.info->type, g.ff, g.latent, g.used, g.experts,
                l.gate_bytes + l.up_bytes, 0, true);
            ck(cudaGetLastError());
        }
        float *destination = owner.device == primary.device ? out : owner.fast_out.as<float>();
        weighted_reduce<<<(rows * g.latent + 255) / 256, 256, 0, stream>>>(
            owner.routes.as<float>(), scale, destination, g.latent, g.used, rows);
        ck(cudaGetLastError());
        if (owner.device != primary.device)
            copy_device(out, primary.device, destination, owner.device, size_t(rows) * g.latent * 4, stream);
        owner.fast_done.record(stream);
        {
            DeviceGuard primary_guard(primary.device);
            owner.fast_done.wait(caller);
            primary.final_done.record(caller);
        }
        counters.routed_entries += entries;
        counters.hits += entries;
        ++counters.resident_calls;
    }

    void run(int layer, int input_device, const float *latent, const int32_t *ids, const float *weights,
             float *out, int rows, cudaStream_t caller) {
        validate_layer(layer);
        if (rows < 1 || rows > max_rows || !latent || !ids || !weights || !out || !caller)
            throw std::runtime_error("Hy4 experts: invalid device input, row count or stream");
        auto at = card_index.find(input_device);
        if (at == card_index.end())
            throw std::runtime_error("Hy4 experts: input device must be a selected expert device");
        Card &primary = *cards[at->second];
        DeviceGuard main_guard(input_device);
        if (previous_output_card >= 0)
            cards[previous_output_card]->final_done.wait(caller);
        previous_output_card = int(at->second);
        ++counters.calls;
        if(rows>8){
            if(auto full=full_layers.find(layer);full!=full_layers.end()&&tc_layer(layers[layer]))run_wide_resident(layer,primary,*cards[full->second],latent,ids,weights,out,rows,caller);
            else {prepare_routes(primary,ids,rows,caller);run_wide_streamed(layer,primary,latent,weights,out,rows,caller);}
            return;
        }
        if (auto full = full_layers.find(layer); full != full_layers.end() && layers[layer].fast) {
            run_resident(layer, primary, *cards[full->second], latent, ids, weights, out, rows, caller);
            return;
        }
        // A native cache scheduler needs only route IDs, not feature vectors. The
        // synchronization also retires all workspace uses from the preceding call.
        ck(cudaMemcpyAsync(primary.host_ids.ptr, ids, size_t(rows) * g.used * sizeof(int32_t),
                           cudaMemcpyDeviceToHost, caller));
        ck(cudaStreamSynchronize(caller));
        ++counters.metadata_readbacks;
        struct Position {
            int row, slot;
        };
        std::map<int, std::vector<Position>> groups;
        const auto *host = primary.host_ids.as<int32_t>();
        for (int row = 0; row < rows; ++row) {
            std::vector<int> seen;
            for (int slot = 0; slot < g.used; ++slot) {
                const int expert = host[row * g.used + slot];
                if (expert < 0 || expert >= g.experts ||
                    std::find(seen.begin(), seen.end(), expert) != seen.end())
                    throw std::runtime_error(
                        "Hy4 experts: routed IDs must be distinct in each token and in range");
                seen.push_back(expert);
                groups[expert].push_back({row, slot});
            }
        }
        counters.routed_entries += uint64_t(rows) * g.used;
        counters.unique_experts += groups.size();
        for (auto &card : cards)
            for (auto &lane : card->lanes)
                lane->touched = false;
        const auto &l = layers[layer];
        for (const auto &[expert, positions] : groups) {
            Card &card = select_card(layer, expert);
            DeviceGuard worker_guard(card.device);
            Lane &lane = *card.lanes[card.next_lane++ % card.lanes.size()];
            lane.touched = true;
            Entry *entry = find(card, key(layer, expert));
            const bool hit=entry!=nullptr;
            if (entry)
                counters.hits += positions.size();
            else {
                counters.misses += positions.size();
                entry = admit(card, key(layer, expert), l.bytes, false);
                upload(card, lane, l, expert, entry ? entry->weights.ptr : lane.transient.ptr, entry);
            }
            if (!hit)
                lane.uploaded.wait(lane.compute.value);
            const auto stream = lane.compute.value;
            for (size_t i = 0; i < positions.size(); ++i)
                copy_device(lane.input.as<float>() + i * g.latent, card.device,
                            latent + size_t(positions[i].row) * g.latent, input_device, size_t(g.latent) * 4,
                            stream);
            const int n = int(positions.size());
            native_quantize_q8_1(lane.input.as<float>(), lane.xq.ptr, g.latent, n, stream);
            const auto *blob = entry ? entry->weights.as<uint8_t>() : lane.transient.as<uint8_t>();
            linear_device(l.gate.info->type, blob, lane.input.as<float>(), lane.xq.ptr, lane.gate.as<float>(),
                          g.latent, g.ff, n, stream);
            linear_device(l.up.info->type, blob + l.gate_bytes, lane.input.as<float>(), lane.xq.ptr,
                          lane.up.as<float>(), g.latent, g.ff, n, stream);
            const size_t elements = size_t(n) * g.ff;
            if (native_mmvq_supported(l.down.info->type))
                clamped_swiglu_quant<<<unsigned((elements + 255) / 256), 256, 0, stream>>>(
                    lane.gate.as<float>(), lane.up.as<float>(), lane.hq.as<Q81Block>(), elements,
                    g.clamp.at(layer));
            else
                clamped_swiglu_float<<<unsigned((elements + 255) / 256), 256, 0, stream>>>(
                    lane.gate.as<float>(), lane.up.as<float>(), lane.hidden.as<float>(), elements,
                    g.clamp.at(layer));
            ck(cudaGetLastError());
            linear_device(l.down.info->type, blob + l.gate_bytes + l.up_bytes, lane.hidden.as<float>(),
                          lane.hq.ptr, lane.down.as<float>(), g.ff, g.latent, n, stream);
            lane.weights_used.record(stream);
            if (entry)entry->last_use=&lane.weights_used;
            for (size_t i = 0; i < positions.size(); ++i) {
                const size_t index = size_t(positions[i].row) * g.used + positions[i].slot;
                copy_device(primary.routes.as<float>() + index * g.latent, input_device,
                            lane.down.as<float>() + i * g.latent, card.device, size_t(g.latent) * 4, stream);
            }
            lane.finished.record(stream);
        }
        ck(cudaSetDevice(input_device));
        for (auto &card : cards)
            for (auto &lane : card->lanes)
                if (lane->touched)
                    lane->finished.wait(caller);
        weighted_reduce<<<(rows * g.latent + 255) / 256, 256, 0, caller>>>(
            primary.routes.as<float>(), weights, out, g.latent, g.used, rows);
        ck(cudaGetLastError());
        primary.final_done.record(caller);
    }

    bool preload(int layer, int device) {
        validate_layer(layer);
        if (auto existing = full_layers.find(layer); existing != full_layers.end())
            return device < 0 || cards[existing->second]->device == device;
        synchronize_all();
        const auto &l = layers[layer];
        const size_t charge=cards.front()->arena_map->allocation_bytes(l.bytes);
        const size_t layer_bytes = charge * g.experts;
        size_t chosen = 0;
        if (device >= 0) {
            auto at = card_index.find(device);
            if (at == card_index.end())
                throw std::runtime_error("Hy4 experts: preload device is not selected");
            chosen = at->second;
        } else
            for (size_t i = 1; i < cards.size(); ++i)
                if (cards[i]->capacity - cards[i]->pinned > cards[chosen]->capacity - cards[chosen]->pinned)
                    chosen = i;
        auto &card = *cards[chosen];
        size_t pinned_here = 0;
        for (int expert = 0; expert < g.experts; ++expert)
            if (auto at = card.cache.find(key(layer, expert)); at != card.cache.end() && at->second->pinned)
                pinned_here += at->second->charge;
        if (layer_bytes > card.capacity - (card.pinned - pinned_here))
            return false;
        DeviceGuard guard(card.device);
        for(auto it=card.lru.begin();it!=card.lru.end();){auto current=it++;if(!card.cache.at(*current)->pinned)evict(card,current);}
        std::vector<unsigned long long> pointers(g.experts);
        for (int expert = 0; expert < g.experts; ++expert) {
            const Key k = key(layer, expert);
            Entry *entry = find(card, k);
            const bool existed = entry != nullptr;
            entry = admit(card, k, l.bytes, true);
            if (!entry)
                throw std::runtime_error("Hy4 experts: preload capacity accounting failed");
            if (!existed) {
                auto &lane = *card.lanes[card.next_lane++ % card.lanes.size()];
                upload(card, lane, l, expert, entry->weights.ptr, entry);
                lane.uploaded.wait(card.fast.value);
            }
            pointers[expert] = reinterpret_cast<unsigned long long>(entry->weights.ptr);
        }
        ck(cudaMemcpyAsync(card.tables.as<unsigned long long>() + size_t(layer) * g.experts, pointers.data(),
                           pointers.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice,
                           card.fast.value));
        ck(cudaStreamSynchronize(card.fast.value));
        full_layers[layer] = chosen;
        return true;
    }
};

ExpertExecutor::ExpertExecutor(const Geometry &g, Lookup lookup, std::vector<ExpertDevice> devices, int rows,
                               int lanes)
    : impl_(std::make_unique<Impl>(g, std::move(lookup), std::move(devices), rows, lanes)) {}
ExpertExecutor::~ExpertExecutor() = default;
void ExpertExecutor::run(int layer, int device, const float *latent, const int32_t *ids, const float *weights,
                         float *out, int rows, void *stream) {
    std::lock_guard lock(impl_->mutex);
    if (impl_->failed)
        throw std::runtime_error("Hy4 experts: previous execution failed; recreate the executor");
    try {
        impl_->run(layer, device, latent, ids, weights, out, rows, static_cast<cudaStream_t>(stream));
    } catch (...) {
        impl_->failed = true;
        throw;
    }
}
bool ExpertExecutor::preload_layer(int layer, int device) {
    std::lock_guard lock(impl_->mutex);
    if (impl_->failed)
        throw std::runtime_error("Hy4 experts: previous execution failed; recreate the executor");
    try {
        return impl_->preload(layer, device);
    } catch (...) {
        impl_->failed = true;
        throw;
    }
}
void ExpertExecutor::synchronize() {
    std::lock_guard lock(impl_->mutex);
    impl_->synchronize_all();
}
size_t ExpertExecutor::reserved_bytes(int device) const {
    std::lock_guard lock(impl_->mutex);
    auto at = impl_->card_index.find(device);
    if (at == impl_->card_index.end())
        throw std::runtime_error("Hy4 experts: unknown device");
    const auto &card = *impl_->cards[at->second];
    return card.fixed + card.arena->bytes;
}
ExpertCacheStats ExpertExecutor::stats() const {
    std::lock_guard lock(impl_->mutex);
    auto result = impl_->counters;
    result.pinned_layers=impl_->full_layers.size();
    for (const auto &card : impl_->cards) {
        result.cache_bytes += card->used;
        result.cache_capacity += card->capacity;
        result.pinned_bytes += card->pinned;
        result.arena_bytes += card->arena->bytes;
        result.observed_driver_bytes += card->observed_driver_bytes;
        for(const auto& [_,entry]:card->cache){result.resident_weight_bytes+=entry->weights.bytes;if(entry->pinned)result.pinned_weight_bytes+=entry->weights.bytes;}
    }
    return result;
}
} // namespace strata::hy4
