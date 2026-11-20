#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace strata::hy4::device_kernels {
constexpr int threads = 256;
template <class T> __device__ T sum(T v) {
    __shared__ T tmp[threads];
    tmp[threadIdx.x] = v;
    __syncthreads();
    for (int s = threads / 2; s; s >>= 1) {
        if (threadIdx.x < s)
            tmp[threadIdx.x] += tmp[threadIdx.x + s];
        __syncthreads();
    }
    const T result = tmp[0];
    __syncthreads();
    return result;
}
__device__ float maximum(float v) {
    __shared__ float tmp[threads];
    tmp[threadIdx.x] = v;
    __syncthreads();
    for (int s = threads / 2; s; s >>= 1) {
        if (threadIdx.x < s)
            tmp[threadIdx.x] = fmaxf(tmp[threadIdx.x], tmp[threadIdx.x + s]);
        __syncthreads();
    }
    const float result = tmp[0];
    __syncthreads();
    return result;
}
__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__global__ void rms(const float *x, const float *weight, float *y, int n, float eps, int weight_rows = 1) {
    const int row = blockIdx.x;
    double sq = 0;
    for (int i = threadIdx.x; i < n; i += threads)
        sq += double(x[size_t(row) * n + i]) * x[size_t(row) * n + i];
    const float r = 1.0f / sqrtf(float(sum(sq) / n) + eps);
    for (int i = threadIdx.x; i < n; i += threads)
        y[size_t(row) * n + i] =
            (x[size_t(row) * n + i] * r) * (weight ? weight[size_t(row % weight_rows) * n + i] : 1.0f);
}
__global__ void hc_mix(const float *x, const float *mixes, const float *scale, const float *base, float *out,
                       float *post, int dim, int hc, float eps, float magnitude) {
    const int row = blockIdx.x;
    for (int h = threadIdx.x; h < hc; h += threads)
        post[row * hc + h] =
            sigmoid(mixes[row * 2 * hc + hc + h] * scale[1] + base[hc + h]) * magnitude + eps;
    for (int i = threadIdx.x; i < dim; i += threads) {
        float v = 0;
        for (int h = 0; h < hc; ++h)
            v += x[(size_t(row) * hc + h) * dim + i] *
                 (sigmoid(mixes[row * 2 * hc + h] * scale[0] + base[h]) + eps);
        out[size_t(row) * dim + i] = v;
    }
}
__global__ void hc_post(float *x, const float *value, const float *post, int dim, int hc) {
    const int row = blockIdx.x;
    for (int i = threadIdx.x; i < dim * hc; i += threads)
        x[size_t(row) * dim * hc + i] += value[size_t(row) * dim + i % dim] * post[row * hc + i / dim];
}
__global__ void hc_head(const float *x, const float *mixes, const float *scale, const float *base, float *out,
                        int dim, int hc, float eps) {
    const int row = blockIdx.x;
    for (int i = threadIdx.x; i < dim; i += threads) {
        float v = 0;
        for (int h = 0; h < hc; ++h)
            v += x[(size_t(row) * hc + h) * dim + i] *
                 (sigmoid(mixes[row * hc + h] * scale[0] + base[h]) + eps);
        out[size_t(row) * dim + i] = v;
    }
}
__global__ void add_bias(float *x, const float *b, int width, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * count)
        x[i] += b[i % width];
}
__global__ void add(float *x, const float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] += y[i];
}
__global__ void weights_float(const void *in, float *out, size_t n, int type) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = type == 1 ? __half2float(static_cast<const half *>(in)[i])
                           : __uint_as_float(uint32_t(static_cast<const uint16_t *>(in)[i]) << 16);
}
__global__ void dense_16(const uint16_t *weight, const float *x, float *out, int ni, int no, int rows,
                         int type) {
    int lane = threadIdx.x % 32, row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= no)
        return;
    float acc[8] = {};
    for (int i = lane; i < ni; i += 32) {
        uint16_t bits = weight[size_t(row) * ni + i];
        float w = type == 1 ? __half2float(__ushort_as_half(bits)) : __uint_as_float(uint32_t(bits) << 16);
        for (int b = 0; b < rows; ++b)
            acc[b] += w * x[size_t(b) * ni + i];
    }
    for (int b = 0; b < rows; ++b) {
        for (int d = 16; d; d >>= 1)
            acc[b] += __shfl_down_sync(0xffffffff, acc[b], d);
        if (lane == 0)
            out[size_t(b) * no + row] = acc[b];
    }
}
__global__ void glu(float *gate, const float *up, int n, float clamp = 0) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = clamp > 0 ? fminf(gate[i], clamp) : gate[i];
        float u = clamp > 0 ? fminf(fmaxf(up[i], -clamp), clamp) : up[i];
        gate[i] = g * sigmoid(g) * u;
    }
}
__global__ void gate(float *x, const float *g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] *= sigmoid(g[i]);
}
__global__ void route(const float *logits, const float *bias, int32_t *ids, float *weights, int experts,
                      int used, float scale) {
    const int row = blockIdx.x;
    __shared__ float scores[4096], probs[4096];
    for (int i = threadIdx.x; i < experts; i += threads) {
        float p = sigmoid(logits[size_t(row) * experts + i]);
        probs[i] = p;
        scores[i] = p + (bias ? bias[i] : 0.f);
    }
    __syncthreads();
    // Selection is deterministic: bias affects ranking only, not normalized probability.
    if (threadIdx.x == 0) {
        float total = 0;
        for (int k = 0; k < used; ++k) {
            int best = 0;
            for (int i = 1; i < experts; ++i)
                if (scores[i] > scores[best])
                    best = i;
            ids[row * used + k] = best;
            weights[row * used + k] = probs[best];
            total += probs[best];
            scores[best] = -INFINITY;
        }
        total = fmaxf(total, 0x1p-14f);
        for (int k = 0; k < used; ++k)
            weights[row * used + k] = weights[row * used + k] / total * scale;
    }
}
__device__ int axis(int pair) { return pair % 3 == 1 && pair < 30 ? 1 : pair % 3 == 2 && pair < 30 ? 2 : 0; }
__global__ void rope(float *x, const int32_t *positions, int rows, int heads, int width, int rot,
                     float base) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs = rot / 2;
    if (i >= rows * heads * pairs)
        return;
    int p = i % pairs, head = (i / pairs) % heads, row = i / (pairs * heads);
    float angle = positions[row * 3 + axis(p)] * powf(base, -2.f * p / rot);
    float c = cosf(angle), s = sinf(angle);
    size_t off = (size_t(row) * heads + head) * width + width - rot + 2 * p;
    float a = x[off], b = x[off + 1];
    x[off] = a * c - b * s;
    x[off + 1] = a * s + b * c;
}
__global__ void heads_pack(const float *x, float *y, int rows, int heads, int in_width, int width,
                           int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * heads * width)
        return;
    int j = i % width, h = (i / width) % heads, r = i / (width * heads);
    y[(size_t(h) * rows + r) * width + j] = x[(size_t(r) * heads + h) * in_width + offset + j];
}
__global__ void heads_unpack(const float *x, float *y, int rows, int heads, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * heads * width)
        return;
    int j = i % width, h = (i / width) % heads, r = i / (width * heads);
    y[i] = x[(size_t(h) * rows + r) * width + j];
}
__device__ int load4(const unsigned char *p) {
    return int(p[0]) | (int(p[1]) << 8) | (int(p[2]) << 16) | (int(p[3]) << 24);
}
__global__ void planes_q8(const unsigned char *w, const unsigned char *q, float *out, int nin, int nout,
                          int rows, int heads) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, r = blockIdx.x * 4 + warp, h = blockIdx.y,
        b = blockIdx.z;
    if (r >= nout)
        return;
    int nb = nin / 32;
    const unsigned char *wr = w + (size_t(h) * nout + r) * nb * 34;
    const unsigned char *qr = q + (size_t(h) * rows + b) * nb * 36;
    float acc = 0;
    for (int j = lane / 8; j < nb; j += 4) {
        auto *wb = wr + j * 34;
        auto *xb = qr + j * 36;
        int d = __dp4a(load4(wb + 2 + (lane % 8) * 4), load4(xb + 4 + (lane % 8) * 4), 0);
        acc += float(d) * __half2float(*reinterpret_cast<const half *>(wb)) *
               __half2float(*reinterpret_cast<const half *>(xb));
    }
    for (int s = 16; s; s >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, s);
    if (lane == 0)
        out[(size_t(b) * heads + h) * nout + r] = acc;
}
__global__ void planes_float(const void *w, const float *x, float *out, int nin, int nout, int rows,
                             int heads, int type) {
    int r = blockIdx.x, b = blockIdx.y, h = blockIdx.z;
    double value = 0;
    size_t off = (size_t(h) * nout + r) * nin;
    for (int i = threadIdx.x; i < nin; i += threads) {
        float a;
        if (type == 0)
            a = static_cast<const float *>(w)[off + i];
        else if (type == 1)
            a = __half2float(static_cast<const half *>(w)[off + i]);
        else {
            uint32_t bits = uint32_t(static_cast<const uint16_t *>(w)[off + i]) << 16;
            a = __uint_as_float(bits);
        }
        value += double(a) * x[(size_t(h) * rows + b) * nin + i];
    }
    value = sum(value);
    if (threadIdx.x == 0)
        out[(size_t(b) * heads + h) * nout + r] = float(value);
}
__global__ void kv_normalize(float *kv, const float *norm, int stride, int latent, float eps) {
    int row = blockIdx.x;
    double v = 0;
    for (int i = threadIdx.x; i < latent; i += threads)
        v += double(kv[size_t(row) * stride + i]) * kv[size_t(row) * stride + i];
    float r = 1.f / sqrtf(float(sum(v) / latent) + eps);
    for (int i = threadIdx.x; i < latent; i += threads)
        kv[size_t(row) * stride + i] = kv[size_t(row) * stride + i] * r * norm[i];
}
template <class T> __device__ float read(const T *p) { return float(*p); }
template <> __device__ float read<half>(const half *p) { return __half2float(*p); }
template <class T> __device__ T store(float x) { return T(x); }
template <> __device__ half store<half>(float x) { return __float2half_rn(x); }
template <class T> __global__ void kv_write(T *dst, const float *kv, const int32_t *cells, int width) {
    int row = blockIdx.x;
    for (int i = threadIdx.x; i < width; i += threads)
        dst[size_t(cells[row]) * width + i] = store<T>(kv[size_t(row) * width + i]);
}
__global__ void index_ring(const float *values, const float *gates, float *ring_values, float *ring_gates,
                           const int32_t *slots, const int32_t *pos, int width, int ring) {
    int row = blockIdx.x;
    size_t cell = size_t(slots[row]) * ring + pos[row] % ring;
    for (int i = threadIdx.x; i < width; i += threads) {
        ring_values[cell * width + i] = values[size_t(row) * width + i];
        ring_gates[cell * width + i] = gates[size_t(row) * width + i];
    }
}
__global__ void pool(const float *values, const float *gates, const float *ape, float *out,
                     const int32_t *slots, const int32_t *starts, int width, int count, int ring) {
    int row = blockIdx.x;
    for (int i = threadIdx.x; i < width; i += threads) {
        float mx = -INFINITY;
        for (int t = 0; t < count; ++t) {
            size_t cell = size_t(slots[row]) * ring + (starts[row] + t) % ring;
            mx = fmaxf(mx, gates[cell * width + i] + ape[t * width + i]);
        }
        float den = 0, num = 0;
        for (int t = 0; t < count; ++t) {
            size_t cell = size_t(slots[row]) * ring + (starts[row] + t) % ring;
            float a = expf(gates[cell * width + i] + ape[t * width + i] - mx);
            den += a;
            num += a * values[cell * width + i];
        }
        out[size_t(row) * width + i] = num / den;
    }
}
__global__ void pool_store(float *cache, const float *value, const int32_t *physical, int width) {
    int row = blockIdx.x;
    for (int i = threadIdx.x; i < width; i += threads)
        cache[size_t(physical[row]) * width + i] = value[size_t(row) * width + i];
}
// Eight independent groups per block. A warp shares one key across all indexer
// heads; FP32 warp reductions avoid the reference implementation's repeated
// full-block FP64 reductions on consumer/Ampere GPUs.
__global__ void index_scores(const float *pooled, const float *query, const float *weight, const int32_t *map,
                             float *scores, int groups, int width, int heads, int pool, float scale) {
    const int lane = threadIdx.x % 32;
    const int group = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (group >= groups)
        return;
    const float *key = pooled + size_t(map[group * pool] / pool) * width;
    float score = 0;
    for (int h = 0; h < heads; ++h) {
        float dot = 0;
        for (int i = lane; i < width; i += 32)
            dot += query[h * width + i] * key[i];
        for (int offset = 16; offset; offset >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        if (lane == 0)
            score += fmaxf(dot, 0.f) * (weight[h] * scale);
    }
    if (lane == 0)
        scores[group] = score;
}
__global__ void group_ids(int32_t *ids, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        ids[i] = i;
}
__global__ void selected_sort(const int32_t *input, int32_t *out, int count) {
    __shared__ int32_t ids[512];
    for (int i = threadIdx.x; i < 512; i += threads)
        ids[i] = i < count ? input[i] : INT32_MAX;
    __syncthreads();
    for (int k = 2; k <= 512; k *= 2)
        for (int j = k / 2; j; j /= 2) {
            for (int i = threadIdx.x; i < 512; i += threads) {
                int other = i ^ j;
                if (other > i && ((ids[i] > ids[other]) == ((i & k) == 0))) {
                    int t = ids[i];
                    ids[i] = ids[other];
                    ids[other] = t;
                }
            }
            __syncthreads();
        }
    for (int i = threadIdx.x; i < count; i += threads)
        out[i] = ids[i];
}
__global__ void map_write(int32_t *map, const int32_t *slots, const int32_t *pos, const int32_t *cells,
                          int rows, int capacity) {
    int i = threadIdx.x;
    if (i < rows)
        map[size_t(slots[i]) * capacity + pos[i]] = cells[i];
}
__global__ void feature_store(const float *hidden, float *ring, const int32_t *slots, const int32_t *pos,
                              int dim, int capacity) {
    int row = blockIdx.x;
    for (int d = threadIdx.x; d < dim; d += threads)
        ring[(size_t(slots[row]) * capacity + pos[row] % capacity) * dim + d] = hidden[size_t(row) * dim + d];
}
__global__ void feature_gather(const float *ring, float *hidden, const int32_t *slots, const int32_t *pos,
                               int dim, int capacity) {
    int row = blockIdx.x;
    for (int d = threadIdx.x; d < dim; d += threads)
        hidden[size_t(row) * dim + d] =
            pos[row] > 0 ? ring[(size_t(slots[row]) * capacity + (pos[row] - 1) % capacity) * dim + d] : 0;
}
__global__ void carry_gather(const float *carry, float *hidden, const int32_t *slots, int dim) {
    int row = blockIdx.x;
    for (int d = threadIdx.x; d < dim; d += threads)
        hidden[size_t(row) * dim + d] = carry[size_t(slots[row]) * dim + d];
}
__global__ void carry_store(const float *hidden, float *carry, const int32_t *slots, int dim, int rows) {
    int row = blockIdx.x;
    for (int later = row + 1; later < rows; ++later)
        if (slots[later] == slots[row])
            return;
    for (int d = threadIdx.x; d < dim; d += threads)
        carry[size_t(slots[row]) * dim + d] = hidden[size_t(row) * dim + d];
}
__global__ void concat(const float *a, const float *b, float *out, int dim) {
    int row = blockIdx.x;
    for (int i = threadIdx.x; i < dim; i += threads) {
        out[size_t(row) * dim * 2 + i] = a[size_t(row) * dim + i];
        out[size_t(row) * dim * 2 + dim + i] = b[size_t(row) * dim + i];
    }
}
__global__ void selection(const int32_t *groups, int32_t *selected, int count, int pos, int pool) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int history = count * pool, tail = pos % pool + 1;
    if (i < history)
        selected[i] = groups[i / pool] * pool + i % pool;
    else if (i < history + tail)
        selected[i] = pos - pos % pool + i - history;
}
template <class T>
__global__ void attn_split(const float *q_abs, const float *query, const T *cache, const int32_t *map,
                           const int32_t *selected, const float *sinks, float *partial, int row, int pos,
                           int heads, int latent, int head_width, int rope_width, int value_width,
                           int selected_count, int chunks, int ring) {
    int h = blockIdx.x, chunk = blockIdx.y, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    constexpr int tile = 128;
    __shared__ float score[tile];
    int start = chunk * tile, count = min(tile, selected_count - start);
    int stride = latent + rope_width;
    for (int j = warp; j < count; j += threads / 32) {
        int logical = selected ? selected[start + j] : pos - selected_count + 1 + start + j;
        int physical = ring ? logical % ring : map[logical];
        const T *key = cache + size_t(physical) * stride;
        float dot = 0;
        for (int i = lane; i < stride; i += 32) {
            float v =
                i < latent
                    ? q_abs[(size_t(row) * heads + h) * latent + i]
                    : query[(size_t(row) * heads + h) * head_width + head_width - rope_width + i - latent];
            dot += v * read(key + i);
        }
        for (int s = 16; s; s >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, s);
        if (lane == 0)
            score[j] = dot * rsqrtf(float(head_width));
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < count; j += threads)
        mx = fmaxf(mx, score[j]);
    if (threadIdx.x == 0 && chunk == 0)
        mx = fmaxf(mx, sinks[h]);
    mx = maximum(mx);
    float den = 0;
    for (int j = threadIdx.x; j < count; j += threads) {
        score[j] = expf(score[j] - mx);
        den += score[j];
    }
    if (threadIdx.x == 0 && chunk == 0)
        den += expf(sinks[h] - mx);
    den = sum(den);
    __syncthreads();
    size_t dest = (size_t(h) * chunks + chunk) * (latent + 2);
    if (threadIdx.x == 0) {
        partial[dest] = mx;
        partial[dest + 1] = den;
    }
    for (int d = threadIdx.x; d < latent; d += threads) {
        float v = 0;
        for (int j = 0; j < count; ++j) {
            int logical = selected ? selected[start + j] : pos - selected_count + 1 + start + j;
            int physical = ring ? logical % ring : map[logical];
            v += score[j] * read(cache + size_t(physical) * stride + d);
        }
        partial[dest + 2 + d] = v;
    }
}
__global__ void attn_merge(const float *partial, float *out, int rows, int heads, int latent, int chunks,
                           int row) {
    int h = blockIdx.x;
    float mx = -INFINITY;
    for (int c = 0; c < chunks; ++c)
        mx = fmaxf(mx, partial[(size_t(h) * chunks + c) * (latent + 2)]);
    float den = 0;
    for (int c = 0; c < chunks; ++c) {
        size_t o = (size_t(h) * chunks + c) * (latent + 2);
        den += expf(partial[o] - mx) * partial[o + 1];
    }
    for (int d = threadIdx.x; d < latent; d += threads) {
        float v = 0;
        for (int c = 0; c < chunks; ++c) {
            size_t o = (size_t(h) * chunks + c) * (latent + 2);
            v += expf(partial[o] - mx) * partial[o + 2 + d];
        }
        out[(size_t(row) * heads + h) * latent + d] = v / den;
    }
}
__global__ void engram_delta(float *x, const float *normalized_x, const float *keys, const float *signal,
                             const float *gate_weight, const float *scales, const int32_t *valid, int dim,
                             int hc) {
    int row = blockIdx.x, h = blockIdx.y;
    double d = 0;
    for (int i = threadIdx.x; i < dim; i += threads)
        d += double(normalized_x[(size_t(row) * hc + h) * dim + i]) * keys[(size_t(row) * hc + h) * dim + i] *
             gate_weight[h * dim + i];
    float dot = float(sum(d)) / sqrtf(float(dim));
    float g = sigmoid(copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot));
    for (int i = threadIdx.x; i < dim; i += threads)
        if (valid[row])
            x[(size_t(row) * hc + h) * dim + i] += signal[size_t(row) * dim + i] * (g + scales[h * dim + i]);
}
__global__ void unpack_engram(const float *kv, float *keys, float *value, int dim, int hc, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * (hc + 1) * dim)
        return;
    int row = i / ((hc + 1) * dim), o = i % ((hc + 1) * dim);
    if (o < hc * dim)
        keys[size_t(row) * hc * dim + o] = kv[i];
    else
        value[size_t(row) * dim + o - hc * dim] = kv[i];
}
} // namespace strata::hy4::device_kernels
