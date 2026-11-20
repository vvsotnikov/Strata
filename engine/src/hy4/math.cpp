#include "strata/hy4/engine.hpp"
#include "strata/quant/act.hpp"
#include "strata/quant/dequant.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/fp16.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace strata::hy4 {

size_t Tensor::row_bytes() const {
    if (info->type == 43) return info->shape.at(0);
    const size_t result = quant::row_bytes(static_cast<int>(info->type), info->shape.at(0));
    if (!result) throw std::runtime_error("hy4: invalid row geometry for " + info->name);
    return result;
}

Vec Tensor::values(int64_t row) const {
    const size_t n = row < 0 ? info->elements() : info->shape[0];
    if (row >= 0 && static_cast<uint64_t>(row) >= info->elements() / info->shape[0])
        throw std::runtime_error("hy4: row out of range for " + info->name);
    const uint8_t* src = data + (row < 0 ? 0 : row_bytes() * row);
    Vec out(n);
    if (info->type == 0) std::memcpy(out.data(), src, n * sizeof(float));
    else if (info->type == 1 || info->type == 30) {
        for (size_t i = 0; i < n; ++i) {
            uint16_t value;
            std::memcpy(&value, src + i * 2, 2);
            if (info->type == 1) out[i] = quant::fp16_to_fp32(value);
            else { uint32_t bits = uint32_t(value) << 16; std::memcpy(&out[i], &bits, 4); }
        }
    } else if (info->type == 43) {
        for (size_t i = 0; i < n; ++i) {
            const unsigned b = src[i], e = (b >> 3) & 15, m = b & 7;
            const float value = e == 15 && m == 7 ? std::numeric_limits<float>::quiet_NaN()
                : std::ldexp(e ? 1.0f + float(m) / 8.0f : float(m), e ? int(e) - 7 : -9);
            out[i] = std::copysign(value, b & 128 ? -1.0f : 1.0f);
        }
    } else if (!quant::dequant_row(static_cast<quant::Type>(info->type), src, out.data(), n)) {
        throw std::runtime_error("hy4: unsupported tensor type for " + info->name);
    }
    return out;
}

Matmul::Matmul(int threads) : threads_(threads) {
    if (threads < 1 || threads > 256) throw std::runtime_error("hy4: threads must be 1..256");
}

Vec Matmul::linear(const Tensor& w, std::span<const float> x, int plane) {
    const auto& shape = w.info->shape;
    if (shape.size() < 2 || shape.size() > 3 || x.size() != shape[0] || plane < 0 ||
        static_cast<uint64_t>(plane) >= (shape.size() == 3 ? shape[2] : 1))
        throw std::runtime_error("hy4: matvec shape mismatch for " + w.info->name);
    const int n = static_cast<int>(shape[0]), rows = static_cast<int>(shape[1]);
    const size_t stride = w.row_bytes();
    const uint8_t* data = w.data + static_cast<size_t>(plane) * rows * stride;
    const auto type = static_cast<quant::Type>(w.info->type);
    const auto* dot = quant::dot_kernel(type, quant::default_arith());
    Vec out(rows);
    if (dot) {
        const auto* act = quant::act_kernel(dot->act, dot->arith);
        if (!act) throw std::runtime_error("hy4: activation quantizer unavailable");
        std::vector<uint8_t> input(quant::row_bytes(static_cast<int>(dot->act), n));
        act->fn(x.data(), input.data(), n);
#pragma omp parallel for schedule(static) num_threads(threads_) if(rows >= 128)
        for (int r = 0; r < rows; ++r) dot->fn(n, &out[r], data + r * stride, input.data());
    } else if (w.info->type == 0 || w.info->type == 1 || w.info->type == 30) {
#pragma omp parallel for schedule(static) num_threads(threads_) if(rows >= 128)
        for (int r = 0; r < rows; ++r) {
            float sum = 0;
            const uint8_t* p = data + r * stride;
            for (int j = 0; j < n; ++j) {
                float value;
                if (w.info->type == 0) std::memcpy(&value, p + j * 4, 4);
                else {
                    uint16_t h; std::memcpy(&h, p + j * 2, 2);
                    if (w.info->type == 1) value = quant::fp16_to_fp32(h);
                    else { uint32_t bits = uint32_t(h) << 16; std::memcpy(&value, &bits, 4); }
                }
                sum += value * x[j];
            }
            out[r] = sum;
        }
    } else throw std::runtime_error("hy4: no native dot for " + w.info->name);
    return out;
}

void rms(std::span<float> x, float epsilon, std::span<const float> weight) {
    if (x.empty() || (!weight.empty() && weight.size() != x.size()))
        throw std::runtime_error("hy4: invalid RMSNorm shape");
    double squares = 0;
    for (float v : x) squares += static_cast<double>(v) * v;
    if (!std::isfinite(squares)) throw std::runtime_error("hy4: non-finite activation");
    const float scale = 1.0f / std::sqrt(static_cast<float>(squares / x.size()) + epsilon);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (x[i] * scale) * (weight.empty() ? 1.0f : weight[i]);
}

void rope(std::span<float> x, int64_t position, float base) {
    rope(x, RopePosition{position, position, position}, base);
}

void rope(std::span<float> x, const RopePosition& position, float base) {
    if (x.size() % 2 || base <= 0 || !std::isfinite(base) ||
        std::any_of(position.begin(), position.end(), [](int64_t p) { return p < 0; }))
        throw std::runtime_error("hy4: invalid RoPE inputs");
    const size_t half = x.size() / 2;
    for (size_t i = 0; i < half; ++i) {
        const size_t axis = i < 30 && i % 3 == 1 ? 1 : i < 30 && i % 3 == 2 ? 2 : 0;
        const float angle = static_cast<float>(position[axis]) * std::pow(base, -2.0f * float(i) / float(x.size()));
        const float c = std::cos(angle), s = std::sin(angle), a = x[2 * i], b = x[2 * i + 1];
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

std::vector<int> route(std::span<const float> probabilities, std::span<const float> bias, int used) {
    if (used < 1 || static_cast<size_t>(used) > probabilities.size() || (!bias.empty() && bias.size() != probabilities.size()))
        throw std::runtime_error("hy4: invalid expert routing shape");
    if (std::any_of(probabilities.begin(), probabilities.end(), [](float x) { return !std::isfinite(x); }) ||
        std::any_of(bias.begin(), bias.end(), [](float x) { return !std::isfinite(x); }))
        throw std::runtime_error("hy4: non-finite expert routing");
    std::vector<int> ids(probabilities.size());
    std::iota(ids.begin(), ids.end(), 0);
    const auto score = [&](int i) { return probabilities[i] + (bias.empty() ? 0.0f : bias[i]); };
    std::partial_sort(ids.begin(), ids.begin() + used, ids.end(), [&](int a, int b) {
        return score(a) > score(b) || (score(a) == score(b) && a < b);
    });
    ids.resize(used);
    return ids;
}

std::vector<int32_t> engram_hash(const Geometry& g, int module, std::span<const int32_t> history, int32_t token) {
    if (module < 0 || static_cast<size_t>(module) >= g.engram_layers.size())
        throw std::runtime_error("hy4: invalid Engram module");
    const auto mapped = [&](int32_t id) -> uint64_t {
        const int64_t v = id >= 0 && static_cast<size_t>(id) < g.token_map.size() ? g.token_map[id] : -1;
        return static_cast<uint64_t>(v < 0 ? g.token_map.at(g.pad) : v);
    };
    const size_t columns = static_cast<size_t>(g.ngram - 1) * g.hash_heads;
    uint64_t hash = mapped(token) * g.multipliers.at(static_cast<size_t>(module) * g.ngram);
    std::vector<int32_t> ids;
    for (int order = 1; order < g.ngram; ++order) {
        const int32_t previous = history.size() >= static_cast<size_t>(order) ? history[history.size() - order] : g.pad;
        hash ^= mapped(previous) * g.multipliers.at(static_cast<size_t>(module) * g.ngram + order);
        for (int head = 0; head < g.hash_heads; ++head) {
            const size_t col = static_cast<size_t>(module) * columns + (order - 1) * g.hash_heads + head;
            ids.push_back(static_cast<int32_t>(hash % g.primes.at(col) + g.offsets.at(col)));
        }
    }
    return ids;
}

Vec pool_key(std::span<const float> values, std::span<const float> gates,
             std::span<const float> ape, int width, int count) {
    if (width <= 0 || count <= 0 || values.size() != static_cast<size_t>(width * count) ||
        gates.size() != values.size() || ape.size() != values.size())
        throw std::runtime_error("hy4: invalid compressed indexer pool");
    Vec result(width);
    for (int i = 0; i < width; ++i) {
        float max = -std::numeric_limits<float>::infinity();
        for (int t = 0; t < count; ++t) max = std::max(max, gates[t * width + i] + ape[t * width + i]);
        float den = 0, num = 0;
        for (int t = 0; t < count; ++t) {
            const float w = std::exp(gates[t * width + i] + ape[t * width + i] - max);
            den += w; num += w * values[t * width + i];
        }
        result[i] = num / den;
    }
    return result;
}

}  // namespace strata::hy4
