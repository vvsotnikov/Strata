#include "device_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t code) {
    if (code != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(code));
}
struct Allocation {
    void *data = nullptr;
    explicit Allocation(size_t bytes) { check(cudaMalloc(&data, bytes)); }
    ~Allocation() { cudaFree(data); }
    template <class T> T *as() { return static_cast<T *>(data); }
};
} // namespace

int main() {
    try {
        // Actual Hy4 geometry, ragged final block, and nonidentity physical pages.
        constexpr int groups = 262143, width = 128, heads = 16, pool = 4;
        const float scale = 1.f / std::sqrt(float(width * heads));
        std::mt19937 random(17);
        std::uniform_real_distribution<float> distribution(-1, 1);
        std::vector<float> keys(size_t(groups) * width), query(heads * width), weights(heads), actual(groups);
        std::vector<int32_t> map(size_t(groups) * pool);
        for (auto &value : keys)
            value = distribution(random);
        for (auto &value : query)
            value = distribution(random);
        for (auto &value : weights)
            value = distribution(random);
        for (int group = 0; group < groups; ++group)
            for (int cell = 0; cell < pool; ++cell)
                map[size_t(group) * pool + cell] = int((int64_t(group) * 7919) % groups) * pool + cell;
        Allocation device_keys(keys.size() * 4), device_query(query.size() * 4),
            device_weights(weights.size() * 4), device_map(map.size() * 4), device_output(actual.size() * 4);
        check(cudaMemcpy(device_keys.data, keys.data(), keys.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(device_query.data, query.data(), query.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(device_weights.data, weights.data(), weights.size() * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(device_map.data, map.data(), map.size() * 4, cudaMemcpyHostToDevice));
        strata::hy4::device_kernels::index_scores<<<(groups + 7) / 8, 256>>>(
            device_keys.as<float>(), device_query.as<float>(), device_weights.as<float>(),
            device_map.as<int32_t>(), device_output.as<float>(), groups, width, heads, pool, scale);
        check(cudaGetLastError());
        check(cudaMemcpy(actual.data(), device_output.data, actual.size() * 4, cudaMemcpyDeviceToHost));
        float maximum = 0;
        // Sample across the entire logical range, including the ragged last warp.
        for (int sample = 0; sample < 258; ++sample) {
            const int group = sample == 257 ? groups - 1 : sample * 1021;
            const float *key = keys.data() + size_t(map[group * pool] / pool) * width;
            float expected = 0;
            for (int head = 0; head < heads; ++head) {
                double dot = 0;
                for (int i = 0; i < width; ++i)
                    dot += double(query[head * width + i]) * key[i];
                expected += std::max(float(dot), 0.f) * (weights[head] * scale);
            }
            maximum = std::max(maximum, std::abs(actual[group] - expected));
        }
        for (float value : actual)
            if (!std::isfinite(value))
                throw std::runtime_error("nonfinite indexer score");
        if (maximum > 1e-5f)
            throw std::runtime_error("GPU indexer differs from scalar reference");
        std::cout << "Hy4 1M-indexer geometry max_abs=" << maximum << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
