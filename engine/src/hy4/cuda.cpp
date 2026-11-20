#include "strata/hy4/cuda.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <stdexcept>

namespace strata::hy4 {
namespace {
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("hy4 CUDA: ") + cudaGetErrorString(e));
}
struct Card {
    int id;
    size_t budget, used = 0, in_bytes = 0, out_bytes = 0, quant_bytes = 0;
    cudaStream_t stream = nullptr;
    void *input = nullptr, *output = nullptr, *quant = nullptr;
    std::vector<void*> weights;
    Card(int device, size_t limit) : id(device), budget(limit) {
        check(cudaSetDevice(id));
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~Card() {
        cudaSetDevice(id);
        if (stream) cudaStreamSynchronize(stream);
        for (void* p : weights) cudaFree(p);
        cudaFree(input); cudaFree(output); cudaFree(quant);
        if (stream) cudaStreamDestroy(stream);
    }
    void reserve(void*& p, size_t& capacity, size_t bytes) {
        if (capacity >= bytes) return;
        check(cudaStreamSynchronize(stream));
        void* replacement = nullptr;
        check(cudaMalloc(&replacement, bytes));
        cudaFree(p); p = replacement; capacity = bytes;
    }
};
class CudaMatmul final : public Matmul {
public:
    CudaMatmul(int threads, const std::vector<int>& devices, const std::vector<size_t>& budgets) : Matmul(threads) {
        if (devices.empty() || devices.size() != budgets.size()) throw std::runtime_error("hy4: provide a cache budget for each CUDA device");
        int count; check(cudaGetDeviceCount(&count));
        std::vector<int> sorted = devices; std::sort(sorted.begin(), sorted.end());
        if (sorted.front() < 0 || sorted.back() >= count || std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            throw std::runtime_error("hy4: invalid or duplicate CUDA device");
        for (size_t i = 0; i < devices.size(); ++i) {
            check(cudaSetDevice(devices[i]));
            size_t free, total; check(cudaMemGetInfo(&free, &total));
            constexpr size_t margin = 256ull << 20;
            if (budgets[i] > free || free - budgets[i] < margin) throw std::runtime_error("hy4: CUDA weight budget leaves less than 256 MiB for scratch");
            cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, devices[i]));
            if (prop.major < 7 || (prop.major == 7 && prop.minor < 5)) throw std::runtime_error("hy4 CUDA requires compute capability 7.5 or newer");
            cards_.push_back(std::make_unique<Card>(devices[i], budgets[i]));
        }
        std::cerr << "hy4: native CUDA dense offload; cold experts, attention state and Engram on CPU\n";
    }
    Vec linear(const Tensor& w, std::span<const float> x, int plane) override {
        // Keep expert execution and per-head attention matrices on the CPU in the first native port.
        // A resident dense matrix never gets evicted by a newly routed expert.
        const auto& shape = w.info->shape;
        if (shape.size() != 2 || plane != 0 || !kernels::native_mmvq_supported(w.info->type) || shape[0] != x.size())
            return Matmul::linear(w, x, plane);
        auto found = resident_.find(w.data);
        if (found == resident_.end()) {
            if (declined_.contains(w.data)) return Matmul::linear(w, x, plane);
            size_t card_index = 0;
            // Stable layer split, not a Qwen compile-time geometry.
            if (w.info->name.starts_with("blk.")) card_index = std::stoul(w.info->name.substr(4)) % cards_.size();
            const size_t bytes = w.row_bytes() * shape[1];
            auto& c = *cards_[card_index];
            if (bytes > c.budget - c.used) { declined_[w.data] = true; return Matmul::linear(w, x, plane); }
            check(cudaSetDevice(c.id));
            void* weights = nullptr;
            check(cudaMalloc(&weights, bytes));
            try { check(cudaMemcpyAsync(weights, w.data, bytes, cudaMemcpyHostToDevice, c.stream)); check(cudaStreamSynchronize(c.stream)); }
            catch (...) { cudaFree(weights); throw; }
            c.weights.push_back(weights); c.used += bytes;
            found = resident_.emplace(w.data, Entry{card_index, weights}).first;
        }
        auto& c = *cards_[found->second.card];
        check(cudaSetDevice(c.id));
        c.reserve(c.input, c.in_bytes, x.size_bytes());
        c.reserve(c.quant, c.quant_bytes, kernels::native_q8_1_bytes(shape[0]));
        c.reserve(c.output, c.out_bytes, shape[1] * sizeof(float));
        Vec result(shape[1]);
        check(cudaMemcpyAsync(c.input, x.data(), x.size_bytes(), cudaMemcpyHostToDevice, c.stream));
        kernels::native_quantize_q8_1(static_cast<float*>(c.input), c.quant, shape[0], 1, c.stream);
        kernels::native_mmvq(w.info->type, found->second.weights, c.quant, static_cast<float*>(c.output), shape[0], shape[1], 1, c.stream);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(result.data(), c.output, result.size() * sizeof(float), cudaMemcpyDeviceToHost, c.stream));
        check(cudaStreamSynchronize(c.stream));
        return result;
    }
private:
    struct Entry { size_t card; void* weights; };
    std::vector<std::unique_ptr<Card>> cards_;
    std::map<const void*, Entry> resident_;
    std::map<const void*, bool> declined_;
};
}
std::unique_ptr<Matmul> cuda_matmul(int threads, const std::vector<int>& devices, const std::vector<size_t>& budgets) {
    return std::make_unique<CudaMatmul>(threads, devices, budgets);
}
}
