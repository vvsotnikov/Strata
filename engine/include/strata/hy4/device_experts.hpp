#pragma once

#include "strata/hy4/engine.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace strata::hy4 {

struct ExpertDevice {
    int device = 0;            // Ordinal within CUDA_VISIBLE_DEVICES.
    size_t resident_bytes = 0; // Cache capacity, in addition to fixed workspace below.
};

struct ExpertMemory {
    size_t device_workspace = 0; // Per selected device, includes transient weight slots.
    size_t host_staging = 0;     // Pinned host bytes per selected device, not VRAM.
};

struct ExpertCacheStats {
    uint64_t calls = 0, routed_entries = 0, unique_experts = 0;
    uint64_t hits = 0, misses = 0; // Routed entries; hits + misses == routed_entries.
    uint64_t evictions = 0, uploads = 0, uploaded_bytes = 0;
    uint64_t resident_calls = 0, metadata_readbacks = 0;
    size_t cache_bytes = 0, cache_capacity = 0, pinned_bytes = 0; // pinned cache bytes, not host staging
};

// Exact Hy4 latent experts. All selected experts execute; no routing approximation.
// Inputs/outputs are F32 device arrays, token-major. Routing weights are the normalized
// original sigmoid probabilities (including expert_scale), not the biased selection scores.
// Calls on one executor are serialized by its owner. Copies and launches are asynchronous
// on private streams; run() joins their completion into caller_stream before returning.
// Only selected expert IDs cross to the host on a streaming/cache-miss call. Activations,
// probabilities, and expert outputs stay on devices. A fully preloaded layer avoids this
// readback. Full layers on another card use peer copies, never host activation staging.
class ExpertExecutor {
  public:
    using Lookup = std::function<const Tensor &(const std::string &)>;
    static ExpertMemory memory_requirements(const Geometry &geometry, const Lookup &lookup, int max_rows = 8,
                                            int lanes = 2);
    ExpertExecutor(const Geometry &geometry, Lookup lookup, std::vector<ExpertDevice> devices,
                   int max_rows = 8, int lanes = 2);
    ~ExpertExecutor();
    ExpertExecutor(const ExpertExecutor &) = delete;
    ExpertExecutor &operator=(const ExpertExecutor &) = delete;

    void run(int layer, int input_device, const float *latent, const int32_t *expert_ids,
             const float *routing_weights, float *weighted_out, int rows, void *caller_stream);

    // Pin all experts of one layer when a card has enough unused cache capacity.
    // Returns false without changing placement when the layer cannot fit. The default
    // device chooses the selected card with the most available resident capacity.
    // Q4_K gate/up with Q4_K/Q6_K down, and floating F32/F16/BF16 triplets
    // use routed fast kernels. Other formats use cached native MMVQs.
    bool preload_layer(int layer, int device = -1);
    void synchronize();
    size_t reserved_bytes(int device) const; // Allocated VRAM including workspace/cache.
    ExpertCacheStats stats() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::hy4
