#pragma once
#include "strata/hy4/engine.hpp"

namespace strata::hy4 {
struct DeviceOptions {
    std::vector<int> devices;
    std::vector<size_t> dense_bytes, expert_bytes;
    int slots = 4, batch = 8;
    int64_t context_pool = 8192, max_context = 8192;
    bool host_kv = true;
    bool f32_kv = false;
    size_t vram_reserve = 1024ull << 20;
};
struct DeviceInput {
    int32_t token = 0;
    int slot = 0;
    // Negative coordinates use the sequence's next text position. Embedding rows carry
    // explicit temporal/height/width coordinates; logical KV positions remain separate.
    std::array<int32_t, 3> rope{-1, -1, -1};
    int32_t next_rope = -1;
    std::span<const float> embedding;
};
class DeviceModel {
  public:
    virtual ~DeviceModel() = default;
    virtual std::vector<Vec> forward(std::span<const DeviceInput> inputs, const Trace &trace = {}) = 0;
    virtual void reset(int slot) = 0;
    virtual void truncate(int slot, size_t length) = 0;
    virtual size_t length(int slot) const = 0;
    virtual std::span<const int32_t> history(int slot) const = 0;
    virtual int slots() const = 0;
    virtual int batch() const = 0;
    virtual size_t free_tokens() const = 0;
    virtual std::string stats() const = 0;
    virtual bool healthy() const = 0;
    virtual bool has_mtp() const = 0;
    virtual std::vector<Vec> draft_forward(std::span<const DeviceInput> inputs, bool target_features) = 0;
    virtual size_t draft_length(int slot) const = 0;
    virtual void draft_truncate(int slot, size_t length) = 0;
    virtual void draft_accept(int slot, size_t absolute_target_position) = 0;
};
std::unique_ptr<DeviceModel> device_model(const Engine &weights, const DeviceOptions &options);
} // namespace strata::hy4
