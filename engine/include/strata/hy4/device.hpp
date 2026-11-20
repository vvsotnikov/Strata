#pragma once
#include "strata/hy4/engine.hpp"

namespace strata::hy4 {
struct DeviceOptions {
    std::vector<int> devices;
    std::vector<size_t> dense_bytes, expert_bytes;
    std::vector<size_t> vram_reserves; // optional per-device margin override, e.g. vision encoder
    int slots = 4, batch = 8;
    int prefill_rows = 8; // wider TC prompt tiles; decode/MTP verification retain batch<=8
    int64_t context_pool = 8192, max_context = 8192;
    bool host_kv = true;
    bool host_swa_kv = true; // A/B: keep bounded sliding-window KV on device when false.
    bool stage_kv = true; // gather selected host KV once before all heads consume it
    bool f32_kv = false;
    bool preload_mtp_first = false; // explicit measured placement alternative
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
    virtual bool checkpoint(int slot) { (void)slot; return false; }
    virtual bool restore_checkpoint(int slot,size_t length) { (void)slot; (void)length; return false; }
    virtual void reset(int slot) = 0;
    virtual void truncate(int slot, size_t length) = 0;
    virtual size_t length(int slot) const = 0;
    virtual std::span<const int32_t> history(int slot) const = 0;
    virtual int slots() const = 0;
    virtual int batch() const = 0;
    virtual int prefill_batch() const { return batch(); }
    virtual Vec prefill(std::span<const DeviceInput> inputs, bool ingest_draft) {
        auto logits = forward(inputs);
        if (ingest_draft) draft_ingest(inputs);
        return logits.back();
    }
    virtual size_t free_tokens() const = 0;
    virtual std::string stats() const = 0;
    virtual bool healthy() const = 0;
    virtual bool has_mtp() const = 0;
    virtual std::vector<Vec> draft_forward(std::span<const DeviceInput> inputs, bool target_features) = 0;
    // Canonical cache catch-up needs features/KV, not vocabulary projection.
    virtual void draft_ingest(std::span<const DeviceInput> inputs) { (void)draft_forward(inputs, true); }
    virtual size_t draft_length(int slot) const = 0;
    virtual void draft_truncate(int slot, size_t length) = 0;
    virtual void draft_accept(int slot, size_t absolute_target_position) = 0;
};
std::unique_ptr<DeviceModel> device_model(const Engine &weights, const DeviceOptions &options);
} // namespace strata::hy4
