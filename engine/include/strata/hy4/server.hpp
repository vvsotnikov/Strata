#pragma once
#include "strata/hy4/device.hpp"
#include <functional>

namespace strata::hy4 {
struct ServerOptions {
    int mtp = 5;
    int prefill_chunk = 8;
    int64_t max_context = 0; // 0 uses model metadata's limit
    bool vision = false;
    int32_t image_pad = -1; // resolved from GGUF tokenizer by the Engine overload
    size_t max_line_bytes = 64ull << 20;
};
// read(line, wait): false means no line now. An EOF is delivered as QUIT once.
// Production transport polls stdin; no reader thread can outlive the server/model.
struct ServerTransport {
    std::function<bool(std::string &, bool)> read;
    std::function<void(const std::string &)> write;
};
int serve_device(DeviceModel &model, const Engine &weights, ServerOptions options = {});
// The same scheduler with injectable transport and geometry, for deterministic CPU tests.
int serve_device(DeviceModel &model, const Geometry &geometry, std::span<const int32_t> eos,
                 ServerOptions options, ServerTransport transport);
} // namespace strata::hy4
