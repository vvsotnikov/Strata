#pragma once
#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>

namespace strata::hy4::detail {
struct DeviceMemoryPlan {
    size_t dense_capacity = 0, transient_bytes = 0, expert_capacity = 0;
};
// available is measured after KV, vectors and every named work buffer exist.
// Dense matrices larger than the cache stream through an independently reserved
// buffer. A head that fits the cache does not need a duplicate transient copy.
inline DeviceMemoryPlan plan_device_memory(size_t available, size_t expert_workspace, size_t reserve,
                                           size_t total_dense, size_t largest_dense,
                                           std::optional<size_t> dense = {},
                                           std::optional<size_t> experts = {}) {
    auto subtract = [](size_t total, size_t bytes) {
        if (bytes > total)
            throw std::runtime_error("Hy4 device budgets exceed available VRAM after KV/scratch");
        return total - bytes;
    };
    size_t room = subtract(subtract(available, reserve), expert_workspace);
    const size_t dense_room = subtract(room, experts.value_or(0));
    DeviceMemoryPlan plan;
    plan.dense_capacity = dense.value_or(std::min(total_dense, dense_room));
    plan.transient_bytes = largest_dense > plan.dense_capacity ? largest_dense : 0;
    if (!dense && plan.transient_bytes) {
        plan.dense_capacity = std::min(total_dense, subtract(dense_room, plan.transient_bytes));
    }
    room = subtract(subtract(room, plan.dense_capacity), plan.transient_bytes);
    plan.expert_capacity = experts.value_or(room);
    (void)subtract(room, plan.expert_capacity);
    return plan;
}
} // namespace strata::hy4::detail
