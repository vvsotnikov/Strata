#include "strata/hy4/device_memory.hpp"
#include <iostream>
#include <cstdint>
using strata::hy4::detail::plan_device_memory;
int main() {
    auto check = [](bool ok) {
        if (!ok)
            throw std::runtime_error("Hy4 device memory plan regression");
    };
    const auto resident = plan_device_memory(1000, 100, 50, 500, 300);
    check(resident.dense_capacity == 500 && resident.transient_bytes == 0 && resident.expert_capacity == 350);
    const auto stream = plan_device_memory(1000, 100, 0, 800, 300, size_t(0));
    check(stream.dense_capacity == 0 && stream.transient_bytes == 300 && stream.expert_capacity == 600);
    const auto mixed = plan_device_memory(1000, 100, 50, 800, 300, size_t(200), size_t(150));
    check(mixed.dense_capacity == 200 && mixed.transient_bytes == 300 && mixed.expert_capacity == 150);
    const auto bounded = plan_device_memory(1000, 100, 50, 800, 300, {}, size_t(100));
    check(bounded.dense_capacity == 750 && bounded.transient_bytes == 0 && bounded.expert_capacity == 100);
    bool failed = false;
    try {
        plan_device_memory(300, 100, 0, 800, 300);
    } catch (const std::runtime_error &) {
        failed = true;
    }
    check(failed);
    failed = false;
    try {
        plan_device_memory(500, 100, 0, 800, 300, size_t(200), size_t(1));
    } catch (const std::runtime_error &) {
        failed = true;
    }
    check(failed);
    failed = false;
    try {
        plan_device_memory(0, SIZE_MAX, SIZE_MAX, 800, 300);
    } catch (const std::runtime_error &) {
        failed = true;
    }
    check(failed);
    std::cout << "Hy4 device memory planner passed\n";
}
