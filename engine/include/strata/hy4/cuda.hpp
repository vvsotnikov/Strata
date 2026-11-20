#pragma once
#include "strata/hy4/engine.hpp"

namespace strata::hy4 {
// Explicit device ordinals in CUDA_VISIBLE_DEVICES. Budgets include weights, not scratch;
// unmapped matrices and cold experts execute on the CPU. Engram stays host mapped.
std::unique_ptr<Matmul> cuda_matmul(int threads, const std::vector<int>& devices,
                                    const std::vector<size_t>& budgets);
}
