#include "strata/hy4/engine.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace strata::hy4;
void check(bool value) { if (!value) throw std::runtime_error("Hy4 numerical check failed"); }
int main() {
    Vec x{1, 2, 3, 4};
    rms(x, 0);
    check(std::abs(x[3] - 4 / std::sqrt(7.5f)) < 1e-6);
    Vec r{1, 0, 0, 1};
    rope(r, 1, 100);
    check(std::abs(r[0] - std::cos(1.0f)) < 1e-6);
    check(std::abs(r[1] - std::sin(1.0f)) < 1e-6);
    check(std::abs(r[2] + std::sin(0.1f)) < 1e-6);
    const auto selected = route(Vec{.8f, .6f, .1f}, Vec{0, 0, 1}, 2);
    check(selected == std::vector<int>({2, 0}));
    const Vec p = pool_key(Vec{2, 4, 8, 12}, Vec{0, 0, 0, 0}, Vec{0, 0, 0, 0}, 2, 2);
    check(p == Vec({5, 8}));
    Geometry g;
    g.ngram = 3; g.hash_heads = 1; g.engram_layers = {0}; g.pad = 0;
    g.token_map = {0, 1, 2, 3}; g.multipliers = {UINT64_MAX, 3, 5};
    g.primes = {7, 11}; g.offsets = {0, 7};
    const auto hash = engram_hash(g, 0, std::vector<int32_t>{1, 2}, 3);
    const uint64_t a = uint64_t(3) * UINT64_MAX ^ uint64_t(2) * 3;
    check(hash[0] == int32_t(a % 7));
    check(hash[1] == int32_t((a ^ uint64_t(1) * 5) % 11 + 7));
    strata::TensorInfo info{"fp8", {1, 3}, 43, 0};
    const uint8_t bytes[]{0x38, 0x7e, 0xb8};
    Tensor t{&info, bytes};
    check(t.values(0)[0] == 1 && t.values(1)[0] == 448 && t.values(2)[0] == -1);
    std::cout << "Hy4: RMSNorm, RoPE, biased selection, gated pooling, uint64 hash overflow and FP8 rows passed\n";
}
