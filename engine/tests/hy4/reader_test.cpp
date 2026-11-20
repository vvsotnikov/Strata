#define STRATA_GGUF_MAIN_DISABLED 1
#include "../core/gguf_fixture.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>

int main() {
    const auto dir = std::filesystem::temp_directory_path() / ("strata-hy4-reader-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    struct Cleanup { std::filesystem::path dir; ~Cleanup() { std::error_code ec; std::filesystem::remove_all(dir, ec); } } cleanup{dir};
    std::vector<uint64_t> values(128); for (size_t i = 0; i < values.size(); ++i) values[i] = i;
    const auto path = dir / "fp8.gguf";
    fixture::write(path, {fixture::u32s("mapping", values)}, {{"table", {96, 3}, 43}});
    const auto check = [](bool ok) { if (!ok) throw std::runtime_error("Hy4 GGUF reader regression"); };
    strata::GgufFile sampled(path.string());
    strata::GgufFile complete(path.string(), true);
    check(sampled.get("mapping")->count == 128 && sampled.get("mapping")->items.size() == 64);
    check(complete.get("mapping")->items.size() == 128 && complete.get("mapping")->items.back().u == 127);
    check(strata::tensor_payload_bytes(complete.tensors()[0]) == 288);
    auto model = strata::GgufModel::open(path.string(), true);
    check(model.in_bounds(model.shard(0).tensors()[0], 0));
    const auto cut = dir / "cut.gguf";
    fixture::write(cut, {}, {{"table", {96, 3}, 43}}, 1);
    auto truncated = strata::GgufModel::open(cut.string(), true);
    check(!truncated.in_bounds(truncated.shard(0).tensors()[0], 0));
    std::cout << "Hy4: opt-in full metadata, default array sample and FP8 tensor bounds passed\n";
}
