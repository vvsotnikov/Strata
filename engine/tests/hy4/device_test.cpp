#include "strata/hy4/device.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace strata::hy4;
namespace {
struct Row {
    int token = 0, slot = 0, next = -1;
    std::array<int32_t, 3> position{-1, -1, -1};
    Vec embedding;
    DeviceInput input() const { return {token, slot, position, next, embedding}; }
    RopePosition rope() const { return {position[0], position[1], position[2]}; }
};
Vec target(Engine &cpu, const Row &row) {
    if (!row.embedding.empty())
        return cpu.forward_embedding(row.embedding, row.rope(), row.next);
    if (row.position[0] < 0)
        return cpu.forward(row.token);
    return cpu.forward_at(row.token, row.rope(), row.next);
}
Vec draft(Engine &cpu, const Row &row, std::span<const float> previous) {
    if (!row.embedding.empty())
        return cpu.draft_forward_embedding(row.embedding, previous, row.rope(), row.next);
    return cpu.draft_forward(row.token, previous, row.rope(), row.next);
}
void compare(const Vec &actual, const Vec &expected, float &maximum, const std::string &stage) {
    if (actual.size() != expected.size())
        throw std::runtime_error(stage + " output width");
    float error = 0;
    for (size_t i = 0; i < actual.size(); ++i)
        error = std::max(error, std::abs(actual[i] - expected[i]));
    maximum = std::max(maximum, error);
    if (error > 1e-4f)
        throw std::runtime_error(stage + " max_abs=" + std::to_string(error));
}
void require(bool value, const char *text) {
    if (!value)
        throw std::runtime_error(text);
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 2)
            throw std::runtime_error("hy4_device_test TINY_MODEL [--actual|--peer]");
        bool actual = false, peer = false;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--actual") actual = true;
            else if (option == "--peer") peer = true;
            else throw std::runtime_error("unknown device test option " + option);
        }
        require(!actual || !peer, "peer mode is restricted to the tiny fixture");
        Engine source(argv[1], 4, actual ? 8192 : 64);
        const auto &g = source.geometry();
        DeviceOptions options;
        // Ordinals within the explicitly selected CUDA_VISIBLE_DEVICES UUIDs.
        options.devices = peer ? std::vector<int>{0, 1} : std::vector<int>{0};
        options.dense_bytes.assign(options.devices.size(), actual ? 128ull << 20 : 16ull << 20);
        options.expert_bytes.assign(options.devices.size(), actual ? 32ull << 20 : 0);
        if (peer) options.vram_reserve = 64ull << 20;
        options.slots = 4;
        options.max_context = actual ? 8192 : 64;
        options.context_pool = actual ? 8192 : 128;
        options.f32_kv = !actual;
        auto model = device_model(source, options);
        std::vector<std::unique_ptr<Engine>> refs;
        for (int slot = 0; slot < options.slots && !actual; ++slot)
            refs.push_back(std::make_unique<Engine>(argv[1], 2, 64));
        float maximum = 0;
        std::vector<Vec> all;
        for (int phase = 0; phase < (actual ? 2 : 4); ++phase) {
            std::vector<DeviceInput> input;
            for (int row = 0; row < (actual ? 1 : 8); ++row) {
                DeviceInput in;
                in.slot = actual ? 0 : phase % 2;
                in.token = 1 + phase * (actual ? 1 : 8) + row;
                input.push_back(in);
            }
            const auto output = model->forward(input);
            for (size_t row = 0; row < input.size(); ++row) {
                if (!actual)
                    compare(output[row], refs[input[row].slot]->forward(input[row].token), maximum,
                            "batched trunk");
                all.push_back(output[row]);
            }
        }
        if (actual) {
            std::ofstream output("device-actual-logits.f32", std::ios::binary);
            for (const auto &row : all) {
                output.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
                std::cout << "top=" << std::max_element(row.begin(), row.end()) - row.begin() << '\n';
            }
        } else {
            const size_t before = model->length(0), free = model->free_tokens();
            DeviceInput invalid[2];
            invalid[0].token = invalid[1].token = 1;
            invalid[1].rope = {0, -1, 0};
            bool rejected = false;
            try {
                model->forward(invalid);
            } catch (const std::exception &) {
                rejected = true;
            }
            require(rejected && model->length(0) == before && model->free_tokens() == free &&
                        model->healthy(),
                    "position validation mutated state");
            for (int slot = 0; slot < options.slots; ++slot)
                model->reset(slot);
            require(model->free_tokens() == size_t(options.context_pool),
                    "unified pool leak after baseline reset");
            if (model->has_mtp()) {
                std::vector<std::vector<Row>> history(options.slots);
                std::vector<std::vector<Vec>> features(options.slots);
                Vec zero(g.dim, 0);
                auto rejects_stale_carry = [&](int slot) {
                    DeviceInput in;
                    in.slot = slot;
                    in.token = 1;
                    const size_t before_draft = model->draft_length(slot);
                    bool rejected_carry = false;
                    try {
                        model->draft_forward(std::array{in}, false);
                    } catch (const std::runtime_error &error) {
                        rejected_carry =
                            std::string(error.what()).find("carry is unavailable") != std::string::npos;
                    }
                    require(rejected_carry && model->healthy() && model->draft_length(slot) == before_draft,
                            "recursive MTP consumed stale or uninitialized carry");
                };
                rejects_stale_carry(0);
                auto make = [&](int slot, int position, bool image) {
                    Row row;
                    row.slot = slot;
                    row.token = 1 + (position + slot * 17) % (g.vocab - 1);
                    int p = position < 6 ? position : position - 2;
                    row.position = {p, p, p};
                    row.next = p + 1;
                    if (image && position >= 2 && position < 6) {
                        row.embedding = source.tensor("token_embd.weight").values(30 + position);
                        for (float &value : row.embedding)
                            value *= 1.7f;
                        row.token = -1;
                        row.position = {2, 2 + (position - 2) / 2, 2 + (position - 2) % 2};
                        row.next = position == 5 ? 4 : 2;
                    }
                    return row;
                };
                auto submit_target = [&](const std::vector<Row> &rows) {
                    std::vector<DeviceInput> input;
                    for (const auto &row : rows)
                        input.push_back(row.input());
                    const auto output = model->forward(input);
                    for (size_t i = 0; i < rows.size(); ++i) {
                        const auto &row = rows[i];
                        compare(output[i], target(*refs[row.slot], row), maximum, "image/MRoPE trunk");
                        history[row.slot].push_back(row);
                        features[row.slot].push_back(refs[row.slot]->last_hidden());
                    }
                };
                auto canonical_draft = [&](const std::vector<Row> &rows) {
                    std::vector<DeviceInput> input;
                    for (const auto &row : rows)
                        input.push_back(row.input());
                    std::vector<size_t> position(options.slots);
                    for (int slot = 0; slot < options.slots; ++slot)
                        position[slot] = model->draft_length(slot);
                    const auto output = model->draft_forward(input, true);
                    for (size_t i = 0; i < rows.size(); ++i) {
                        const auto &row = rows[i];
                        const size_t p = position[row.slot]++;
                        compare(output[i], draft(*refs[row.slot], row, p ? features[row.slot][p - 1] : zero),
                                maximum, "canonical MTP/image");
                    }
                };
                auto rebuild = [&](int slot) {
                    refs[slot] = std::make_unique<Engine>(argv[1], 2, 64);
                    features[slot].clear();
                    for (const auto &row : history[slot]) {
                        target(*refs[slot], row);
                        features[slot].push_back(refs[slot]->last_hidden());
                    }
                };
                for (int slot = 0; slot < options.slots; ++slot) {
                    refs[slot]->reset();
                    for (int first = 0; first < 12; first += 8) {
                        std::vector<Row> rows;
                        for (int p = first; p < std::min(first + 8, 12); ++p)
                            rows.push_back(make(slot, p, true));
                        submit_target(rows);
                        canonical_draft(rows);
                    }
                }
                // Recursive drafts use accepted target features first, then the previous
                // draft's own post-norm feature; distinct slots execute together.
                for (int step = 0; step < 2; ++step) {
                    std::vector<Row> rows;
                    std::vector<Vec> previous;
                    for (int slot = 0; slot < options.slots; ++slot) {
                        if (!step)
                            model->draft_accept(slot, 11);
                        previous.push_back(step ? refs[slot]->last_draft_hidden() : features[slot][11]);
                        rows.push_back(make(slot, 12 + step, false));
                    }
                    std::vector<DeviceInput> input;
                    for (const auto &row : rows)
                        input.push_back(row.input());
                    const auto output = model->draft_forward(input, false);
                    for (size_t i = 0; i < rows.size(); ++i)
                        compare(output[i], draft(*refs[i], rows[i], previous[i]), maximum, "recursive MTP");
                }
                // Verify multiple candidates, retain one, discard two, and replay the
                // canonical draft through both a completed pooling group and SWA wrap.
                for (int slot = 0; slot < options.slots; ++slot) {
                    std::vector<Row> verify;
                    for (int p = 12; p < 15; ++p)
                        verify.push_back(make(slot, p, false));
                    submit_target(verify);
                    model->truncate(slot, 13);
                    // A feature cached for a discarded target position cannot authorize
                    // canonical MTP execution beyond the new target sequence length.
                    DeviceInput stale;
                    stale.slot = slot;
                    stale.token = 1;
                    bool rejected_feature = false;
                    try {
                        model->draft_forward(std::array{stale}, true);
                    } catch (const std::runtime_error &error) {
                        rejected_feature = std::string(error.what()).find("outside the retained window") !=
                                           std::string::npos;
                    }
                    require(rejected_feature && model->draft_length(slot) == 14 && model->healthy(),
                            "canonical MTP used a discarded target feature");
                    model->draft_truncate(slot, 12);
                    rejects_stale_carry(slot);
                    history[slot].resize(13);
                    rebuild(slot);
                    for (int p = 0; p < 12; ++p)
                        draft(*refs[slot], history[slot][p], p ? features[slot][p - 1] : zero);
                    canonical_draft({history[slot][12]});
                    model->draft_accept(slot, 12);
                    Row continuation = make(slot, 13, false);
                    continuation.token = (continuation.token + 7) % g.vocab;
                    const auto output = model->draft_forward(std::array{continuation.input()}, false);
                    compare(output[0], draft(*refs[slot], continuation, features[slot][12]), maximum,
                            "MTP after rejection");
                    submit_target({continuation});
                    require(model->length(slot) == 14 && model->draft_length(slot) == 14,
                            "rejection replay length");
                }
                // Reuse every pool slot in a new order; no KV, feature or Engram state
                // from the old image-bearing sessions may affect the replacement.
                for (int slot : {2, 0, 3, 1}) {
                    model->reset(slot);
                    history[slot].clear();
                    features[slot].clear();
                    refs[slot]->reset();
                    require(model->draft_length(slot) == 0, "slot reset leaked draft history");
                    rejects_stale_carry(slot);
                }
                require(model->free_tokens() == size_t(options.context_pool),
                        "slot reset leaked unified blocks");
                for (int p = 0; p < 8; ++p) {
                    std::vector<Row> rows;
                    for (int slot : {3, 1, 0, 2})
                        rows.push_back(make(slot, p, false));
                    submit_target(rows);
                    canonical_draft(rows);
                }
                std::cout << "GPU MTP/image, recursive features, rejection replay and slot reuse passed\n";
            }
            for (int slot = 0; slot < options.slots; ++slot)
                model->reset(slot);
            require(model->free_tokens() == size_t(options.context_pool), "final unified pool leak");
        }
        std::cout << "GPU batch/reference devices=" << options.devices.size() << " max_abs=" << maximum
                  << " stats=" << model->stats() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
