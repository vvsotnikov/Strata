#include "strata/hy4/engine.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace strata::hy4;

namespace {
template <class T> T read(std::ifstream &input) {
    T value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!input)
        throw std::runtime_error("truncated reference input");
    return value;
}
void save(const std::filesystem::path &path, const Vec &values) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    if (!output)
        throw std::runtime_error("cannot write reference output");
}
void append(Vec &destination, std::span<const float> row) {
    destination.insert(destination.end(), row.begin(), row.end());
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 4)
            throw std::runtime_error("reference_test MODEL INPUT OUTPUT_DIR");
        Engine engine(argv[1], 2, 64);
        std::ifstream input(argv[2], std::ios::binary);
        const uint32_t count = read<uint32_t>(input), dim = read<uint32_t>(input);
        if (dim != static_cast<uint32_t>(engine.geometry().dim) || count > 64)
            throw std::runtime_error("invalid reference fixture dimensions");
        Vec logits, hidden, draft_logits, draft_hidden, first_hidden, first_draft;
        RopePosition first_position{};
        int32_t first_candidate = 0;
        for (uint32_t row = 0; row < count; ++row) {
            const int32_t token = read<int32_t>(input);
            RopePosition position{};
            for (auto &axis : position)
                axis = read<int64_t>(input);
            const int64_t next = read<int64_t>(input);
            Vec embedding(dim);
            for (auto &value : embedding)
                value = read<float>(input);
            const int32_t candidate = read<int32_t>(input);
            RopePosition draft_position{};
            for (auto &axis : draft_position)
                axis = read<int64_t>(input);
            append(logits, token < 0 ? engine.forward_embedding(embedding, position, next)
                                     : engine.forward_at(token, position, next));
            append(hidden, engine.last_hidden());
            const Vec draft =
                candidate < 0
                    ? engine.draft_forward_embedding(embedding, engine.last_hidden(), draft_position)
                    : engine.draft_forward(candidate, engine.last_hidden(), draft_position);
            append(draft_logits, draft);
            append(draft_hidden, engine.last_draft_hidden());
            if (!row) {
                first_hidden = engine.last_hidden();
                first_draft = draft;
                first_position = draft_position;
                first_candidate = candidate;
            }
            if (engine.history().size() != row + 1)
                throw std::runtime_error("draft contaminated target history");
        }
        const size_t length = engine.history().size();
        try {
            engine.forward_at(1, RopePosition{0, -1, 0});
            throw std::logic_error("accepted negative image axis");
        } catch (const std::runtime_error &) {
        }
        if (engine.history().size() != length)
            throw std::runtime_error("invalid position mutated target history");
        engine.reset_draft();
        if (engine.draft_forward(first_candidate, first_hidden, first_position) != first_draft ||
            engine.history().size() != length)
            throw std::runtime_error("draft cache reset did not reproduce its first step");
        engine.reset();
        if (!engine.history().empty() || !engine.last_hidden().empty() || !engine.last_draft_hidden().empty())
            throw std::runtime_error("reset left target/draft state behind");
        const std::filesystem::path output(argv[3]);
        save(output / "target_logits.f32", logits);
        save(output / "target_hidden.f32", hidden);
        save(output / "draft_logits.f32", draft_logits);
        save(output / "draft_hidden.f32", draft_hidden);
        std::cout << "Hy4 target/image and separate MTP reference completed " << count << " rows\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
