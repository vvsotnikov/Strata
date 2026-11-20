#include "strata/hy4/server.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#ifndef _WIN32
#include <poll.h>
#include <unistd.h>
#endif

using namespace strata::hy4;
namespace {
void check(bool value, const std::string &message) {
    if (!value)
        throw std::runtime_error(message);
}
struct Seen {
    int slot, token;
    std::array<int32_t, 3> rope;
    Vec embedding;
};
struct Synthetic final : DeviceModel {
    std::array<std::vector<int32_t>, 4> histories, drafts;
    std::array<bool, 4> carry_valid{};
    bool disagree_once = false;
    size_t provisional = 0;
    std::vector<Seen> seen;
    size_t capacity = 512, largest_batch = 0, draft_calls = 0, target_replays = 0, accepted_features = 0;
    bool mtp = false, cross_slot_tile = false, slow = false;
    size_t forwards = 0;
    std::function<void()> after_forward;
    std::function<void(bool)> after_draft;
    Vec logits(int32_t token) const {
        Vec out(8, -8);
        out.at(size_t((token + 1) % 8)) = 8;
        return out;
    }
    std::vector<Vec> forward(std::span<const DeviceInput> inputs, const Trace & = {}) override {
        if (slow)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        check(inputs.size() <= 8, "scheduler exceeded native tile capacity");
        check(inputs.size() <= free_tokens(), "unified pool exhausted");
        for (const auto &in : inputs)
            check(in.slot >= 0 && in.slot < 4 && in.token >= 0 && in.token < 8, "invalid device input");
        largest_batch = std::max(largest_batch, inputs.size());
        if (!inputs.empty())
            for (const auto &in : inputs)
                if (in.slot != inputs.front().slot)
                    cross_slot_tile = true;
        std::vector<Vec> out;
        for (const auto &in : inputs) {
            histories[in.slot].push_back(in.token);
            seen.push_back({in.slot, in.token, in.rope, Vec(in.embedding.begin(), in.embedding.end())});
            out.push_back(logits(in.token));
        }
        ++forwards;
        if (after_forward)
            after_forward();
        return out;
    }
    void reset(int slot) override {
        histories.at(slot).clear();
        drafts.at(slot).clear();
        carry_valid.at(slot) = false;
    }
    void truncate(int slot, size_t n) override {
        check(n <= histories.at(slot).size(), "invalid target truncation");
        if (n != histories.at(slot).size()) carry_valid.at(slot) = false;
        histories.at(slot).resize(n);
    }
    size_t length(int slot) const override { return histories.at(slot).size(); }
    std::span<const int32_t> history(int slot) const override { return histories.at(slot); }
    int slots() const override { return 4; }
    int batch() const override { return 8; }
    size_t free_tokens() const override {
        size_t used = 0;
        for (const auto &h : histories)
            used += h.size();
        return capacity - used;
    }
    std::string stats() const override { return "synthetic=1"; }
    bool healthy() const override { return true; }
    bool has_mtp() const override { return mtp; }
    std::vector<Vec> draft_forward(std::span<const DeviceInput> inputs, bool target_features) override {
        ++draft_calls;
        if (target_features)
            ++target_replays;
        std::vector<Vec> out;
        for (const auto &in : inputs) {
            check(target_features || carry_valid.at(in.slot), "proposer used invalidated target feature carry after " + std::to_string(forwards) + " target calls and " + std::to_string(accepted_features) + " accepted feature restores");
            drafts.at(in.slot).push_back(in.token);
            const bool disagree = !target_features && disagree_once && ++provisional == 1;
            out.push_back(logits(disagree ? (in.token + 2) % 8 : in.token));
            carry_valid.at(in.slot) = true;
        }
        if (after_draft)
            after_draft(target_features);
        return out;
    }
    size_t draft_length(int slot) const override { return drafts.at(slot).size(); }
    void draft_truncate(int slot, size_t n) override {
        check(n <= drafts.at(slot).size(), "invalid draft truncation");
        if (n != drafts.at(slot).size()) carry_valid.at(slot) = false;
        drafts.at(slot).resize(n);
    }
    void draft_accept(int slot, size_t) override { ++accepted_features; carry_valid.at(slot) = true; }
};
struct Harness {
    std::deque<std::string> input;
    std::vector<std::string> output;
    std::function<void(const std::string &)> after_write;
    size_t reads = 0;
    ServerTransport transport() {
        return {[&](std::string &line, bool wait) {
                    check(++reads < 100000, "scheduler did not terminate");
                    if (!input.empty()) {
                        line = input.front();
                        input.pop_front();
                        return true;
                    }
                    if (wait) {
                        line = "QUIT";
                        return true;
                    }
                    return false;
                },
                [&](const std::string &line) {
                    output.push_back(line);
                    if (after_write)
                        after_write(line);
                }};
    }
    bool contains(const std::string &prefix) const {
        return std::any_of(output.begin(), output.end(),
                           [&](const auto &s) { return s.starts_with(prefix); });
    }
    size_t count(const std::string &prefix) const {
        return std::count_if(output.begin(), output.end(),
                             [&](const auto &s) { return s.starts_with(prefix); });
    }
    std::vector<int> tokens(int slot = -1) const {
        std::vector<int> out;
        for (const auto &line : output) {
            std::istringstream in(line);
            std::string kind;
            int a, b;
            if (!(in >> kind >> a))
                continue;
            if (slot < 0 && kind == "T")
                out.push_back(a);
            else if (slot >= 0 && kind == "BT" && (in >> b) && a == slot)
                out.push_back(b);
        }
        return out;
    }
};
Geometry geometry() {
    Geometry g;
    g.dim = 4;
    g.vocab = 8;
    g.pad = 7;
    g.max_context = 128;
    return g;
}
void run(Synthetic &model, Harness &h, int mtp = 0, bool vision = false) {
    ServerOptions o;
    o.mtp = mtp;
    o.prefill_chunk = 8;
    o.max_context = 128;
    o.vision = vision;
    o.image_pad = 7;
    const std::array<int32_t, 1> eos{6};
    check(serve_device(model, geometry(), eos, o, h.transport()) == 0, "server returned failure");
}

void single_and_cache() {
    Synthetic model;
    Harness h;
    h.input = {"GEN 3 temperature=0 top_k=0 1,2"};
    bool second = false;
    h.after_write = [&](const std::string &line) {
        if (!second && line.starts_with("DONE ")) {
            second = true;
            h.input.push_back("GEN 2 temperature=0 1,2");
        }
    };
    run(model, h);
    check(std::any_of(h.output.begin(), h.output.end(),
                      [](const auto &line) {
                          return line.starts_with("INFO ") && line.find("batch_slots=4") != std::string::npos;
                      }),
          "native slot count missing from discovery");
    check(h.contains("READY 128 stop"), "READY does not expose cancellation");
    check(h.tokens() == std::vector<int>({3, 4, 5, 3, 4}),
          "solo output or prefix reuse changed next-token output");
    check(h.count("DONE ") == 2 && h.contains("DONE 3 ") && h.contains("DONE 2 "),
          "solo completion is not delimited");
    check(h.contains("RESUME 1"), "second prompt did not reuse its held prefix");
    check(!h.contains("ERR "), "solo request failed");
}
void batch_isolation_and_cancel() {
    Synthetic model;
    Harness h;
    h.input = {"BGEN 0 20 temperature=0 slot_read=1 0,1", "BGEN 1 3 temperature=0 slot_read=1 1,2"};
    bool stopped = false;
    h.after_write = [&](const std::string &line) {
        if (!stopped && line.starts_with("BT 0 ")) {
            stopped = true;
            h.input.push_back("BSTOP 0");
        }
    };
    run(model, h);
    check(stopped, "batch slot never emitted");
    check(h.contains("BADM 0 ") && h.contains("BADM 1 "), "batch admissions were not acknowledged");
    check(h.count("BDONE 0 ") == 1 && h.count("BDONE 1 ") == 1, "batch slot is not released exactly once");
    const auto stopped_line = std::find_if(h.output.begin(), h.output.end(),
                                           [](const auto &s) { return s.starts_with("BDONE 0 "); });
    check(stopped_line != h.output.end() && stopped_line->find("cancel") != std::string::npos,
          "BSTOP did not cancel its own slot");
    check(h.tokens(1) == std::vector<int>({3, 4, 5}), "cancelling slot zero corrupted another slot");
    check(!h.contains("ERR "), "batch request failed");
}
void batch_uses_shared_device_tile() {
    Synthetic model;
    Harness h;
    h.input = {"BGEN 0 4 temperature=0 slot_read=1 0,1", "BGEN 1 3 temperature=0 slot_read=1 1,2"};
    run(model, h);
    check(!h.contains("ERR "), "concurrent requests failed");
    check(h.tokens(0) == std::vector<int>({2, 3, 4, 5}) && h.tokens(1) == std::vector<int>({3, 4, 5}),
          "concurrent outputs crossed slot histories");
    check(model.cross_slot_tile, "concurrent slots never shared a native forward tile");
}
void embedded_prompt_and_invalid_input() {
    const auto unique = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto path = std::filesystem::temp_directory_path() / ("hy4-server-" + unique + ".sve");
    struct Remove {
        std::filesystem::path path;
        ~Remove() {
            std::error_code e;
            std::filesystem::remove(path, e);
        }
    } remove{path};
    const int32_t header[]{0x31455653, 2, 2, 1, 4};
    const float rows[]{1, 2, 3, 4, 5, 6, 7, 8};
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(header), sizeof(header));
        f.write(reinterpret_cast<const char *>(rows), sizeof(rows));
    }
    Synthetic model;
    Harness h;
    h.input = {"GENI 1 temperature=0 " + path.string() + " 1,7,7,2"};
    run(model, h, 0, true);
    check(!h.contains("ERR "), "valid image embeddings rejected");
    std::vector<Vec> received;
    for (const auto &in : model.seen)
        if (!in.embedding.empty())
            received.push_back(in.embedding);
    check(received == std::vector<Vec>({{1, 2, 3, 4}, {5, 6, 7, 8}}),
          "SVE1 rows or learned newline lost in image ingestion");
    check(model.seen.size() >= 4 && model.seen[1].rope == std::array<int32_t, 3>{1, 1, 1} &&
              model.seen[2].rope == std::array<int32_t, 3>{1, 1, 2} &&
              model.seen[3].rope == std::array<int32_t, 3>{3, 3, 3},
          "image RoPE axes/newline or following text position differ from the model template");
    check(h.tokens() == std::vector<int>({3}), "image request did not complete");
    Synthetic invalid;
    Harness bad;
    bad.input = {"BGENI 0 1 temperature=0 " + path.string() + " 1,7,2", "GEN 1 temperature=0 1,2"};
    run(invalid, bad, 0, true);
    check(bad.contains("ERR "), "mismatched image row count accepted");
    check(bad.tokens() == std::vector<int>({3}), "invalid image request poisoned the next admission");
}
void speculation_matches_greedy_and_uses_target_features() {
    Synthetic model;
    model.mtp = true;
    Harness h;
    h.input = {"GEN 5 temperature=0 seed=37 0,1"};
    run(model, h, 3);
    check(!h.contains("ERR "), "native MTP request failed");
    check(h.tokens() == std::vector<int>({2, 3, 4, 5, 6}),
          "MTP changed greedy outputs or continued after EOS");
    check(model.draft_calls > 0, "enabled MTP never called the native draft block");
    check(model.accepted_features > 0 && model.target_replays > 0,
          "MTP did not ingest accepted target features");
    check(model.largest_batch > 1, "MTP verification was serialized into single-token calls");
}
void rejection_commit_restores_carry_after_target_truncate() {
    Synthetic model; model.mtp = true; model.disagree_once = true;
    Harness h; h.input = {"GEN 5 temperature=0 0,1"}; run(model, h, 3);
    check(!h.contains("ERR "), "rejection invalidated the next round's target feature carry");
    check(h.tokens() == std::vector<int>({2,3,4,5,6}), "rejection/replay changed target outputs");
    check(model.provisional >= 2, "rejection test did not execute another proposal round");
}
void cancellation_at_last_gpu_boundary_does_not_emit() {
    for (int phase = 0; phase < 3; ++phase) {
        Synthetic model;
        model.mtp = true;
        Harness h;
        h.input = {"GEN 5 temperature=0 0,1"};
        bool sent = false;
        if (phase != 1)
            model.after_forward = [&]() {
                if (!sent && model.forwards == size_t(phase == 0 ? 1 : 2)) {
                    sent = true;
                    h.input.push_back("STOP");
                }
            };
        else
            model.after_draft = [&](bool target) {
                if (!sent && !target) {
                    sent = true;
                    h.input.push_back("STOP");
                }
            };
        run(model, h, 3);
        check(sent, "test failed to inject cancellation during a GPU window");
        check(h.tokens() == (phase == 0 ? std::vector<int>{} : std::vector<int>{2}),
              "cancelled GPU window leaked a new token");
        check(std::any_of(h.output.begin(), h.output.end(),
                          [](const auto &x) {
                              return x.starts_with("DONE ") && x.find("cancel") != std::string::npos;
                          }),
              "GPU boundary cancellation did not finish");
        check(model.draft_length(0) == model.length(0),
              "cancelled MTP window left draft and target lengths inconsistent");
    }
}
void queued_cancel_is_not_lost() {
    Synthetic model;
    Harness h;
    h.input = {"BGEN 0 20 temperature=0 slot_read=1 0,1", "BSTOP 0",
               "BGEN 1 2 temperature=0 slot_read=1 1,2"};
    run(model, h);
    check(!h.contains("ERR "), "queued cancellation failed");
    check(h.tokens(0).empty(), "BSTOP arriving before admission was lost");
    check(h.contains("BADM 0 0") ||
              (h.contains("BADM 0 1") && std::any_of(h.output.begin(), h.output.end(),
                                                     [](const auto &x) {
                                                         return x.starts_with("BDONE 0 ") &&
                                                                x.find("cancel") != std::string::npos;
                                                     })),
          "cancelled pending admission was not released");
    check(h.tokens(1) == std::vector<int>({3, 4}), "queued cancellation affected another slot");
}
void reading_slots_are_admitted_and_interleaved() {
    Synthetic model;
    Harness h;
    const std::string prompt = "0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5";
    h.input = {"BGEN 0 2 temperature=0 slot_read=1 " + prompt,
               "BGEN 1 2 temperature=0 slot_read=1 " + prompt};
    run(model, h);
    check(!h.contains("ERR "), "parallel long prompts failed");
    const auto a = std::find_if(h.output.begin(), h.output.end(),
                                [](const auto &x) { return x.starts_with("BADM 1 1"); });
    const auto b =
        std::find_if(h.output.begin(), h.output.end(), [](const auto &x) { return x.starts_with("BT "); });
    check(a != h.output.end() && b != h.output.end() && a < b,
          "second slot was not admitted before the first long prompt completed");
    std::vector<int> prompt_tiles;
    size_t pos = 0;
    // The prompt is longer than two tiles and terminates on token 5. The initial
    // slot ownership sequence must alternate rather than consume one whole prompt.
    while (pos < model.seen.size() && prompt_tiles.size() < 4) {
        const int slot = model.seen[pos].slot;
        prompt_tiles.push_back(slot);
        pos += 8;
    }
    check(prompt_tiles.size() == 4 && prompt_tiles[0] != prompt_tiles[1] &&
              prompt_tiles[0] == prompt_tiles[2],
          "long-prompt prefill was not scheduled in fair bounded chunks");
}
#ifndef _WIN32
int stdio_mode() {
    Synthetic model;
    model.mtp = true;
    model.slow = true;
    ServerOptions o;
    o.mtp = 2;
    o.prefill_chunk = 8;
    o.max_context = 128;
    std::string buffer;
    bool eof = false;
    ServerTransport transport{[&](std::string &line, bool wait) {
                                  for (;;) {
                                      const auto nl = buffer.find('\n');
                                      if (nl != std::string::npos) {
                                          line = buffer.substr(0, nl);
                                          buffer.erase(0, nl + 1);
                                          return true;
                                      }
                                      if (eof)
                                          return false;
                                      pollfd fd{STDIN_FILENO, POLLIN, 0};
                                      const int ready = ::poll(&fd, 1, wait ? -1 : 0);
                                      if (!ready)
                                          return false;
                                      check(ready > 0, "test stdin poll failed");
                                      char bytes[4096];
                                      const auto n = ::read(STDIN_FILENO, bytes, sizeof(bytes));
                                      check(n >= 0, "test stdin read failed");
                                      if (!n) {
                                          eof = true;
                                          line = "QUIT";
                                          return true;
                                      }
                                      buffer.append(bytes, size_t(n));
                                  }
                              },
                              [](const std::string &line) { std::cout << line << '\n'
                                                                      << std::flush; }};
    const std::array<int32_t, 1> eos{6};
    return serve_device(model, geometry(), eos, o, std::move(transport));
}
#endif
void bad_request_recovers() {
    Synthetic model;
    Harness h;
    h.input = {"BGEN 99 1 temperature=0 1,2", "GEN 1 temperature=0 1,2"};
    run(model, h);
    check(h.contains("ERR "), "invalid slot accepted");
    check(h.tokens() == std::vector<int>({3}) && h.count("DONE ") == 1,
          "invalid slot poisoned subsequent requests");
}
} // namespace
int main(int argc, char **argv) {
    try {
#ifndef _WIN32
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "--stdio")
                return stdio_mode();
#else
        (void)argc;
        (void)argv;
#endif
        single_and_cache();
        batch_isolation_and_cancel();
        batch_uses_shared_device_tile();
        embedded_prompt_and_invalid_input();
        speculation_matches_greedy_and_uses_target_features();
        queued_cancel_is_not_lost();
        rejection_commit_restores_carry_after_target_truncate();
        cancellation_at_last_gpu_boundary_does_not_emit();
        reading_slots_are_admitted_and_interleaved();
        bad_request_recovers();
        std::cout << "Hy4 scheduler CPU protocol tests passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
