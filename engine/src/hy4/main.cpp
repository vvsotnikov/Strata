#include "strata/hy4/engine.hpp"
#include "strata/qwen35/sampling.hpp"
#ifdef STRATA_HY4_CUDA
#include "strata/hy4/cuda.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {
using namespace strata::hy4;
using Sampling = strata::qwen35::Sampling;
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

std::vector<int32_t> ids(const std::string& text) {
    std::vector<int32_t> out;
    std::istringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        size_t end;
        const long long value = std::stoll(item, &end);
        if (end != item.size() || value < 0 || value > INT32_MAX) throw std::runtime_error("invalid token ID");
        out.push_back(static_cast<int32_t>(value));
    }
    if (out.empty()) throw std::runtime_error("empty token prompt");
    return out;
}

int sample(Vec logits, const Sampling& sampling, const std::vector<int32_t>& history, std::mt19937_64& rng) {
    strata::qwen35::apply_penalties(logits, sampling, history);
    for (float value : logits) if (!std::isfinite(value))
        throw std::runtime_error("sampling penalties overflowed logits");
    if (sampling.temperature <= 0) return std::max_element(logits.begin(), logits.end()) - logits.begin();
    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
    if (sampling.top_k > 0 && static_cast<size_t>(sampling.top_k) < order.size()) order.resize(sampling.top_k);
    const float maximum = logits[order[0]];
    std::vector<double> probs;
    double total = 0;
    for (int id : order) {
        const double p = std::exp(double((logits[id] - maximum) / sampling.temperature));
        if (p < sampling.min_p) break;
        probs.push_back(p); total += p;
    }
    size_t take = probs.size();
    if (sampling.top_p < 1) {
        double cumulative = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumulative += probs[i];
            if (cumulative >= total * sampling.top_p) { take = i + 1; break; }
        }
    }
    if (!take) throw std::runtime_error("sampling discarded every token");
    probs.resize(take);
    std::discrete_distribution<size_t> distribution(probs.begin(), probs.end());
    return order[distribution(rng)];
}

void sampling_key(Sampling& s, const std::string& key, const std::string& value) {
    size_t end = 0;
    if (key == "seed") {
        if (value.empty() || value[0] == '-') throw std::runtime_error("invalid sampling seed");
        s.seed = std::stoull(value, &end);
        if (end != value.size()) throw std::runtime_error("invalid sampling seed");
        return;
    }
    const double number = std::stod(value, &end);
    if (end != value.size() || !std::isfinite(number) || std::abs(number) > std::numeric_limits<float>::max())
        throw std::runtime_error("invalid sampling parameter");
    if ((key == "top_k" || key == "last_n" || key == "penalty_last_n") &&
        (number < 0 || number > INT32_MAX || number != std::floor(number)))
        throw std::runtime_error("sampling integer out of range");
    if ((key == "temp" || key == "temperature")) s.temperature = number;
    else if (key == "top_p") s.top_p = number;
    else if (key == "top_k") s.top_k = static_cast<int>(number);
    else if (key == "min_p") s.min_p = number;
    else if ((key == "repeat" || key == "penalty_repeat")) s.repeat = number;
    else if ((key == "last_n" || key == "penalty_last_n")) s.last_n = static_cast<int>(number);
    else if ((key == "freq" || key == "penalty_freq")) s.freq = number;
    else if ((key == "present" || key == "penalty_present")) s.present = number;
    else if (key != "prefill") throw std::runtime_error("unsupported sampling key: " + key);
    if (s.temperature < 0 || s.top_p <= 0 || s.top_p > 1 || s.min_p < 0 || s.min_p > 1 || s.repeat <= 0)
        throw std::runtime_error("sampling parameter out of range");
}

void serve(Engine& engine) {
    std::mutex mutex;
    std::condition_variable cv;
    struct Request { std::string line; std::shared_ptr<std::atomic<bool>> cancel; };
    std::deque<Request> queue;
    std::shared_ptr<std::atomic<bool>> active;
    bool eof = false;
    std::thread reader([&] {
        std::string line;
        while (std::getline(std::cin, line)) {
            {
                std::lock_guard lock(mutex);
                if (line == "STOP") {
                    if (active) active->store(true);
                    else if (!queue.empty()) queue.front().cancel->store(true);
                    continue;
                }
                queue.push_back({line, std::make_shared<std::atomic<bool>>(false)});
            }
            cv.notify_one();
            if (line == "QUIT") break;
        }
        { std::lock_guard lock(mutex); eof = true; }
        cv.notify_one();
    });
    std::cout << "READY " << engine.max_context() << " stop\n" << std::flush;
    for (;;) {
        Request request;
        { std::unique_lock lock(mutex); cv.wait(lock, [&] { return !queue.empty() || eof; });
          if (queue.empty()) break;
          request = std::move(queue.front()); queue.pop_front(); active = request.cancel; }
        const std::string& line = request.line;
        const auto& stop = *request.cancel;
        if (line == "QUIT") break;
        try {
            std::istringstream input(line);
            std::string command, part, token_text;
            int count;
            if (!(input >> command >> count) || command != "GEN" || count < 0)
                throw std::runtime_error("expected GEN <max_new> [sampling keys] <id,id,...>");
            Sampling settings; settings.top_k = 0;
            int64_t pen_from = -1;
            while (input >> part) {
                const size_t eq = part.find('=');
                if (eq != std::string::npos && part.substr(0, eq) == "pen_from") {
                    size_t end;
                    const auto value = part.substr(eq + 1);
                    pen_from = std::stoll(value, &end);
                    if (end != value.size() || pen_from < 0) throw std::runtime_error("invalid pen_from");
                } else if (eq != std::string::npos) sampling_key(settings, part.substr(0, eq), part.substr(eq + 1));
                else if (token_text.empty()) token_text = part;
                else throw std::runtime_error("extra GEN argument");
            }
            const auto prompt = ids(token_text);
            if (prompt.size() > static_cast<size_t>(engine.max_context())) throw std::runtime_error("prompt exceeds context");
            if (pen_from >= static_cast<int64_t>(prompt.size())) throw std::runtime_error("pen_from exceeds prompt");
            const auto* penalty_prompt = std::getenv("STRATA_PENALTY_PROMPT");
            auto penalty_history = strata::qwen35::penalty_history_start(prompt, penalty_prompt && std::string(penalty_prompt) == "1");
            if (pen_from >= 0) penalty_history.assign(prompt.begin() + pen_from, prompt.end());
            engine.reset();
            const auto t0 = Clock::now();
            Vec logits;
            bool cancelled = false;
            size_t prompt_read = 0;
            for (size_t i = 0; i < prompt.size(); ++i) {
                if (stop.load()) { cancelled = true; break; }
                logits = engine.forward(prompt[i]);
                ++prompt_read;
                if ((i + 1) % 8 == 0 || i + 1 == prompt.size())
                    std::cout << "PP " << i + 1 << ' ' << prompt.size() << ' ' << milliseconds(t0) << '\n' << std::flush;
            }
            const double prompt_ms = milliseconds(t0);
            const auto d0 = Clock::now();
            std::mt19937_64 rng(settings.seed);
            int produced = 0;
            std::string finish = cancelled ? "cancel" : "length";
            for (; produced < count && !cancelled; ++produced) {
                if (stop.load()) { finish = "cancel"; break; }
                const int token = sample(logits, settings, penalty_history, rng);
                penalty_history.push_back(token);
                std::cout << "T " << token << '\n' << std::flush;
                const auto* eos = engine.metadata().get("tokenizer.ggml.eos_token_id");
                if (eos && token == static_cast<int>(eos->u)) { ++produced; finish = "stop"; break; }
                if (engine.history().size() >= static_cast<size_t>(engine.max_context())) { ++produced; break; }
                if (produced + 1 < count) logits = engine.forward(token);
            }
            { std::lock_guard lock(mutex); active.reset(); }
            std::cout << "DONE " << produced << ' ' << prompt.size() << ' ' << prompt_ms << ' ' << milliseconds(d0)
                      << ' ' << finish << " 0 0 0 0 0 0 0 0.0 " << prompt_read << " 0\n" << std::flush;
        } catch (const std::exception& error) {
            engine.reset();
            { std::lock_guard lock(mutex); active.reset(); }
            std::cout << "ERR " << error.what() << '\n' << std::flush;
        }
    }
    // The reader exits on QUIT or EOF; cancellation is scoped to its request.
    if (reader.joinable()) reader.join();
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::string model, token_text, dump_file, dump_dir, cuda_devices, cuda_budgets;
        int threads = 4, count = 1;
        int64_t context = 8192;
        bool server = false, inspect = false;
        Sampling settings; settings.top_k = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            const auto value = [&]() -> std::string {
                if (++i >= argc) throw std::runtime_error(a + " needs a value");
                return argv[i];
            };
            if (a == "--model") model = value();
            else if (a == "--tokens") token_text = value();
            else if (a == "--threads") threads = std::stoi(value());
            else if (a == "--max-context") context = std::stoll(value());
            else if (a == "--n") count = std::stoi(value());
            else if (a == "--dump-logits") dump_file = value();
            else if (a == "--dump-dir") dump_dir = value();
#ifdef STRATA_HY4_CUDA
            else if (a == "--cuda-devices") cuda_devices = value();
            else if (a == "--cuda-cache-mib") cuda_budgets = value();
#endif
            else if (a == "--serve") server = true;
            else if (a == "--inspect") inspect = true;
            else if (a == "--temp") sampling_key(settings, "temp", value());
            else if (a == "--help") {
                std::cout << "strata-hy4 --model <first.gguf> [--threads N] [--max-context N]\n"
                             "  --tokens ID,ID,... [--n N] [--dump-logits FILE] [--dump-dir DIR]\n"
                             "  --serve | --inspect\n";
#ifdef STRATA_HY4_CUDA
                std::cout << "  --cuda-devices 0,1 --cuda-cache-mib 8192,8192 (dense offload; experts and Engram remain on CPU)\n";
#endif
                return 0;
            } else throw std::runtime_error("unknown option: " + a);
        }
        if (model.empty() || count < 0) throw std::runtime_error("--model required; --n must be nonnegative");
        Engine engine(model, threads, context);
#ifdef STRATA_HY4_CUDA
        if (!cuda_devices.empty() || !cuda_budgets.empty()) {
            const auto devices = ids(cuda_devices), limits = ids(cuda_budgets);
            const std::vector<int> cards(devices.begin(), devices.end());
            std::vector<size_t> budgets; for (int32_t m : limits) budgets.push_back(static_cast<size_t>(m) << 20);
            engine.set_matmul(cuda_matmul(threads, cards, budgets));
        }
#endif
        if (inspect) {
            const auto& g = engine.geometry();
            std::cout << "{\"architecture\":\"hy_v4_flash\",\"layers\":" << g.layers
                      << ",\"experts_per_token\":" << g.used << ",\"model_context\":" << g.max_context
                      << ",\"context\":" << context << ",\"engram\":\"host\",\"vision\":false,\"speculation\":false}\n";
            return 0;
        }
        if (server) { serve(engine); return 0; }
        const auto prompt = ids(token_text);
        std::ofstream dump;
        if (!dump_file.empty()) { dump.open(dump_file, std::ios::binary); if (!dump) throw std::runtime_error("cannot create logits file"); }
        if (!dump_dir.empty()) std::filesystem::create_directories(dump_dir);
        Vec logits;
        size_t position = 0;
        const Trace trace = dump_dir.empty() ? Trace{} : Trace([&](const std::string& stage, int layer, std::span<const float> data) {
            const auto name = std::filesystem::path(dump_dir) / (std::to_string(position) + "-" + std::to_string(layer) + "-" + stage + ".f32");
            std::ofstream f(name, std::ios::binary);
            f.write(reinterpret_cast<const char*>(data.data()), data.size_bytes());
        });
        for (int32_t token : prompt) {
            logits = engine.forward(token, trace);
            if (dump) dump.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
            ++position;
        }
        std::mt19937_64 rng(settings.seed);
        for (int i = 0; i < count; ++i) {
            const int token = sample(logits, settings, engine.history(), rng);
            std::cout << (i ? "," : "") << token << std::flush;
            const auto* eos = engine.metadata().get("tokenizer.ggml.eos_token_id");
            if ((eos && token == static_cast<int>(eos->u)) || i + 1 == count) break;
            logits = engine.forward(token, trace); ++position;
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "strata-hy4: " << error.what() << '\n';
        return 2;
    }
}
