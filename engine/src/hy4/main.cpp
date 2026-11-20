#include "strata/hy4/engine.hpp"
#include "strata/qwen35/sampling.hpp"
#ifdef STRATA_HY4_CUDA
#include "strata/hy4/device.hpp"
#include "strata/hy4/server.hpp"
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
double milliseconds(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

std::vector<int32_t> ids(const std::string &text) {
    if (text.empty() || text.back() == ',')
        throw std::runtime_error("empty value in comma-separated IDs/budgets");
    std::vector<int32_t> out;
    std::istringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        size_t end;
        const long long value = std::stoll(item, &end);
        if (end != item.size() || value < 0 || value > INT32_MAX)
            throw std::runtime_error("invalid token ID");
        out.push_back(static_cast<int32_t>(value));
    }
    if (out.empty())
        throw std::runtime_error("empty token prompt");
    return out;
}

int sample(Vec logits, const Sampling &sampling, const std::vector<int32_t> &history, std::mt19937_64 &rng) {
    strata::qwen35::apply_penalties(logits, sampling, history);
    for (float value : logits)
        if (!std::isfinite(value))
            throw std::runtime_error("sampling penalties overflowed logits");
    if (sampling.temperature <= 0)
        return std::max_element(logits.begin(), logits.end()) - logits.begin();
    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
    if (sampling.top_k > 0 && static_cast<size_t>(sampling.top_k) < order.size())
        order.resize(sampling.top_k);
    const float maximum = logits[order[0]];
    std::vector<double> probs;
    double total = 0;
    for (int id : order) {
        const double p = std::exp(double((logits[id] - maximum) / sampling.temperature));
        if (p < sampling.min_p)
            break;
        probs.push_back(p);
        total += p;
    }
    size_t take = probs.size();
    if (sampling.top_p < 1) {
        double cumulative = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumulative += probs[i];
            if (cumulative >= total * sampling.top_p) {
                take = i + 1;
                break;
            }
        }
    }
    if (!take)
        throw std::runtime_error("sampling discarded every token");
    probs.resize(take);
    std::discrete_distribution<size_t> distribution(probs.begin(), probs.end());
    return order[distribution(rng)];
}

void sampling_key(Sampling &s, const std::string &key, const std::string &value) {
    size_t end = 0;
    if (key == "seed") {
        if (value.empty() || value[0] == '-')
            throw std::runtime_error("invalid sampling seed");
        s.seed = std::stoull(value, &end);
        if (end != value.size())
            throw std::runtime_error("invalid sampling seed");
        return;
    }
    const double number = std::stod(value, &end);
    if (end != value.size() || !std::isfinite(number) || std::abs(number) > std::numeric_limits<float>::max())
        throw std::runtime_error("invalid sampling parameter");
    if ((key == "top_k" || key == "last_n" || key == "penalty_last_n") &&
        (number < 0 || number > INT32_MAX || number != std::floor(number)))
        throw std::runtime_error("sampling integer out of range");
    if ((key == "temp" || key == "temperature"))
        s.temperature = number;
    else if (key == "top_p")
        s.top_p = number;
    else if (key == "top_k")
        s.top_k = static_cast<int>(number);
    else if (key == "min_p")
        s.min_p = number;
    else if ((key == "repeat" || key == "penalty_repeat"))
        s.repeat = number;
    else if ((key == "last_n" || key == "penalty_last_n"))
        s.last_n = static_cast<int>(number);
    else if ((key == "freq" || key == "penalty_freq"))
        s.freq = number;
    else if ((key == "present" || key == "penalty_present"))
        s.present = number;
    else if (key != "prefill")
        throw std::runtime_error("unsupported sampling key: " + key);
    if (s.temperature < 0 || s.top_p <= 0 || s.top_p > 1 || s.min_p < 0 || s.min_p > 1 || s.repeat <= 0)
        throw std::runtime_error("sampling parameter out of range");
}

void serve(Engine &engine) {
    std::mutex mutex;
    std::condition_variable cv;
    struct Request {
        std::string line;
        std::shared_ptr<std::atomic<bool>> cancel;
    };
    std::deque<Request> queue;
    std::shared_ptr<std::atomic<bool>> active;
    bool eof = false;
    std::thread reader([&] {
        std::string line;
        while (std::getline(std::cin, line)) {
            {
                std::lock_guard lock(mutex);
                if (line == "STOP") {
                    if (active)
                        active->store(true);
                    else if (!queue.empty())
                        queue.front().cancel->store(true);
                    continue;
                }
                queue.push_back({line, std::make_shared<std::atomic<bool>>(false)});
            }
            cv.notify_one();
            if (line == "QUIT")
                break;
        }
        {
            std::lock_guard lock(mutex);
            eof = true;
        }
        cv.notify_one();
    });
    std::string end_tokens;
    for (const char *key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"}) {
        if (const auto *value = engine.metadata().get(key)) {
            if (!end_tokens.empty()) end_tokens += ',';
            end_tokens += std::to_string(value->u);
        }
    }
    if (!end_tokens.empty()) std::cout << "INFO eos_ids=" << end_tokens << '\n';
    std::cout << "READY " << engine.max_context() << " stop\n" << std::flush;
    for (;;) {
        Request request;
        {
            std::unique_lock lock(mutex);
            cv.wait(lock, [&] { return !queue.empty() || eof; });
            if (queue.empty())
                break;
            request = std::move(queue.front());
            queue.pop_front();
            active = request.cancel;
        }
        const std::string &line = request.line;
        const auto &stop = *request.cancel;
        if (line == "QUIT")
            break;
        try {
            std::istringstream input(line);
            std::string command, part, token_text;
            int count;
            if (!(input >> command >> count) || command != "GEN" || count < 0)
                throw std::runtime_error("expected GEN <max_new> [sampling keys] <id,id,...>");
            Sampling settings;
            settings.top_k = 0;
            int64_t pen_from = -1;
            while (input >> part) {
                const size_t eq = part.find('=');
                if (eq != std::string::npos && part.substr(0, eq) == "pen_from") {
                    size_t end;
                    const auto value = part.substr(eq + 1);
                    pen_from = std::stoll(value, &end);
                    if (end != value.size() || pen_from < 0)
                        throw std::runtime_error("invalid pen_from");
                } else if (eq != std::string::npos)
                    sampling_key(settings, part.substr(0, eq), part.substr(eq + 1));
                else if (token_text.empty())
                    token_text = part;
                else
                    throw std::runtime_error("extra GEN argument");
            }
            const auto prompt = ids(token_text);
            if (prompt.size() > static_cast<size_t>(engine.max_context()))
                throw std::runtime_error("prompt exceeds context");
            if (pen_from >= static_cast<int64_t>(prompt.size()))
                throw std::runtime_error("pen_from exceeds prompt");
            const auto *penalty_prompt = std::getenv("STRATA_PENALTY_PROMPT");
            auto penalty_history = strata::qwen35::penalty_history_start(
                prompt, penalty_prompt && std::string(penalty_prompt) == "1");
            if (pen_from >= 0)
                penalty_history.assign(prompt.begin() + pen_from, prompt.end());
            engine.reset();
            const auto t0 = Clock::now();
            Vec logits;
            bool cancelled = false;
            size_t prompt_read = 0;
            for (size_t i = 0; i < prompt.size(); ++i) {
                if (stop.load()) {
                    cancelled = true;
                    break;
                }
                logits = engine.forward(prompt[i]);
                ++prompt_read;
                if ((i + 1) % 8 == 0 || i + 1 == prompt.size())
                    std::cout << "PP " << i + 1 << ' ' << prompt.size() << ' ' << milliseconds(t0) << '\n'
                              << std::flush;
            }
            const double prompt_ms = milliseconds(t0);
            const auto d0 = Clock::now();
            std::mt19937_64 rng(settings.seed);
            int produced = 0;
            std::string finish = cancelled ? "cancel" : "length";
            for (; produced < count && !cancelled; ++produced) {
                if (stop.load()) {
                    finish = "cancel";
                    break;
                }
                const int token = sample(logits, settings, penalty_history, rng);
                penalty_history.push_back(token);
                std::cout << "T " << token << '\n' << std::flush;
                const auto *eos = engine.metadata().get("tokenizer.ggml.eos_token_id");
                if (eos && token == static_cast<int>(eos->u)) {
                    ++produced;
                    finish = "stop";
                    break;
                }
                if (engine.history().size() >= static_cast<size_t>(engine.max_context())) {
                    ++produced;
                    break;
                }
                if (produced + 1 < count)
                    logits = engine.forward(token);
            }
            {
                std::lock_guard lock(mutex);
                active.reset();
            }
            std::cout << "DONE " << produced << ' ' << prompt.size() << ' ' << prompt_ms << ' '
                      << milliseconds(d0) << ' ' << finish << " 0 0 0 0 0 0 0 0.0 " << prompt_read << " 0\n"
                      << std::flush;
        } catch (const std::exception &error) {
            engine.reset();
            {
                std::lock_guard lock(mutex);
                active.reset();
            }
            std::cout << "ERR " << error.what() << '\n' << std::flush;
        }
    }
    // The reader exits on QUIT or EOF; cancellation is scoped to its request.
    if (reader.joinable())
        reader.join();
}
} // namespace

int main(int argc, char **argv) {
    try {
        std::string model, token_text, dump_file, dump_dir, cuda_devices, cuda_budgets, expert_budgets,
            mmproj;
        int slots = 1, prefill_chunk = 8, mtp = 0;
        int64_t pool = 0;
        size_t reserve_mib = 1024;
        std::string kv_location = "ram", kv_type;
        bool pool_given = false;
        int threads = 4, count = 1;
        int64_t context = 8192;
        bool server = false, inspect = false, capabilities = false;
        Sampling settings;
        settings.top_k = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            const auto value = [&]() -> std::string {
                if (++i >= argc)
                    throw std::runtime_error(a + " needs a value");
                return argv[i];
            };
            const auto integer_value = [&](int64_t minimum, int64_t maximum) -> int64_t {
                const auto text = value();
                size_t used = 0;
                int64_t result;
                try {
                    result = std::stoll(text, &used);
                } catch (const std::exception &) {
                    throw std::runtime_error(a + " requires an integer");
                }
                if (used != text.size() || result < minimum || result > maximum)
                    throw std::runtime_error(a + " integer is out of range");
                return result;
            };
            if (a == "--model")
                model = value();
            else if (a == "--tokens")
                token_text = value();
            else if (a == "--threads")
                threads = int(integer_value(1, 256));
            else if (a == "--max-context")
                context = integer_value(1, int64_t(1) << 26);
            else if (a == "--n")
                count = int(integer_value(0, INT32_MAX));
            else if (a == "--dump-logits")
                dump_file = value();
            else if (a == "--dump-dir")
                dump_dir = value();
            else if (a == "--cuda-devices")
                cuda_devices = value();
            else if (a == "--cuda-cache-mib")
                cuda_budgets = value();
            else if (a == "--expert-cache-mib")
                expert_budgets = value();
            else if (a == "--slots")
                slots = int(integer_value(1, 8));
            else if (a == "--context-pool") {
                pool = integer_value(1, int64_t(8) << 26);
                pool_given = true;
            } else if (a == "--prefill-chunk")
                prefill_chunk = int(integer_value(1, 8));
            else if (a == "--mtp")
                mtp = int(integer_value(0, 5));
            else if (a == "--no-mtp")
                mtp = 0;
            else if (a == "--mmproj")
                mmproj = value();
            else if (a == "--kv-location")
                kv_location = value();
            else if (a == "--kv-type")
                kv_type = value();
            else if (a == "--vram-reserve-mib")
                reserve_mib = size_t(integer_value(0, 1048576));
            else if (a == "--capabilities")
                capabilities = true;
            else if (a == "--serve")
                server = true;
            else if (a == "--inspect")
                inspect = true;
            else if (a == "--temp")
                sampling_key(settings, "temp", value());
            else if (a == "--help") {
                std::cout << "strata-hy4 --model <first.gguf> [--threads N] [--max-context N]\n"
                             "  --tokens ID,ID,... [--n N] [--dump-logits FILE] [--dump-dir DIR]\n"
                             "  --serve | --inspect\n";
#ifdef STRATA_HY4_CUDA
                std::cout << "  --cuda-devices 0,1 [--cuda-cache-mib M,M] [--expert-cache-mib M,M]\n";
#endif
                std::cout << "  --slots N --context-pool N --prefill-chunk 1..8 --mtp 0..5\n"
                             "  --kv-location ram|gpu --kv-type f16|f32 --vram-reserve-mib N\n"
                             "  --mmproj FILE (image embedding input) | --capabilities (no model load)\n";
                return 0;
            } else
                throw std::runtime_error("unknown option: " + a);
        }
        if (capabilities) {
#ifdef STRATA_HY4_CUDA
            std::cout
                << R"({"architecture":"hy_v4_flash","backend":"cuda","features":{"gpu_experts":true,"vision":true,"mtp":true,"batching":true,"prompt_cache":true,"kv_gpu":true,"kv_ram":true}})"
                << '\n';
#else
            std::cout
                << R"({"architecture":"hy_v4_flash","backend":"cpu","features":{"gpu_experts":false,"vision":false,"mtp":false,"batching":false,"prompt_cache":false,"kv_gpu":false,"kv_ram":true}})"
                << '\n';
#endif
            return 0;
        }
        if (model.empty() || count < 0 || context <= 0 || threads < 1 || threads > 256 || slots < 1 ||
            slots > 8 || prefill_chunk < 1 || prefill_chunk > 8 || mtp < 0 || mtp > 5 ||
            (kv_location != "ram" && kv_location != "gpu") ||
            (!kv_type.empty() && kv_type != "f16" && kv_type != "f32"))
            throw std::runtime_error(
                "invalid model, context, threading, batching, MTP or KV option; see --help");
        if (!pool_given)
            pool = context;
        if (pool < context)
            throw std::runtime_error("context pool must cover one full request");
        if (!mmproj.empty() && !std::filesystem::is_regular_file(mmproj))
            throw std::runtime_error("projector file not found");
        const bool device_requested = !cuda_devices.empty();
        if (kv_type.empty())
            kv_type = device_requested ? "f16" : "f32";
        if (!device_requested && (kv_type != "f32" || pool != context))
            throw std::runtime_error("CPU reference uses F32 KV in one context; it has no shared KV pool");
        if (!device_requested && (!cuda_budgets.empty() || !expert_budgets.empty() || slots != 1 || mtp ||
                                  !mmproj.empty() || kv_location != "ram"))
            throw std::runtime_error(
                "GPU serving features require --cuda-devices; CPU mode is the serial reference backend");
#ifndef STRATA_HY4_CUDA
        if (device_requested)
            throw std::runtime_error("this executable was built without CUDA");
#endif
        Engine engine(model, threads, context);
#ifdef STRATA_HY4_CUDA
        std::unique_ptr<DeviceModel> device;
        if (device_requested && !inspect) {
            DeviceOptions options;
            const auto cards = ids(cuda_devices);
            options.devices.assign(cards.begin(), cards.end());
            auto limits = [&](const std::string &text) {
                std::vector<size_t> result;
                if (!text.empty())
                    for (auto m : ids(text))
                        result.push_back(size_t(m) << 20);
                return result;
            };
            options.dense_bytes = limits(cuda_budgets);
            options.expert_bytes = limits(expert_budgets);
            options.slots = slots;
            options.batch = 8;
            options.context_pool = pool;
            options.max_context = context;
            options.host_kv = kv_location == "ram";
            options.f32_kv = kv_type == "f32";
            options.vram_reserve = reserve_mib << 20;
            device = device_model(engine, options);
        }
#endif
        if (inspect) {
            const auto &g = engine.geometry();
            std::cout << "{\"architecture\":\"hy_v4_flash\",\"layers\":" << g.layers
                      << ",\"experts_per_token\":" << g.used << ",\"model_context\":" << g.max_context
                      << ",\"context\":" << context << ",\"context_pool\":" << pool
                      << ",\"engram\":\"host\",\"mtp_layers\":" << g.draft_layers << "}\n";
            return 0;
        }
        if (server) {
#ifdef STRATA_HY4_CUDA
            if (device) {
                ServerOptions options;
                options.mtp = mtp;
                options.prefill_chunk = prefill_chunk;
                options.max_context = context;
                options.vision = !mmproj.empty();
                return serve_device(*device, engine, options);
            }
#endif
            serve(engine);
            return 0;
        }
        const auto prompt = ids(token_text);
        std::ofstream dump;
        if (!dump_file.empty()) {
            dump.open(dump_file, std::ios::binary);
            if (!dump)
                throw std::runtime_error("cannot create logits file");
        }
        if (!dump_dir.empty())
            std::filesystem::create_directories(dump_dir);
        Vec logits;
        size_t position = 0;
        const Trace trace =
            dump_dir.empty()
                ? Trace{}
                : Trace([&](const std::string &stage, int layer, std::span<const float> data) {
                      const auto name =
                          std::filesystem::path(dump_dir) /
                          (std::to_string(position) + "-" + std::to_string(layer) + "-" + stage + ".f32");
                      std::ofstream f(name, std::ios::binary);
                      f.write(reinterpret_cast<const char *>(data.data()), data.size_bytes());
                  });
        auto forward = [&](int32_t token) {
#ifdef STRATA_HY4_CUDA
            if (device) {
                DeviceInput row;
                row.token = token;
                return device->forward(std::span(&row, 1), trace).front();
            }
#endif
            return engine.forward(token, trace);
        };
        for (int32_t token : prompt) {
            logits = forward(token);
            if (dump)
                dump.write(reinterpret_cast<const char *>(logits.data()), logits.size() * sizeof(float));
            ++position;
        }
        std::mt19937_64 rng(settings.seed);
        for (int i = 0; i < count; ++i) {
            const int token =
                sample(logits, settings,
#ifdef STRATA_HY4_CUDA
                       device ? std::vector<int32_t>(device->history(0).begin(), device->history(0).end()) :
#endif
                              engine.history(),
                       rng);
            std::cout << (i ? "," : "") << token << std::flush;
            const auto *eos = engine.metadata().get("tokenizer.ggml.eos_token_id");
            if ((eos && token == static_cast<int>(eos->u)) || i + 1 == count)
                break;
            logits = forward(token);
            ++position;
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "strata-hy4: " << error.what() << '\n';
        return 2;
    }
}
