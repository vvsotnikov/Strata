#include "strata/hy4/vision.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char **argv) {
    try {
        strata::hy4::VisionOptions options;
        std::string mmproj, image, output, dump;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            auto value = [&]() {
                if (++i >= argc)
                    throw std::runtime_error(key + " requires a value");
                return std::string(argv[i]);
            };
            if (key == "--mmproj")
                mmproj = value();
            else if (key == "--gpu")
                options.cuda_device = 0;
            else if (key == "--device")
                options.cuda_device = std::stoi(value());
            else if (key == "--threads")
                options.threads = std::stoi(value());
            else if (key == "--max-tokens")
                options.max_tokens = std::stoi(value());
            else if (key == "--min-tokens")
                options.min_tokens = std::stoi(value());
            else if (key == "--image")
                image = value();
            else if (key == "--output")
                output = value();
            else if (key == "--dump-dir")
                dump = value();
            else if (key == "--precision") {
                auto p = value();
                if (p != "f32" && p != "f16")
                    throw std::runtime_error("precision must be f32 or f16");
                options.fp32 = p == "f32";
            } else if (key == "--model" || key == "--tower" || key == "--flash-attn")
                value();
            else if (key == "--version") {
                std::cout << "strata-hy4-vision\n";
                return 0;
            } else if (key == "--help") {
                std::cout << "strata-hy4-vision --mmproj FILE [--gpu] [--device N] [--threads N] "
                             "[--max-tokens N] [--min-tokens N]\n  [--image FILE --output FILE] [--dump-dir "
                             "DIR] [--precision f16|f32]\n";
                return 0;
            } else
                throw std::runtime_error("unknown vision option: " + key);
        }
        if (mmproj.empty())
            throw std::runtime_error("--mmproj is required");
        strata::hy4::VisionEncoder encoder(mmproj, options);
        const auto encode = [&](const std::string &path, const std::string &out) {
            auto start = std::chrono::steady_clock::now();
            strata::hy4::VisionTrace trace;
            if (!dump.empty()) {
                std::filesystem::create_directories(dump);
                trace = [&](const std::string &stage, int layer, std::span<const float> values, int) {
                    std::ofstream f(std::filesystem::path(dump) /
                                        (stage + "-" + std::to_string(layer) + ".f32"),
                                    std::ios::binary);
                    f.write(reinterpret_cast<const char *>(values.data()), values.size_bytes());
                };
            }
            auto result = encoder.encode_file(path, trace);
            result.write_sve(out);
            std::cout
                << "OK " << result.tokens() << ' ' << result.width << ' ' << result.height << ' '
                << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()
                << '\n'
                << std::flush;
        };
        if (!image.empty()) {
            if (output.empty())
                throw std::runtime_error("--image needs --output");
            encode(image, output);
            return 0;
        }
        std::cout << "READY " << encoder.output_dim() << " own\n" << std::flush;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT")
                break;
            try {
                std::istringstream request(line);
                std::string command, path, out, extra;
                if (!(request >> command >> path >> out) || command != "ENC" || (request >> extra))
                    throw std::runtime_error("expected ENC <image> <output>");
                encode(path, out);
            } catch (const std::exception &error) {
                std::cout << "ERR " << error.what() << '\n' << std::flush;
            }
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "strata-hy4-vision: " << error.what() << '\n';
        return 2;
    }
}
