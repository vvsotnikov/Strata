#include "strata/hy4/vision.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace strata::hy4;

namespace {
void check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
template <class Action> void rejects(Action action) {
    try {
        action();
    } catch (const std::runtime_error &) {
        return;
    }
    throw std::runtime_error("expected invalid image position to be rejected");
}
} // namespace

int main() {
    // An actual 64x64 image has four 32x32 merged patches plus a newline
    // embedding at the end of each row. Newlines occupy an image-pad position.
    VisionEmbedding image;
    image.width = 3;
    image.height = 2;
    const int64_t image_start_token_position = 9;
    const int64_t first_image_position = image_start_token_position + 1;
    const std::array<std::array<int64_t, 3>, 6> expected{{
        {10, 10, 10},
        {10, 10, 11},
        {10, 10, 12},
        {10, 11, 10},
        {10, 11, 11},
        {10, 11, 12},
    }};
    check(image.tokens() == expected.size(), "image/newline token count");
    for (size_t row = 0; row < expected.size(); ++row)
        check(image.position(row, first_image_position) == expected[row], "image three-axis position");
    check(image.next_text_position(first_image_position) == 13, "image end position");
    check(image.next_text_position(first_image_position) + 1 == 14, "text after image end position");
    image.width = 2;
    image.height = 5;
    check(image.next_text_position(10) == 15, "tall image advancement uses height");
    check(image.position(9, 10) == std::array<int64_t, 3>{10, 14, 11}, "tall image last newline");
    rejects([&] { image.position(image.tokens(), 10); });
    rejects([&] { image.position(0, -1); });
    rejects([&] { image.next_text_position(std::numeric_limits<int64_t>::max()); });
    image.width = 0;
    rejects([&] { image.next_text_position(10); });
    // Official [12,10,10] interleaving assigns the final two frequency
    // pairs to time; it does not repeat H,W after pair 29.
    constexpr std::array<int, 32> axes{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
                                       1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0};
    const RopePosition position{7, 13, 29};
    Vec rotary(64, 1.f);
    rope(rotary, position, 10000.f);
    for (size_t pair = 0; pair < axes.size(); ++pair) {
        const float angle = position[axes[pair]] * std::pow(10000.f, -2.f * pair / 64.f);
        check(std::abs(rotary[2 * pair] - (std::cos(angle) - std::sin(angle))) < 1e-6f, "MRoPE even channel");
        check(std::abs(rotary[2 * pair + 1] - (std::sin(angle) + std::cos(angle))) < 1e-6f,
              "MRoPE odd channel");
    }
    std::cout << "Hy4 image boundaries, newline rows and three-axis positions passed\n";
}
