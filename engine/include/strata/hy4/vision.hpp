#pragma once

#include "strata/hy4/engine.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>

namespace strata::hy4 {

struct VisionOptions {
    int threads = 4;
    int cuda_device = -1; // -1: CPU; ordinal within CUDA_VISIBLE_DEVICES otherwise
    int min_tokens = 0, max_tokens = 0;
    bool fp32 = false; // CUDA: exact F32 weights/GEMMs for numerical qualification
};

struct VisionEmbedding {
    Vec rows; // [height][width][dim], width includes the learned row newline
    int dim = 0, width = 0, height = 0;
    int patch_width = 0, patch_height = 0;
    size_t tokens() const { return static_cast<size_t>(width) * height; }
    std::array<int64_t, 3> position(size_t row, int64_t start) const;
    int64_t next_text_position(int64_t start) const;
    void write_sve(const std::string &path) const;
};

using VisionTrace = std::function<void(const std::string &, int, std::span<const float>, int)>;

class VisionEncoder {
  public:
    explicit VisionEncoder(const std::string &mmproj, VisionOptions options = {});
    ~VisionEncoder();
    VisionEncoder(const VisionEncoder &) = delete;
    VisionEncoder &operator=(const VisionEncoder &) = delete;
    int output_dim() const;
    VisionEmbedding encode_file(const std::string &image, const VisionTrace &trace = {});
    VisionEmbedding encode_rgb(std::span<const uint8_t> rgb, int width, int height,
                               const VisionTrace &trace = {});
    // Numerical-oracle entry: already normalized CHW pixels, both sides divisible by patch*merge.
    VisionEmbedding encode_normalized(std::span<const float> chw, int width, int height,
                                      const VisionTrace &trace = {});

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

namespace vision_detail {
struct Geometry {
    int dim = 0, heads = 0, ff = 0, layers = 0, patch = 0, merge = 0, grid = 0;
    int out = 0, hidden = 0, min_pixels = 0, max_pixels = 0;
    float eps = 0;
    std::array<float, 3> mean{}, stddev{};
};
// Exact live activation allocation for the current tiled CUDA encoder. Includes
// merger outputs and newline rows; excludes resident weights and cuBLAS state.
size_t cuda_workspace_bytes(const Geometry &, int patch_width, int patch_height);
bool cuda_memory_fits(size_t free_bytes, size_t new_weights, size_t workspace,
                      size_t conversion_temporary, size_t reserve_bytes);
using Lookup = std::function<const Tensor &(const std::string &)>;
class Backend {
  public:
    virtual ~Backend() = default;
    // An empty result admits the image. CUDA backends release prior scratch
    // before checking; rejection selects the native CPU encoder for this image.
    virtual std::string admission_error(int, int) { return {}; }
    // Patches: [patch_height*patch_width][3*patch*patch], planar within each patch.
    virtual Vec encode(const Vec &patches, const Vec &positions, int pw, int ph,
                       const VisionTrace &trace) = 0;
};
std::unique_ptr<Backend> cuda_backend(const Geometry &, Lookup, const VisionOptions &);
} // namespace vision_detail
} // namespace strata::hy4
