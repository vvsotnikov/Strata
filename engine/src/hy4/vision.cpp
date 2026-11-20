#include "strata/hy4/vision.hpp"
#include "../../../terminal/tools/vision/own/image_prep.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

namespace strata::hy4 {
namespace {
using G = vision_detail::Geometry;
using Lookup = vision_detail::Lookup;

std::string layer_name(int layer, const std::string &suffix) {
    return "v.blk." + std::to_string(layer) + "." + suffix;
}
float gelu(float x) { return .5f * x * (1.f + std::erf(x * .7071067811865475244f)); }
void require(bool yes, const std::string &message) {
    if (!yes)
        throw std::runtime_error("Hy4 vision: " + message);
}

void normalize(const Vec &in, Vec &out, int n, const Vec &w, const Vec *b, float eps, bool rms_only,
               int threads) {
    const int rows = in.size() / n;
    out.resize(in.size());
#pragma omp parallel for num_threads(threads) schedule(static)
    for (int r = 0; r < rows; ++r) {
        const float *x = in.data() + static_cast<size_t>(r) * n;
        double sum = 0;
        if (!rms_only)
            for (int i = 0; i < n; ++i)
                sum += x[i];
        const float mean = rms_only ? 0 : static_cast<float>(sum / n);
        double squares = 0;
        for (int i = 0; i < n; ++i) {
            const double v = x[i] - mean;
            squares += v * v;
        }
        const float scale = 1.f / std::sqrt(static_cast<float>(squares / n) + eps);
        for (int i = 0; i < n; ++i)
            out[static_cast<size_t>(r) * n + i] = ((x[i] - mean) * scale) * w[i] + (b ? (*b)[i] : 0);
    }
}

class Cpu final : public vision_detail::Backend {
  public:
    Cpu(const G &geometry, Lookup lookup, int threads)
        : g_(geometry), lookup_(std::move(lookup)), threads_(threads) {}
    Vec encode(const Vec &patches, const Vec &positions, int pw, int ph, const VisionTrace &trace) override {
        const int count = pw * ph, d = g_.dim, hd = d / g_.heads;
        Vec x = mm("v.patch_embd.weight", patches, count, 3 * g_.patch * g_.patch, d, "v.patch_embd.bias");
        for (size_t i = 0; i < x.size(); ++i)
            x[i] += positions[i];
        if (trace)
            trace("vision_patch", -1, x, d);
        for (int layer = 0; layer < g_.layers; ++layer) {
            const auto n = [&](const char *suffix) { return layer_name(layer, suffix); };
            Vec norm;
            const Vec &b1 = weight(n("ln1.bias"));
            normalize(x, norm, d, weight(n("ln1.weight")), &b1, g_.eps, false, threads_);
            Vec q = mm(n("attn_q.weight"), norm, count, d, d, n("attn_q.bias"));
            Vec k = mm(n("attn_k.weight"), norm, count, d, d, n("attn_k.bias"));
            Vec v = mm(n("attn_v.weight"), norm, count, d, d, n("attn_v.bias"));
            Vec context(x.size());
#pragma omp parallel for num_threads(threads_) schedule(static)
            for (int pair = 0; pair < count * g_.heads; ++pair) {
                const int row = pair / g_.heads, head = pair % g_.heads;
                const float *query = q.data() + static_cast<size_t>(row) * d + head * hd;
                Vec scores(count);
                float maximum = -std::numeric_limits<float>::infinity();
                for (int key = 0; key < count; ++key) {
                    double sum = 0;
                    for (int j = 0; j < hd; ++j)
                        sum +=
                            static_cast<double>(query[j]) * k[static_cast<size_t>(key) * d + head * hd + j];
                    scores[key] = static_cast<float>(sum) / std::sqrt(static_cast<float>(hd));
                    maximum = std::max(maximum, scores[key]);
                }
                double denominator = 0;
                for (float &s : scores) {
                    s = std::exp(s - maximum);
                    denominator += s;
                }
                for (int j = 0; j < hd; ++j) {
                    double sum = 0;
                    for (int key = 0; key < count; ++key)
                        sum += static_cast<double>(scores[key]) *
                               v[static_cast<size_t>(key) * d + head * hd + j];
                    context[static_cast<size_t>(row) * d + head * hd + j] =
                        static_cast<float>(sum / denominator);
                }
            }
            Vec out = mm(n("attn_out.weight"), context, count, d, d, n("attn_out.bias"));
            for (size_t i = 0; i < x.size(); ++i)
                x[i] += out[i];
            const Vec &b2 = weight(n("ln2.bias"));
            normalize(x, norm, d, weight(n("ln2.weight")), &b2, g_.eps, false, threads_);
            Vec up = mm(n("ffn_up.weight"), norm, count, d, g_.ff, n("ffn_up.bias"));
            for (float &z : up)
                z = gelu(z);
            out = mm(n("ffn_down.weight"), up, count, g_.ff, d, n("ffn_down.bias"));
            for (size_t i = 0; i < x.size(); ++i)
                x[i] += out[i];
            if (trace)
                trace("vision_layer", layer, x, d);
        }
        Vec norm;
        normalize(x, norm, d, weight("mm.pre_norm.weight"), nullptr, g_.eps, true, threads_);
        const int ow = pw / g_.merge, oh = ph / g_.merge, merged = ow * oh, inner = d * g_.merge * g_.merge;
        Vec packed(static_cast<size_t>(merged) * inner);
        for (int y = 0; y < oh; ++y)
            for (int x0 = 0; x0 < ow; ++x0)
                for (int c = 0; c < d; ++c)
                    for (int dy = 0; dy < g_.merge; ++dy)
                        for (int dx = 0; dx < g_.merge; ++dx)
                            packed[(static_cast<size_t>(y) * ow + x0) * inner + c * g_.merge * g_.merge +
                                   dy * g_.merge + dx] =
                                norm[(static_cast<size_t>(y * g_.merge + dy) * pw + x0 * g_.merge + dx) * d +
                                     c];
        Vec mid = mm("mm.0.weight", packed, merged, inner, g_.hidden, "mm.0.bias");
        for (float &z : mid)
            z = gelu(z);
        Vec projected = mm("mm.2.weight", mid, merged, g_.hidden, g_.out, "mm.2.bias");
        if (trace)
            trace("vision_projected", -1, projected, g_.out);
        const Vec &newline = weight("v.image_newline");
        Vec result(static_cast<size_t>(ow + 1) * oh * g_.out);
        for (int y = 0; y < oh; ++y) {
            std::copy_n(projected.begin() + static_cast<size_t>(y) * ow * g_.out,
                        static_cast<size_t>(ow) * g_.out,
                        result.begin() + static_cast<size_t>(y) * (ow + 1) * g_.out);
            std::copy(newline.begin(), newline.end(),
                      result.begin() + (static_cast<size_t>(y) * (ow + 1) + ow) * g_.out);
        }
        return result;
    }

  private:
    const Vec &weight(const std::string &name) {
        auto it = weights_.find(name);
        if (it == weights_.end())
            it = weights_.emplace(name, lookup_(name).values()).first;
        return it->second;
    }
    Vec mm(const std::string &name, const Vec &x, int count, int inner, int outer, const std::string &bias) {
        const Vec &w = weight(name);
        const Vec &b = weight(bias);
        require(w.size() == static_cast<size_t>(inner) * outer &&
                    x.size() == static_cast<size_t>(count) * inner && b.size() == static_cast<size_t>(outer),
                "matrix dimensions: " + name);
        Vec out(static_cast<size_t>(count) * outer);
#pragma omp parallel for num_threads(threads_) schedule(static)
        for (int row = 0; row < count; ++row) {
            const float *input = x.data() + static_cast<size_t>(row) * inner;
            for (int col = 0; col < outer; ++col) {
                const float *weights = w.data() + static_cast<size_t>(col) * inner;
                float sum = 0;
#pragma omp simd reduction(+ : sum)
                for (int k = 0; k < inner; ++k)
                    sum += input[k] * weights[k];
                out[static_cast<size_t>(row) * outer + col] = sum + b[col];
            }
        }
        return out;
    }
    G g_;
    Lookup lookup_;
    int threads_;
    std::map<std::string, Vec> weights_;
};

int round_even_multiple(int n, int factor) {
    int q = n / factor, r = n % factor;
    if (r * 2 > factor || (r * 2 == factor && (q & 1)))
        ++q;
    return q * factor;
}
} // namespace

size_t vision_detail::cuda_workspace_bytes(const Geometry &g, int pw, int ph) {
    require(pw > 0 && ph > 0 && g.merge > 0 && pw % g.merge == 0 && ph % g.merge == 0,
            "CUDA workspace image geometry");
    const size_t count = size_t(pw) * ph, merged = count / (g.merge * g.merge);
    const size_t inner = size_t(g.dim) * g.merge * g.merge;
    const size_t floats = count * (3 * g.patch * g.patch + 12 * g.dim + g.ff) +
                          count * std::min<size_t>(count, 128) * g.heads + merged * inner +
                          merged * g.hidden + merged * g.out +
                          size_t(pw / g.merge + 1) * (ph / g.merge) * g.out;
    const size_t halves = count * std::max<size_t>({size_t(3 * g.patch * g.patch), size_t(g.dim),
                                                  size_t(g.ff), size_t(g.hidden), inner});
    return floats * sizeof(float) + halves * sizeof(uint16_t);
}
bool vision_detail::cuda_memory_fits(size_t available, size_t weights, size_t workspace,
                                     size_t temporary, size_t reserve) {
    // Subtract rather than sum so a malformed or huge estimate cannot wrap.
    for (const size_t bytes : {weights, workspace, temporary, reserve}) {
        if (bytes > available)
            return false;
        available -= bytes;
    }
    return true;
}

struct VisionEncoder::Impl {
    GgufModel model;
    std::map<std::string, Tensor> tensors;
    G g;
    VisionOptions options;
    std::unique_ptr<vision_detail::Backend> backend;
    std::unique_ptr<vision_detail::Backend> cpu_fallback;
    explicit Impl(const std::string &path, VisionOptions o) : model(GgufModel::open(path, true)), options(o) {
        require(o.threads >= 1 && o.threads <= 256 && o.min_tokens >= 0 && o.max_tokens >= 0,
                "invalid options");
        const auto &meta = model.meta();
        const auto text = [&](const char *k) {
            const auto *v = meta.get(k);
            require(v && v->type == MetaType::STRING, std::string("missing ") + k);
            return v->s;
        };
        const auto integer = [&](const char *k, int low, int high) {
            const auto *v = meta.get(k);
            require(v && v->is_num() && v->num() >= low && v->num() <= high &&
                        v->num() == std::floor(v->num()),
                    std::string("invalid ") + k);
            return static_cast<int>(v->num());
        };
        require(text("general.architecture") == "clip" && text("clip.projector_type") == "hy_v4_flash",
                "expected hy_v4_flash projector");
        g.dim = integer("clip.vision.embedding_length", 1, 8192);
        g.heads = integer("clip.vision.attention.head_count", 1, 128);
        g.ff = integer("clip.vision.feed_forward_length", 1, 65536);
        g.layers = integer("clip.vision.block_count", 1, 128);
        g.patch = integer("clip.vision.patch_size", 1, 64);
        g.merge = integer("clip.vision.spatial_merge_size", 1, 8);
        g.out = integer("clip.vision.projection_dim", 1, 16384);
        g.min_pixels = integer("clip.vision.image_min_pixels", 1, 1 << 26);
        g.max_pixels = integer("clip.vision.image_max_pixels", g.min_pixels, 1 << 26);
        require(g.dim % g.heads == 0, "head geometry");
        const auto *eps = meta.get("clip.vision.attention.layer_norm_epsilon");
        require(eps && eps->is_num() && eps->num() > 0 && std::isfinite(eps->num()), "norm epsilon");
        g.eps = eps->num();
        for (const auto [key, dst] :
             {std::pair{"clip.vision.image_mean", &g.mean}, std::pair{"clip.vision.image_std", &g.stddev}}) {
            const auto *v = meta.get(key);
            require(v && v->type == MetaType::ARRAY && v->items.size() == 3, std::string("invalid ") + key);
            for (int i = 0; i < 3; ++i) {
                const double n = v->items[i].num();
                require(std::isfinite(n), std::string("nonfinite ") + key);
                (*dst)[i] = n;
            }
        }
        for (float s : g.stddev)
            require(s > 0, "image standard deviation must be positive");
        for (size_t shard = 0; shard < model.size(); ++shard)
            for (const auto &t : model.shard(shard).tensors()) {
                require(model.in_bounds(t, shard), "truncated tensor " + t.name);
                require(t.type == 0 || t.type == 1 || t.type == 30, "unsupported tensor type for " + t.name);
                tensors.emplace(t.name, Tensor{&t, model.shard(shard).tensor_data(t)});
            }
        const auto shape = [&](const std::string &name, std::initializer_list<uint64_t> dims) {
            require(at(name).info->shape == std::vector<uint64_t>(dims), "tensor shape: " + name);
        };
        const uint64_t d = g.dim, f = g.ff, p = g.patch, merge = g.merge;
        shape("v.patch_embd.weight", {p, p, 3, d});
        shape("v.patch_embd.bias", {d});
        const auto &pos = at("v.position_embd.weight");
        require(pos.info->shape.size() == 2 && pos.info->shape[0] == d, "position table");
        g.grid = std::lround(std::sqrt(static_cast<double>(pos.info->shape[1])));
        require(static_cast<uint64_t>(g.grid) * g.grid == pos.info->shape[1], "position grid must be square");
        for (int i = 0; i < g.layers; ++i) {
            for (const char *norm : {"ln1", "ln2"})
                for (const char *suffix : {"weight", "bias"})
                    shape(layer_name(i, std::string(norm) + "." + suffix), {d});
            for (const char *proj : {"attn_q", "attn_k", "attn_v", "attn_out"}) {
                shape(layer_name(i, std::string(proj) + ".weight"), {d, d});
                shape(layer_name(i, std::string(proj) + ".bias"), {d});
            }
            shape(layer_name(i, "ffn_up.weight"), {d, f});
            shape(layer_name(i, "ffn_up.bias"), {f});
            shape(layer_name(i, "ffn_down.weight"), {f, d});
            shape(layer_name(i, "ffn_down.bias"), {d});
        }
        shape("mm.pre_norm.weight", {d});
        const auto &m0 = at("mm.0.weight");
        require(m0.info->shape.size() == 4, "merger convolution");
        require(m0.info->shape[3] > 0 && m0.info->shape[3] <= 65536, "merger hidden width");
        g.hidden = m0.info->shape[3];
        shape("mm.0.weight", {merge, merge, d, uint64_t(g.hidden)});
        shape("mm.0.bias", {uint64_t(g.hidden)});
        shape("mm.2.weight", {1, 1, uint64_t(g.hidden), uint64_t(g.out)});
        shape("mm.2.bias", {uint64_t(g.out)});
        shape("v.image_newline", {uint64_t(g.out)});
        Lookup lookup = [this](const std::string &n) -> const Tensor & { return at(n); };
        if (o.cuda_device >= 0) {
#ifdef STRATA_HY4_CUDA
            backend = vision_detail::cuda_backend(g, lookup, o);
#else
            throw std::runtime_error("Hy4 vision: this build has no CUDA backend");
#endif
        } else
            backend = std::make_unique<Cpu>(g, lookup, o.threads);
    }
    const Tensor &at(const std::string &n) const {
        auto it = tensors.find(n);
        require(it != tensors.end(), "missing tensor " + n);
        return it->second;
    }
};

VisionEncoder::VisionEncoder(const std::string &mmproj, VisionOptions options)
    : impl_(std::make_unique<Impl>(mmproj, options)) {}
VisionEncoder::~VisionEncoder() = default;
int VisionEncoder::output_dim() const { return impl_->g.out; }

VisionEmbedding VisionEncoder::encode_file(const std::string &path, const VisionTrace &trace) {
    strata_vit::ImageU8 image;
    std::string error;
    require(strata_vit::image_load_file(path, image, error), error);
    return encode_rgb(image.rgb, image.w, image.h, trace);
}

VisionEmbedding VisionEncoder::encode_rgb(std::span<const uint8_t> rgb, int width, int height,
                                          const VisionTrace &trace) {
    const auto &g = impl_->g;
    const auto &o = impl_->options;
    require(width > 0 && height > 0 && static_cast<uint64_t>(width) * height * 3 == rgb.size(),
            "invalid RGB image");
    require(static_cast<double>(std::max(width, height)) / std::min(width, height) <= 200,
            "image aspect ratio exceeds 200");
    const int factor = g.patch * g.merge;
    int min_pixels = g.min_pixels, max_pixels = g.max_pixels;
    const int64_t area_per_token = int64_t(factor) * factor;
    if (o.min_tokens)
        min_pixels = std::min<int64_t>(int64_t(o.min_tokens) * area_per_token, g.max_pixels);
    if (o.max_tokens)
        max_pixels = std::min<int64_t>(int64_t(o.max_tokens) * area_per_token, g.max_pixels);
    require(min_pixels <= max_pixels, "minimum image size exceeds maximum");
    int h = round_even_multiple(height, factor), w = round_even_multiple(width, factor);
    if (int64_t(h) * w > max_pixels) {
        const double beta = std::sqrt(static_cast<double>(width) * height / max_pixels);
        h = std::max(factor, int(std::floor(height / beta / factor)) * factor);
        w = std::max(factor, int(std::floor(width / beta / factor)) * factor);
    } else if (int64_t(h) * w < min_pixels) {
        const double beta =
            std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(width) * height));
        h = int(std::ceil(height * beta / factor)) * factor;
        w = int(std::ceil(width * beta / factor)) * factor;
    }
    require(w > 0 && h > 0 && int64_t(w) * h <= g.max_pixels * 2ll, "resized image exceeds supported area");
    strata_vit::ImageU8 original{width, height, std::vector<uint8_t>(rgb.begin(), rgb.end())}, resized;
    if (width == w && height == h)
        resized = std::move(original);
    else
        strata_vit::resize_bicubic(original, resized, w, h);
    Vec chw(static_cast<size_t>(w) * h * 3);
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i)
            chw[static_cast<size_t>(c) * w * h + i] =
                (resized.rgb[i * 3 + c] / 255.f - g.mean[c]) / g.stddev[c];
    return encode_normalized(chw, w, h, trace);
}

VisionEmbedding VisionEncoder::encode_normalized(std::span<const float> chw, int width, int height,
                                                 const VisionTrace &trace) {
    const auto &g = impl_->g;
    require(width > 0 && height > 0 && width % (g.patch * g.merge) == 0 &&
                height % (g.patch * g.merge) == 0 && chw.size() == static_cast<size_t>(width) * height * 3,
            "normalized image dimensions");
    require(static_cast<int64_t>(width) * height <= g.max_pixels * 2ll, "normalized image too large");
    for (float x : chw)
        require(std::isfinite(x), "nonfinite normalized pixel");
    const int pw = width / g.patch, ph = height / g.patch, pk = 3 * g.patch * g.patch;
    Vec patches(static_cast<size_t>(pw) * ph * pk);
    for (int y = 0; y < ph; ++y)
        for (int x = 0; x < pw; ++x)
            for (int c = 0; c < 3; ++c)
                for (int dy = 0; dy < g.patch; ++dy)
                    for (int dx = 0; dx < g.patch; ++dx)
                        patches[(static_cast<size_t>(y) * pw + x) * pk + (c * g.patch + dy) * g.patch + dx] =
                            chw[(static_cast<size_t>(c) * height + y * g.patch + dy) * width + x * g.patch +
                                dx];
    Vec positions(static_cast<size_t>(pw) * ph * g.dim);
    const Vec table = impl_->at("v.position_embd.weight").values();
    const float sx = (pw + .1f) / g.grid, sy = (ph + .1f) / g.grid;
    for (int y = 0; y < ph; ++y)
        for (int x = 0; x < pw; ++x) {
            const float fy = (y + .5f) / sy - .5f, fx = (x + .5f) / sx - .5f;
            const int y0 = std::clamp(int(std::floor(fy)), 0, g.grid - 1),
                      y1 = std::clamp(int(std::floor(fy)) + 1, 0, g.grid - 1);
            const int x0 = std::clamp(int(std::floor(fx)), 0, g.grid - 1),
                      x1 = std::clamp(int(std::floor(fx)) + 1, 0, g.grid - 1);
            const float wy = std::clamp(fy - y0, 0.f, 1.f), wx = std::clamp(fx - x0, 0.f, 1.f);
            for (int c = 0; c < g.dim; ++c)
                positions[(static_cast<size_t>(y) * pw + x) * g.dim + c] =
                    (1 - wy) * (1 - wx) * table[(static_cast<size_t>(y0) * g.grid + x0) * g.dim + c] +
                    (1 - wy) * wx * table[(static_cast<size_t>(y0) * g.grid + x1) * g.dim + c] +
                    wy * (1 - wx) * table[(static_cast<size_t>(y1) * g.grid + x0) * g.dim + c] +
                    wy * wx * table[(static_cast<size_t>(y1) * g.grid + x1) * g.dim + c];
        }
    VisionEmbedding result;
    result.dim = g.out;
    result.patch_width = pw;
    result.patch_height = ph;
    result.width = pw / g.merge + 1;
    result.height = ph / g.merge;
    auto *backend = impl_->backend.get();
    const std::string admission = backend->admission_error(pw, ph);
    if (!admission.empty()) {
        std::cerr << "Hy4 vision: " << admission << "; using native CPU encoder for this image\n";
        if (!impl_->cpu_fallback) {
            auto lookup = [p = impl_.get()](const std::string &name) -> const Tensor & { return p->at(name); };
            impl_->cpu_fallback = std::make_unique<Cpu>(g, lookup, impl_->options.threads);
        }
        backend = impl_->cpu_fallback.get();
    }
    result.rows = backend->encode(patches, positions, pw, ph, trace);
    require(result.rows.size() == result.tokens() * g.out, "backend output dimensions");
    for (float x : result.rows)
        require(std::isfinite(x), "nonfinite image embedding");
    return result;
}

std::array<int64_t, 3> VisionEmbedding::position(size_t row, int64_t start) const {
    require(width > 0 && height > 0 && row < tokens() && start >= 0 &&
                start <= INT64_MAX - std::max(width, height),
            "image position out of range");
    return {start, start + static_cast<int64_t>(row / width), start + static_cast<int64_t>(row % width)};
}
int64_t VisionEmbedding::next_text_position(int64_t start) const {
    require(width > 0 && height > 0 && start >= 0 && start <= INT64_MAX - std::max(width, height),
            "image positions invalid");
    return start + std::max(width, height);
}
void VisionEmbedding::write_sve(const std::string &path) const {
    require(dim > 0 && width > 0 && height > 0 && rows.size() == tokens() * dim && tokens() <= INT32_MAX,
            "invalid embedding layout");
    std::ofstream file(path, std::ios::binary);
    require(bool(file), "cannot create image embedding file");
    const int32_t header[]{0x31455653, static_cast<int32_t>(tokens()), width, height, dim};
    file.write(reinterpret_cast<const char *>(header), sizeof(header));
    file.write(reinterpret_cast<const char *>(rows.data()), rows.size() * sizeof(float));
    require(bool(file), "writing image embeddings failed");
}
} // namespace strata::hy4
