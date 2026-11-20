#pragma once

#define STRATA_GGUF_MAIN_DISABLED 1
#include "strata/artifact/gguf_reader.hpp"

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace strata::hy4 {

using Vec = std::vector<float>;
using Trace = std::function<void(const std::string&, int, std::span<const float>)>;

struct Geometry {
    int dim = 0, layers = 0, draft_layers = 0, vocab = 0, hc = 0;
    int qrank = 0, krank = 0, head_dim = 0, value_dim = 0, rope_dim = 0;
    int dense_ff = 0, ff = 0, latent = 0, experts = 0, used = 0, dense_layers = 0;
    int window = 0, index_dim = 0, index_heads = 0, topk = 0, pool = 0;
    int ngram = 0, hash_heads = 0, engram_dim = 0, pad = 0;
    int64_t max_context = 0;
    float eps = 0, hc_eps = 0, hc_magnitude = 0, expert_scale = 0;
    float rope_base = 0, rope_base_swa = 0;
    bool force_tail = true;
    std::vector<int> heads, swa, indexer, engram_layers;
    Vec clamp;
    std::vector<uint64_t> multipliers, offsets, primes;
    std::vector<int64_t> token_map;
    static Geometry read(const GgufFile& file);
};

struct Tensor {
    const TensorInfo* info = nullptr;
    const uint8_t* data = nullptr;
    Vec values(int64_t row = -1) const;
    size_t row_bytes() const;
};

class Matmul {
public:
    explicit Matmul(int threads = 4);
    virtual ~Matmul() = default;
    virtual Vec linear(const Tensor& w, std::span<const float> x, int plane = 0);
    int threads() const { return threads_; }
private:
    int threads_;
};

void rms(std::span<float> x, float epsilon, std::span<const float> weight = {});
void rope(std::span<float> x, int64_t position, float base);
std::vector<int> route(std::span<const float> probabilities, std::span<const float> bias, int used);
std::vector<int32_t> engram_hash(const Geometry& g, int module, std::span<const int32_t> history, int32_t token);
Vec pool_key(std::span<const float> values, std::span<const float> gates,
             std::span<const float> ape, int width, int count);

class Engine {
public:
    Engine(const std::string& gguf, int threads = 4, int64_t max_context = 8192);
    const Geometry& geometry() const { return g_; }
    const GgufFile& metadata() const { return model_.meta(); }
    const Tensor& tensor(const std::string& name) const;
    void reset();
    Vec forward(int32_t token, const Trace& trace = {});
    const std::vector<int32_t>& history() const { return history_; }
    int64_t max_context() const { return max_context_; }
    void set_matmul(std::unique_ptr<Matmul> backend) { mm_ = std::move(backend); }
private:
    struct Cache {
        Vec kv, pool_values, pool_gates, pooled;
        int64_t count = 0;
    };
    struct Mix { Vec value, post; };
    Mix mix(const Vec& x, int layer, const char* kind);
    Vec finish_mix(const Vec& residual, const Vec& value, const Vec& post);
    Vec engram(Vec x, int layer, int module, int32_t token);
    Vec attention(const Vec& x, int layer, const Trace& trace);
    Vec ffn(const Vec& x, int layer, const Trace& trace);
    std::vector<int> select(const Vec& x, const Vec& q, int layer);
    Vec linear(const std::string& name, std::span<const float> x, int plane = 0);
    Vec norm(Vec x, const std::string& name, float eps = -1);
    std::string layer_name(int layer, const std::string& suffix) const;
    void validate();

    GgufModel model_;
    Geometry g_;
    std::map<std::string, Tensor> tensors_;
    std::unique_ptr<Matmul> mm_;
    std::vector<Cache> cache_;
    std::vector<int32_t> history_;
    int64_t max_context_;
};

}  // namespace strata::hy4
