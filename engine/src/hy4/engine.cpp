#include "strata/hy4/engine.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace strata::hy4 {
namespace {
constexpr const char* prefix = "hy_v4_flash.";
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

const MetaValue& required(const GgufFile& f, const std::string& key) {
    const auto* v = f.get(key);
    if (!v) throw std::runtime_error("hy4: missing metadata " + key);
    return *v;
}

double scalar(const GgufFile& f, const std::string& key) {
    const auto& v = required(f, std::string(prefix) + key);
    if (!v.is_num() || !std::isfinite(v.num())) throw std::runtime_error("hy4: invalid scalar " + key);
    return v.num();
}

int integer(const GgufFile& f, const std::string& key, int low, int high) {
    const double n = scalar(f, key);
    if (n < low || n > high || n != std::floor(n)) throw std::runtime_error("hy4: invalid size " + key);
    return static_cast<int>(n);
}

template<class T> std::vector<T> array(const GgufFile& f, const std::string& key, int count = -1) {
    const auto& a = required(f, std::string(prefix) + key);
    if (a.type != MetaType::ARRAY || a.items.size() != a.count || (count >= 0 && a.count != static_cast<size_t>(count)))
        throw std::runtime_error("hy4: invalid or truncated array " + key);
    std::vector<T> out;
    for (const auto& v : a.items) {
        if (!v.is_num()) throw std::runtime_error("hy4: nonnumeric array " + key);
        const bool signed_int = v.type == MetaType::I8 || v.type == MetaType::I16 || v.type == MetaType::I32 || v.type == MetaType::I64;
        if constexpr (std::is_integral_v<T>) {
            if (v.type == MetaType::F32 || v.type == MetaType::F64) throw std::runtime_error("hy4: integer array required: " + key);
            if (signed_int) {
                const int64_t value = std::bit_cast<int64_t>(v.u);
                if constexpr (std::is_unsigned_v<T>) {
                    if (value < 0 || static_cast<uint64_t>(value) > std::numeric_limits<T>::max())
                        throw std::runtime_error("hy4: array integer out of range: " + key);
                } else if (value < std::numeric_limits<T>::min() || value > std::numeric_limits<T>::max())
                    throw std::runtime_error("hy4: array integer out of range: " + key);
                out.push_back(static_cast<T>(value));
            } else {
                if (v.u > static_cast<uint64_t>(std::numeric_limits<T>::max())) throw std::runtime_error("hy4: array integer out of range: " + key);
                out.push_back(static_cast<T>(v.u));
            }
        } else {
            const double value = signed_int ? static_cast<double>(std::bit_cast<int64_t>(v.u)) : v.num();
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<T>::max()) throw std::runtime_error("hy4: nonfinite array " + key);
            out.push_back(static_cast<T>(value));
        }
    }
    return out;
}

float dot(std::span<const float> a, std::span<const float> b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) sum += static_cast<double>(a[i]) * b[i];
    return static_cast<float>(sum);
}

void swiglu(Vec& gate, const Vec& up, float limit = 0) {
    for (size_t j = 0; j < gate.size(); ++j) {
        const float g = limit > 0 ? std::min(gate[j], limit) : gate[j];
        const float u = limit > 0 ? std::clamp(up[j], -limit, limit) : up[j];
        gate[j] = (g * sigmoid(g)) * u;
    }
}
int64_t checked_next_position(const RopePosition& position, int64_t next) {
    if (std::any_of(position.begin(), position.end(), [](int64_t p) { return p < 0 || p == INT64_MAX; }) || next < -1)
        throw std::runtime_error("hy4: invalid rotary position");
    return next < 0 ? *std::max_element(position.begin(), position.end()) + 1 : next;
}
void finite_row(std::span<const float> row, int width) {
    if (row.size() != static_cast<size_t>(width) ||
        std::any_of(row.begin(), row.end(), [](float v) { return !std::isfinite(v); }))
        throw std::runtime_error("hy4: invalid input embedding or hidden state");
}
}  // namespace

Geometry Geometry::read(const GgufFile& f) {
    if (required(f, "general.architecture").s != "hy_v4_flash")
        throw std::runtime_error("hy4: expected architecture hy_v4_flash");
    Geometry g;
    const int total = integer(f, "block_count", 1, 256);
    g.draft_layers = integer(f, "nextn_predict_layers", 0, total - 1);
    g.layers = total - g.draft_layers;
    g.dim = integer(f, "embedding_length", 1, 32768);
    g.vocab = integer(f, "vocab_size", 1, 1 << 22);
    g.max_context = integer(f, "context_length", 1, 1 << 26);
    g.hc = integer(f, "hyper_connection.count", 1, 16);
    g.qrank = integer(f, "attention.q_lora_rank", 1, 32768);
    g.krank = integer(f, "attention.kv_lora_rank", 1, 4096);
    g.head_dim = integer(f, "attention.key_length_mla", 2, 4096);
    g.value_dim = integer(f, "attention.value_length_mla", 1, 4096);
    g.rope_dim = integer(f, "rope.dimension_count", 2, g.head_dim - 1);
    if (g.rope_dim % 2) throw std::runtime_error("hy4: RoPE width must be even");
    g.dense_ff = integer(f, "feed_forward_length", 1, 131072);
    g.ff = integer(f, "expert_feed_forward_length", 1, 65536);
    g.latent = integer(f, "expert_latent_length", 1, g.dim);
    g.experts = integer(f, "expert_count", 1, 4096);
    g.used = integer(f, "expert_used_count", 1, std::min(64, g.experts));
    g.dense_layers = integer(f, "leading_dense_block_count", 0, g.layers);
    if (integer(f, "expert_shared_count", 1, 64) != 1 || scalar(f, "expert_gating_func") != 2 ||
        scalar(f, "expert_weights_norm") != 1 || scalar(f, "attention.head_count_kv") != 1)
        throw std::runtime_error("hy4: requires sigmoid routing, normalized weights, one shared expert and MLA");
    g.window = integer(f, "attention.sliding_window", 1, 1 << 20);
    g.index_dim = integer(f, "attention.indexer.key_length", g.rope_dim + 1, 4096);
    g.index_heads = integer(f, "attention.indexer.head_count", 1, 128);
    g.topk = integer(f, "attention.indexer.top_k", 1, 65536);
    g.pool = integer(f, "attention.indexer.kpool", 2, 64);
    g.force_tail = scalar(f, "attention.indexer.kpool_select_tail") != 0;
    if (!g.force_tail || g.topk % g.pool || g.topk / g.pool < 2)
        throw std::runtime_error("hy4: requires pooled selection with a reserved current group");
    g.eps = scalar(f, "attention.layer_norm_rms_epsilon");
    g.hc_eps = scalar(f, "hyper_connection.epsilon");
    g.hc_magnitude = scalar(f, "hyper_connection.magnitude");
    g.expert_scale = scalar(f, "expert_weights_scale");
    g.rope_base = scalar(f, "rope.freq_base");
    g.rope_base_swa = scalar(f, "rope.freq_base_swa");
    if (g.eps <= 0 || g.hc_eps < 0 || g.hc_magnitude <= 0 || g.rope_base <= 0 || g.rope_base_swa <= 0)
        throw std::runtime_error("hy4: invalid norm or positional parameters");
    const auto& heads = required(f, std::string(prefix) + "attention.head_count");
    if (heads.type == MetaType::ARRAY) g.heads = array<int>(f, "attention.head_count", total);
    else g.heads.assign(total, static_cast<int>(heads.num()));
    g.swa = array<int>(f, "attention.sliding_window_pattern", total);
    g.indexer = array<int>(f, "attention.indexer.types", g.layers);
    g.clamp = array<float>(f, "swiglu_clamp_exp", total);
    for (int i = 0; i < g.layers; ++i) {
        if (g.heads[i] < 1 || g.heads[i] > 256 || (g.swa[i] != 0 && g.swa[i] != 1) ||
            (g.indexer[i] != 0 && g.indexer[i] != 1) || g.swa[i] == g.indexer[i])
            throw std::runtime_error("hy4: unsupported attention/indexer layer pattern");
    }
    g.ngram = integer(f, "ple.ngram_size", 2, 8);
    g.hash_heads = integer(f, "ple.heads_per_ngram", 1, 64);
    g.engram_dim = integer(f, "embedding_length_per_layer_input", 1, 4096);
    g.pad = integer(f, "engram.pad_token_id", 0, g.vocab - 1);
    g.engram_layers = array<int>(f, "ple.layers");
    const int modules = g.engram_layers.size(), columns = (g.ngram - 1) * g.hash_heads;
    if (modules < 1 || modules > g.layers) throw std::runtime_error("hy4: invalid Engram modules");
    g.multipliers = array<uint64_t>(f, "ple.layer_multipliers", modules * g.ngram);
    g.offsets = array<uint64_t>(f, "ple.head_offsets", modules * columns);
    g.primes = array<uint64_t>(f, "ple.head_vocab_sizes", modules * columns);
    g.token_map = array<int64_t>(f, "engram.token_map", g.vocab);
    if (std::any_of(g.token_map.begin(), g.token_map.end(), [](int64_t v) { return v < -1; }))
        throw std::runtime_error("hy4: Engram token mapping must be nonnegative or -1");
    if (g.token_map[g.pad] < 0) throw std::runtime_error("hy4: invalid Engram padding mapping");
    for (size_t i = 0; i < g.primes.size(); ++i)
        if (g.primes[i] < 2 || g.offsets[i] > INT32_MAX || g.primes[i] > INT32_MAX - g.offsets[i])
            throw std::runtime_error("hy4: Engram bucket does not fit int32 row indices");
    for (int i = 0; i < modules; ++i)
        if (g.engram_layers[i] < 0 || g.engram_layers[i] >= g.layers ||
            (i && g.engram_layers[i] <= g.engram_layers[i - 1]))
            throw std::runtime_error("hy4: Engram layers must be ordered trunk indices");
    return g;
}

Engine::Engine(const std::string& gguf, int threads, int64_t max_context)
    : model_(GgufModel::open(gguf, true)), g_(Geometry::read(model_.meta())),
      mm_(std::make_unique<Matmul>(threads)), cache_(g_.layers + g_.draft_layers), max_context_(max_context) {
    if (max_context < 1 || max_context > g_.max_context) throw std::runtime_error("hy4: context exceeds model limit");
    if (g_.draft_layers > 1) throw std::runtime_error("hy4: CPU reference supports one MTP block");
    for (size_t shard = 0; shard < model_.size(); ++shard) {
        for (const auto& t : model_.shard(shard).tensors()) {
            if (!model_.in_bounds(t, shard)) throw std::runtime_error("hy4: truncated or unknown tensor " + t.name);
            tensors_.emplace(t.name, Tensor{&t, model_.shard(shard).tensor_data(t)});
        }
    }
    validate();
}

const Tensor& Engine::tensor(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("hy4: missing tensor " + name);
    return it->second;
}

std::string Engine::layer_name(int layer, const std::string& suffix) const {
    return "blk." + std::to_string(layer) + "." + suffix;
}

void Engine::validate() {
    const auto check = [&](const std::string& name, std::initializer_list<uint64_t> shape) {
        const auto& w = tensor(name);
        if (w.info->shape != std::vector<uint64_t>(shape)) throw std::runtime_error("hy4: tensor shape mismatch: " + name);
        w.row_bytes();
    };
    const uint64_t d = g_.dim, hc = g_.hc, qr = g_.qrank, kr = g_.krank, rot = g_.rope_dim;
    const uint64_t no = g_.head_dim - g_.rope_dim, vd = g_.value_dim, idx = g_.index_dim;
    const uint64_t ff = g_.ff, latent = g_.latent, cols = (g_.ngram - 1) * g_.hash_heads;
    check("token_embd.weight", {d, uint64_t(g_.vocab)});
    check("output.weight", {d, uint64_t(g_.vocab)});
    check("output_norm.weight", {d});
    check("output_hc_fn.weight", {d * hc, hc});
    check("output_hc_base.weight", {hc});
    check("output_hc_scale.weight", {1});
    uint64_t engram_rows = 0;
    for (size_t i = 0; i < g_.offsets.size(); ++i) engram_rows = std::max(engram_rows, g_.offsets[i] + g_.primes[i]);
    check("per_layer_token_embd.weight", {uint64_t(g_.engram_dim), engram_rows});
    for (int l = 0; l < g_.layers + g_.draft_layers; ++l) {
        const uint64_t nh = g_.heads[l];
        const auto c = [&](const char* suffix, std::initializer_list<uint64_t> shape) { check(layer_name(l, suffix), shape); };
        for (const auto* name : {"attn_norm.weight", "ffn_norm.weight", "post_attention_norm.weight", "post_ffw_norm.weight"}) c(name, {d});
        c("attn_q_a.weight", {d, qr}); c("attn_q_a_norm.weight", {qr});
        c("attn_q_b.weight", {qr, nh * uint64_t(g_.head_dim)});
        c("attn_kv_a_mqa.weight", {d, kr + rot}); c("attn_kv_a_norm.weight", {kr});
        c("attn_k_b.weight", {no, kr, nh}); c("attn_v_b.weight", {kr, vd, nh});
        c("attn_gate.weight", {d, nh * vd}); c("attn_output.weight", {nh * vd, d});
        c("attn_sinks.weight", {nh});
        if (l < g_.layers) for (const auto* kind : {"attn", "ffn"}) {
            check(layer_name(l, std::string("hc_") + kind + "_fn.weight"), {d * hc, 2 * hc});
            check(layer_name(l, std::string("hc_") + kind + "_base.weight"), {2 * hc});
            check(layer_name(l, std::string("hc_") + kind + "_scale.weight"), {2});
        }
        if (l >= g_.layers || g_.indexer[l]) {
            c("indexer.attn_k.weight", {d, idx}); c("indexer.k_norm.weight", {idx});
            c("indexer.attn_q_b.weight", {qr, idx * uint64_t(g_.index_heads)});
            c("indexer.proj.weight", {d, uint64_t(g_.index_heads)});
            c("indexer_compressor_value.weight", {idx, idx}); c("indexer_compressor_gate.weight", {idx, idx});
            c("indexer_compressor_ape.weight", {idx, uint64_t(g_.pool)}); c("indexer_compressor_norm.weight", {idx});
        }
        if (std::find(g_.engram_layers.begin(), g_.engram_layers.end(), l) != g_.engram_layers.end()) {
            c("engram_kv.weight", {cols * uint64_t(g_.engram_dim), d * (hc + 1)});
            c("engram_kv_b.weight", {d, hc + 1}); c("engram_gate.weight", {d, hc});
            c("engram_lookup_norm.weight", {uint64_t(g_.engram_dim), cols});
            c("engram_pre_gamma.weight", {d}); c("engram_hc_scale.weight", {d, hc});
        }
        if (l >= g_.layers) {
            c("nextn.enorm.weight", {d}); c("nextn.hnorm.weight", {d});
            c("nextn.eh_proj.weight", {2*d, d}); c("nextn.shared_head_norm.weight", {d});
        }
        if (l < g_.dense_layers) {
            c("ffn_gate.weight", {d, uint64_t(g_.dense_ff)}); c("ffn_up.weight", {d, uint64_t(g_.dense_ff)});
            c("ffn_down.weight", {uint64_t(g_.dense_ff), d});
        } else {
            c("ffn_gate_inp.weight", {d, uint64_t(g_.experts)});
            if (tensors_.contains(layer_name(l, "exp_probs_b.bias"))) c("exp_probs_b.bias", {uint64_t(g_.experts)});
            c("ffn_routed_down.weight", {d, latent}); c("ffn_routed_up.weight", {latent, d});
            c("ffn_gate_exps.weight", {latent, ff, uint64_t(g_.experts)});
            c("ffn_up_exps.weight", {latent, ff, uint64_t(g_.experts)});
            c("ffn_down_exps.weight", {ff, latent, uint64_t(g_.experts)});
            c("ffn_gate_shexp.weight", {d, ff}); c("ffn_up_shexp.weight", {d, ff});
            c("ffn_down_shexp.weight", {ff, d});
        }
    }
}

Vec Engine::linear(const std::string& name, std::span<const float> x, int plane) { return mm_->linear(tensor(name), x, plane); }
Vec Engine::norm(Vec x, const std::string& name, float eps) {
    rms(x, eps < 0 ? g_.eps : eps, tensor(name).values());
    return x;
}

void Engine::reset() {
    history_.clear(); positions_.clear(); last_hidden_.clear(); next_rope_ = 0;
    for (auto& c : cache_) c = Cache{};
    reset_draft();
}

void Engine::reset_draft() {
    draft_history_.clear(); draft_positions_.clear(); last_draft_hidden_.clear(); draft_next_rope_ = 0;
    for (size_t i = g_.layers; i < cache_.size(); ++i) cache_[i] = Cache{};
}

Engine::Mix Engine::mix(const Vec& x, int layer, const char* kind) {
    Vec flat = x;
    rms(flat, g_.eps);
    const std::string name = std::string("hc_") + kind;
    Vec mixes = linear(layer_name(layer, name + "_fn.weight"), flat);
    const Vec scale = tensor(layer_name(layer, name + "_scale.weight")).values();
    const Vec base = tensor(layer_name(layer, name + "_base.weight")).values();
    Mix out{Vec(g_.dim, 0), Vec(g_.hc)};
    for (int h = 0; h < g_.hc; ++h) {
        const float pre = sigmoid(mixes[h] * scale[0] + base[h]) + g_.hc_eps;
        out.post[h] = sigmoid(mixes[h + g_.hc] * scale[1] + base[h + g_.hc]) * g_.hc_magnitude + g_.hc_eps;
        for (int i = 0; i < g_.dim; ++i) out.value[i] += x[h * g_.dim + i] * pre;
    }
    return out;
}

Vec Engine::finish_mix(const Vec& residual, const Vec& value, const Vec& post) {
    Vec out = residual;
    for (int h = 0; h < g_.hc; ++h)
        for (int i = 0; i < g_.dim; ++i) out[h * g_.dim + i] += value[i] * post[h];
    return out;
}

Vec Engine::engram(Vec x, int layer, int module, int32_t token) {
    if (g_.token_map.at(token) < 0) return x;
    const auto ids = engram_hash(g_, module, history_, token);
    const Vec lookup_norm = tensor(layer_name(layer, "engram_lookup_norm.weight")).values();
    Vec rows;
    for (size_t h = 0; h < ids.size(); ++h) {
        Vec row = tensor("per_layer_token_embd.weight").values(ids[h]);
        rms(row, 1.1920929e-7f, std::span(lookup_norm).subspan(h * g_.engram_dim, g_.engram_dim));
        rows.insert(rows.end(), row.begin(), row.end());
    }
    Vec kv = linear(layer_name(layer, "engram_kv.weight"), rows);
    const Vec bias = tensor(layer_name(layer, "engram_kv_b.weight")).values();
    for (size_t i = 0; i < kv.size(); ++i) kv[i] += bias[i];
    Vec value(kv.begin() + g_.hc * g_.dim, kv.end());
    rms(value, 1.1920929e-7f, tensor(layer_name(layer, "engram_pre_gamma.weight")).values());
    const Vec gate_weight = tensor(layer_name(layer, "engram_gate.weight")).values();
    const Vec hc_scale = tensor(layer_name(layer, "engram_hc_scale.weight")).values();
    for (int h = 0; h < g_.hc; ++h) {
        Vec key(kv.begin() + h * g_.dim, kv.begin() + (h + 1) * g_.dim);
        Vec query(x.begin() + h * g_.dim, x.begin() + (h + 1) * g_.dim);
        rms(key, 1.1920929e-7f); rms(query, 1.1920929e-7f);
        float score = 0;
        for (int i = 0; i < g_.dim; ++i) score += (query[i] * key[i]) * gate_weight[h * g_.dim + i];
        score /= std::sqrt(float(g_.dim));
        const float gate = sigmoid(std::copysign(std::sqrt(std::max(std::abs(score), 1e-6f)), score));
        for (int i = 0; i < g_.dim; ++i) x[h * g_.dim + i] += value[i] * (gate + hc_scale[h * g_.dim + i]);
    }
    return x;
}

std::vector<int> Engine::select(const Vec& x, const Vec& q, int layer, int64_t pos,
                                const RopePosition& position, const std::vector<RopePosition>& rope_history) {
    auto& c = cache_[layer];
    const auto ln = [&](const char* name) { return layer_name(layer, name); };
    const int width = g_.index_dim;
    const float base = g_.swa[layer] ? g_.rope_base_swa : g_.rope_base;
    Vec key = norm(linear(ln("indexer.attn_k.weight"), x), ln("indexer.k_norm.weight"));
    Vec value = linear(ln("indexer_compressor_value.weight"), key);
    Vec gate = linear(ln("indexer_compressor_gate.weight"), key);
    if (pos % g_.pool == 0) { c.pool_values.clear(); c.pool_gates.clear(); }
    c.pool_values.insert(c.pool_values.end(), value.begin(), value.end());
    c.pool_gates.insert(c.pool_gates.end(), gate.begin(), gate.end());
    if (pos % g_.pool == g_.pool - 1) {
        Vec pooled = pool_key(c.pool_values, c.pool_gates,
                             tensor(ln("indexer_compressor_ape.weight")).values(), width, g_.pool);
        pooled = norm(std::move(pooled), ln("indexer_compressor_norm.weight"));
        rope(std::span(pooled).last(g_.rope_dim), rope_history.at(pos - g_.pool + 1), base);
        c.pooled.insert(c.pooled.end(), pooled.begin(), pooled.end());
    }
    Vec query = linear(ln("indexer.attn_q_b.weight"), q);
    for (int h = 0; h < g_.index_heads; ++h)
        rope(std::span(query).subspan(h * width + width - g_.rope_dim, g_.rope_dim), position, base);
    Vec weights = linear(ln("indexer.proj.weight"), x);
    const int groups = static_cast<int>(pos / g_.pool);
    Vec scores(groups, 0);
    const float scale = 1.0f / std::sqrt(float(width * g_.index_heads));
    for (int group = 0; group < groups; ++group) {
        for (int h = 0; h < g_.index_heads; ++h) {
            const float s = dot(std::span(query).subspan(h * width, width), std::span(c.pooled).subspan(group * width, width));
            scores[group] += std::max(s, 0.0f) * (weights[h] * scale);
        }
    }
    std::vector<int> picked(groups);
    std::iota(picked.begin(), picked.end(), 0);
    const int take = std::min(groups, g_.topk / g_.pool - 1);
    std::partial_sort(picked.begin(), picked.begin() + take, picked.end(), [&](int a, int b) {
        return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
    });
    std::vector<int> positions;
    for (int i = 0; i < take; ++i)
        for (int t = 0; t < g_.pool; ++t) positions.push_back(picked[i] * g_.pool + t);
    for (int64_t p = pos - pos % g_.pool; p <= pos; ++p) positions.push_back(static_cast<int>(p));
    std::sort(positions.begin(), positions.end());
    return positions;
}

Vec Engine::attention(const Vec& x, int layer, int64_t pos, const RopePosition& position,
                      const std::vector<RopePosition>& rope_history, const Trace& trace) {
    const int heads = g_.heads[layer], dim = g_.dim, no = g_.head_dim - g_.rope_dim;
    const int stride = g_.krank + g_.rope_dim;
    const auto ln = [&](const char* name) { return layer_name(layer, name); };
    const float base = g_.swa[layer] ? g_.rope_base_swa : g_.rope_base;
    Vec qr = norm(linear(ln("attn_q_a.weight"), x), ln("attn_q_a_norm.weight"));
    Vec query = linear(ln("attn_q_b.weight"), qr);
    if (trace) trace("attn_query_plain", layer, query);
    Vec kv = linear(ln("attn_kv_a_mqa.weight"), x);
    rms(std::span(kv).first(g_.krank), g_.eps, tensor(ln("attn_kv_a_norm.weight")).values());
    rope(std::span(kv).last(g_.rope_dim), position, base);
    if (trace) trace("attn_kv", layer, kv);
    auto& c = cache_[layer];
    if (g_.swa[layer]) {
        if (c.kv.empty()) c.kv.resize(static_cast<size_t>(g_.window) * stride);
        std::copy(kv.begin(), kv.end(), c.kv.begin() + (pos % g_.window) * stride);
    } else c.kv.insert(c.kv.end(), kv.begin(), kv.end());
    ++c.count;
    std::vector<int> selected;
    if (layer >= g_.layers || g_.indexer[layer]) selected = select(x, qr, layer, pos, position, rope_history);
    else {
        const int64_t first = std::max<int64_t>(0, pos - g_.window + 1);
        for (int64_t p = first; p <= pos; ++p) selected.push_back(static_cast<int>(p));
    }
    Vec output(heads * g_.value_dim);
    Vec absorbed_trace, scores_trace, latent_trace;
    if (trace) trace("attn_selected", layer, Vec(selected.begin(), selected.end()));
    const Vec sinks = tensor(ln("attn_sinks.weight")).values();
    const float scale = 1.0f / std::sqrt(float(g_.head_dim));
    for (int h = 0; h < heads; ++h) {
        Vec q = linear(ln("attn_k_b.weight"), std::span(query).subspan(h * g_.head_dim, no), h);
        q.insert(q.end(), query.begin() + h * g_.head_dim + no, query.begin() + (h + 1) * g_.head_dim);
        rope(std::span(q).last(g_.rope_dim), position, base);
        if (trace) absorbed_trace.insert(absorbed_trace.end(), q.begin(), q.end());
        Vec scores(selected.size());
        float maximum = sinks[h];
        for (size_t j = 0; j < selected.size(); ++j) {
            const size_t cell = g_.swa[layer] ? selected[j] % g_.window : selected[j];
            scores[j] = dot(q, std::span(c.kv).subspan(cell * stride, stride)) * scale;
            maximum = std::max(maximum, scores[j]);
        }
        if (trace) scores_trace.insert(scores_trace.end(), scores.begin(), scores.end());
        Vec latent(g_.krank, 0);
        float den = std::exp(sinks[h] - maximum);
        for (size_t j = 0; j < selected.size(); ++j) {
            const float probability = std::exp(scores[j] - maximum);
            den += probability;
            const size_t cell = g_.swa[layer] ? selected[j] % g_.window : selected[j];
            for (int i = 0; i < g_.krank; ++i) latent[i] += probability * c.kv[cell * stride + i];
        }
        for (float& v : latent) v /= den;
        if (trace) latent_trace.insert(latent_trace.end(), latent.begin(), latent.end());
        Vec head = linear(ln("attn_v_b.weight"), latent, h);
        std::copy(head.begin(), head.end(), output.begin() + h * g_.value_dim);
    }
    if (trace) {
        trace("attn_q_absorbed", layer, absorbed_trace);
        trace("attn_scores_scaled", layer, scores_trace);
        trace("attn_latent", layer, latent_trace);
        trace("attn_head_output", layer, output);
    }
    Vec gate = linear(ln("attn_gate.weight"), x);
    for (size_t i = 0; i < output.size(); ++i) output[i] *= sigmoid(gate[i]);
    if (trace) trace("attn_gated", layer, output);
    if (trace) trace("attn_projection_input", layer, output);
    Vec result = norm(linear(ln("attn_output.weight"), output), ln("post_attention_norm.weight"));
    if (trace) { trace("q_resid", layer, qr); trace("attn_out", layer, result); }
    return result;
}

Vec Engine::ffn(const Vec& x, int layer, const Trace& trace) {
    const auto ln = [&](const char* name) { return layer_name(layer, name); };
    Vec out;
    if (layer < g_.dense_layers) {
        Vec gate = linear(ln("ffn_gate.weight"), x);
        Vec up = linear(ln("ffn_up.weight"), x);
        if (trace) { trace("ffn_gate", layer, gate); trace("ffn_up", layer, up); }
        swiglu(gate, up);
        if (trace) trace("ffn_swiglu", layer, gate);
        out = linear(ln("ffn_down.weight"), gate);
        if (trace) trace("ffn_down", layer, out);
    } else {
        Vec probs = linear(ln("ffn_gate_inp.weight"), x);
        if (trace) trace("ffn_moe_logits", layer, probs);
        for (float& p : probs) p = sigmoid(p);
        Vec bias;
        if (tensors_.contains(ln("exp_probs_b.bias"))) bias = tensor(ln("exp_probs_b.bias")).values();
        const auto ids = route(probs, bias, g_.used);
        if (trace) {
            const Vec selected(ids.begin(), ids.end());
            trace("ffn_moe_topk", layer, selected);
        }
        float den = 0;
        for (int id : ids) den += probs[id];
        den = std::max(den, 6.103515625e-5f);
        Vec latent = linear(ln("ffn_routed_down.weight"), x);
        if (trace) trace("ffn_routed_input", layer, latent);
        Vec sum(g_.latent, 0);
        for (int id : ids) {
            Vec gate = linear(ln("ffn_gate_exps.weight"), latent, id);
            Vec up = linear(ln("ffn_up_exps.weight"), latent, id);
            swiglu(gate, up, g_.clamp[layer]);
            Vec down = linear(ln("ffn_down_exps.weight"), gate, id);
            const float w = (probs[id] / den) * g_.expert_scale;
            for (int j = 0; j < g_.latent; ++j) sum[j] += down[j] * w;
        }
        out = linear(ln("ffn_routed_up.weight"), sum);
        if (trace) {
            trace("ffn_moe_out", layer, sum);
            trace("ffn_moe_up", layer, out);
        }
        Vec shared = linear(ln("ffn_gate_shexp.weight"), x);
        Vec up = linear(ln("ffn_up_shexp.weight"), x);
        swiglu(shared, up);
        shared = linear(ln("ffn_down_shexp.weight"), shared);
        if (trace) trace("ffn_shexp", layer, shared);
        for (int i = 0; i < g_.dim; ++i) out[i] += shared[i];
    }
    return norm(std::move(out), ln("post_ffw_norm.weight"));
}

Vec Engine::forward(int32_t token, const Trace& trace) {
    return forward_at(token, {}, -1, trace);
}

Vec Engine::forward_at(int32_t token, std::optional<RopePosition> position, int64_t next_rope, const Trace& trace) {
    if (token < 0 || token >= g_.vocab || history_.size() >= static_cast<size_t>(max_context_))
        throw std::runtime_error("hy4: token or context is out of range");
    const RopePosition p = position.value_or(RopePosition{next_rope_, next_rope_, next_rope_});
    const int64_t next = checked_next_position(p, next_rope);
    const Vec embedding = tensor("token_embd.weight").values(token);
    finite_row(embedding, g_.dim);
    return forward_row(embedding, token, p, next, trace);
}

Vec Engine::forward_embedding(std::span<const float> embedding, const RopePosition& position,
                              int64_t next_rope, const Trace& trace) {
    if (history_.size() >= static_cast<size_t>(max_context_)) throw std::runtime_error("hy4: context is out of range");
    finite_row(embedding, g_.dim);
    return forward_row(embedding, -1, position, checked_next_position(position, next_rope), trace);
}

Vec Engine::forward_row(std::span<const float> embedding, int32_t token, const RopePosition& position,
                        int64_t next_rope, const Trace& trace) {
    Vec streams;
    for (int h = 0; h < g_.hc; ++h) streams.insert(streams.end(), embedding.begin(), embedding.end());
    for (int layer = 0; layer < g_.layers; ++layer) {
        const auto it = std::find(g_.engram_layers.begin(), g_.engram_layers.end(), layer);
        if (token >= 0 && it != g_.engram_layers.end()) streams = engram(std::move(streams), layer, it - g_.engram_layers.begin(), token);
        const auto attn = mix(streams, layer, "attn");
        Vec normalized = norm(attn.value, layer_name(layer, "attn_norm.weight"));
        streams = finish_mix(streams, attention(normalized, layer, history_.size(), position, positions_, trace), attn.post);
        const auto feed = mix(streams, layer, "ffn");
        normalized = norm(feed.value, layer_name(layer, "ffn_norm.weight"));
        if (trace) trace("ffn_norm", layer, normalized);
        Vec f = ffn(normalized, layer, trace);
        streams = finish_mix(streams, f, feed.post);
        if (trace) { trace("ffn_out", layer, f); trace("l_out", layer, streams); }
    }
    Vec flat = streams;
    rms(flat, g_.eps);
    Vec mixes = linear("output_hc_fn.weight", flat);
    const Vec scale = tensor("output_hc_scale.weight").values();
    const Vec base = tensor("output_hc_base.weight").values();
    Vec hidden(g_.dim, 0);
    for (int h = 0; h < g_.hc; ++h) {
        const float weight = sigmoid(mixes[h] * scale[0] + base[h]) + g_.hc_eps;
        for (int i = 0; i < g_.dim; ++i) hidden[i] += streams[h * g_.dim + i] * weight;
    }
    hidden = norm(std::move(hidden), "output_norm.weight");
    if (trace) trace("result_norm", -1, hidden);
    Vec logits = linear("output.weight", hidden);
    for (float v : logits) if (!std::isfinite(v)) throw std::runtime_error("hy4: nonfinite logits");
    history_.push_back(token); positions_.push_back(position); next_rope_ = next_rope;
    last_hidden_ = std::move(hidden);
    return logits;
}

Vec Engine::draft_forward(int32_t token, std::span<const float> previous_hidden,
                          std::optional<RopePosition> position, int64_t next_rope, const Trace& trace) {
    if (token < 0 || token >= g_.vocab) throw std::runtime_error("hy4: draft token is out of range");
    const RopePosition p = position.value_or(RopePosition{draft_next_rope_, draft_next_rope_, draft_next_rope_});
    const Vec embedding = tensor("token_embd.weight").values(token);
    return draft_row(embedding, previous_hidden, token, p, checked_next_position(p, next_rope), trace);
}

Vec Engine::draft_forward_embedding(std::span<const float> embedding, std::span<const float> previous_hidden,
                                    const RopePosition& position, int64_t next_rope, const Trace& trace) {
    return draft_row(embedding, previous_hidden, -1, position, checked_next_position(position, next_rope), trace);
}

Vec Engine::draft_row(std::span<const float> embedding, std::span<const float> previous_hidden,
                      int32_t token, const RopePosition& position, int64_t next_rope, const Trace& trace) {
    if (!g_.draft_layers || draft_history_.size() >= static_cast<size_t>(max_context_))
        throw std::runtime_error("hy4: draft block unavailable or draft context is full");
    finite_row(embedding, g_.dim); finite_row(previous_hidden, g_.dim);
    const int layer = g_.layers;
    const auto ln = [&](const char* suffix) { return layer_name(layer, suffix); };
    Vec joined = norm(Vec(embedding.begin(), embedding.end()), ln("nextn.enorm.weight"));
    const Vec normalized_hidden = norm(Vec(previous_hidden.begin(), previous_hidden.end()), ln("nextn.hnorm.weight"));
    joined.insert(joined.end(), normalized_hidden.begin(), normalized_hidden.end());
    Vec x = linear(ln("nextn.eh_proj.weight"), joined);
    if (trace) trace("mtp_eh_proj", layer, x);
    Vec delta = attention(norm(x, ln("attn_norm.weight")), layer, draft_history_.size(), position, draft_positions_, trace);
    for (int i = 0; i < g_.dim; ++i) x[i] += delta[i];
    delta = ffn(norm(x, ln("ffn_norm.weight")), layer, trace);
    for (int i = 0; i < g_.dim; ++i) x[i] += delta[i];
    Vec hidden = norm(std::move(x), ln("nextn.shared_head_norm.weight"));
    if (trace) trace("mtp_result_norm", layer, hidden);
    Vec logits = linear("output.weight", hidden);
    for (float v : logits) if (!std::isfinite(v)) throw std::runtime_error("hy4: nonfinite draft logits");
    draft_history_.push_back(token); draft_positions_.push_back(position); draft_next_rope_ = next_rope;
    last_draft_hidden_ = std::move(hidden);
    return logits;
}

}  // namespace strata::hy4
