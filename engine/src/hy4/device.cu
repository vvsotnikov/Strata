#include "strata/hy4/device.hpp"
#include "strata/hy4/device_experts.hpp"
#include "strata/hy4/device_memory.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "device_kernels.cuh"

#include <cublas_v2.h>
#include <cub/device/device_radix_sort.cuh>
#include <algorithm>
#include <list>
#include <numeric>
#include <sstream>
#include <unordered_map>

namespace strata::hy4 {
namespace {
namespace K = device_kernels;
void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("Hy4 GPU: ") + cudaGetErrorString(e));
}
void bc(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Hy4 cuBLAS error " + std::to_string(e));
}
int blocks(size_t n) { return int((n + 255) / 256); }
struct Buffer {
    void *p = nullptr;
    size_t bytes = 0;
    int device = -1;
    bool host = false;
    ~Buffer() { clear(); }
    Buffer() = default;
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    void clear() {
        if (p) {
            if (device >= 0)
                cudaSetDevice(device);
            if (host)
                cudaFreeHost(p);
            else
                cudaFree(p);
        }
        p = nullptr;
        bytes = 0;
    }
    void reserve(size_t n, int d, bool h = false) {
        if (n <= bytes && device == d && host == h)
            return;
        clear();
        ck(cudaSetDevice(d));
        if (h)
            ck(cudaHostAlloc(&p, n, cudaHostAllocMapped | cudaHostAllocPortable));
        else
            ck(cudaMalloc(&p, n));
        bytes = n;
        device = d;
        host = h;
    }
    template <class T> T *data() const {
        if (!host)
            return static_cast<T *>(p);
        void *out = nullptr;
        ck(cudaHostGetDevicePointer(&out, p, 0));
        return static_cast<T *>(out);
    }
};
struct Matrix {
    Buffer raw;
    uint64_t used = 0;
};
struct Card {
    int device;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    cudaEvent_t done = nullptr;
    size_t dense_limit = 0, dense_used = 0;
    uint64_t clock = 0, uploads = 0, upload_bytes = 0;
    std::unordered_map<std::string, std::unique_ptr<Matrix>> weights;
    std::unordered_map<std::string, std::unique_ptr<Buffer>> vectors;
    std::map<std::string, std::unique_ptr<Buffer>> scratch;
    Buffer blas_workspace, transient, map, draft_map, features, carry, slots, pos, cells, ringcells, rope, groupslots,
        groupstarts, groupphysical, grouprope, valid;
    Card(int d) : device(d) {
        ck(cudaSetDevice(d));
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ck(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        bc(cublasCreate(&blas));
        bc(cublasSetStream(blas, stream));
        bc(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
        // Own the cuBLAS workspace before residency budgets are derived. The
        // handle otherwise may allocate it lazily during the first dense GEMM.
        blas_workspace.reserve(4ull << 20, device);
        bc(cublasSetWorkspace(blas, blas_workspace.p, blas_workspace.bytes));
    }
    ~Card() {
        cudaSetDevice(device);
        cudaStreamSynchronize(stream);
        if (blas)
            cublasDestroy(blas);
        if (done)
            cudaEventDestroy(done);
        if (stream)
            cudaStreamDestroy(stream);
    }
    template <class T> T *work(const std::string &name, size_t n) {
        auto &b = scratch[name];
        if (!b)
            b = std::make_unique<Buffer>();
        b->reserve(n * sizeof(T), device);
        return b->data<T>();
    }
    template <class T> void copy(Buffer &b, std::span<T> x) {
        b.reserve(x.size_bytes(), device);
        if (!x.empty())
            ck(cudaMemcpyAsync(b.p, x.data(), x.size_bytes(), cudaMemcpyHostToDevice, stream));
    }
    const float *vector(const Tensor &t) {
        auto found = vectors.find(t.info->name);
        if (found != vectors.end())
            return found->second->data<float>();
        Vec values = t.values();
        auto b = std::make_unique<Buffer>();
        b->reserve(values.size() * sizeof(float), device);
        ck(cudaMemcpyAsync(b->p, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice,
                           stream));
        ck(cudaStreamSynchronize(stream));
        const float *result = b->data<float>();
        vectors.emplace(t.info->name, std::move(b));
        return result;
    }
    const void *matrix(const Tensor &t) {
        auto found = weights.find(t.info->name);
        if (found != weights.end()) {
            found->second->used = ++clock;
            return found->second->raw.p;
        }
        size_t bytes = t.row_bytes() * (t.info->elements() / t.info->shape[0]);
        if (bytes > dense_limit) {
            transient.reserve(bytes, device);
            ck(cudaMemcpyAsync(transient.p, t.data, bytes, cudaMemcpyHostToDevice, stream));
            ++uploads;
            upload_bytes += bytes;
            return transient.p;
        }
        while (dense_used + bytes > dense_limit) {
            auto victim = std::min_element(weights.begin(), weights.end(), [](const auto &a, const auto &b) {
                return a.second->used < b.second->used;
            });
            if (victim == weights.end())
                throw std::runtime_error("Hy4 dense cache accounting");
            ck(cudaStreamSynchronize(stream));
            dense_used -= victim->second->raw.bytes;
            weights.erase(victim);
        }
        auto value = std::make_unique<Matrix>();
        value->raw.reserve(bytes, device);
        value->used = ++clock;
        ck(cudaMemcpyAsync(value->raw.p, t.data, bytes, cudaMemcpyHostToDevice, stream));
        const void *result = value->raw.p;
        dense_used += bytes;
        ++uploads;
        upload_bytes += bytes;
        weights.emplace(t.info->name, std::move(value));
        return result;
    }
};
struct LayerCache {
    int card = 0;
    Buffer kv, pooled, raw_value, raw_gate;
};
struct Sequence {
    std::vector<int32_t> tokens, physical_blocks, next_after;
    std::vector<std::array<int32_t, 3>> positions;
    int32_t next_rope = 0;
};
class Model final : public DeviceModel {
  public:
    Model(const Engine &source, const DeviceOptions &options)
        : source_(source), g_(source.geometry()), o_(options), sequences_(options.slots),
          draft_sequences_(options.slots), feature_positions_(options.slots, std::vector<int32_t>(32, -1)),
          carry_valid_(options.slots, false) {
        if (o_.devices.empty() || o_.slots < 1 || o_.slots > 64 || o_.batch < 1 || o_.batch > 8 ||
            o_.max_context < 1 || o_.max_context > g_.max_context || o_.context_pool < g_.pool ||
            o_.context_pool > g_.max_context * int64_t(o_.slots))
            throw std::runtime_error("invalid Hy4 device geometry or capacities");
        if ((!o_.dense_bytes.empty() && o_.dense_bytes.size() != o_.devices.size()) ||
            (!o_.expert_bytes.empty() && o_.expert_bytes.size() != o_.devices.size()))
            throw std::runtime_error("Hy4 dense/expert budget vectors must match selected CUDA devices");
        if (g_.topk / g_.pool - 1 > 512)
            throw std::runtime_error("Hy4 pooled selection supports up to 512 historical groups");
        auto sorted = o_.devices;
        std::sort(sorted.begin(), sorted.end());
        if (sorted.front() < 0 || std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            throw std::runtime_error("invalid/duplicate Hy4 device");
        int count;
        ck(cudaGetDeviceCount(&count));
        if (sorted.back() >= count)
            throw std::runtime_error("Hy4 CUDA device unavailable");
        ring_ = g_.window + o_.batch;
        pool_blocks_ = o_.context_pool / g_.pool;
        for (int i = int(pool_blocks_) - 1; i >= 0; --i) {
            free_.push_back(i);
            draft_free_.push_back(i);
        }
        g_.indexer.resize(g_.layers + g_.draft_layers, 1);
        const int max_heads = *std::max_element(g_.heads.begin(), g_.heads.end());
        for (int id : o_.devices) {
            cards_.push_back(std::make_unique<Card>(id));
            auto &c = *cards_.back();
            c.map.reserve(size_t(o_.slots) * o_.max_context * sizeof(int32_t), id);
            c.draft_map.reserve(c.map.bytes, id);
        }
        for (size_t i = 0; i < cards_.size(); ++i)
            for (size_t j = 0; j < cards_.size(); ++j)
                if (i != j) {
                    ck(cudaSetDevice(cards_[i]->device));
                    int can = 0;
                    ck(cudaDeviceCanAccessPeer(&can, cards_[i]->device, cards_[j]->device));
                    if (can) {
                        auto e = cudaDeviceEnablePeerAccess(cards_[j]->device, 0);
                        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                            ck(e);
                        cudaGetLastError();
                    }
                }
        const size_t kv_width = g_.krank + g_.rope_dim, kv_item = o_.f32_kv ? sizeof(float) : sizeof(half);
        layers_.reserve(g_.layers + g_.draft_layers);
        for (int l = 0; l < g_.layers + g_.draft_layers; ++l) {
            auto cache = std::make_unique<LayerCache>();
            cache->card = l % cards_.size();
            auto &c = *cards_[cache->card];
            ck(cudaSetDevice(c.device));
            size_t cells = g_.swa[l] ? size_t(o_.slots) * ring_ : pool_blocks_ * g_.pool;
            cache->kv.reserve(cells * kv_width * kv_item, c.device, o_.host_kv);
            if (g_.indexer[l]) {
                cache->pooled.reserve(pool_blocks_ * g_.index_dim * sizeof(float), c.device);
                cache->raw_value.reserve(
                    size_t(o_.slots) * (g_.pool + o_.batch) * g_.index_dim * sizeof(float), c.device);
                cache->raw_gate.reserve(cache->raw_value.bytes, c.device);
            }
            layers_.push_back(std::move(cache));
        }
        auto &head = *cards_[(g_.layers - 1) % cards_.size()];
        head.features.reserve(size_t(o_.slots) * 32 * g_.dim * 4, head.device);
        if (g_.draft_layers) {
            auto &d = *cards_[g_.layers % cards_.size()];
            d.carry.reserve(size_t(o_.slots) * g_.dim * 4, d.device);
            ck(cudaMemset(d.carry.p, 0, d.carry.bytes));
        }
        // Establish all persistent scratch before deriving automatic expert budgets.
        for (auto &ptr : cards_) {
            auto &c = *ptr;
            ck(cudaSetDevice(c.device));
            const size_t b = o_.batch, d = g_.dim, hc = g_.hc, heads = max_heads,
                         largest = std::max({g_.dense_ff, g_.ff, g_.latent, g_.qrank,
                                             int(heads) * g_.head_dim, int(heads) * g_.value_dim,
                                             g_.index_dim * g_.index_heads, g_.dim * (g_.hc + 1)});
            for (const char *key : {"x", "flat", "keys", "query_norm"})
                c.work<float>(key, b * d * hc);
            for (const char *key :
                 {"norm", "mix_value", "branch", "latent", "expert_out", "shared", "value", "hidden"})
                c.work<float>(key, b * std::max(d, size_t(largest)));
            for (const char *key :
                 {"tmp", "gate", "up", "query", "packed", "head_out", "attention", "linear_scratch"})
                c.work<float>(
                    key, b * std::max({size_t(largest), heads * size_t(g_.krank), heads * size_t(g_.head_dim),
                                       heads * size_t(g_.value_dim), d * (hc + 1)}));
            c.work<float>("qr", b * g_.qrank);
            c.work<float>("kv", b * kv_width);
            c.work<float>("mixes", b * hc * 2);
            c.work<float>("post", b * hc);
            c.work<float>("logits", b * g_.vocab);
            c.work<float>("router", b * g_.experts);
            c.work<int32_t>("ids", b * g_.used);
            c.work<float>("route_weights", b * g_.used);
            c.work<uint8_t>("q8", b * std::max(size_t(largest), heads * size_t(g_.krank)) * 36 / 32 + 4096);
            c.work<float>("partial", heads * ((std::max(g_.topk, g_.window) + 127) / 128) * (g_.krank + 2));
            const size_t engram_width = size_t(g_.ngram - 1) * g_.hash_heads * g_.engram_dim;
            for (const char *key : {"engram_rows", "engram_normalized"})
                c.work<float>(key, b * engram_width);
            c.work<float>("engram_kv", b * d * (hc + 1));
            for (const char *key : {"index_key", "index_value", "index_gate", "pool_values"})
                c.work<float>(key, b * g_.index_dim);
            c.work<float>("index_query", b * g_.index_heads * g_.index_dim);
            c.work<float>("index_weights", b * g_.index_heads);
            if (g_.draft_layers) {
                for (const char *key : {"mtp_previous", "mtp_embedding", "mtp_x", "mtp_gather", "mtp_head"})
                    c.work<float>(key, b * d);
                c.work<float>("mtp_combined", b * d * 2);
            }
            c.work<uint8_t>("q8", b * std::max({size_t(largest), heads * size_t(g_.krank), engram_width, 2 * d}) * 36 / 32 + 4096);
            c.work<int32_t>("selected", b * g_.topk);
            c.work<int32_t>("selected_groups", 512);
            size_t groups = (o_.max_context + g_.pool - 1) / g_.pool;
            c.work<float>("scores", groups);
            c.work<float>("sorted_scores", groups);
            auto *ids = c.work<int32_t>("group_ids", groups);
            auto *sorted_ids = c.work<int32_t>("sorted_group_ids", groups);
            K::group_ids<<<blocks(groups), 256, 0, c.stream>>>(ids, int(groups));
            size_t temp = 0;
            ck(cub::DeviceRadixSort::SortPairsDescending(nullptr, temp, c.work<float>("scores", groups),
                                                         c.work<float>("sorted_scores", groups), ids,
                                                         sorted_ids, int(groups), 0, 32, c.stream));
            c.work<uint8_t>("sort_temp", temp);
            c.slots.reserve(b * 4, c.device);
            c.pos.reserve(b * 4, c.device);
            c.cells.reserve(b * 4, c.device);
            c.ringcells.reserve(b * 4, c.device);
            c.rope.reserve(b * 3 * 4, c.device);
            c.groupslots.reserve(b * 4, c.device);
            c.groupstarts.reserve(b * 4, c.device);
            c.groupphysical.reserve(b * 4, c.device);
            c.grouprope.reserve(b * 3 * 4, c.device);
            c.valid.reserve(b * 4, c.device);
            ck(cudaStreamSynchronize(c.stream));
        }
        // Materialize every norm/bias/scale vector before measuring free VRAM.
        // Some vectors are stored as two-dimensional GGUF tensors; identify them
        // by their executed role rather than tensor rank.
        for (int l = 0; l < g_.layers + g_.draft_layers; ++l) {
            auto &c = *cards_[l % cards_.size()];
            std::vector<std::string> names{
                "attn_norm.weight", "ffn_norm.weight", "post_attention_norm.weight", "post_ffw_norm.weight",
                "attn_q_a_norm.weight", "attn_kv_a_norm.weight", "attn_sinks.weight"};
            if (l < g_.layers) {
                for (const char *name : {"hc_attn_scale.weight", "hc_attn_base.weight", "hc_ffn_scale.weight", "hc_ffn_base.weight"})
                    names.push_back(name);
            } else {
                for (const char *name : {"nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight"})
                    names.push_back(name);
            }
            if (g_.indexer[l]) {
                for (const char *name : {"indexer.k_norm.weight", "indexer_compressor_norm.weight", "indexer_compressor_ape.weight"})
                    names.push_back(name);
            }
            if (std::find(g_.engram_layers.begin(), g_.engram_layers.end(), l) != g_.engram_layers.end()) {
                for (const char *name : {"engram_lookup_norm.weight", "engram_kv_b.weight", "engram_pre_gamma.weight", "engram_gate.weight", "engram_hc_scale.weight"})
                    names.push_back(name);
            }
            if (source_.tensors().contains(ln(l, "exp_probs_b.bias")))
                names.push_back("exp_probs_b.bias");
            for (const auto &name : names)
                c.vector(tensor(ln(l, name.c_str())));
        }
        for (const char *name : {"output_hc_scale.weight", "output_hc_base.weight", "output_norm.weight"})
            head.vector(tensor(name));
        std::vector<size_t> dense(cards_.size()), largest_matrix(cards_.size());
        auto inventory = [&](size_t card, const Tensor &t) {
            const size_t bytes = t.row_bytes() * (t.info->elements() / t.info->shape[0]);
            dense[card] += bytes;
            largest_matrix[card] = std::max(largest_matrix[card], bytes);
        };
        // Executed trunk/draft matrices only; host embeddings and preloaded
        // multi-dimensional norm vectors never enter the dense cache.
        for (int l = 0; l < g_.layers + g_.draft_layers; ++l) {
            const size_t ci = l % cards_.size();
            for (const auto &[name, t] : source_.tensors())
                if (name.starts_with("blk." + std::to_string(l) + ".") && t.info->shape.size() >= 2 &&
                    name.find("_exps.") == std::string::npos && !cards_[ci]->vectors.contains(name))
                    inventory(ci, t);
        }
        inventory((g_.layers - 1) % cards_.size(), tensor("output.weight"));
        inventory((g_.layers - 1) % cards_.size(), tensor("output_hc_fn.weight"));
        const auto requirement = ExpertExecutor::memory_requirements(
            g_, [&](const std::string &n) -> const Tensor & { return tensor(n); }, o_.batch);
        std::vector<ExpertDevice> expert;
        for (size_t i = 0; i < cards_.size(); ++i) {
            auto &c = *cards_[i];
            ck(cudaSetDevice(c.device));
            ck(cudaStreamSynchronize(c.stream));
            size_t avail, total;
            ck(cudaMemGetInfo(&avail, &total));
            const auto plan = detail::plan_device_memory(
                avail, requirement.device_workspace, o_.vram_reserve, dense[i], largest_matrix[i],
                o_.dense_bytes.empty() ? std::optional<size_t>{} : std::optional<size_t>{o_.dense_bytes[i]},
                o_.expert_bytes.empty() ? std::optional<size_t>{} : std::optional<size_t>{o_.expert_bytes[i]});
            c.dense_limit = plan.dense_capacity;
            // This is needed only for matrices too large for the actual dense
            // cache. Reserving every largest matrix unconditionally would waste
            // hundreds of MiB on a head which is already GPU resident.
            if (plan.transient_bytes)
                c.transient.reserve(plan.transient_bytes, c.device);
            expert.push_back({c.device, plan.expert_capacity});
        }
        experts_ = std::make_unique<ExpertExecutor>(
            g_, [&](const std::string &n) -> const Tensor & { return tensor(n); }, expert, o_.batch);
        // Whole-layer preloading enables the device-only routed fast path where it fits.
        for (int l = g_.dense_layers; l < g_.layers; ++l)
            experts_->preload_layer(l);
    }
    ~Model() {
        try {
            if (experts_)
                experts_->synchronize();
            for (auto &c : cards_) {
                cudaSetDevice(c->device);
                cudaStreamSynchronize(c->stream);
            }
        } catch (...) {
        }
    }
    int slots() const override { return o_.slots; }
    int batch() const override { return o_.batch; }
    bool healthy() const override { return !failed_; }
    bool has_mtp() const override { return g_.draft_layers > 0; }
    size_t draft_length(int slot) const override {
        sequence(slot);
        return draft_sequences_[slot].tokens.size();
    }
    void draft_truncate(int slot, size_t length) override {
        sequence(slot);
        const auto before = draft_length(slot);
        rewind(draft_sequences_[slot], draft_free_, length);
        if (draft_length(slot) != before)
            carry_valid_[slot] = false;
    }
    void draft_accept(int slot, size_t position) override;
    std::vector<Vec> draft_forward(std::span<const DeviceInput>, bool) override;
    size_t length(int slot) const override { return sequence(slot).tokens.size(); }
    std::span<const int32_t> history(int slot) const override { return sequence(slot).tokens; }
    size_t free_tokens() const override { return free_.size() * g_.pool; }
    void reset(int slot) override {
        auto &q = sequence(slot);
        for (auto block : q.physical_blocks)
            free_.push_back(block);
        q = Sequence{};
        rewind(draft_sequences_[slot], draft_free_, 0);
        std::fill(feature_positions_[slot].begin(), feature_positions_[slot].end(), -1);
        carry_valid_[slot] = false;
    }
    void truncate(int slot, size_t length) override {
        auto &q = sequence(slot);
        if (q.tokens.size() - std::min(length, q.tokens.size()) > size_t(o_.batch)) {
            reset(slot);
            return;
        }
        const auto before = q.tokens.size();
        rewind(q, free_, length);
        if (q.tokens.size() != before)
            carry_valid_[slot] = false;
    }
    std::string stats() const override {
        auto e = experts_->stats();
        std::ostringstream s;
        s << "kv_pool=" << pool_blocks_ * g_.pool << " kv_free=" << free_tokens() << " expert_hits=" << e.hits
          << " expert_misses=" << e.misses << " expert_upload_bytes=" << e.uploaded_bytes
          << " resident_calls=" << e.resident_calls;
        return s.str();
    }
    std::vector<Vec> forward(std::span<const DeviceInput> input, const Trace &trace) override;

  private:
    const Tensor &tensor(const std::string &n) const { return source_.tensor(n); }
    void rewind(Sequence &q, std::vector<int32_t> &free, size_t length) {
        if (length > q.tokens.size())
            throw std::runtime_error("Hy4 rewind beyond sequence");
        if (q.tokens.size() - length > size_t(o_.batch))
            length = 0;
        size_t keep = (length + g_.pool - 1) / g_.pool;
        while (q.physical_blocks.size() > keep) {
            free.push_back(q.physical_blocks.back());
            q.physical_blocks.pop_back();
        }
        q.tokens.resize(length);
        q.positions.resize(length);
        q.next_after.resize(length);
        q.next_rope = length ? q.next_after.back() : 0;
    }
    Sequence &sequence(int slot) {
        if (slot < 0 || slot >= o_.slots)
            throw std::runtime_error("Hy4 slot out of range");
        return sequences_[slot];
    }
    const Sequence &sequence(int slot) const {
        if (slot < 0 || slot >= o_.slots)
            throw std::runtime_error("Hy4 slot out of range");
        return sequences_[slot];
    }
    std::string ln(int l, const char *s) const { return "blk." + std::to_string(l) + "." + s; }
    const float *vector(Card &c, const std::string &n) { return c.vector(tensor(n)); }
    float *work(Card &c, const char *n, size_t values) { return c.work<float>(n, values); }
    void mm(Card &c, const std::string &name, const float *x, float *out, int rows) {
        const auto &w = tensor(name);
        int ni = w.info->shape[0], no = w.info->shape[1];
        const void *data = c.matrix(w);
        int type = w.info->type;
        if (kernels::native_mmvq_supported(type)) {
            void *q = c.work<uint8_t>("q8", size_t(rows) * ni / 32 * 36);
            kernels::native_quantize_q8_1(x, q, ni, rows, c.stream);
            kernels::native_mmvq(type, data, q, out, ni, no, rows, c.stream);
        } else if (type == 0) {
            const float one = 1, zero = 0;
            bc(cublasSgemm(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, no, rows, ni, &one,
                           static_cast<const float *>(data), ni, x, ni, &zero, out, no));
        } else if (type == 1 || type == 30)
            K::dense_16<<<(no + 3) / 4, 128, 0, c.stream>>>(static_cast<const uint16_t *>(data), x, out, ni,
                                                            no, rows, type);
        else
            throw std::runtime_error("unsupported Hy4 GPU matrix type " + std::to_string(type));
    }
    void norm(Card &c, const float *x, float *out, int rows, int width, const std::string &w = {},
              float eps = -1, int weight_rows = 1) {
        K::rms<<<rows, 256, 0, c.stream>>>(x, w.empty() ? nullptr : vector(c, w), out, width,
                                           eps < 0 ? g_.eps : eps, weight_rows);
    }
    void planes(Card &c, const std::string &name, const float *input, float *out, int rows) {
        const auto &w = tensor(name);
        int ni = w.info->shape[0], no = w.info->shape[1], nh = w.info->shape[2];
        const void *data = c.matrix(w);
        int type = w.info->type;
        if (type == 8) {
            auto *q = c.work<uint8_t>("q8", size_t(rows) * nh * ni / 32 * 36);
            kernels::native_quantize_q8_1(input, q, rows * nh * ni, 1, c.stream);
            K::planes_q8<<<dim3((no + 3) / 4, nh, rows), 128, 0, c.stream>>>(
                static_cast<const uint8_t *>(data), q, out, ni, no, rows, nh);
        } else if (type == 0 || type == 1 || type == 30)
            K::planes_float<<<dim3(no, rows, nh), 256, 0, c.stream>>>(data, input, out, ni, no, rows, nh,
                                                                      type);
        else
            throw std::runtime_error("unsupported Hy4 GPU head matrix format");
    }
    void mix(Card &c, float *streams, float *normalized, float *post, int layer, const char *kind, int rows) {
        float *flat = work(c, "flat", size_t(rows) * g_.dim * g_.hc);
        norm(c, streams, flat, rows, g_.dim * g_.hc);
        float *mixes = work(c, "mixes", size_t(rows) * 2 * g_.hc);
        std::string base = ln(layer, (std::string("hc_") + kind).c_str());
        mm(c, base + "_fn.weight", flat, mixes, rows);
        float *value = work(c, "mix_value", size_t(rows) * g_.dim);
        K::hc_mix<<<rows, 256, 0, c.stream>>>(streams, mixes, vector(c, base + "_scale.weight"),
                                              vector(c, base + "_base.weight"), value, post, g_.dim, g_.hc,
                                              g_.hc_eps, g_.hc_magnitude);
        norm(c, value, normalized, rows, g_.dim,
             ln(layer, std::string(kind) == "attn" ? "attn_norm.weight" : "ffn_norm.weight"));
    }
    void engram(Card &c, float *streams, int layer, int module, int rows, const Vec &data) {
        auto *lookup = work(c, "engram_rows", data.size());
        ck(cudaMemcpyAsync(lookup, data.data(), data.size() * 4, cudaMemcpyHostToDevice, c.stream));
        int cols = (g_.ngram - 1) * g_.hash_heads;
        float *normalized = work(c, "engram_normalized", data.size());
        norm(c, lookup, normalized, rows * cols, g_.engram_dim, ln(layer, "engram_lookup_norm.weight"),
             0x1p-23f, cols);
        float *kv = work(c, "engram_kv", size_t(rows) * g_.dim * (g_.hc + 1));
        mm(c, ln(layer, "engram_kv.weight"), normalized, kv, rows);
        K::add_bias<<<blocks(size_t(rows) * g_.dim * (g_.hc + 1)), 256, 0, c.stream>>>(
            kv, vector(c, ln(layer, "engram_kv_b.weight")), g_.dim * (g_.hc + 1), rows);
        float *keys = work(c, "keys", size_t(rows) * g_.dim * g_.hc),
              *value = work(c, "value", size_t(rows) * g_.dim);
        K::unpack_engram<<<blocks(size_t(rows) * (g_.hc + 1) * g_.dim), 256, 0, c.stream>>>(
            kv, keys, value, g_.dim, g_.hc, rows);
        norm(c, keys, keys, rows * g_.hc, g_.dim, {}, 0x1p-23f);
        float *query = work(c, "query_norm", size_t(rows) * g_.dim * g_.hc);
        norm(c, streams, query, rows * g_.hc, g_.dim, {}, 0x1p-23f);
        norm(c, value, value, rows, g_.dim, ln(layer, "engram_pre_gamma.weight"), 0x1p-23f);
        K::engram_delta<<<dim3(rows, g_.hc), 256, 0, c.stream>>>(
            streams, query, keys, value, vector(c, ln(layer, "engram_gate.weight")),
            vector(c, ln(layer, "engram_hc_scale.weight")), c.valid.data<int32_t>(), g_.dim, g_.hc);
    }
    void attention(Card &c, float *input, float *out, int layer, int rows, const std::vector<int32_t> &slots,
                   const std::vector<int32_t> &pos, int groups, const Trace &trace);
    void ffn(Card &c, float *input, float *out, int layer, int rows) {
        float *gate = work(c, "gate", size_t(rows) * std::max(g_.dense_ff, g_.ff));
        float *up = work(c, "up", size_t(rows) * std::max(g_.dense_ff, g_.ff));
        if (layer < g_.dense_layers) {
            mm(c, ln(layer, "ffn_gate.weight"), input, gate, rows);
            mm(c, ln(layer, "ffn_up.weight"), input, up, rows);
            K::glu<<<blocks(size_t(rows) * g_.dense_ff), 256, 0, c.stream>>>(gate, up, rows * g_.dense_ff);
            mm(c, ln(layer, "ffn_down.weight"), gate, out, rows);
        } else {
            float *scores = work(c, "router", size_t(rows) * g_.experts);
            mm(c, ln(layer, "ffn_gate_inp.weight"), input, scores, rows);
            auto *ids = c.work<int32_t>("ids", size_t(rows) * g_.used);
            float *weights = work(c, "route_weights", size_t(rows) * g_.used);
            const float *bias = nullptr;
            if (source_.tensors().contains(ln(layer, "exp_probs_b.bias")))
                bias = vector(c, ln(layer, "exp_probs_b.bias"));
            K::route<<<rows, 256, 0, c.stream>>>(scores, bias, ids, weights, g_.experts, g_.used,
                                                 g_.expert_scale);
            float *latent = work(c, "latent", size_t(rows) * g_.latent),
                  *experts = work(c, "expert_out", size_t(rows) * g_.latent);
            mm(c, ln(layer, "ffn_routed_down.weight"), input, latent, rows);
            experts_->run(layer, c.device, latent, ids, weights, experts, rows, c.stream);
            mm(c, ln(layer, "ffn_routed_up.weight"), experts, out, rows);
            mm(c, ln(layer, "ffn_gate_shexp.weight"), input, gate, rows);
            mm(c, ln(layer, "ffn_up_shexp.weight"), input, up, rows);
            K::glu<<<blocks(size_t(rows) * g_.ff), 256, 0, c.stream>>>(gate, up, rows * g_.ff);
            float *shared = work(c, "shared", size_t(rows) * g_.dim);
            mm(c, ln(layer, "ffn_down_shexp.weight"), gate, shared, rows);
            K::add<<<blocks(size_t(rows) * g_.dim), 256, 0, c.stream>>>(out, shared, rows * g_.dim);
        }
        norm(c, out, out, rows, g_.dim, ln(layer, "post_ffw_norm.weight"));
    }
    void dump(Card &c, const Trace &trace, const char *name, int layer, const float *p, size_t n) {
        if (!trace)
            return;
        Vec host(n);
        ck(cudaMemcpyAsync(host.data(), p, n * 4, cudaMemcpyDeviceToHost, c.stream));
        ck(cudaStreamSynchronize(c.stream));
        trace(name, layer, host);
    }
    const Engine &source_;
    Geometry g_;
    DeviceOptions o_;
    std::vector<std::unique_ptr<Card>> cards_;
    std::vector<std::unique_ptr<LayerCache>> layers_;
    std::vector<Sequence> sequences_, draft_sequences_;
    std::vector<std::vector<int32_t>> feature_positions_;
    std::vector<bool> carry_valid_;
    std::vector<int32_t> free_, draft_free_;
    std::unique_ptr<ExpertExecutor> experts_;
    int ring_ = 0;
    size_t pool_blocks_ = 0;
    bool failed_ = false;
};

void Model::attention(Card &c, float *input, float *out, int layer, int rows,
                      const std::vector<int32_t> &slots, const std::vector<int32_t> &pos, int complete_groups,
                      const Trace &trace) {
    auto &cache = *layers_[layer];
    int heads = g_.heads[layer], latent = g_.krank, rot = g_.rope_dim, hd = g_.head_dim, idx = g_.index_dim;
    float base = g_.swa[layer] ? g_.rope_base_swa : g_.rope_base;
    float *qr = work(c, "qr", size_t(rows) * g_.qrank);
    mm(c, ln(layer, "attn_q_a.weight"), input, qr, rows);
    norm(c, qr, qr, rows, g_.qrank, ln(layer, "attn_q_a_norm.weight"));
    dump(c, trace, "q_resid", layer, qr, size_t(rows) * g_.qrank);
    float *query = work(c, "query", size_t(rows) * heads * hd);
    mm(c, ln(layer, "attn_q_b.weight"), qr, query, rows);
    K::rope<<<blocks(size_t(rows) * heads * rot / 2), 256, 0, c.stream>>>(query, c.rope.data<int32_t>(), rows,
                                                                          heads, hd, rot, base);
    float *kv = work(c, "kv", size_t(rows) * (latent + rot));
    mm(c, ln(layer, "attn_kv_a_mqa.weight"), input, kv, rows);
    K::kv_normalize<<<rows, 256, 0, c.stream>>>(kv, vector(c, ln(layer, "attn_kv_a_norm.weight")),
                                                latent + rot, latent, g_.eps);
    K::rope<<<blocks(size_t(rows) * rot / 2), 256, 0, c.stream>>>(kv, c.rope.data<int32_t>(), rows, 1,
                                                                  latent + rot, rot, base);
    const auto *cells = g_.swa[layer] ? c.ringcells.data<int32_t>() : c.cells.data<int32_t>();
    if (o_.f32_kv)
        K::kv_write<<<rows, 256, 0, c.stream>>>(cache.kv.data<float>(), kv, cells, latent + rot);
    else
        K::kv_write<<<rows, 256, 0, c.stream>>>(cache.kv.data<half>(), kv, cells, latent + rot);
    float *packed = work(c, "packed", size_t(rows) * heads * std::max(latent, hd));
    float *absorbed = work(c, "head_out", size_t(rows) * heads * latent);
    K::heads_pack<<<blocks(size_t(rows) * heads * (hd - rot)), 256, 0, c.stream>>>(query, packed, rows, heads,
                                                                                   hd, hd - rot, 0);
    planes(c, ln(layer, "attn_k_b.weight"), packed, absorbed, rows);
    float *iq = nullptr;
    float *iw = nullptr;
    if (g_.indexer[layer]) {
        float *key = work(c, "index_key", size_t(rows) * idx);
        mm(c, ln(layer, "indexer.attn_k.weight"), input, key, rows);
        norm(c, key, key, rows, idx, ln(layer, "indexer.k_norm.weight"));
        float *v = work(c, "index_value", size_t(rows) * idx),
              *gate = work(c, "index_gate", size_t(rows) * idx);
        mm(c, ln(layer, "indexer_compressor_value.weight"), key, v, rows);
        mm(c, ln(layer, "indexer_compressor_gate.weight"), key, gate, rows);
        K::index_ring<<<rows, 256, 0, c.stream>>>(v, gate, cache.raw_value.data<float>(),
                                                  cache.raw_gate.data<float>(), c.slots.data<int32_t>(),
                                                  c.pos.data<int32_t>(), idx, g_.pool + o_.batch);
        if (complete_groups) {
            float *p = work(c, "pool_values", size_t(complete_groups) * idx);
            K::pool<<<complete_groups, 256, 0, c.stream>>>(
                cache.raw_value.data<float>(), cache.raw_gate.data<float>(),
                vector(c, ln(layer, "indexer_compressor_ape.weight")), p, c.groupslots.data<int32_t>(),
                c.groupstarts.data<int32_t>(), idx, g_.pool, g_.pool + o_.batch);
            norm(c, p, p, complete_groups, idx, ln(layer, "indexer_compressor_norm.weight"));
            K::rope<<<blocks(size_t(complete_groups) * rot / 2), 256, 0, c.stream>>>(
                p, c.grouprope.data<int32_t>(), complete_groups, 1, idx, rot, base);
            K::pool_store<<<complete_groups, 256, 0, c.stream>>>(cache.pooled.data<float>(), p,
                                                                 c.groupphysical.data<int32_t>(), idx);
        }
        if (*std::max_element(pos.begin(), pos.end()) + 1 > g_.topk) {
            iq = work(c, "index_query", size_t(rows) * g_.index_heads * idx);
            iw = work(c, "index_weights", size_t(rows) * g_.index_heads);
            mm(c, ln(layer, "indexer.attn_q_b.weight"), qr, iq, rows);
            K::rope<<<blocks(size_t(rows) * g_.index_heads * rot / 2), 256, 0, c.stream>>>(
                iq, c.rope.data<int32_t>(), rows, g_.index_heads, idx, rot, base);
            mm(c, ln(layer, "indexer.proj.weight"), input, iw, rows);
        }
    }
    float *latent_out = work(c, "attention", size_t(rows) * heads * latent);
    float *partial = work(c, "partial", size_t(heads) * ((g_.topk + 127) / 128) * (latent + 2));
    for (int row = 0; row < rows; ++row) {
        int selected_count = g_.swa[layer] ? std::min(g_.window, pos[row] + 1) : pos[row] + 1;
        const int32_t *selected = nullptr;
        const int32_t *map =
            (layer >= g_.layers ? c.draft_map : c.map).data<int32_t>() + size_t(slots[row]) * o_.max_context;
        if (g_.indexer[layer] && selected_count > g_.topk) {
            int ng = pos[row] / g_.pool, take = std::min(ng, g_.topk / g_.pool - 1);
            float *scores = work(c, "scores", (o_.max_context + g_.pool - 1) / g_.pool);
            float *sorted = work(c, "sorted_scores", (o_.max_context + g_.pool - 1) / g_.pool);
            auto *ids = c.work<int32_t>("group_ids", (o_.max_context + g_.pool - 1) / g_.pool);
            auto *sorted_ids = c.work<int32_t>("sorted_group_ids", (o_.max_context + g_.pool - 1) / g_.pool);
            K::index_scores<<<(ng + 7) / 8, 256, 0, c.stream>>>(
                cache.pooled.data<float>(), iq + size_t(row) * g_.index_heads * idx,
                iw + row * g_.index_heads, map, scores, ng, idx, g_.index_heads, g_.pool,
                1.f / std::sqrt(float(idx * g_.index_heads)));
            auto &temp = *c.scratch.at("sort_temp");
            size_t bytes = temp.bytes;
            ck(cub::DeviceRadixSort::SortPairsDescending(temp.p, bytes, scores, sorted, ids, sorted_ids, ng,
                                                         0, 32, c.stream));
            auto *selected_groups = c.work<int32_t>("selected_groups", 512);
            K::selected_sort<<<1, 256, 0, c.stream>>>(sorted_ids, selected_groups, take);
            auto *sel = c.work<int32_t>("selected", size_t(o_.batch) * g_.topk) + size_t(row) * g_.topk;
            selected_count = take * g_.pool + pos[row] % g_.pool + 1;
            K::selection<<<blocks(selected_count), 256, 0, c.stream>>>(selected_groups, sel, take, pos[row],
                                                                       g_.pool);
            selected = sel;
        }
        int chunks = (selected_count + 127) / 128;
        const float *sinks = vector(c, ln(layer, "attn_sinks.weight"));
        size_t offset = g_.swa[layer] ? size_t(slots[row]) * ring_ * (latent + rot) : 0;
        if (o_.f32_kv)
            K::attn_split<<<dim3(heads, chunks), 256, 0, c.stream>>>(
                absorbed, query, cache.kv.data<float>() + offset, map, selected, sinks, partial, row,
                pos[row], heads, latent, hd, rot, g_.value_dim, selected_count, chunks,
                g_.swa[layer] ? ring_ : 0);
        else
            K::attn_split<<<dim3(heads, chunks), 256, 0, c.stream>>>(
                absorbed, query, cache.kv.data<half>() + offset, map, selected, sinks, partial, row, pos[row],
                heads, latent, hd, rot, g_.value_dim, selected_count, chunks, g_.swa[layer] ? ring_ : 0);
        K::attn_merge<<<heads, 256, 0, c.stream>>>(partial, latent_out, rows, heads, latent, chunks, row);
    }
    K::heads_pack<<<blocks(size_t(rows) * heads * latent), 256, 0, c.stream>>>(latent_out, packed, rows,
                                                                               heads, latent, latent, 0);
    float *context = work(c, "head_out", size_t(rows) * heads * g_.value_dim);
    planes(c, ln(layer, "attn_v_b.weight"), packed, context, rows);
    float *gate = work(c, "gate", size_t(rows) * heads * g_.value_dim);
    mm(c, ln(layer, "attn_gate.weight"), input, gate, rows);
    K::gate<<<blocks(size_t(rows) * heads * g_.value_dim), 256, 0, c.stream>>>(context, gate,
                                                                               rows * heads * g_.value_dim);
    mm(c, ln(layer, "attn_output.weight"), context, out, rows);
    norm(c, out, out, rows, g_.dim, ln(layer, "post_attention_norm.weight"));
    dump(c, trace, "attn_out", layer, out, size_t(rows) * g_.dim);
}

void Model::draft_accept(int slot, size_t position) {
    sequence(slot);
    if (!has_mtp())
        throw std::runtime_error("Hy4 artifact has no MTP block");
    if (position >= length(slot) || feature_positions_[slot][position % 32] != int64_t(position))
        throw std::runtime_error("Hy4 target MTP feature is unavailable");
    auto &source = *cards_[(g_.layers - 1) % cards_.size()];
    auto &destination = *cards_[g_.layers % cards_.size()];
    const float *from = source.features.data<float>() + (size_t(slot) * 32 + position % 32) * g_.dim;
    float *to = destination.carry.data<float>() + size_t(slot) * g_.dim;
    ck(cudaSetDevice(destination.device));
    if (source.device == destination.device)
        ck(cudaMemcpyAsync(to, from, size_t(g_.dim) * 4, cudaMemcpyDeviceToDevice, destination.stream));
    else {
        ck(cudaEventRecord(source.done, source.stream));
        ck(cudaStreamWaitEvent(destination.stream, source.done, 0));
        ck(cudaMemcpyPeerAsync(to, destination.device, from, source.device, size_t(g_.dim) * 4,
                               destination.stream));
        ck(cudaEventRecord(destination.done, destination.stream));
        ck(cudaStreamWaitEvent(source.stream, destination.done, 0));
    }
    carry_valid_[slot] = true;
}

std::vector<Vec> Model::draft_forward(std::span<const DeviceInput> inputs, bool target_features) {
    if (failed_)
        throw std::runtime_error("Hy4 GPU model must be recreated after an execution failure");
    if (!has_mtp() || inputs.empty() || inputs.size() > size_t(o_.batch))
        throw std::runtime_error("invalid Hy4 MTP batch");
    const int rows = inputs.size(), layer = g_.layers;
    auto &c = *cards_[layers_[layer]->card];
    auto &head = *cards_[(g_.layers - 1) % cards_.size()];
    std::vector<size_t> next(o_.slots);
    std::vector<int32_t> next_rope(o_.slots);
    std::vector<bool> seen(o_.slots);
    size_t required = 0;
    for (int s = 0; s < o_.slots; ++s) {
        next[s] = draft_sequences_[s].tokens.size();
        next_rope[s] = draft_sequences_[s].next_rope;
    }
    for (const auto &in : inputs) {
        sequence(in.slot);
        if (!target_features && !carry_valid_[in.slot])
            throw std::runtime_error("MTP carry is unavailable after reset or rewind; accept a target "
                                     "feature or replay canonical rows first");
        if (!target_features && seen[in.slot])
            throw std::runtime_error("recursive MTP rows must belong to distinct sequences");
        seen[in.slot] = true;
        size_t p = next[in.slot];
        if (p >= size_t(o_.max_context))
            throw std::runtime_error("MTP context exhausted");
        if (in.embedding.empty() && (in.token < 0 || in.token >= g_.vocab))
            throw std::runtime_error("invalid MTP token");
        if (!in.embedding.empty() &&
            (in.embedding.size() != size_t(g_.dim) || std::any_of(in.embedding.begin(), in.embedding.end(),
                                                                  [](float x) { return !std::isfinite(x); })))
            throw std::runtime_error("invalid MTP embedding");
        if (target_features &&
            (p > length(in.slot) || (p > 0 && feature_positions_[in.slot][(p - 1) % 32] != int64_t(p - 1))))
            throw std::runtime_error("MTP target feature is outside the retained window");
        auto rp = in.rope;
        if (target_features && p < sequence(in.slot).positions.size())
            rp = sequence(in.slot).positions[p];
        else if (std::all_of(rp.begin(), rp.end(), [](int32_t v) { return v < 0; }))
            rp = {next_rope[in.slot], next_rope[in.slot], next_rope[in.slot]};
        if (in.next_rope < -1 ||
            std::any_of(
                rp.begin(), rp.end(), [](int32_t v) { return v < 0; }) ||
            (in.next_rope < 0 && rp[0] == INT32_MAX))
            throw std::runtime_error("invalid MTP rotary position");
        next_rope[in.slot] = in.next_rope >= 0 ? in.next_rope : rp[0] + 1;
        if (p % g_.pool == 0)
            ++required;
        ++next[in.slot];
    }
    if (required > draft_free_.size())
        throw std::runtime_error("Hy4 MTP KV pool exhausted");
    struct FailureGuard {
        bool &failed;
        int before = std::uncaught_exceptions();
        ~FailureGuard() {
            if (std::uncaught_exceptions() > before)
                failed = true;
        }
    } failure{failed_};
    std::vector<int32_t> slots, pos, cells, ringcells, positions, gs, gstart, gphysical, gpositions;
    Vec embedding;
    for (const auto &in : inputs) {
        auto &q = draft_sequences_[in.slot];
        int p = q.tokens.size();
        if (p % g_.pool == 0) {
            q.physical_blocks.push_back(draft_free_.back());
            draft_free_.pop_back();
        }
        auto rp = in.rope;
        if (target_features && p < sequence(in.slot).positions.size())
            rp = sequence(in.slot).positions[p];
        else if (rp[0] < 0)
            rp = {q.next_rope, q.next_rope, q.next_rope};
        auto v = in.embedding.empty() ? tensor("token_embd.weight").values(in.token)
                                      : Vec(in.embedding.begin(), in.embedding.end());
        embedding.insert(embedding.end(), v.begin(), v.end());
        slots.push_back(in.slot);
        pos.push_back(p);
        cells.push_back(q.physical_blocks[p / g_.pool] * g_.pool + p % g_.pool);
        ringcells.push_back(in.slot * ring_ + p % ring_);
        positions.insert(positions.end(), rp.begin(), rp.end());
        q.tokens.push_back(in.embedding.empty() ? in.token : -1);
        q.positions.push_back(rp);
        q.next_rope = target_features && size_t(p) < sequence(in.slot).next_after.size()
                          ? sequence(in.slot).next_after[p]
                          : (in.next_rope >= 0 ? in.next_rope : rp[0] + 1);
        q.next_after.push_back(q.next_rope);
        if (p % g_.pool == g_.pool - 1) {
            int start = p - g_.pool + 1;
            gs.push_back(in.slot);
            gstart.push_back(start);
            gphysical.push_back(q.physical_blocks[start / g_.pool]);
            auto first = q.positions[start];
            gpositions.insert(gpositions.end(), first.begin(), first.end());
        }
    }
    ck(cudaSetDevice(c.device));
    c.copy(c.slots, std::span(slots));
    c.copy(c.pos, std::span(pos));
    c.copy(c.cells, std::span(cells));
    c.copy(c.ringcells, std::span(ringcells));
    c.copy(c.rope, std::span(positions));
    if (!gs.empty()) {
        c.copy(c.groupslots, std::span(gs));
        c.copy(c.groupstarts, std::span(gstart));
        c.copy(c.groupphysical, std::span(gphysical));
        c.copy(c.grouprope, std::span(gpositions));
    }
    K::map_write<<<1, 256, 0, c.stream>>>(c.draft_map.data<int32_t>(), c.slots.data<int32_t>(),
                                          c.pos.data<int32_t>(), c.cells.data<int32_t>(), rows,
                                          o_.max_context);
    float *previous = work(c, "mtp_previous", size_t(rows) * g_.dim);
    if (target_features) {
        if (c.device == head.device)
            K::feature_gather<<<rows, 256, 0, c.stream>>>(head.features.data<float>(), previous,
                                                          c.slots.data<int32_t>(), c.pos.data<int32_t>(),
                                                          g_.dim, 32);
        else {
            ck(cudaSetDevice(head.device));
            head.copy(head.slots, std::span(slots));
            head.copy(head.pos, std::span(pos));
            float *gather = work(head, "mtp_gather", size_t(rows) * g_.dim);
            K::feature_gather<<<rows, 256, 0, head.stream>>>(head.features.data<float>(), gather,
                                                             head.slots.data<int32_t>(),
                                                             head.pos.data<int32_t>(), g_.dim, 32);
            ck(cudaEventRecord(head.done, head.stream));
            ck(cudaSetDevice(c.device));
            ck(cudaStreamWaitEvent(c.stream, head.done, 0));
            ck(cudaMemcpyPeerAsync(previous, c.device, gather, head.device, size_t(rows) * g_.dim * 4,
                                   c.stream));
        }
    } else
        K::carry_gather<<<rows, 256, 0, c.stream>>>(c.carry.data<float>(), previous, c.slots.data<int32_t>(),
                                                    g_.dim);
    float *token = work(c, "mtp_embedding", embedding.size());
    ck(cudaMemcpyAsync(token, embedding.data(), embedding.size() * 4, cudaMemcpyHostToDevice, c.stream));
    norm(c, token, token, rows, g_.dim, ln(layer, "nextn.enorm.weight"));
    norm(c, previous, previous, rows, g_.dim, ln(layer, "nextn.hnorm.weight"));
    float *combined = work(c, "mtp_combined", size_t(rows) * g_.dim * 2);
    K::concat<<<rows, 256, 0, c.stream>>>(token, previous, combined, g_.dim);
    float *x = work(c, "mtp_x", size_t(rows) * g_.dim);
    mm(c, ln(layer, "nextn.eh_proj.weight"), combined, x, rows);
    float *normalized = work(c, "norm", size_t(rows) * g_.dim),
          *branch = work(c, "branch", size_t(rows) * g_.dim);
    norm(c, x, normalized, rows, g_.dim, ln(layer, "attn_norm.weight"));
    attention(c, normalized, branch, layer, rows, slots, pos, gs.size(), {});
    K::add<<<blocks(size_t(rows) * g_.dim), 256, 0, c.stream>>>(x, branch, rows * g_.dim);
    norm(c, x, normalized, rows, g_.dim, ln(layer, "ffn_norm.weight"));
    ffn(c, normalized, branch, layer, rows);
    K::add<<<blocks(size_t(rows) * g_.dim), 256, 0, c.stream>>>(x, branch, rows * g_.dim);
    norm(c, x, x, rows, g_.dim, ln(layer, "nextn.shared_head_norm.weight"));
    K::carry_store<<<rows, 256, 0, c.stream>>>(x, c.carry.data<float>(), c.slots.data<int32_t>(), g_.dim,
                                               rows);
    const float *projected = x;
    ck(cudaSetDevice(head.device));
    if (head.device != c.device) {
        float *gathered = work(head, "mtp_head", size_t(rows) * g_.dim);
        ck(cudaEventRecord(c.done, c.stream));
        ck(cudaStreamWaitEvent(head.stream, c.done, 0));
        ck(cudaMemcpyPeerAsync(gathered, head.device, x, c.device, size_t(rows) * g_.dim * 4, head.stream));
        projected = gathered;
    }
    float *logits = work(head, "logits", size_t(rows) * g_.vocab);
    mm(head, "output.weight", projected, logits, rows);
    Vec host(size_t(rows) * g_.vocab);
    ck(cudaMemcpyAsync(host.data(), logits, host.size() * 4, cudaMemcpyDeviceToHost, head.stream));
    ck(cudaStreamSynchronize(head.stream));
    for (float v : host)
        if (!std::isfinite(v))
            throw std::runtime_error("nonfinite MTP logits");
    std::vector<Vec> result;
    for (int r = 0; r < rows; ++r) {
        result.emplace_back(host.begin() + size_t(r) * g_.vocab, host.begin() + size_t(r + 1) * g_.vocab);
        carry_valid_[slots[r]] = true;
    }
    return result;
}

std::vector<Vec> Model::forward(std::span<const DeviceInput> inputs, const Trace &trace) {
    if (failed_)
        throw std::runtime_error("Hy4 GPU model must be recreated after an execution failure");
    if (inputs.empty() || inputs.size() > size_t(o_.batch))
        throw std::runtime_error("Hy4 GPU batch must contain 1..8 rows");
    int rows = inputs.size();
    std::vector<int32_t> slots, pos, cells, ringcells, rope, valid, group_slots, group_starts, group_physical,
        group_rope;
    Vec embeddings;
    std::vector<Vec> engrams(g_.engram_layers.size());
    // Validate the entire submission before mutating state or reserving pooled blocks.
    std::vector<size_t> next(o_.slots);
    std::vector<int32_t> next_rope(o_.slots);
    for (int s = 0; s < o_.slots; ++s) {
        next[s] = length(s);
        next_rope[s] = sequence(s).next_rope;
    }
    size_t required = 0;
    for (const auto &in : inputs) {
        sequence(in.slot);
        if (in.embedding.empty() && (in.token < 0 || in.token >= g_.vocab))
            throw std::runtime_error("Hy4 token out of range");
        if (!in.embedding.empty() &&
            (in.embedding.size() != size_t(g_.dim) || std::any_of(in.embedding.begin(), in.embedding.end(),
                                                                  [](float v) { return !std::isfinite(v); })))
            throw std::runtime_error("Hy4 invalid image embedding");
        if (next[in.slot] >= size_t(o_.max_context))
            throw std::runtime_error("Hy4 sequence context exhausted");
        if (next[in.slot] % g_.pool == 0)
            ++required;
        ++next[in.slot];
        auto p = in.rope;
        bool automatic = std::all_of(p.begin(), p.end(), [](int32_t v) { return v < 0; });
        if (automatic)
            p = {next_rope[in.slot], next_rope[in.slot], next_rope[in.slot]};
        if (in.next_rope < -1 ||
            std::any_of(
                p.begin(), p.end(), [](int32_t v) { return v < 0; }) ||
            (in.next_rope < 0 && p[0] == INT32_MAX))
            throw std::runtime_error("Hy4 invalid rotary position");
        next_rope[in.slot] = in.next_rope >= 0 ? in.next_rope : p[0] + 1;
    }
    if (required > free_.size())
        throw std::runtime_error("Hy4 unified KV pool exhausted");
    struct FailureGuard {
        bool &failed;
        int before = std::uncaught_exceptions();
        ~FailureGuard() {
            if (std::uncaught_exceptions() > before)
                failed = true;
        }
    } failure{failed_};
    for (const auto &in : inputs) {
        auto &q = sequence(in.slot);
        int p = q.tokens.size();
        if (p % g_.pool == 0) {
            q.physical_blocks.push_back(free_.back());
            free_.pop_back();
        }
        int cell = q.physical_blocks[p / g_.pool] * g_.pool + p % g_.pool;
        auto rp = in.rope;
        if (rp[0] < 0)
            rp = {q.next_rope, q.next_rope, q.next_rope};
        if (std::any_of(rp.begin(), rp.end(), [](int x) { return x < 0; }))
            throw std::runtime_error("negative Hy4 rotary position");
        slots.push_back(in.slot);
        pos.push_back(p);
        cells.push_back(cell);
        ringcells.push_back(in.slot * ring_ + p % ring_);
        rope.insert(rope.end(), rp.begin(), rp.end());
        bool is_valid = in.embedding.empty() && g_.token_map[in.token] >= 0;
        valid.push_back(is_valid);
        Vec embedding = in.embedding.empty() ? tensor("token_embd.weight").values(in.token)
                                             : Vec(in.embedding.begin(), in.embedding.end());
        for (int h = 0; h < g_.hc; ++h)
            embeddings.insert(embeddings.end(), embedding.begin(), embedding.end());
        for (size_t e = 0; e < g_.engram_layers.size(); ++e) {
            if (is_valid) {
                auto hashes = engram_hash(g_, e, q.tokens, in.token);
                for (int row : hashes) {
                    auto v = tensor("per_layer_token_embd.weight").values(row);
                    engrams[e].insert(engrams[e].end(), v.begin(), v.end());
                }
            } else
                engrams[e].insert(engrams[e].end(), size_t(g_.ngram - 1) * g_.hash_heads * g_.engram_dim, 0);
        }
        q.tokens.push_back(in.embedding.empty() ? in.token : -1);
        q.positions.push_back(rp);
        q.next_rope = in.next_rope >= 0 ? in.next_rope : rp[0] + 1;
        q.next_after.push_back(q.next_rope);
        if (p % g_.pool == g_.pool - 1) {
            int start = p - g_.pool + 1;
            group_slots.push_back(in.slot);
            group_starts.push_back(start);
            group_physical.push_back(q.physical_blocks[start / g_.pool]);
            auto first = q.positions[start];
            group_rope.insert(group_rope.end(), first.begin(), first.end());
        }
    }
    for (auto &ptr : cards_) {
        auto &c = *ptr;
        ck(cudaSetDevice(c.device));
        c.copy(c.slots, std::span(slots));
        c.copy(c.pos, std::span(pos));
        c.copy(c.cells, std::span(cells));
        c.copy(c.ringcells, std::span(ringcells));
        c.copy(c.rope, std::span(rope));
        c.copy(c.valid, std::span(valid));
        if (!group_slots.empty()) {
            c.copy(c.groupslots, std::span(group_slots));
            c.copy(c.groupstarts, std::span(group_starts));
            c.copy(c.groupphysical, std::span(group_physical));
            c.copy(c.grouprope, std::span(group_rope));
        }
        K::map_write<<<1, 256, 0, c.stream>>>(c.map.data<int32_t>(), c.slots.data<int32_t>(),
                                              c.pos.data<int32_t>(), c.cells.data<int32_t>(), rows,
                                              o_.max_context);
    }
    int previous = 0;
    auto &first = *cards_[0];
    ck(cudaSetDevice(first.device));
    float *x = work(first, "x", embeddings.size());
    ck(cudaMemcpyAsync(x, embeddings.data(), embeddings.size() * 4, cudaMemcpyHostToDevice, first.stream));
    for (int l = 0; l < g_.layers; ++l) {
        int ci = layers_[l]->card;
        auto &c = *cards_[ci];
        ck(cudaSetDevice(c.device));
        float *current = work(c, "x", embeddings.size());
        if (ci != previous) {
            auto &src = *cards_[previous];
            ck(cudaEventRecord(src.done, src.stream));
            ck(cudaStreamWaitEvent(c.stream, src.done, 0));
            ck(cudaMemcpyPeerAsync(current, c.device, work(src, "x", embeddings.size()), src.device,
                                   embeddings.size() * 4, c.stream));
        }
        x = current;
        previous = ci;
        auto it = std::find(g_.engram_layers.begin(), g_.engram_layers.end(), l);
        if (it != g_.engram_layers.end())
            engram(c, x, l, it - g_.engram_layers.begin(), rows, engrams[it - g_.engram_layers.begin()]);
        float *normalized = work(c, "norm", size_t(rows) * g_.dim),
              *post = work(c, "post", size_t(rows) * g_.hc),
              *branch = work(c, "branch", size_t(rows) * g_.dim);
        mix(c, x, normalized, post, l, "attn", rows);
        attention(c, normalized, branch, l, rows, slots, pos, group_slots.size(), trace);
        K::hc_post<<<rows, 256, 0, c.stream>>>(x, branch, post, g_.dim, g_.hc);
        mix(c, x, normalized, post, l, "ffn", rows);
        dump(c, trace, "ffn_norm", l, normalized, size_t(rows) * g_.dim);
        ffn(c, normalized, branch, l, rows);
        dump(c, trace, "ffn_out", l, branch, size_t(rows) * g_.dim);
        K::hc_post<<<rows, 256, 0, c.stream>>>(x, branch, post, g_.dim, g_.hc);
        dump(c, trace, "l_out", l, x, size_t(rows) * g_.dim * g_.hc);
        ck(cudaGetLastError());
    }
    auto &c = *cards_[previous];
    float *flat = work(c, "flat", embeddings.size());
    norm(c, x, flat, rows, g_.dim * g_.hc);
    float *mixes = work(c, "mixes", size_t(rows) * g_.hc);
    mm(c, "output_hc_fn.weight", flat, mixes, rows);
    float *hidden = work(c, "hidden", size_t(rows) * g_.dim);
    K::hc_head<<<rows, 256, 0, c.stream>>>(x, mixes, vector(c, "output_hc_scale.weight"),
                                           vector(c, "output_hc_base.weight"), hidden, g_.dim, g_.hc,
                                           g_.hc_eps);
    norm(c, hidden, hidden, rows, g_.dim, "output_norm.weight");
    K::feature_store<<<rows, 256, 0, c.stream>>>(hidden, c.features.data<float>(), c.slots.data<int32_t>(),
                                                 c.pos.data<int32_t>(), g_.dim, 32);
    for (int r = 0; r < rows; ++r)
        feature_positions_[slots[r]][pos[r] % 32] = pos[r];
    dump(c, trace, "result_norm", -1, hidden, size_t(rows) * g_.dim);
    float *logits = work(c, "logits", size_t(rows) * g_.vocab);
    mm(c, "output.weight", hidden, logits, rows);
    Vec host(size_t(rows) * g_.vocab);
    ck(cudaMemcpyAsync(host.data(), logits, host.size() * 4, cudaMemcpyDeviceToHost, c.stream));
    ck(cudaStreamSynchronize(c.stream));
    for (float v : host)
        if (!std::isfinite(v))
            throw std::runtime_error("Hy4 GPU nonfinite logits");
    std::vector<Vec> result;
    for (int r = 0; r < rows; ++r)
        result.emplace_back(host.begin() + size_t(r) * g_.vocab, host.begin() + size_t(r + 1) * g_.vocab);
    return result;
}
} // namespace
std::unique_ptr<DeviceModel> device_model(const Engine &source, const DeviceOptions &options) {
    return std::make_unique<Model>(source, options);
}
} // namespace strata::hy4
