// A Model without weights or a GPU, for tests of the serve loop (ds4_main.cpp) alone: what a request reuses, what is
// set aside and taken back, and the lines it writes.  Linked in place of ds4_model.cu (target ds4_serve_fake); the
// state a conversation leaves is its tokens and its restore points, and a slot is those in bytes.  Every answer token
// is kAnswerToken.  tests/dsv41/serve_scope_test.py drives it.
//
//   DS4_FAKE_MARKS_INSIDE=1   restore points inside a read are kept in one run (feed_marks), as a long read's are
//   DS4_FAKE_ENCODE_LOG=<f>   a line for every picture the tower encodes (width height), appended to <f>
#include "ds4_model.hpp"
#include "ds4_packinfo.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::ds4 {

namespace {
constexpr int kAnswerToken = 7;
constexpr int kImageToken = 3;
constexpr int kDim = 8;
constexpr int kSpan = 4;   // the tokens every picture takes
constexpr uint32_t kSlotMagic = 0x4b414644u;

struct Seq {
    std::vector<int32_t> tokens;
    int64_t ck[2] = {-1, -1};
    int64_t anchor = -1;
    std::vector<int64_t> marks;
    bool changed = false;   // since the state was last set aside or taken back
};
}  // namespace

struct Model::Impl {
    std::vector<Seq> seqs;
    int bound = 0;
    int64_t max_context = 32768;
    int last_fed = 0;
    Stats stats;
    Seq& s() { return seqs[(size_t) bound]; }
    const Seq& s() const { return seqs[(size_t) bound]; }
    void push(const int* t, int n) {
        s().tokens.insert(s().tokens.end(), t, t + n);
        s().changed = true;
        last_fed = n;
    }
    std::vector<int64_t> points(const Seq& q) const {
        std::vector<int64_t> v = {q.ck[0], q.ck[1], q.anchor};
        v.insert(v.end(), q.marks.begin(), q.marks.end());
        v.erase(std::remove_if(v.begin(), v.end(), [&](int64_t p) { return p <= 0 || p > (int64_t) q.tokens.size(); }),
                v.end());
        std::sort(v.begin(), v.end(), std::greater<int64_t>());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    }
};

Model::Model() : p_(new Impl) {}
Model::~Model() = default;

bool Model::load(const Options& o, std::string&) {
    p_->seqs.assign((size_t) std::max(1, o.seqs), Seq{});
    p_->max_context = o.max_context;
    return true;
}

bool Model::step(int token, bool, std::string&) {
    p_->push(&token, 1);
    return true;
}
bool Model::feed(const int* tokens, int n, int, std::string&, int) {
    p_->push(tokens, n);
    return true;
}
bool Model::feed_mark(const int* tokens, int n, int, int slot, std::string&) {
    if (n > 1) p_->push(tokens, n - 1);
    checkpoint(slot);
    p_->push(tokens + n - 1, 1);
    p_->last_fed = n;
    return true;
}
bool Model::feed_marks(const int* tokens, int n, int rows, int slot, const std::vector<std::pair<int64_t, bool>>& inner,
                       std::string& err) {
    const int64_t start = pos();
    int done = 0;
    for (const auto& [at, shared] : inner) {
        const int upto = (int) (at - start);
        p_->push(tokens + done, upto - done);
        done = upto;
        mark(shared);
    }
    const bool ok = feed_mark(tokens + done, n - done, rows, slot, err);
    p_->last_fed = n;
    return ok;
}
bool Model::marks_inside(int, const std::vector<int64_t>&) const { return std::getenv("DS4_FAKE_MARKS_INSIDE") != nullptr; }
int Model::sample(float, float, int, float, uint64_t, uint64_t) { return kAnswerToken; }
void Model::set_penalties(int, float, float, float) {}
bool Model::logits(std::vector<float>&) { return false; }

int Model::draft_block() const { return 0; }
bool Model::gpus_are_gfx12() const { return false; }
bool Model::draft(int, std::vector<int>& drafts, std::vector<float>& conf, std::string&, float, float, int, float,
                  uint64_t, uint64_t) {
    drafts.clear();
    conf.clear();
    return true;
}
bool Model::verify(int y, const std::vector<int>&, float, float, int, float, uint64_t, uint64_t, int& accepted, int& next,
                   std::string&) {
    p_->push(&y, 1);
    accepted = 0;
    next = kAnswerToken;
    return true;
}

int Model::feed_unit() const { return 16; }
void Model::set_progress(std::function<void(int64_t)>) {}
void Model::set_cancel(std::function<bool()>) {}
bool Model::cancelled() const { return false; }
void Model::set_yield(std::function<bool()>) {}
int Model::fed() const { return p_->last_fed; }
int Model::prompt_chunk() const { return 0; }
int Model::pipeline_stages() const { return 1; }
void Model::begin_request() {}
int64_t Model::pos() const { return (int64_t) p_->s().tokens.size(); }
int64_t Model::max_context() const { return p_->max_context; }
int Model::vocab() const { return 1024; }
int Model::dim() const { return kDim; }
const std::vector<int32_t>& Model::tokens() const { return p_->s().tokens; }

int Model::seq_count() const { return (int) p_->seqs.size(); }
void Model::seq_bind(int s) { p_->bound = s; }
int Model::seq_bound() const { return p_->bound; }
int64_t Model::seq_pos(int s) const { return (int64_t) p_->seqs[(size_t) s].tokens.size(); }
const std::vector<int32_t>& Model::seq_tokens(int s) const { return p_->seqs[(size_t) s].tokens; }
bool Model::step_multi(const int* seqs, const int* tokens, int n, std::string&) {
    for (int i = 0; i < n; ++i) {
        Seq& q = p_->seqs[(size_t) seqs[i]];
        q.tokens.push_back(tokens[i]);
        q.changed = true;
    }
    return true;
}
int Model::sample_multi(int, float, float, int, float, uint64_t, uint64_t) { return kAnswerToken; }
bool Model::step_multi_spec(const int* seqs, const int* const* toks, const int*, int n, std::string&) {
    for (int i = 0; i < n; ++i) {
        Seq& q = p_->seqs[(size_t) seqs[i]];
        q.tokens.push_back(toks[i][0]);
        q.changed = true;
    }
    return true;
}
bool Model::can_draft_multi(const int*, int) const { return false; }
bool Model::draft_multi(const int*, const int*, int n, std::vector<std::vector<int>>& drafts,
                        std::vector<std::vector<float>>& conf, std::string&) {
    drafts.assign((size_t) n, {});
    conf.assign((size_t) n, {});
    return true;
}
bool Model::verify_multi(int, const std::vector<int>&, float, float, int, float, uint64_t, uint64_t, int& accepted,
                         int& next, std::string&) {
    accepted = 0;
    next = kAnswerToken;
    return true;
}
int Model::verify_max() const { return 16; }

void Model::reset() { p_->s() = Seq{}; }
void Model::settle_idle(const std::atomic<bool>&) {}
bool Model::router_nonfinite() { return false; }
void Model::checkpoint(int slot) { p_->s().ck[slot] = pos(); }
int64_t Model::restorable(int64_t limit) const {
    for (int64_t p : p_->points(p_->s()))
        if (p <= limit) return p;
    return -1;
}
bool Model::restore(int64_t limit) {
    const int64_t at = restorable(limit);
    if (at < 0) return false;
    Seq& q = p_->s();
    q.tokens.resize((size_t) at);
    for (int64_t& c : q.ck)
        if (c > at) c = -1;
    if (q.anchor > at) q.anchor = -1;
    q.marks.erase(std::remove_if(q.marks.begin(), q.marks.end(), [&](int64_t p) { return p > at; }), q.marks.end());
    q.changed = true;
    return true;
}
void Model::pin() { p_->s().anchor = p_->s().ck[1]; }
void Model::mark(bool) {
    Seq& q = p_->s();
    q.marks.push_back(pos());
    if ((int) q.marks.size() > marks_cap()) q.marks.erase(q.marks.begin());
}
int Model::marks_cap() const { return 4; }
size_t Model::check_bytes() const { return 0; }
std::vector<int64_t> Model::check_positions() const {
    std::vector<int64_t> v = p_->points(p_->s());
    v.erase(std::remove(v.begin(), v.end(), pos()), v.end());
    return v;
}

uint64_t Model::slot_save(std::vector<uint8_t>& out, std::string& err, int64_t at) {
    Seq& q = p_->s();
    if (q.tokens.empty()) {
        err = "there is no conversation";
        return 0;
    }
    if (at < 0 && !q.changed) {
        err = "it is already set aside";
        return 0;
    }
    const uint64_t n = at >= 0 ? (uint64_t) at : q.tokens.size();
    std::vector<int64_t> pts;
    for (int64_t p : p_->points(q))
        if ((uint64_t) p <= n) pts.push_back(p);
    const uint64_t np = pts.size();
    out.resize(4 + 8 + n * 4 + 8 + np * 8);
    uint8_t* w = out.data();
    std::memcpy(w, &kSlotMagic, 4), w += 4;
    std::memcpy(w, &n, 8), w += 8;
    std::memcpy(w, q.tokens.data(), n * 4), w += n * 4;
    std::memcpy(w, &np, 8), w += 8;
    std::memcpy(w, pts.data(), np * 8);
    if (at < 0) q.changed = false;
    return out.size();
}
bool Model::slot_load(const uint8_t* data, size_t bytes, int64_t n_pos, std::string& err) {
    uint32_t magic = 0;
    uint64_t n = 0, np = 0;
    if (bytes < 20 || (std::memcpy(&magic, data, 4), magic != kSlotMagic) || (std::memcpy(&n, data + 4, 8), n != (uint64_t) n_pos) ||
        bytes < 20 + n * 4) {
        err = "not this model's slot";
        return false;
    }
    Seq q;
    q.tokens.resize(n);
    std::memcpy(q.tokens.data(), data + 12, n * 4);
    std::memcpy(&np, data + 12 + n * 4, 8);
    q.marks.resize(np);
    std::memcpy(q.marks.data(), data + 20 + n * 4, np * 8);
    p_->s() = q;
    return true;
}
std::string Model::slot_identity() const { return "ds4-fake-model-1"; }

int Model::image_token() const { return kImageToken; }
bool Model::encode_image(const uint8_t*, int width, int height, std::vector<uint16_t>& span, std::string&) {
    if (const char* log = std::getenv("DS4_FAKE_ENCODE_LOG")) {
        if (std::FILE* f = std::fopen(log, "a")) {
            std::fprintf(f, "%d %d\n", width, height);
            std::fclose(f);
        }
    }
    span.assign((size_t) kSpan * kDim, 0);
    return true;
}
void Model::set_images(std::vector<Image>) {}
void Model::vision_idle() {}
bool Model::vision_lazy() const { return false; }

void Model::save_profile() const {}
const Stats& Model::stats() const { return p_->stats; }
std::string Model::describe() const { return "a fake model for serve-loop tests"; }
std::vector<std::string> Model::info_lines() const { return {}; }

int pack_info(const std::string&, std::FILE*) { return 0; }

}  // namespace strata::ds4
