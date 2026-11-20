#pragma once

#include "strata/qwen35/sampling.hpp"
#include <any>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace strata::hy4::spec {
using Token = int32_t;
using Logits = std::vector<float>;
using Sampling = strata::qwen35::Sampling;
using Stop = std::function<bool(Token)>;

// Dense token-ID indexed probabilities. Constructing normalizes finite nonnegative
// weights once. No sparse support is omitted from the rejection residual.
class Distribution {
  public:
    explicit Distribution(std::vector<double> weights);
    const std::vector<double> &probabilities() const { return p_; }
    double at(Token token) const;
    size_t size() const { return p_.size(); }
    Token mode() const;
    Token sample(std::mt19937_64 &rng) const;

  private:
    std::vector<double> p_;
};

// The history must be the same history the ordinary sampler would count, including
// the currently consumed anchor token. Logit bias/grammar masks, if any, must already
// have been applied to the logits by the caller for both ordinary and speculative runs.
Distribution filter_logits(std::span<const float> logits, const Sampling &sampling,
                           std::span<const Token> penalty_history);

struct Proposal {
    Token token;
    Distribution q;
};
struct Verification {
    std::vector<Token> emitted; // Accepted drafts followed by one correction/bonus, except after EOS.
    size_t accepted = 0;
    bool rejected = false, bonus = false, stopped = false;
    std::optional<Token> next_seed; // Emitted correction/bonus not consumed by the target yet.
};

// target[i] predicts proposals[i], conditioned on seed + proposals[0:i].
// target.back() predicts the bonus. In stochastic mode proposals must actually have
// been sampled from their recorded q; acceptance_rng is independent of proposal_rng.
Verification verify(std::span<const Proposal> proposals, std::span<const Distribution> target, bool greedy,
                    std::mt19937_64 &acceptance_rng, std::mt19937_64 &target_rng, const Stop &stop = {});

struct RoundOptions {
    int max_draft_tokens = 5;
    double min_draft_probability = 0;
    Sampling target_sampling, draft_sampling;
    Stop stop;
};
struct RoundCallbacks {
    // These model calls consume the named tokens at their absolute positions.
    // draft_logits uses the current draft hidden carry (initially last target h).
    std::function<Logits(Token, int64_t)> draft_logits;
    std::function<std::vector<Logits>(std::span<const Token>, int64_t)> target_logits;
    std::function<std::any()> target_save, draft_save;
    std::function<void(const std::any &)> target_restore, draft_restore;
    // Keep accepted candidate KV cells while restoring any overwritten prefix/ring/
    // pool state. Merely resetting a logical cursor is not a valid implementation.
    std::function<void(const std::any &, int64_t)> target_keep_prefix;
    // Rebuild accepted MTP positions with the corresponding target h_{p-1}, not the
    // provisional draft h. Preserve verification h rows until this and keep complete.
    std::function<void(int64_t, std::span<const Token>)> draft_replay;
    // Set next round's seed carry to the last accepted TARGET h, verification row m.
    // Canonical replay leaves a draft h' which must never be used as the next seed's h.
    std::function<void(size_t)> draft_accept;
};
struct RoundResult {
    Verification verification;
    size_t drafted = 0, target_inputs_kept = 0;
    int64_t committed_length = 0;
    bool retained_draft_seed = false;
};
struct RollbackFailure final : std::runtime_error {
    std::exception_ptr original;
    explicit RollbackFailure(std::exception_ptr error)
        : std::runtime_error("Hy4 speculation: cache rollback failed; discard this session"),
          original(std::move(error)) {}
};

// Seed has already been emitted/sampled but is not in target KV at position N.
// penalty_history includes seed; output_budget is the output room AFTER seed.
// This round verifies [seed, drafts], commits only seed + accepted drafts to KV,
// and emits accepted drafts + one unconsumed replacement/bonus (unless EOS).
// On failure both target and draft restore their original checkpoints. A
// RollbackFailure means one model could not restore and the session must be discarded.
RoundResult run_round(Token seed, int64_t position, size_t output_budget,
                      std::span<const Token> penalty_history, const RoundOptions &options,
                      const RoundCallbacks &callbacks, std::mt19937_64 &proposal_rng,
                      std::mt19937_64 &acceptance_rng, std::mt19937_64 &target_rng);
} // namespace strata::hy4::spec
