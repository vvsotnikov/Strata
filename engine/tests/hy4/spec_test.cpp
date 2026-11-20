#include "strata/hy4/spec.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace strata::hy4::spec;
namespace {
void check(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
Distribution delta(int token, int vocab = 4) {
    std::vector<double> p(vocab, 0);
    p.at(token) = 1;
    return Distribution(std::move(p));
}
Logits greedy(int token) {
    Logits logits(4, -4);
    logits.at(token) = 4;
    return logits;
}
std::mt19937_64 proposal_rng(11), acceptance_rng(12), target_rng(13);

void filters() {
    Sampling s;
    s.temperature = 1;
    s.top_k = 0;
    const auto p = filter_logits(Logits{2, 1, 0}, s, {});
    check(std::abs(p.at(0) - std::exp(2) / (std::exp(2) + std::exp(1) + 1)) < 1e-12, "softmax");
    s.top_k = 1;
    check(filter_logits(Logits{2, 1, 0}, s, {}).at(0) == 1, "top-k");
    s.top_k = 0;
    s.top_p = .5;
    check(filter_logits(Logits{2, 1, 0}, s, {}).at(0) == 1, "top-p");
    s.top_p = 1;
    s.min_p = 1;
    const auto tie = filter_logits(Logits{2, 2, 0}, s, {});
    check(tie.at(0) == .5 && tie.at(1) == .5 && tie.at(2) == 0, "min-p ties");
    s.min_p = 0;
    const auto masked = filter_logits(Logits{-std::numeric_limits<float>::infinity(), 2, 0}, s, {});
    check(masked.at(0) == 0 && masked.mode() == 1, "grammar/allowed-token masks preserve support");
    s.min_p = 0;
    s.temperature = 0;
    s.repeat = 2;
    s.freq = 1;
    s.present = 1;
    s.last_n = 2;
    check(filter_logits(Logits{2, 1, 0}, s, std::vector<Token>{0, 0}).mode() == 1, "history penalties");
    s.repeat = 1e-42f;
    s.freq = s.present = 0;
    bool threw = false;
    try {
        filter_logits(Logits{2, 1, 0}, s, std::vector<Token>{0});
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, "penalty overflow must fail before sorting/sampling");
}
void acceptance() {
    std::vector<Proposal> draft{{1, delta(1)}, {2, delta(2)}};
    std::vector<Distribution> p{delta(1), delta(2), delta(3)};
    auto result = verify(draft, p, true, acceptance_rng, target_rng);
    check(result.accepted == 2 && result.bonus && result.emitted == std::vector<Token>({1, 2, 3}) &&
              result.next_seed == 3,
          "all accepted + bonus");
    p[1] = delta(0);
    result = verify(draft, p, true, acceptance_rng, target_rng);
    check(result.accepted == 1 && result.rejected && result.emitted == std::vector<Token>({1, 0}) &&
              result.next_seed == 0,
          "greedy rejection");
    p[0] = delta(0);
    result = verify(draft, p, false, acceptance_rng, target_rng);
    check(result.accepted == 0 && result.rejected && result.emitted == std::vector<Token>({0}),
          "disjoint support residual");
    p[0] = delta(1);
    result = verify(draft, p, true, acceptance_rng, target_rng, [](Token t) { return t == 1; });
    check(result.accepted == 1 && result.stopped && !result.next_seed &&
              result.emitted == std::vector<Token>({1}),
          "accepted EOS has no bonus");
    bool threw = false;
    try {
        std::vector<Proposal> bad{{0, delta(1)}};
        verify(bad, std::vector<Distribution>{delta(0), delta(0)}, false, acceptance_rng, target_rng);
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw, "a recorded proposal cannot have zero q probability");

    // This catches sample-and-match, an unnormalized residual and omitted q support.
    Distribution q({.2, .8}), target({.8, .2});
    int count = 0;
    constexpr int draws = 100000;
    for (int i = 0; i < draws; ++i) {
        const Token proposed = q.sample(proposal_rng);
        const std::vector<Proposal> proposals{{proposed, q}};
        const auto r =
            verify(proposals, std::vector<Distribution>{target, target}, false, acceptance_rng, target_rng);
        count += r.emitted[0] == 0;
    }
    check(std::abs(double(count) / draws - .8) < .01, "rejection must preserve target distribution");
}

struct Model {
    struct State {
        std::vector<Token> kv;
        int carry = 1002;
    };
    State target{{0, 1, 2}}, draft{{0, 1, 2}};
    std::vector<int> target_modes{1, 2, 0}, draft_modes{1, 2};
    std::vector<int> verified_h;
    std::vector<std::pair<int64_t, Token>> replay;
    std::vector<int> replay_inputs;
    size_t accepted_h = 999;
    bool fail_verify = false, fail_restore = false, constant_logits = false, draft_restored = false;
    RoundCallbacks callbacks() {
        RoundCallbacks cb;
        cb.target_save = [&]() { return std::any(target); };
        cb.draft_save = [&]() { return std::any(draft); };
        cb.target_restore = [&](const std::any &cp) {
            if (fail_restore)
                throw std::runtime_error("failed restore");
            target = std::any_cast<State>(cp);
        };
        cb.draft_restore = [&](const std::any &cp) {
            draft = std::any_cast<State>(cp);
            draft_restored = true;
        };
        cb.draft_logits = [&](Token token, int64_t pos) {
            check(int64_t(draft.kv.size()) == pos, "draft position");
            draft.kv.push_back(token);
            draft.carry = 2000 + int(pos);
            return constant_logits ? Logits{4, 3, 2, 1} : greedy(draft_modes.at(size_t(pos - 3)));
        };
        cb.target_logits = [&](std::span<const Token> input, int64_t pos) {
            check(int64_t(target.kv.size()) == pos, "target position");
            verified_h.clear();
            std::vector<Logits> logits;
            for (size_t i = 0; i < input.size(); ++i) {
                target.kv.push_back(input[i]);
                target.carry = 1000 + int(pos + i);
                verified_h.push_back(target.carry);
                logits.push_back(constant_logits ? Logits{4, 3, 2, 1} : greedy(target_modes.at(i)));
            }
            if (fail_verify)
                throw std::runtime_error("verification failed");
            return logits;
        };
        cb.target_keep_prefix = [&](const std::any &base, int64_t n) {
            check(std::any_cast<const State &>(base).kv.size() == 3, "base checkpoint");
            check(n >= 3 && size_t(n) <= target.kv.size(), "target prefix keep");
            target.kv.resize(size_t(n));
            target.carry = 1000 + int(n) - 1;
            draft.carry = -1; // A target rewind invalidates a previously published draft carry.
        };
        cb.draft_replay = [&](int64_t start, std::span<const Token> tokens) {
            check(int64_t(draft.kv.size()) == start, "canonical replay position");
            for (size_t i = 0; i < tokens.size(); ++i) {
                const auto pos = start + int64_t(i);
                const int input_h = pos == 3 ? 1002 : verified_h.at(size_t(pos - 4));
                replay_inputs.push_back(input_h);
                replay.push_back({pos, tokens[i]});
                draft.kv.push_back(tokens[i]);
                draft.carry = 2000 + int(pos);
            }
        };
        cb.draft_accept = [&](size_t row) {
            accepted_h = row;
            draft.carry = verified_h.at(row);
        };
        return cb;
    }
};
RoundResult round(Model &m, RoundOptions options = {}, size_t budget = 3) {
    options.target_sampling.top_k = options.draft_sampling.top_k = 0;
    options.max_draft_tokens = 2;
    return run_round(3, 3, budget, std::vector<Token>{2, 3}, options, m.callbacks(), proposal_rng,
                     acceptance_rng, target_rng);
}
void state_semantics() {
    Model full;
    auto result = round(full);
    check(result.verification.accepted == 2 && result.committed_length == 6 && result.target_inputs_kept == 3,
          "committed prefix length");
    check(full.target.kv == std::vector<Token>({0, 1, 2, 3, 1, 2}) && full.draft.kv == full.target.kv,
          "target/draft accepted KV");
    check(full.replay == std::vector<std::pair<int64_t, Token>>({{4, 1}, {5, 2}}),
          "replay accepted last draft even when it was not drafted as input");
    check(full.replay_inputs == std::vector<int>({1003, 1004}) && full.draft.carry == 1005 &&
              full.accepted_h == 2,
          "canonical target feature shift");
    check(result.verification.next_seed == 0 && full.target.kv.back() != 0,
          "bonus is emitted but unconsumed");

    Model reject0;
    reject0.target_modes[0] = 2;
    result = round(reject0);
    check(result.verification.accepted == 0 && reject0.target.kv == std::vector<Token>({0, 1, 2, 3}),
          "rejection0 target truncation");
    check(reject0.replay.empty() && reject0.draft.kv == reject0.target.kv && reject0.draft.carry == 1003 &&
              reject0.accepted_h == 0,
          "rejection0 keeps seed KV but target carry");
    Model reject1;
    reject1.target_modes[1] = 3;
    result = round(reject1);
    check(result.verification.accepted == 1 && reject1.replay.size() == 1 && reject1.draft.carry == 1004,
          "accepted prefix replay");
    check(result.verification.emitted == std::vector<Token>({1, 3}) && reject1.target.kv.back() == 1,
          "replacement absent from KV");

    Model no_draft;
    result = round(no_draft, {}, 1);
    check(result.drafted == 0 && !result.retained_draft_seed &&
              no_draft.replay == std::vector<std::pair<int64_t, Token>>({{3, 3}}),
          "one-token budget canonical seed replay");
    check(no_draft.draft.carry == 1003 && no_draft.accepted_h == 0, "no-draft seed target feature");
    Model eos;
    RoundOptions stop;
    stop.stop = [](Token t) { return t == 1; };
    result = round(eos, stop);
    check(result.drafted == 1 && result.verification.stopped && !result.verification.next_seed &&
              eos.target.kv.back() == 1,
          "EOS draft stops window");

    Model penalties;
    penalties.constant_logits = true;
    RoundOptions penalized;
    penalized.target_sampling.freq = penalized.draft_sampling.freq = 5;
    penalized.target_sampling.last_n = penalized.draft_sampling.last_n = 64;
    const auto with_penalties = run_round(2, 3, 3, std::vector<Token>{2}, penalized, penalties.callbacks(),
                                          proposal_rng, acceptance_rng, target_rng);
    check(with_penalties.verification.emitted == std::vector<Token>({0, 1, 3}),
          "p and q use identical incremental penalty history");

    Model error;
    error.fail_verify = true;
    bool threw = false;
    try {
        round(error);
    } catch (const std::exception &) {
        threw = true;
    }
    check(threw && error.target.kv == std::vector<Token>({0, 1, 2}) && error.draft.kv == error.target.kv &&
              error.draft.carry == 1002,
          "failure restores both complete states");
    error.fail_restore = true;
    threw = false;
    error.draft_restored = false;
    try {
        round(error);
    } catch (const RollbackFailure &e) {
        threw = bool(e.original);
    }
    check(threw && error.draft_restored, "rollback failure is fatal but still restores the other side");
}
} // namespace
int main() {
    try {
        filters();
        acceptance();
        state_semantics();
        std::cout << "Hy4: exact rejection distribution, filters/penalties, accepted-prefix replay, carry, "
                     "budget, EOS and rollback checks passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
