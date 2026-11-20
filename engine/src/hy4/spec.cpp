#include "strata/hy4/spec.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace strata::hy4::spec {
Distribution::Distribution(std::vector<double> weights) : p_(std::move(weights)) {
    if (p_.empty() || p_.size() > size_t(std::numeric_limits<Token>::max()))
        throw std::runtime_error("Hy4 sampler: invalid vocabulary size");
    long double sum = 0;
    for (double p : p_) {
        if (!std::isfinite(p) || p < 0)
            throw std::runtime_error("Hy4 sampler: invalid probability");
        sum += p;
    }
    if (!(sum > 0) || !std::isfinite(sum))
        throw std::runtime_error("Hy4 sampler: empty probability support");
    for (double &p : p_)
        p = double(static_cast<long double>(p) / sum);
}
double Distribution::at(Token token) const {
    if (token < 0 || size_t(token) >= p_.size())
        throw std::runtime_error("Hy4 sampler: token outside distribution");
    return p_[token];
}
Token Distribution::mode() const { return Token(std::max_element(p_.begin(), p_.end()) - p_.begin()); }
Token Distribution::sample(std::mt19937_64 &rng) const {
    std::discrete_distribution<size_t> choose(p_.begin(), p_.end());
    return Token(choose(rng));
}

Distribution filter_logits(std::span<const float> raw, const Sampling &s, std::span<const Token> history) {
    if (raw.empty() || raw.size() > size_t(std::numeric_limits<Token>::max()))
        throw std::runtime_error("Hy4 sampler: invalid logits size");
    const auto finite = [](float x) { return std::isfinite(x); };
    if (!finite(s.temperature) || s.temperature < 0 || !finite(s.top_p) || s.top_p <= 0 || s.top_p > 1 ||
        !finite(s.min_p) || s.min_p < 0 || s.min_p > 1 || s.top_k < 0 || !finite(s.repeat) || s.repeat <= 0 ||
        !finite(s.freq) || !finite(s.present))
        throw std::runtime_error("Hy4 sampler: invalid sampling settings");
    Logits logits(raw.begin(), raw.end());
    for (float value : raw)
        if (!std::isfinite(value) && value != -std::numeric_limits<float>::infinity())
            throw std::runtime_error("Hy4 sampler: invalid raw logits");
    strata::qwen35::apply_penalties(logits, s, std::vector<Token>(history.begin(), history.end()));
    for (size_t i = 0; i < logits.size(); ++i) {
        if (raw[i] == -std::numeric_limits<float>::infinity())
            logits[i] = raw[i];
        else if (!std::isfinite(logits[i]))
            throw std::runtime_error("Hy4 sampler: nonfinite logits after penalties");
    }
    const auto best = std::max_element(logits.begin(), logits.end());
    if (!std::isfinite(*best))
        throw std::runtime_error("Hy4 sampler: all logits masked");
    std::vector<double> probs(logits.size(), 0);
    if (s.temperature == 0) {
        probs[size_t(best - logits.begin())] = 1;
        return Distribution(std::move(probs));
    }
    const float maximum = *best;
    const auto larger = [&](Token a, Token b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
    };
    // Hy4's default is unrestricted top-k/top-p: no O(V log V) sort is needed.
    const bool capped = s.top_k > 0 && size_t(s.top_k) < logits.size();
    if (!capped && s.top_p == 1) {
        for (size_t i = 0; i < logits.size(); ++i) {
            const double p = std::exp(double((logits[i] - maximum) / s.temperature));
            if (p >= s.min_p)
                probs[i] = p;
        }
        return Distribution(std::move(probs));
    }
    std::vector<Token> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    if (capped) {
        std::nth_element(order.begin(), order.begin() + s.top_k, order.end(), larger);
        order.resize(s.top_k);
    }
    if (s.top_p < 1)
        std::sort(order.begin(), order.end(), larger);
    double total = 0;
    for (Token id : order) {
        const double p = std::exp(double((logits[id] - maximum) / s.temperature));
        if (p >= s.min_p) {
            probs[id] = p;
            total += p;
        }
    }
    if (s.top_p < 1) {
        double cumulative = 0;
        bool cut = false;
        for (Token id : order) {
            if (cut)
                probs[id] = 0;
            else {
                cumulative += probs[id];
                if (cumulative >= total * s.top_p)
                    cut = true;
            }
        }
    }
    return Distribution(std::move(probs));
}

Verification verify(std::span<const Proposal> drafts, std::span<const Distribution> target, bool greedy,
                    std::mt19937_64 &acceptance_rng, std::mt19937_64 &target_rng, const Stop &stop) {
    if (target.size() != drafts.size() + 1)
        throw std::runtime_error("Hy4 speculation: target needs K+1 rows");
    const size_t vocab = target.front().size();
    for (const auto &p : target)
        if (p.size() != vocab)
            throw std::runtime_error("Hy4 speculation: target vocabulary changed");
    Verification result;
    result.emitted.reserve(drafts.size() + 1);
    const auto terminal = [&](Token token) { return stop && stop(token); };
    for (size_t i = 0; i < drafts.size(); ++i) {
        const auto &q = drafts[i].q;
        const auto &p = target[i];
        const Token proposed = drafts[i].token;
        if (q.size() != vocab)
            throw std::runtime_error("Hy4 speculation: draft vocabulary mismatch");
        const double qy = q.at(proposed), py = p.at(proposed);
        if (!(qy > 0))
            throw std::runtime_error("Hy4 speculation: proposal has zero q probability");
        const bool accepted =
            greedy ? proposed == p.mode()
                   : (py >= qy || std::generate_canonical<double, 53>(acceptance_rng) < py / qy);
        if (accepted) {
            ++result.accepted;
            result.emitted.push_back(proposed);
            if (terminal(proposed)) {
                result.stopped = true;
                return result;
            }
            continue;
        }
        result.rejected = true;
        Token replacement;
        if (greedy)
            replacement = p.mode();
        else {
            std::vector<double> residual(vocab);
            for (size_t token = 0; token < vocab; ++token)
                residual[token] = std::max(0.0, p.probabilities()[token] - q.probabilities()[token]);
            replacement = Distribution(std::move(residual)).sample(target_rng);
        }
        result.emitted.push_back(replacement);
        result.stopped = terminal(replacement);
        if (!result.stopped)
            result.next_seed = replacement;
        return result;
    }
    const Token bonus = greedy ? target.back().mode() : target.back().sample(target_rng);
    result.bonus = true;
    result.emitted.push_back(bonus);
    result.stopped = terminal(bonus);
    if (!result.stopped)
        result.next_seed = bonus;
    return result;
}

RoundResult run_round(Token seed, int64_t position, size_t output_budget,
                      std::span<const Token> penalty_history, const RoundOptions &options,
                      const RoundCallbacks &cb, std::mt19937_64 &proposal_rng,
                      std::mt19937_64 &acceptance_rng, std::mt19937_64 &target_rng) {
    if (position < 0 || !output_budget || options.max_draft_tokens < 0 || options.max_draft_tokens > 64 ||
        !std::isfinite(options.min_draft_probability) || options.min_draft_probability < 0 ||
        options.min_draft_probability > 1)
        throw std::runtime_error("Hy4 speculation: invalid round limits");
    if (penalty_history.empty() || penalty_history.back() != seed)
        throw std::runtime_error("Hy4 speculation: penalty history must include seed");
    if (options.stop && options.stop(seed))
        throw std::runtime_error("Hy4 speculation: do not run a terminal seed");
    if (!cb.target_logits || !cb.draft_logits || !cb.target_save || !cb.draft_save || !cb.target_restore ||
        !cb.draft_restore || !cb.target_keep_prefix || !cb.draft_replay || !cb.draft_accept)
        throw std::runtime_error("Hy4 speculation: missing model/cache callbacks");
    const size_t desired = std::min(size_t(options.max_draft_tokens), output_budget - 1);
    if (uint64_t(position) > uint64_t(std::numeric_limits<int64_t>::max()) - desired - 1)
        throw std::runtime_error("Hy4 speculation: position overflow");
    const std::any target_base = cb.target_save(), draft_base = cb.draft_save();
    try {
        std::optional<std::any> after_seed;
        std::vector<Proposal> proposals;
        proposals.reserve(desired);
        std::vector<Token> draft_history(penalty_history.begin(), penalty_history.end());
        Token input = seed;
        for (size_t i = 0; i < desired; ++i) {
            const auto logits = cb.draft_logits(input, position + int64_t(i));
            if (i == 0)
                after_seed = cb.draft_save();
            auto q = filter_logits(logits, options.draft_sampling, draft_history);
            const Token best = q.mode();
            if (q.at(best) < options.min_draft_probability)
                break;
            const Token token = options.draft_sampling.temperature == 0 ? best : q.sample(proposal_rng);
            proposals.push_back(Proposal{token, std::move(q)});
            draft_history.push_back(token);
            input = token;
            if (options.stop && options.stop(token))
                break;
        }
        std::vector<Token> inputs{seed};
        for (const auto &p : proposals)
            inputs.push_back(p.token);
        const auto logits = cb.target_logits(inputs, position);
        if (logits.size() != inputs.size())
            throw std::runtime_error("Hy4 speculation: verifier did not return every row");
        std::vector<Distribution> p;
        p.reserve(logits.size());
        std::vector<Token> history(penalty_history.begin(), penalty_history.end());
        for (size_t i = 0; i < logits.size(); ++i) {
            p.push_back(filter_logits(logits[i], options.target_sampling, history));
            if (i < proposals.size())
                history.push_back(proposals[i].token);
        }
        auto verified = verify(proposals, p, options.target_sampling.temperature == 0, acceptance_rng,
                               target_rng, options.stop);
        const size_t kept = 1 + verified.accepted;
        if (after_seed) {
            cb.draft_restore(*after_seed);
            if (verified.accepted)
                cb.draft_replay(position + 1, std::span(inputs).subspan(1, verified.accepted));
        } else {
            cb.draft_restore(draft_base);
            cb.draft_replay(position, std::span(inputs).first(kept));
        }
        // The replacement/bonus is emitted but is deliberately absent from both KV caches.
        cb.target_keep_prefix(target_base, position + int64_t(kept));
        cb.draft_accept(verified.accepted);
        return {std::move(verified), proposals.size(), kept, position + int64_t(kept),
                after_seed.has_value()};
    } catch (...) {
        // Restore both sides even if one callback fails. Never silently reuse a
        // context for which rollback itself failed.
        const auto original = std::current_exception();
        bool restored = true;
        try {
            cb.target_restore(target_base);
        } catch (...) {
            restored = false;
        }
        try {
            cb.draft_restore(draft_base);
        } catch (...) {
            restored = false;
        }
        if (!restored)
            throw RollbackFailure(original);
        std::rethrow_exception(original);
    }
}
} // namespace strata::hy4::spec
