#include "strata/hy4/server.hpp"
#include "strata/hy4/spec.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#ifdef _WIN32
#include <windows.h>
#else
#include <poll.h>
#include <unistd.h>
#endif
#include <random>
#include <sstream>
#include <stdexcept>

namespace strata::hy4 {
namespace {
using Clock = std::chrono::steady_clock;
using Sampling = spec::Sampling;
double ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
void require(bool b, const std::string &text) {
    if (!b)
        throw std::runtime_error("Hy4 serve: " + text);
}
uint64_t hash_bytes(const void *data, size_t n, uint64_t h = 0xcbf29ce484222325ull) {
    const auto *p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < n; ++i)
        h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}
int64_t number(const std::string &text) {
    size_t used = 0;
    const auto n = std::stoll(text, &used);
    require(used == text.size(), "invalid integer " + text);
    return n;
}
float real(const std::string &text) {
    size_t used = 0;
    const float n = std::stof(text, &used);
    require(used == text.size() && std::isfinite(n), "invalid float " + text);
    return n;
}
struct Cell {
    std::array<int32_t, 3> rope{-1, -1, -1};
    int32_t next = -1;
    size_t embedding = SIZE_MAX;
    uint64_t key = 0;
};
struct Request {
    bool batch = false, images = false, slot_read = false, checkpoint = true;
    int slot = -1;
    size_t max_new = 0, pen_from = SIZE_MAX;
    size_t hold = 0;
    int32_t hold_id = -1;
    Sampling sampling;
    std::vector<int32_t> ids;
    std::vector<Cell> cells;
    Vec embeddings;
};
void images(Request &r, const std::string &path, const Geometry &g, const ServerOptions &o,
            const std::string &keys) {
    require(o.vision && o.image_pad >= 0, "this engine was started without --vision");
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "cannot open image embeddings");
    struct Image {
        int32_t n, w, h;
        size_t offset;
        uint64_t key;
    };
    std::vector<Image> records;
    std::vector<uint64_t> given;
    if (!keys.empty()) {
        std::istringstream list(keys);
        std::string s;
        while (std::getline(list, s, ',')) {
            size_t end = 0;
            const auto k = std::stoull(s, &end, 16);
            require(end == s.size(), "bad img_keys");
            given.push_back(k);
        }
    }
    for (;;) {
        int32_t h[5];
        file.read(reinterpret_cast<char *>(h), sizeof(h));
        const auto got = file.gcount();
        if (!got)
            break;
        require(got == sizeof(h) && h[0] == 0x31455653 && h[1] > 0 && h[2] > 0 && h[3] > 0 &&
                    int64_t(h[2]) * h[3] == h[1] && h[4] == g.dim,
                "bad SVE image dimensions");
        require(size_t(h[1]) <= r.ids.size() && size_t(h[1]) <= SIZE_MAX / size_t(g.dim),
                "image exceeds prompt");
        const size_t off = r.embeddings.size(), count = size_t(h[1]) * g.dim;
        require(count <= SIZE_MAX - off && (off + count) / size_t(g.dim) <= r.ids.size(),
                "image embedding size overflow");
        r.embeddings.resize(off + count);
        file.read(reinterpret_cast<char *>(r.embeddings.data() + off),
                  std::streamsize(count * sizeof(float)));
        require(size_t(file.gcount()) == count * sizeof(float), "short image embedding record");
        for (size_t i = off; i < off + count; ++i)
            require(std::isfinite(r.embeddings[i]), "nonfinite image embedding");
        uint64_t key = hash_bytes(h + 1, 3 * sizeof(int32_t));
        const size_t rec = records.size();
        if (!given.empty()) {
            require(rec < given.size(), "img_keys count mismatch");
            key = hash_bytes(&given[rec], sizeof(uint64_t), key);
        }
        key = hash_bytes(r.embeddings.data() + off, count * sizeof(float), key);
        records.push_back({h[1], h[2], h[3], off, key});
    }
    require(!records.empty() && (given.empty() || given.size() == records.size()), "img_keys count mismatch");
    size_t rec = 0;
    int64_t position = 0;
    for (size_t i = 0; i < r.ids.size();) {
        if (r.ids[i] != o.image_pad) {
            r.cells[i].rope = {int32_t(position), int32_t(position), int32_t(position)};
            r.cells[i].next = int32_t(++position);
            ++i;
            continue;
        }
        require(rec < records.size(), "prompt contains more images than SVE file");
        const auto &im = records[rec++];
        require(position + std::max(im.w, im.h) <= INT32_MAX, "image RoPE position overflow");
        const int32_t next = int32_t(position + std::max(im.w, im.h));
        for (int j = 0; j < im.n; ++j) {
            require(i + size_t(j) < r.ids.size() && r.ids[i + j] == o.image_pad,
                    "image pad run does not cover all SVE cells/newlines");
            auto &c = r.cells[i + j];
            c.rope = {int32_t(position), int32_t(position + j / im.w), int32_t(position + j % im.w)};
            c.next = next;
            c.embedding = im.offset + size_t(j) * g.dim;
            c.key = hash_bytes(&j, sizeof(j), im.key);
        }
        i += im.n;
        position = next;
    }
    require(rec == records.size(), "SVE file contains more images than prompt");
}
Request parse(const std::string &line, const Geometry &g, const ServerOptions &o) {
    std::istringstream in(line);
    std::string command, word;
    in >> command;
    Request r;
    r.batch = command == "BGEN" || command == "BGENI";
    r.images = command == "GENI" || command == "BGENI";
    require(r.batch || command == "GEN" || command == "GENI", "expected GEN/BGEN or GENI/BGENI");
    r.sampling.top_k = 0;
    if (r.batch) {
        require(bool(in >> word), "missing slot");
        const auto n = number(word);
        require(n >= 0 && n <= INT32_MAX, "invalid slot");
        r.slot = int(n);
    }
    require(bool(in >> word), "missing max_new");
    const auto count = number(word);
    require(count > 0, "max_new must be positive");
    r.max_new = size_t(count);
    std::string image_path, img_keys;
    while (in >> word) {
        const auto eq = word.find('=');
        if (eq == std::string::npos)
            break;
        const auto k = word.substr(0, eq), v = word.substr(eq + 1);
        if (k == "temperature" || k == "temp")
            r.sampling.temperature = real(v);
        else if (k == "top_p")
            r.sampling.top_p = real(v);
        else if (k == "min_p")
            r.sampling.min_p = real(v);
        else if (k == "penalty_repeat")
            r.sampling.repeat = real(v);
        else if (k == "penalty_freq")
            r.sampling.freq = real(v);
        else if (k == "penalty_present")
            r.sampling.present = real(v);
        else if (k == "top_k" || k == "penalty_last_n") {
            const auto n = number(v);
            require(n >= (k == "top_k" ? 0 : -1) && n <= INT32_MAX, "sampling integer out of range");
            if (k == "top_k")
                r.sampling.top_k = int(n);
            else
                r.sampling.last_n = int(n);
        } else if (k == "seed") {
            require(!v.empty() && v[0] != '-', "invalid seed");
            size_t end = 0;
            r.sampling.seed = std::stoull(v, &end);
            require(end == v.size(), "invalid seed");
        } else if (k == "slot_read") {
            require(v == "0" || v == "1", "invalid slot_read");
            r.slot_read = v == "1";
        } else if (k == "ckpt") {
            require(v == "0" || v == "1", "invalid ckpt");
            r.checkpoint = v == "1";
        } else if (k == "pen_from") {
            const auto n = number(v);
            require(n >= 0, "invalid pen_from");
            r.pen_from = size_t(n);
        } else if (k == "hold") {
            const auto colon = v.find(':');
            require(colon != std::string::npos, "invalid hold");
            const auto n = number(v.substr(0, colon)), id = number(v.substr(colon + 1));
            require(n > 0 && id >= 0 && id < g.vocab, "invalid hold");
            r.hold = size_t(n);
            r.hold_id = int32_t(id);
        } else if (k == "img_keys")
            img_keys = v;
        else
            throw std::runtime_error("Hy4 serve: unsupported request key " + k);
    }
    require(r.sampling.temperature >= 0 && r.sampling.top_p > 0 && r.sampling.top_p <= 1 &&
                r.sampling.min_p >= 0 && r.sampling.min_p <= 1 && r.sampling.repeat > 0,
            "invalid sampler settings");
    if (r.images) {
        image_path = word;
        require(bool(in >> word), "missing image prompt");
    }
    require(!word.empty(), "missing token prompt");
    std::string extra;
    require(!(in >> extra), "extra request fields");
    require(word.back() != ',', "empty trailing token ID");
    std::istringstream ids(word);
    while (std::getline(ids, word, ',')) {
        const auto id = number(word);
        require(id >= 0 && id < g.vocab, "token outside vocabulary");
        r.ids.push_back(int32_t(id));
    }
    require(!r.ids.empty() && r.ids.size() <= size_t(o.max_context) &&
                r.max_new <= size_t(o.max_context) - r.ids.size(),
            "prompt + max_new exceed context");
    require(r.pen_from == SIZE_MAX || r.pen_from < r.ids.size(), "pen_from outside prompt");
    r.cells.resize(r.ids.size());
    for (size_t i = 0; i < r.ids.size(); ++i)
        r.cells[i].key = hash_bytes(&r.ids[i], sizeof(int32_t));
    if (r.images)
        images(r, image_path, g, o, img_keys);
    else
        for (int32_t t : r.ids)
            require(!o.vision || t != o.image_pad, "image pads require GENI embeddings");
    return r;
}
struct Active {
    Request request;
    int physical = -1;
    bool reading = true, admitted = false, cancel = false, hold_seen = false;
    size_t read = 0, reused = 0, produced = 0, offered = 0, accepted = 0;
    int32_t seed = 0;
    std::vector<int32_t> penalties;
    std::mt19937_64 proposal, acceptance, target;
    Clock::time_point begin = Clock::now(), decode = Clock::now();
    double prompt_ms = 0;
};
class Server {
    DeviceModel &m;
    const Geometry &g;
    ServerOptions o;
    ServerTransport io;
    std::vector<int32_t> eos;
    std::vector<std::optional<Active>> active;
    std::vector<std::vector<uint64_t>> keys;
    std::vector<uint64_t> used;
    std::deque<std::pair<std::string, bool>> pending;
    uint64_t clock = 0;
    size_t prefill_turn = 0;
    int control = -1;
    bool quitting = false;
    int status = 0;
    void say(const std::string &s) { io.write(s); }
    bool terminal(int32_t t) const { return std::find(eos.begin(), eos.end(), t) != eos.end(); }
    bool busy(int physical) const {
        for (const auto &a : active)
            if (a && a->physical == physical)
                return true;
        return false;
    }
    bool reclaim(size_t room, int protected_slot = -1) {
        std::vector<int> idle;
        for (int p = 0; p < m.slots(); ++p)
            if (p != protected_slot && !busy(p) && m.length(p))
                idle.push_back(p);
        std::sort(idle.begin(), idle.end(), [&](int a, int b) { return used[a] < used[b]; });
        for (int p : idle) {
            if (m.free_tokens() >= room)
                break;
            m.reset(p);
            keys[p].clear();
        }
        return m.free_tokens() >= room;
    }
    DeviceInput input(Active &a, size_t at) {
        DeviceInput row;
        row.slot = a.physical;
        row.token = a.request.ids[at];
        const auto &c = a.request.cells[at];
        row.rope = c.rope;
        row.next_rope = c.next;
        if (c.embedding != SIZE_MAX)
            row.embedding = std::span<const float>(a.request.embeddings).subspan(c.embedding, g.dim);
        return row;
    }
    std::string done(const Active &a, const std::string &why) const {
        std::ostringstream s;
        s << std::fixed << std::setprecision(1) << "DONE " << a.produced << ' ' << a.request.ids.size() << ' '
          << a.prompt_ms << ' ' << (a.reading ? 0 : ms(a.decode)) << ' ' << why << ' ' << a.accepted << ' '
          << a.offered << ' ' << a.reused << " 0 0 0 0 0.0 " << (a.read - a.reused) << " 0";
        return s.str();
    }
    void finish(int logical, const std::string &why) {
        auto &a = *active[logical];
        if (!a.request.batch || !a.admitted) {
            say(done(a, why));
            if (a.request.batch)
                say("BADM " + std::to_string(logical) + " 0");
        } else {
            std::ostringstream s;
            s << std::fixed << std::setprecision(1) << "BDONE " << logical << ' ' << a.produced << ' ' << why
              << ' ' << (a.reading ? 0 : ms(a.decode)) << " drafts=" << a.offered << ':' << a.accepted;
            if (a.request.slot_read)
                s << " read=" << a.prompt_ms << " prompt_read=" << (a.read - a.reused);
            say(s.str());
        }
        if (!a.request.checkpoint) {
            m.reset(a.physical);
            keys[a.physical].clear();
        }
        used[a.physical] = ++clock;
        if (control == logical)
            control = -1;
        active[logical].reset();
    }
    void fault(int logical, const std::exception &error) {
        std::fprintf(stderr, "strata Hy4 slot %d: %s\n", logical, error.what());
        auto &a = *active[logical];
        m.reset(a.physical);
        keys[a.physical].clear();
        // A model execution failure may have left shared CUDA work incomplete. End
        // the process if its runtime rejects continued use, rather than recycle a slot.
        if (!m.healthy()) {
            say("ERR fatal GPU error: " + std::string(error.what()));
            quitting = true;
            status = 1;
            return;
        }
        if (a.request.batch && a.admitted)
            finish(logical, "nonfinite");
        else {
            say("ERR " + std::string(error.what()));
            if (control == logical)
                control = -1;
            active[logical].reset();
        }
    }
    bool emit(int logical, int32_t token) {
        auto &a = *active[logical];
        say(std::string(a.request.batch && a.admitted ? "BT " + std::to_string(logical) + " " : "T ") +
            std::to_string(token));
        ++a.produced;
        a.penalties.push_back(token);
        a.seed = token;
        if (token == a.request.hold_id)
            a.hold_seen = true;
        const char *why = terminal(token)                                                  ? "stop"
                          : a.cancel                                                       ? "cancel"
                          : a.produced >= a.request.max_new                                ? "length"
                          : a.request.hold && !a.hold_seen && a.produced >= a.request.hold ? "hold"
                                                                                           : nullptr;
        if (why) {
            finish(logical, why);
            return false;
        }
        return true;
    }
    void admission(int logical, Vec logits) {
        auto &a = *active[logical];
        a.reading = false;
        a.prompt_ms = ms(a.begin);
        a.decode = Clock::now();
        if (a.cancel) {
            finish(logical, "cancel");
            return;
        }
        if (o.mtp)
            m.draft_accept(a.physical, m.length(a.physical) - 1);
        if (a.request.slot_read && a.request.batch && !a.admitted) {
            say(done(a, "length"));
            say("BADM " + std::to_string(logical) + " 1");
            a.admitted = true;
            control = -1;
        }
        const auto p = spec::filter_logits(logits, a.request.sampling, a.penalties);
        const auto token = a.request.sampling.temperature == 0 ? p.mode() : p.sample(a.target);
        if (a.request.batch && !a.request.slot_read) {
            // Admission owns T/DONE/BADM, while subsequent slot windows own BT/BDONE.
            say("T " + std::to_string(token));
            ++a.produced;
            a.seed = token;
            a.penalties.push_back(token);
            a.hold_seen = token == a.request.hold_id;
            const char *why = terminal(token)                                                  ? "stop"
                              : a.produced >= a.request.max_new                                ? "length"
                              : a.request.hold && !a.hold_seen && a.produced >= a.request.hold ? "hold"
                                                                                               : nullptr;
            say(done(a, why ? why : "length"));
            say("BADM " + std::to_string(logical) + (why ? " 0" : " 1"));
            control = -1;
            if (why) {
                used[a.physical] = ++clock;
                if (!a.request.checkpoint) {
                    m.reset(a.physical);
                    keys[a.physical].clear();
                }
                active[logical].reset();
            } else
                a.admitted = true;
        } else
            emit(logical, token);
    }
    int admit(const std::string &line) {
        Request r = parse(line, g, o);
        int logical = r.batch ? r.slot : 0;
        if (r.batch) {
            require(logical >= 0 && logical < m.slots(), "BGEN: no such free slot");
            require(!active[logical], "BGEN: no such free slot");
        } else {
            logical = -1;
            for (int i = 0; i < m.slots(); ++i)
                if (!active[i]) {
                    logical = i;
                    break;
                }
            require(logical >= 0, "no free session");
        }
        size_t best = 0;
        int physical = -1;
        uint64_t oldest = UINT64_MAX;
        for (int p = 0; p < m.slots(); ++p)
            if (!busy(p)) {
                const auto hist = m.history(p);
                const size_t limit = std::min({hist.size(), r.ids.size() - 1, keys[p].size()});
                size_t same = 0;
                while (same < limit &&
                       (hist[same] == r.ids[same] ||
                        (hist[same] == -1 && r.cells[same].embedding != SIZE_MAX)) &&
                       keys[p][same] == r.cells[same].key)
                    ++same;
                if (physical < 0 || same > best || (same == best && used[p] < oldest)) {
                    physical = p;
                    best = same;
                    oldest = used[p];
                }
            }
        require(physical >= 0, "no free KV session");
        if (best > 0 && best < r.cells.size() && r.cells[best - 1].embedding != SIZE_MAX &&
            r.cells[best].embedding != SIZE_MAX && r.cells[best - 1].rope[0] == r.cells[best].rope[0])
            while (best > 0 && r.cells[best - 1].embedding != SIZE_MAX)
                --best;
        m.truncate(physical, best);
        best = m.length(physical);
        require(best <= r.ids.size() - 1, "runtime prefix truncate failed");
        if (o.mtp) {
            m.draft_truncate(physical, best);
            if (m.draft_length(physical) != best) {
                m.reset(physical);
                best = 0;
            }
        }
        keys[physical].resize(best);
        // Reclaim only idle cached prefixes, never another live conversation.
        const size_t need = r.ids.size() - best + std::min(r.max_new, size_t(8));
        require(reclaim(need, physical), "BGEN: shared KV pool is full");
        Active a;
        a.request = std::move(r);
        a.physical = physical;
        a.read = a.reused = best;
        a.target.seed(a.request.sampling.seed);
        a.proposal.seed(a.request.sampling.seed ^ 0x9e3779b97f4a7c15ull);
        a.acceptance.seed(a.request.sampling.seed ^ 0xd1b54a32d192ed03ull);
        const char *pen = std::getenv("STRATA_PENALTY_PROMPT");
        const bool prompt = pen && std::string(pen) == "1";
        const size_t from = a.request.pen_from != SIZE_MAX ? a.request.pen_from
                            : prompt                       ? 0
                                                           : a.request.ids.size() - 1;
        a.penalties.assign(a.request.ids.begin() + from, a.request.ids.end());
        active[logical] = std::move(a);
        control = logical;
        say("RESUME " + std::to_string(best));
        if (active[logical]->request.batch && active[logical]->request.slot_read) {
            say(done(*active[logical], "length"));
            say("BADM " + std::to_string(logical) + " 1");
            active[logical]->admitted = true;
            control = -1;
        }
        return logical;
    }
    void commands(bool wait) {
        std::string line;
        while (io.read(line, wait)) {
            wait = false;
            if (line.empty())
                continue;
            if (line == "QUIT") {
                quitting = true;
                for (auto &a : active)
                    if (a)
                        a->cancel = true;
                pending.clear();
                return;
            }
            if (line == "STOP") {
                if (!pending.empty())
                    pending.back().second = true;
                else if (control >= 0 && active[control])
                    active[control]->cancel = true;
                continue;
            }
            if (line.rfind("BSTOP ", 0) == 0) {
                try {
                    const auto n = number(line.substr(6));
                    if (n >= 0 && size_t(n) < active.size()) {
                        bool queued = false;
                        for (auto it = pending.rbegin(); it != pending.rend(); ++it) {
                            std::istringstream request(it->first);
                            std::string command;
                            int slot = -1;
                            request >> command >> slot;
                            if ((command == "BGEN" || command == "BGENI") && slot == n) {
                                it->second = true;
                                queued = true;
                                break;
                            }
                        }
                        if (!queued && active[size_t(n)])
                            active[size_t(n)]->cancel = true;
                    }
                } catch (const std::exception &e) {
                    say("ERR " + std::string(e.what()));
                }
                continue;
            }
            if (line == "BYIELD" || line.rfind("BYIELD ", 0) == 0) {
                continue;
            } // no yield capability advertised
            if (line == "STATS") {
                say("INFO " + m.stats());
                continue;
            }
            if (pending.size() >= size_t(m.slots())) {
                say("ERR request queue is full");
                continue;
            }
            pending.emplace_back(std::move(line), false);
        }
    }
    void prefill() {
        for (size_t turn = 0; turn < active.size(); ++turn) {
            const int logical = int((prefill_turn + turn) % active.size());
            if (!active[logical] || !active[logical]->reading)
                continue;
            prefill_turn = (size_t(logical) + 1) % active.size();
            auto &a = *active[logical];
            if (a.cancel) {
                a.prompt_ms = ms(a.begin);
                finish(logical, "cancel");
                return;
            }
            try {
                const size_t take =
                    std::min({size_t(o.prefill_chunk), size_t(m.batch()), a.request.ids.size() - a.read});
                std::vector<DeviceInput> rows;
                require(reclaim(take), "shared KV pool filled during prompt processing");
                for (size_t i = 0; i < take; ++i)
                    rows.push_back(input(a, a.read + i));
                const auto logits = m.forward(rows);
                require(logits.size() == rows.size(), "runtime returned wrong logits row count");
                if (o.mtp) {
                    const auto draft = m.draft_forward(rows, true);
                    require(draft.size() == rows.size(), "MTP prefill row count");
                }
                for (size_t i = 0; i < take; ++i)
                    keys[a.physical].push_back(a.request.cells[a.read + i].key);
                a.read += take;
                const double elapsed = ms(a.begin);
                std::ostringstream progress;
                progress << "PP " << (a.read - a.reused) << ' ' << (a.request.ids.size() - a.reused) << ' '
                         << elapsed << ' ' << 1000. * double(a.read - a.reused) / std::max(1e-9, elapsed);
                if (!a.admitted)
                    say(progress.str());
                commands(false);
                if (a.read == a.request.ids.size())
                    admission(logical, logits.back());
            } catch (const std::exception &e) {
                fault(logical, e);
            }
            return;
        }
    }
    void decode_plain(const std::vector<int> &logicals) {
        std::vector<DeviceInput> rows;
        for (int l : logicals) {
            const auto &a = *active[l];
            rows.push_back(DeviceInput{a.seed, a.physical});
        }
        const auto logits = m.forward(rows);
        require(logits.size() == rows.size(), "decode logits row count");
        if (o.mtp) {
            m.draft_forward(rows, true);
            for (int l : logicals)
                m.draft_accept(active[l]->physical, m.length(active[l]->physical) - 1);
        }
        commands(false);
        for (size_t i = 0; i < logicals.size(); ++i) {
            const int l = logicals[i];
            auto &a = *active[l];
            keys[a.physical].push_back(hash_bytes(&a.seed, sizeof(a.seed)));
            if (a.cancel) {
                finish(l, "cancel");
                continue;
            }
            const auto p = spec::filter_logits(logits[i], a.request.sampling, a.penalties);
            emit(l, a.request.sampling.temperature == 0 ? p.mode() : p.sample(a.target));
        }
    }
    void decode_spec(const std::vector<int> &logicals, int width) {
        struct Lane {
            int logical, physical;
            size_t base, row = 0;
            std::vector<spec::Proposal> drafts;
            std::vector<int32_t> history;
            int32_t next;
            spec::Verification result;
        };
        std::vector<Lane> lanes;
        for (int l : logicals) {
            auto &a = *active[l];
            lanes.push_back({l, a.physical, m.length(a.physical), 0, {}, a.penalties, a.seed, {}});
        }
        try {
            for (int step = 0; step < width; ++step) {
                std::vector<DeviceInput> rows;
                std::vector<size_t> which;
                for (size_t i = 0; i < lanes.size(); ++i) {
                    auto &lane = lanes[i];
                    const auto &a = *active[lane.logical];
                    if (a.cancel || size_t(step) >= a.request.max_new - a.produced - 1 ||
                        (!lane.drafts.empty() && terminal(lane.next)))
                        continue;
                    rows.push_back(DeviceInput{lane.next, lane.physical});
                    which.push_back(i);
                }
                if (rows.empty())
                    break;
                const auto logits = m.draft_forward(rows, false);
                require(logits.size() == rows.size(), "draft logits row count");
                for (size_t j = 0; j < which.size(); ++j) {
                    auto &lane = lanes[which[j]];
                    auto &a = *active[lane.logical];
                    auto q = spec::filter_logits(logits[j], a.request.sampling, lane.history);
                    const int32_t token =
                        a.request.sampling.temperature == 0 ? q.mode() : q.sample(a.proposal);
                    lane.drafts.push_back({token, std::move(q)});
                    lane.history.push_back(token);
                    lane.next = token;
                }
                commands(false);
            }
            for (auto it = lanes.begin(); it != lanes.end();) {
                if (active[it->logical]->cancel) {
                    m.draft_truncate(it->physical, it->base);
                    if (it->base)
                        m.draft_accept(it->physical, it->base - 1);
                    require(m.draft_length(it->physical) == it->base, "cancelled draft rewind failed");
                    finish(it->logical, "cancel");
                    it = lanes.erase(it);
                } else
                    ++it;
            }
            if (lanes.empty())
                return;
            std::vector<DeviceInput> target;
            for (auto &lane : lanes) {
                lane.row = target.size();
                target.push_back(DeviceInput{active[lane.logical]->seed, lane.physical});
                for (const auto &d : lane.drafts)
                    target.push_back(DeviceInput{d.token, lane.physical});
            }
            require(target.size() <= size_t(m.batch()), "verification exceeds native batch");
            const auto logits = m.forward(target);
            require(logits.size() == target.size(), "verification logits row count");
            commands(false);
            for (auto &lane : lanes) {
                auto &a = *active[lane.logical];
                std::vector<spec::Distribution> p;
                auto hist = a.penalties;
                for (size_t j = 0; j <= lane.drafts.size(); ++j) {
                    p.push_back(spec::filter_logits(logits[lane.row + j], a.request.sampling, hist));
                    if (j < lane.drafts.size())
                        hist.push_back(lane.drafts[j].token);
                }
                // Stop at the hold boundary too; otherwise later accepted inputs
                // would enter KV after the response had already ended there.
                size_t prospective = a.produced;
                bool hold_seen = a.hold_seen;
                if (a.cancel)
                    lane.result.stopped = true;
                else
                    lane.result = spec::verify(lane.drafts, p, a.request.sampling.temperature == 0,
                                               a.acceptance, a.target, [&](int32_t t) {
                                                   ++prospective;
                                                   hold_seen |= t == a.request.hold_id;
                                                   return terminal(t) || (a.request.hold && !hold_seen &&
                                                                          prospective >= a.request.hold);
                                               });
                const size_t keep_seed = lane.drafts.empty() ? lane.base : lane.base + 1;
                m.draft_truncate(lane.physical, keep_seed);
                require(m.draft_length(lane.physical) == keep_seed, "draft rewind lost accepted prefix");
            }
            // Canonical draft KV uses target h_(p-1). The last accepted token was
            // never processed by the proposer on a full acceptance and is replayed too.
            for (size_t step = 0;; ++step) {
                std::vector<DeviceInput> replay;
                for (auto &lane : lanes) {
                    const auto &a = *active[lane.logical];
                    if (lane.drafts.empty()) {
                        if (step == 0)
                            replay.push_back(DeviceInput{a.seed, lane.physical});
                    } else if (step < lane.result.accepted)
                        replay.push_back(DeviceInput{lane.drafts[step].token, lane.physical});
                }
                if (replay.empty())
                    break;
                m.draft_forward(replay, true);
            }
            for (auto &lane : lanes) {
                auto &a = *active[lane.logical];
                const size_t keep = lane.base + 1 + lane.result.accepted;
                m.truncate(lane.physical, keep);
                m.draft_accept(lane.physical, keep - 1);
                require(m.length(lane.physical) == keep && m.draft_length(lane.physical) == keep,
                        "accepted prefix rewind failed");
                keys[lane.physical].push_back(hash_bytes(&a.seed, sizeof(a.seed)));
                for (size_t j = 0; j < lane.result.accepted; ++j)
                    keys[lane.physical].push_back(hash_bytes(&lane.drafts[j].token, sizeof(int32_t)));
                a.offered += lane.drafts.size();
                a.accepted += lane.result.accepted;
            }
            for (auto &lane : lanes) {
                if (active[lane.logical]->cancel) {
                    finish(lane.logical, "cancel");
                    continue;
                }
                for (int32_t token : lane.result.emitted)
                    if (!emit(lane.logical, token))
                        break;
            }
        } catch (...) {
            const auto original = std::current_exception();
            bool restored = true;
            for (const auto &lane : lanes) {
                try {
                    m.truncate(lane.physical, lane.base);
                    m.draft_truncate(lane.physical, lane.base);
                    if (lane.base)
                        m.draft_accept(lane.physical, lane.base - 1);
                    require(m.length(lane.physical) == lane.base &&
                                m.draft_length(lane.physical) == lane.base,
                            "failed speculative rollback");
                } catch (...) {
                    restored = false;
                }
            }
            if (!restored)
                throw spec::RollbackFailure(original);
            std::rethrow_exception(original);
        }
    }
    void decode() {
        std::vector<int> logicals;
        for (int l = 0; l < int(active.size()); ++l)
            if (active[l] && !active[l]->reading) {
                if (active[l]->cancel)
                    finish(l, "cancel");
                else
                    logicals.push_back(l);
            }
        if (logicals.empty())
            return;
        try {
            int width = o.mtp ? std::min(o.mtp, m.batch() / int(logicals.size()) - 1) : 0;
            if (width > 0 && !reclaim(logicals.size() * size_t(width + 1)))
                width = 0;
            if (!reclaim(logicals.size())) {
                for (int l : logicals)
                    finish(l, "length");
                return;
            }
            if (width > 0)
                decode_spec(logicals, width);
            else
                decode_plain(logicals);
        } catch (const std::exception &e) {
            for (int l : logicals)
                if (active[l])
                    fault(l, e);
        }
    }

  public:
    Server(DeviceModel &model, const Geometry &geometry, std::span<const int32_t> end, ServerOptions options,
           ServerTransport transport)
        : m(model), g(geometry), o(options), io(std::move(transport)), eos(end.begin(), end.end()),
          active(m.slots()), keys(m.slots()), used(m.slots(), 0) {
        require(m.slots() >= 1 && m.slots() <= m.batch() && m.batch() <= 8,
                "invalid native slot/batch geometry");
        require(bool(io.read) && bool(io.write), "missing transport");
        if (!o.max_context)
            o.max_context = g.max_context;
        require(o.max_context > 0 && o.max_context <= g.max_context, "invalid max context");
        require(o.prefill_chunk > 0 && o.prefill_chunk <= m.batch() && o.mtp >= 0 &&
                    o.mtp <= std::min(5, m.batch() - 1),
                "invalid prefill/MTP bounds");
        require(!o.mtp || m.has_mtp(), "MTP requested but no native draft layer");
        require(!eos.empty(), "missing EOS metadata");
    }
    int run() {
        std::string end_tokens;
        for (int32_t token : eos) {
            if (!end_tokens.empty()) end_tokens += ',';
            end_tokens += std::to_string(token);
        }
        say("INFO eos_ids=" + end_tokens);
        say("INFO engine=hy4 device_forward=1 batch_slots=" + std::to_string(m.slots()) +
            " slot_cache=1 slot_read=1 kv_unified=1 speculation=" + (o.mtp ? "mtp" : "none") +
            " vision=" + (o.vision ? "1" : "0"));
        say("INFO " + m.stats());
        say("READY " + std::to_string(o.max_context) + " stop hold holdb");
        for (;;) {
            bool any = false;
            for (const auto &a : active)
                any |= a.has_value();
            if (quitting && !any)
                break;
            commands(!any && pending.empty() && !quitting);
            if (quitting) {
                for (int l = 0; l < int(active.size()); ++l)
                    if (active[l]) {
                        active[l]->prompt_ms =
                            active[l]->reading ? ms(active[l]->begin) : active[l]->prompt_ms;
                        finish(l, "cancel");
                    }
                continue;
            }
            if (control < 0 && !pending.empty()) {
                auto queued = std::move(pending.front());
                pending.pop_front();
                try {
                    const int logical = admit(queued.first);
                    if (queued.second && active[logical])
                        active[logical]->cancel = true;
                } catch (const std::exception &e) {
                    say("ERR " + std::string(e.what()));
                }
            }
            // Every bounded prefill chunk is preceded by a decode window. Model
            // weights and physical KV pages are shared by all four live sessions.
            decode();
            if (!quitting)
                prefill();
        }
        return status;
    }
};
struct StdinTransport {
    std::string pending;
    bool ended = false;
    size_t limit;
    bool read(std::string &line, bool wait) {
        for (;;) {
            const auto nl = pending.find('\n');
            if (nl != std::string::npos) {
                line = pending.substr(0, nl);
                pending.erase(0, nl + 1);
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                return true;
            }
            if (ended) {
                if (!pending.empty()) {
                    line = std::move(pending);
                    pending.clear();
                    return true;
                }
                return false;
            }
            char buffer[65536];
            size_t n = 0;
#ifdef _WIN32
            const HANDLE handle = GetStdHandle(STD_INPUT_HANDLE);
            require(handle && handle != INVALID_HANDLE_VALUE, "invalid stdin handle");
            const DWORD kind = GetFileType(handle);
            if (kind == FILE_TYPE_PIPE) {
                DWORD available = 0;
                for (;;) {
                    if (!PeekNamedPipe(handle, nullptr, 0, nullptr, &available, nullptr)) {
                        if (GetLastError() == ERROR_BROKEN_PIPE) {
                            available = 0;
                            ended = true;
                            break;
                        }
                        throw std::runtime_error("Hy4 serve: stdin pipe query failed");
                    }
                    if (available || !wait)
                        break;
                    Sleep(2);
                }
                if (!ended && !available)
                    return false;
            } else if (kind == FILE_TYPE_CHAR) {
                const DWORD ready = WaitForSingleObject(handle, wait ? INFINITE : 0);
                if (ready == WAIT_TIMEOUT)
                    return false;
                require(ready == WAIT_OBJECT_0, "stdin console wait failed");
            } else
                require(kind == FILE_TYPE_DISK, "unsupported stdin handle type");
            if (!ended) {
                DWORD count = 0;
                if (!ReadFile(handle, buffer, sizeof(buffer), &count, nullptr)) {
                    require(GetLastError() == ERROR_BROKEN_PIPE, "stdin read failed");
                    count = 0;
                }
                n = count;
            }
#else
            pollfd fd{STDIN_FILENO, POLLIN, 0};
            int ready;
            do {
                ready = ::poll(&fd, 1, wait ? -1 : 0);
            } while (ready < 0 && errno == EINTR);
            if (ready < 0)
                throw std::runtime_error("Hy4 serve: stdin poll failed");
            if (!ready)
                return false;
            const auto count = ::read(STDIN_FILENO, buffer, sizeof(buffer));
            if (count < 0) {
                if (errno == EINTR)
                    continue;
                throw std::runtime_error("Hy4 serve: stdin read failed");
            }
            n = size_t(count);
#endif
            if (!n) {
                ended = true;
                if (pending.empty()) {
                    line = "QUIT";
                    return true;
                }
                pending += "\nQUIT\n";
                continue;
            }
            pending.append(buffer, size_t(n));
            require(pending.size() <= limit, "request line exceeds limit");
        }
    }
};
} // namespace
int serve_device(DeviceModel &model, const Geometry &geometry, std::span<const int32_t> eos,
                 ServerOptions options, ServerTransport transport) {
    return Server(model, geometry, eos, options, std::move(transport)).run();
}
int serve_device(DeviceModel &model, const Engine &weights, ServerOptions options) {
    std::vector<int32_t> eos;
    for (const char *key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"})
        if (const auto *v = weights.metadata().get(key)) {
            require(v->is_num() && v->u < uint64_t(weights.geometry().vocab), "invalid EOS metadata");
            eos.push_back(int32_t(v->u));
        }
    if (!options.max_context)
        options.max_context = weights.max_context();
    if (options.vision && options.image_pad < 0) {
        const auto *tokens = weights.metadata().get("tokenizer.ggml.tokens");
        require(tokens && tokens->items.size() == size_t(weights.geometry().vocab),
                "missing retained tokenizer for vision");
        for (size_t i = 0; i < tokens->items.size(); ++i)
            if (tokens->items[i].s.find("hy_image▁pad:") != std::string::npos) {
                require(options.image_pad < 0, "ambiguous image pad tokenizer");
                options.image_pad = int32_t(i);
            }
        require(options.image_pad >= 0, "missing Hy4 image pad tokenizer");
    }
    auto stdin_io = std::make_shared<StdinTransport>();
    stdin_io->limit = options.max_line_bytes;
    return serve_device(model, weights.geometry(), eos, options,
                        {[stdin_io](std::string &line, bool wait) { return stdin_io->read(line, wait); },
                         [](const std::string &line) {
                             std::printf("%s\n", line.c_str());
                             std::fflush(stdout);
                         }});
}
} // namespace strata::hy4
