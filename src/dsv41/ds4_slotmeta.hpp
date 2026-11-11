// The kept-conversation (slot) bookkeeping that needs no GPU: a slot's .meta file (identity, size, tokens, checkpoint
// positions) and the choice of which kept conversation a prompt continues, and from where.  The engine's serve loop
// (ds4_main.cpp) uses these; tests/dsv41/slot_test.cpp checks them alone.
//
// Slot .meta layout (little endian), "DS4META3":
//   u64 magic, u64 identity length, identity bytes, u64 scope length (0..kScopeMax), scope bytes, u64 .bin bytes,
//   u64 token count, i32 tokens[], u64 checkpoint count (0..19), i64 checkpoints[], u64 magic
// The magic at both ends and the .bin size make a half-written or foreign file refused, never read.  A file of the
// layout before scopes ("DS4META2") names no owner, so it is refused like a foreign one and removed at start.
//
// CACHE SCOPE: the opaque id a request names with GEN scope=<id> (the server's tenant; "" without).  Every kept state
// - a conversation, a shared start, a picture's rows, the state in the model - belongs to the scope that read it, and
// only a request of the same scope goes on from it.
#pragma once

#include "ds4_marks.hpp"   // kMarksMax

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace strata::ds4::slotmeta {

constexpr uint64_t kMagic = 0x334154454d345344ull;   // "DS4META3"
constexpr size_t kScopeMax = 128;

// A scope a GEN line can carry and a .meta can hold: empty, or up to kScopeMax of [A-Za-z0-9._-].
inline bool scope_ok(const std::string& s) {
    if (s.size() > kScopeMax) return false;
    for (const char ch : s) {
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '.' ||
                        ch == '_' || ch == '-';
        if (!ok) return false;
    }
    return true;
}

// Written under a temporary name and renamed over `path` (an interrupted write leaves no half file).
inline bool write(const std::string& path, const std::string& identity, const std::string& scope,
                  const std::vector<int32_t>& toks, uint64_t bytes, const std::vector<int64_t>& cks) {
    if (!scope_ok(scope)) return false;
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        const uint64_t il = identity.size(), sl = scope.size(), nt = toks.size(), nc = cks.size();
        f.write((const char*) &kMagic, 8);
        f.write((const char*) &il, 8);
        f.write(identity.data(), (std::streamsize) il);
        f.write((const char*) &sl, 8);
        f.write(scope.data(), (std::streamsize) sl);
        f.write((const char*) &bytes, 8);
        f.write((const char*) &nt, 8);
        f.write((const char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t)));
        f.write((const char*) &nc, 8);
        f.write((const char*) cks.data(), (std::streamsize) (nc * sizeof(int64_t)));
        f.write((const char*) &kMagic, 8);
        f.flush();
        if (!f) return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    return !ec;
}

// false: foreign (another identity), damaged, cut short, or not plausible (no tokens, more than max_ctx, a checkpoint
// outside the tokens, a scope no GEN line can carry).  `scope`: the slot's owner, as written.
inline bool read(const std::string& path, const std::string& identity, int64_t max_ctx, std::string& scope,
                 std::vector<int32_t>& toks, uint64_t& bytes, std::vector<int64_t>& cks) {
    std::ifstream f(path, std::ios::binary);
    uint64_t m0 = 0, il = 0, sl = 0, nt = 0, nc = 0, m1 = 0;
    if (!f.read((char*) &m0, 8) || m0 != kMagic || !f.read((char*) &il, 8) || il != identity.size()) return false;
    std::string id(il, '\0');
    if (!f.read(id.data(), (std::streamsize) il) || id != identity) return false;
    if (!f.read((char*) &sl, 8) || sl > kScopeMax) return false;
    scope.assign(sl, '\0');
    if (!f.read(scope.data(), (std::streamsize) sl) || !scope_ok(scope)) return false;
    if (!f.read((char*) &bytes, 8) || !f.read((char*) &nt, 8) || nt == 0 || nt > (uint64_t) max_ctx) return false;
    toks.resize(nt);
    if (!f.read((char*) toks.data(), (std::streamsize) (nt * sizeof(int32_t))) || !f.read((char*) &nc, 8) || nc > 3 + (uint64_t) kMarksMax)
        return false;
    cks.resize(nc);
    if (!f.read((char*) cks.data(), (std::streamsize) (nc * sizeof(int64_t))) || !f.read((char*) &m1, 8) || m1 != kMagic)
        return false;
    for (int64_t c : cks)
        if (c <= 0 || (uint64_t) c > nt) return false;
    return true;
}

// The kept conversation of `scope` that `prompt` continues the furthest, and where: *len is the number of the slot's tokens that are
// the prompt's (0: none).  A conversation the prompt continues whole comes back whole (len == its size); one the prompt
// leaves part way - a retried prompt, a history re-rendered without the earlier reasoning - comes back at its latest
// checkpoint the prompt still agrees with.  Only a match longer than `floor` (what the model itself can go back to)
// counts.  The last prompt token is never covered (a prompt always reads its last token).  Ties keep the earlier slot.
// A conversation of another scope is never picked, whatever it shares with the prompt.
struct Cand {
    const std::vector<int32_t>* tokens;
    const std::vector<int64_t>* cks;
    const std::string* scope;
};
inline size_t pick(const std::vector<Cand>& slots, const std::string& scope, const std::vector<int>& prompt,
                   size_t floor, size_t* len) {
    *len = 0;
    if (prompt.size() < 2) return SIZE_MAX;
    size_t best = SIZE_MAX;
    for (size_t i = 0; i < slots.size(); ++i) {
        if (*slots[i].scope != scope) continue;
        const std::vector<int32_t>& t = *slots[i].tokens;
        const size_t lim = std::min(t.size(), prompt.size() - 1);
        size_t c = 0;
        while (c < lim && t[c] == prompt[c]) ++c;
        size_t u = c == t.size() ? c : 0;
        if (u == 0)
            for (int64_t k : *slots[i].cks)
                if ((size_t) k <= c && (size_t) k > u) u = (size_t) k;
        if (u > *len && u > floor) {
            best = i;
            *len = u;
        }
    }
    return best;
}

}  // namespace strata::ds4::slotmeta
