#include "strata/hy4/offset_arena.hpp"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
using strata::hy4::OffsetArena;
namespace {
void check(bool value) { if (!value) throw std::runtime_error("Hy4 offset arena regression"); }
template<class F> void rejects(F fn) { bool rejected=false; try { fn(); } catch (const std::invalid_argument&) { rejected=true; } check(rejected); }
}
int main() { try {
    rejects([]{OffsetArena invalid(32,3);});
    OffsetArena empty(0); check(!empty.allocate(1));
    OffsetArena arena(4097);
    rejects([&]{arena.allocate(0);}); rejects([&]{arena.allocate(SIZE_MAX);});
    auto a=arena.allocate(1), b=arena.allocate(257), c=arena.allocate(2048);
    check(a==0 && b==256 && c==768 && arena.used()==2816);
    rejects([&]{arena.release(*a,2);}); rejects([&]{arena.release(*a+1,1);});
    arena.release(*b,257); arena.release(*a,1); rejects([&]{arena.release(*a,1);});
    auto merged=arena.allocate(768);check(merged==0);arena.release(*c,2048);arena.release(*merged,768);
    auto full=arena.allocate(4096);check(full==0 && !arena.allocate(1));arena.release(*full,4096);
    struct Block {size_t offset, bytes;}; std::vector<Block> live;
    std::mt19937 random(71); OffsetArena mixed(1<<20);
    for (int step=0;step<30000;++step) {
        if (!live.empty() && (random()%3==0 || live.size()>100)) {
            size_t i=random()%live.size();mixed.release(live[i].offset,live[i].bytes);live.erase(live.begin()+i);
        } else {
            const size_t bytes=1+random()%32768;
            auto offset=mixed.allocate(bytes);
            if (offset) {
                check(*offset%256==0 && *offset+mixed.allocation_bytes(bytes)<=mixed.capacity());
                for (const auto& old:live) check(*offset+mixed.allocation_bytes(bytes)<=old.offset || old.offset+mixed.allocation_bytes(old.bytes)<=*offset);
                live.push_back({*offset,bytes});
            }
        }
        size_t used=0;for(const auto& b:live)used+=mixed.allocation_bytes(b.bytes);check(used==mixed.used());
    }
    for(const auto& b:live)mixed.release(b.offset,b.bytes);
    check(mixed.used()==0 && mixed.allocate(mixed.capacity())==0);
    std::cout<<"Hy4 offset arena alignment, lifetime and randomized coalescing passed\n";return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;} }
