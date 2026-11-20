#pragma once
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>

namespace strata::hy4 {
// Host bookkeeping for one externally owned device allocation. Offsets are
// aligned; release takes the original requested byte count, not its padded size.
class OffsetArena {
  public:
    explicit OffsetArena(size_t capacity, size_t alignment = 256)
        : capacity_(capacity), alignment_(alignment) {
        if (!alignment || (alignment & (alignment - 1)))
            throw std::invalid_argument("Hy4 arena alignment must be a power of two");
        if (capacity) free_.emplace(0, capacity);
    }
    OffsetArena(const OffsetArena &) = delete;
    OffsetArena &operator=(const OffsetArena &) = delete;
    size_t allocation_bytes(size_t bytes) const {
        if (!bytes || bytes > std::numeric_limits<size_t>::max() - (alignment_ - 1))
            throw std::invalid_argument("Hy4 arena allocation size is zero or overflows");
        return (bytes + alignment_ - 1) & ~(alignment_ - 1);
    }
    std::optional<size_t> allocate(size_t bytes) {
        const size_t charged = allocation_bytes(bytes);
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            if (it->second < charged) continue;
            const size_t offset = it->first, remainder = it->second - charged;
            // Insert bookkeeping before mutation so allocation failure leaves
            // the existing ranges and accounting intact.
            allocated_.emplace(offset, Allocation{bytes, charged});
            if (remainder) {
                try { free_.emplace(offset + charged, remainder); }
                catch (...) { allocated_.erase(offset); throw; }
            }
            free_.erase(it);
            used_ += charged;
            return offset;
        }
        return {};
    }
    void release(size_t offset, size_t bytes) {
        const auto allocation = allocated_.find(offset);
        if (allocation == allocated_.end() || allocation->second.requested != bytes)
            throw std::invalid_argument("Hy4 arena invalid, stale or double release");
        const size_t charged = allocation->second.charged;
        auto [it, inserted] = free_.emplace(offset, charged);
        if (!inserted) throw std::logic_error("Hy4 arena overlapping free range");
        allocated_.erase(allocation);
        used_ -= charged;
        if (it != free_.begin()) {
            auto previous = std::prev(it);
            if (previous->first + previous->second == it->first) {
                previous->second += it->second;
                free_.erase(it);
                it = previous;
            }
        }
        const auto next = std::next(it);
        if (next != free_.end() && it->first + it->second == next->first) {
            it->second += next->second;
            free_.erase(next);
        }
    }
    size_t used() const { return used_; }
    size_t capacity() const { return capacity_; }
  private:
    struct Allocation { size_t requested, charged; };
    size_t capacity_, alignment_, used_ = 0;
    std::map<size_t, size_t> free_;
    std::map<size_t, Allocation> allocated_;
};
}
