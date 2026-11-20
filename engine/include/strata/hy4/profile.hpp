#pragma once
#include <cstdlib>
#ifdef STRATA_HY4_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

namespace strata::hy4 {
// Optional profiler annotations; disabled in ordinary serving. Ranges describe
// CPU submission windows. CUDA event timestamps establish GPU execution/overlap.
class ProfileScope {
  public:
    explicit ProfileScope(const char *label) {
#ifdef STRATA_HY4_CUDA
        static const bool enabled = [] {
            const char *value = std::getenv("STRATA_HY4_PROFILE");
            return value && value[0] == '1' && value[1] == '\0';
        }();
        active_ = enabled;
        if (active_) nvtxRangePushA(label);
#else
        (void)label;
#endif
    }
    ~ProfileScope() {
#ifdef STRATA_HY4_CUDA
        if (active_) nvtxRangePop();
#endif
    }
    ProfileScope(const ProfileScope &) = delete;
    ProfileScope &operator=(const ProfileScope &) = delete;
  private:
    bool active_ = false;
};
}
