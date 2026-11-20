// The standalone Hy4 offloader uses eager copies and a host synchronization after every projection.
// It never records a verify graph, so programmatic graph edges must stay disabled. The full Strata
// kernel target supplies its own PDL implementation instead when that target is linked.
#include <cuda_runtime_api.h>
namespace strata::kernels {
bool& pdl_scope() { static thread_local bool value = false; return value; }
bool pdl_supported() { return false; }
bool pdl_launch_ok(const void*, cudaStream_t) { return false; }
}
