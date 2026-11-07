// The F32 router's exact lane/FMA/XOR reduction contract, including partial tiles.
#include "../../src/dsv41/ds4_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
#define CK(v) do { auto e=(v); if(e!=cudaSuccess){std::fprintf(stderr,"CUDA %s: %s\n",#v,cudaGetErrorString(e));return 2;} }while(0)

float reference(const float* w, const float* x, int k) {
    float lanes[32] = {};
    for (int lane = 0; lane < 32; ++lane)
        for (int c = lane * 4; c < k; c += 128)
            for (int j = 0; j < 4; ++j) lanes[lane] = std::fma(w[c+j], x[c+j], lanes[lane]);
    for (int d = 16; d; d >>= 1) {
        float next[32];
        for (int lane = 0; lane < 32; ++lane) next[lane] = lanes[lane] + lanes[lane ^ d];
        std::memcpy(lanes, next, sizeof lanes);
    }
    return lanes[0];
}

int main() {
    long cases = 0;
    for (auto [n,k] : {std::pair{1,4}, {3,124}, {4,128}, {5,132}, {17,5120}, {384,128}, {385,128}}) {
        for (int T : {1,16,31,127,128,129,255,257,512}) {
            const int ldy = n + 13;
            std::vector<float> w((size_t)n*k), x((size_t)T*k), want((size_t)(T+1)*ldy+16), got(want.size());
            for (size_t i=0;i<w.size();++i) w[i] = ((int)((i*13)%137)-68)*0.0031f;
            for (size_t i=0;i<x.size();++i) x[i] = ((int)((i*7)%113)-56)*0.0073f;
            std::memset(want.data(), 0xcd, want.size()*sizeof(float));
            for (int t=0;t<T;++t)
                for (int row=0;row<n;++row) want[(size_t)t*ldy+row] = reference(w.data()+(size_t)row*k,x.data()+(size_t)t*k,k);
            float *dw,*dx,*dy;
            CK(cudaMalloc(&dw,w.size()*4)); CK(cudaMalloc(&dx,x.size()*4)); CK(cudaMalloc(&dy,want.size()*4));
            CK(cudaMemcpy(dw,w.data(),w.size()*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice)); CK(cudaMemset(dy,0xcd,want.size()*4));
            strata::ds4::gemv_f32(dw,dx,n,k,T,dy,ldy,0); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(got.data(),dy,got.size()*4,cudaMemcpyDeviceToHost));
            if (std::memcmp(want.data(),got.data(),want.size()*4)!=0) {
                for (size_t i=0;i<got.size();++i) if(std::memcmp(&want[i],&got[i],4)!=0) {
                    std::fprintf(stderr,"Mismatch n=%d k=%d T=%d at %zu expected=%a got=%a\n",n,k,T,i,want[i],got[i]); return 3;
                }
            }
            CK(cudaFree(dw)); CK(cudaFree(dx)); CK(cudaFree(dy)); ++cases;
        }
    }
    std::printf("PASS %ld F32 router cases; full output/padding/guards match CPU bits\n",cases);
}
