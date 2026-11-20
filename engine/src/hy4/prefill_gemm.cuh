#pragma once
// Quantized-weight tensor-core tiles for Hy4 prefill. Original GGUF weights
// remain resident/streamed in their stored format; decoding occurs only in shared
// memory. Inputs are rounded to FP16 (BF16 for BF16 weights), FP32 accumulation.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

namespace strata::hy4::prefill_detail {
namespace wmma = nvcuda::wmma;
struct Q4K {half d,m;uint8_t scales[12],q[128];};
struct Q6K {uint8_t lo[128],hi[64];int8_t scales[16];half d;};
static_assert(sizeof(Q4K)==144&&sizeof(Q6K)==210);
inline bool supported(int type){return type==0||type==1||type==30||type==8||type==12||type==14;}
inline size_t matrix_bytes(int type,int ni,int no){return type==0?size_t(ni)*no*4:(type==1||type==30)?size_t(ni)*no*2:type==8?size_t(ni)*no/32*34:type==12?size_t(ni)*no/256*144:size_t(ni)*no/256*210;}
__device__ inline float value(const uint8_t* w,int type,int ni,int row,int col){
    const size_t at=size_t(row)*ni+col;
    if(type==0)return reinterpret_cast<const float*>(w)[at];
    if(type==1)return __half2float(reinterpret_cast<const half*>(w)[at]);
    if(type==30)return __uint_as_float(uint32_t(reinterpret_cast<const uint16_t*>(w)[at])<<16);
    if(type==8){const auto* b=w+(at/32)*34;return __half2float(*reinterpret_cast<const half*>(b))*reinterpret_cast<const int8_t*>(b+2)[at%32];}
    if(type==12){const auto& b=reinterpret_cast<const Q4K*>(w)[at/256];const int i=at%256,j=i/32;
        const int sc=j<4?b.scales[j]&63:(b.scales[j+4]&15)|((b.scales[j-4]>>6)<<4);
        const int mn=j<4?b.scales[j+4]&63:(b.scales[j+4]>>4)|((b.scales[j]>>6)<<4);
        const int q=(b.q[(i/64)*32+i%32]>>((i%64)/32*4))&15;
        return (__half2float(b.d)*sc)*q-__half2float(b.m)*mn;
    }
    const auto& b=reinterpret_cast<const Q6K*>(w)[at/256];const int i=at%256,part=i/128,chunk=(i%128)/32,lane=i%32;
    const int q=((b.lo[part*64+(chunk%2)*32+lane]>>((chunk/2)*4))&15)|(((b.hi[part*32+lane]>>(chunk*2))&3)<<4);
    return __half2float(b.d)*b.scales[part*8+chunk*2+lane/16]*(q-32);
}
template<class T>__device__ inline T round16(float x);
template<>__device__ inline half round16<half>(float x){return __float2half_rn(x);}
template<>__device__ inline __nv_bfloat16 round16<__nv_bfloat16>(float x){return __float2bfloat16_rn(x);}

// Dense tiles: four warps share 16 weight rows across 64 activation rows.
// Grouped tiles: four warps share 16 activation rows across 64 weight rows.
// Grouped task lists are GPU-resident; a task is (expert, first compacted row).
// A null table describes one streamed blob. A null indices array is dense input.
template<class T,bool Dense>
__global__ void gemm_tile(const uint8_t* blob,const unsigned long long* table,
                         const float* x,float* y,float* up,int type,int up_type,int ni,int no,int rows,
                         const int32_t* offsets,const int32_t* counts,const int32_t* indices,
                         const int32_t* task_expert,const int32_t* task_start,const int32_t* task_count,
                         int group,int group_count,int used,bool per_route,bool compact_output,
                         size_t offset,size_t up_offset,int ldx,int ldy){
    constexpr int MT=Dense?1:4,NT=Dense?4:1;
    const int warp=threadIdx.x/32;
    int expert=group,first=blockIdx.y*(NT*16),count=group_count;
    if(task_count){if(int(blockIdx.y)>=*task_count)return;expert=task_expert[blockIdx.y];first=task_start[blockIdx.y];count=counts[expert];}
    if constexpr(Dense)count=rows;
    if(first>=count)return;
    const int wm=Dense?0:warp,wn=Dense?warp:0;
    const int row0=(blockIdx.x*MT+wm)*16,token0=first+wn*16;
    if(int(blockIdx.x)*MT*16>=no)return; // uniform CTA exit; edge warps still participate in barriers
    const int format=Dense?type:(blockIdx.z?up_type:type);
    const uint8_t* weights=(table?reinterpret_cast<const uint8_t*>(table[expert]):blob)+(Dense?size_t(blockIdx.z)*up_offset:(blockIdx.z?up_offset:offset));
    float* out=Dense?y+size_t(blockIdx.z)*no:(blockIdx.z?up:y);
    if constexpr(Dense)x+=size_t(blockIdx.z)*rows*ldx;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    // Legacy CUDA builds keep a correct bounded fallback; the optimized path
    // below targets tensor-core devices. Do not emit unsupported WMMA PTX.
    for(int i=threadIdx.x;i<MT*NT*256;i+=128){const int r=int(blockIdx.x)*MT*16+i/(NT*16),t=first+i%(NT*16);if(r>=no||t>=count)continue;
        int input=t,output=t;if(indices){const int route=indices[offsets?offsets[expert]+t:t];input=per_route?route:route/used;if(!compact_output)output=route;}
        float v=0;for(int k=0;k<ni;++k)v+=value(weights,format,ni,r,k)*x[size_t(input)*ldx+k];out[size_t(output)*ldy+r]=v;}
#else
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    using Elem=half; // Turing/Volta use the upstream FP16 fallback for BF16.
#else
    using Elem=T;
#endif
    __shared__ __align__(32) Elem a[MT*256],b[NT*256];
    __shared__ __align__(32) float c[4*256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<ni;k0+=16){
        for(int i=threadIdx.x;i<MT*256;i+=128){const int tile=i/256,j=i%256,r=blockIdx.x*MT*16+tile*16+j/16,k=k0+j%16;
            a[i]=round16<Elem>(r<no&&k<ni?value(weights,format,ni,r,k):0.f);}
        for(int i=threadIdx.x;i<NT*256;i+=128){const int tile=i/256,j=i%256,t=first+tile*16+j/16,k=k0+j%16;int input=t;
            if(indices&&t<count){const int route=indices[offsets?offsets[expert]+t:t];input=per_route?route:route/used;}
            b[i]=round16<Elem>(t<count&&k<ni?x[size_t(input)*ldx+k]:0.f);}
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,Elem,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,Elem,wmma::col_major> bf;
        wmma::load_matrix_sync(af,a+wm*256,16);wmma::load_matrix_sync(bf,b+wn*256,16);wmma::mma_sync(acc,af,bf,acc);
        __syncthreads();
    }
    wmma::store_matrix_sync(c+warp*256,acc,16,wmma::mem_row_major);__syncthreads();
    for(int i=threadIdx.x%32;i<256;i+=32){const int r=row0+i/16,t=token0+i%16;if(r>=no||t>=count)continue;int output=t;
        if(indices&&!compact_output)output=indices[offsets?offsets[expert]+t:t];
        out[size_t(output)*ldy+r]=c[warp*256+i];}
#endif
}
inline void dense(int type,const void* w,const float* x,float* y,int ni,int no,int rows,cudaStream_t stream,int ldx=0,int ldy=0,int planes=1){
    const dim3 grid((no+15)/16,(rows+63)/64,planes);if(!ldx)ldx=ni;if(!ldy)ldy=no;
    if(type==30)gemm_tile<__nv_bfloat16,true><<<grid,128,0,stream>>>(static_cast<const uint8_t*>(w),nullptr,x,y,nullptr,type,type,ni,no,rows,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,rows,1,false,true,0,planes>1?matrix_bytes(type,ni,no):0,ldx,ldy);
    else gemm_tile<half,true><<<grid,128,0,stream>>>(static_cast<const uint8_t*>(w),nullptr,x,y,nullptr,type,type,ni,no,rows,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,rows,1,false,true,0,planes>1?matrix_bytes(type,ni,no):0,ldx,ldy);
}
inline void grouped(int type,int up_type,const void* blob,const unsigned long long* table,const float* x,float* y,float* up,
                    int ni,int no,int rows,const int32_t* offsets,const int32_t* counts,const int32_t* indices,
                    const int32_t* task_expert,const int32_t* task_start,const int32_t* task_count,int max_tasks,
                    int group,int count,int used,bool per_route,bool compact_output,size_t offset,size_t up_offset,cudaStream_t stream){
    const dim3 grid((no+63)/64,task_count?max_tasks:(count+15)/16,up?2:1);
    if(type==30&&(!up||up_type==30))gemm_tile<__nv_bfloat16,false><<<grid,128,0,stream>>>(static_cast<const uint8_t*>(blob),table,x,y,up,type,up_type,ni,no,rows,offsets,counts,indices,task_expert,task_start,task_count,group,count,used,per_route,compact_output,offset,up_offset,ni,no);
    else gemm_tile<half,false><<<grid,128,0,stream>>>(static_cast<const uint8_t*>(blob),table,x,y,up,type,up_type,ni,no,rows,offsets,counts,indices,task_expert,task_start,task_count,group,count,used,per_route,compact_output,offset,up_offset,ni,no);
}
}
