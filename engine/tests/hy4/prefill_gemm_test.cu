#include "prefill_gemm.cuh"
#include "strata/hy4/engine.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
using namespace strata::hy4;
namespace P=strata::hy4::prefill_detail;
namespace {
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer{void* p=nullptr;Buffer(size_t n){ck(cudaMalloc(&p,n));}~Buffer(){cudaFree(p);}template<class T>T* as(){return static_cast<T*>(p);}};
void check(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
float rounded(float x,int type){if(type==30)return __bfloat162float(__float2bfloat16_rn(x));return __half2float(__float2half_rn(x));}
void test(int type,int ni,int no,int rows,int planes){
    std::mt19937 random(51+type);
    const size_t bytes=P::matrix_bytes(type,ni,no)*planes;
    std::vector<uint8_t> raw(bytes);
    if(type==0||type==1||type==30){for(size_t i=0;i<size_t(ni)*no*planes;++i){float v=std::sin(float(i)*.193f)*.2f;
        if(type==0)std::memcpy(raw.data()+i*4,&v,4);
        else if(type==1){half h=__float2half_rn(v);std::memcpy(raw.data()+i*2,&h,2);}
        else{auto b=__float2bfloat16_rn(v);std::memcpy(raw.data()+i*2,&b,2);}}}
    else {const size_t block=type==8?34:type==12?144:210;for(size_t off=0;off<bytes;off+=block){for(size_t i=0;i<block;++i)raw[off+i]=uint8_t(random());
        half d=__float2half(.0078125f),m=__float2half(.00390625f);if(type==14){std::memcpy(raw.data()+off+208,&d,2);for(int i=0;i<16;++i)raw[off+192+i]=uint8_t(int(random()%9)-4);}
        else{std::memcpy(raw.data()+off,&d,2);if(type==12)std::memcpy(raw.data()+off+2,&m,2);}}}
    strata::TensorInfo info;info.name="prefill.weight";info.type=type;info.shape={uint64_t(ni),uint64_t(no),uint64_t(planes)};
    Tensor weight{&info,raw.data()};auto decoded=weight.values();
    for(float& v:decoded)v=rounded(v,type);
    Vec x(size_t(planes)*rows*ni),out(size_t(rows)*planes*no);
    for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.017f)*.4f;
    Buffer w(bytes),dx(x.size()*4),dy(out.size()*4);cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ck(cudaMemcpyAsync(w.p,raw.data(),bytes,cudaMemcpyHostToDevice,stream));ck(cudaMemcpyAsync(dx.p,x.data(),x.size()*4,cudaMemcpyHostToDevice,stream));
    P::dense(type,w.p,dx.as<float>(),dy.as<float>(),ni,no,rows,stream,ni,no*planes,planes);
    ck(cudaGetLastError());ck(cudaMemcpyAsync(out.data(),dy.p,out.size()*4,cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));ck(cudaStreamDestroy(stream));
    double maximum=0,error=0;
    // Include ragged final tile coordinates and sample the actual large geometry.
    for(int h=0;h<planes;++h)for(int t:{0,rows/2,rows-1})for(int r:{0,1,std::min(17,no-1),no/2,no-1}){
        double expected=0;for(int k=0;k<ni;++k)expected+=double(decoded[(size_t(h)*no+r)*ni+k])*rounded(x[(size_t(h)*rows+t)*ni+k],type);
        const float actual=out[(size_t(t)*planes+h)*no+r];check(std::isfinite(actual),"TC produced nonfinite output");maximum=std::max(maximum,std::abs(expected));error=std::max(error,std::abs(double(actual)-expected));}
    std::cout<<"type="<<type<<" "<<ni<<"->"<<no<<" rows="<<rows<<" planes="<<planes<<" max_abs="<<error<<" relative="<<error/(1+maximum)<<'\n';
    // Reference explicitly models the input/weight FP16 or BF16 rounding. This
    // allowance covers only FP32 accumulation order, not that conversion error.
    check(error<=2e-5*(1+maximum),"TC differs from independent rounded CPU dot");
}
}
int main(){try{
    for(int type:{0,1,30,8,12,14})for(int rows:{8,9,15,16,31,128,256,511})test(type,256,70,rows,1);
    for(int type:{0,1,30,8})test(type,192,512,31,4);
    for(int type:{8,12,14,30})test(type,3072,1024,128,1);
    std::cout<<"Hy4 tensor-core prefill CPU-reference parity passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
