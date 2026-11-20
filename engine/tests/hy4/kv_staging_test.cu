#include "device_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace K=strata::hy4::device_kernels;
namespace {
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Buffer{void* p=nullptr;Buffer(size_t bytes){ck(cudaMalloc(&p,bytes));}~Buffer(){cudaFree(p);}template<class T>T* as(){return static_cast<T*>(p);}};
template<class T>T cv(float x){return T(x);}template<>half cv(float x){return __float2half_rn(x);}
template<class T>void test(bool swa,int rows,int window){
    constexpr int heads=4,latent=512,rot=64,hd=256,value=128,width=latent+rot,topk=2048,pool=4,capacity=4096;
    const int ring=swa?window+rows:0,stage_capacity=std::max(topk,window);
    const size_t cells=swa?size_t(2)*ring:size_t(2)*capacity;
    T* host=nullptr;ck(cudaHostAlloc(reinterpret_cast<void**>(&host),cells*width*sizeof(T),cudaHostAllocMapped|cudaHostAllocPortable));
    struct HostFree{void* p;~HostFree(){cudaFreeHost(p);}}cleanup{host};
    for(size_t i=0;i<cells*width;++i)host[i]=cv<T>(std::sin(float(i)*.013f)*.1f);
    void* mapped=nullptr;ck(cudaHostGetDevicePointer(&mapped,host,0));
    std::vector<int32_t> slots(rows),positions(rows),map(size_t(2)*capacity),selected(size_t(rows)*topk,-1);
    int max_count=0;
    for(int slot=0;slot<2;++slot)for(int i=0;i<capacity;++i)map[size_t(slot)*capacity+i]=slot*capacity+(i*37)%capacity;
    for(int r=0;r<rows;++r){slots[r]=r%2;positions[r]=(swa?window+31:capacity-rows)+r;
        const int take=std::min(positions[r]/pool,topk/pool-1),count=swa?std::min(window,positions[r]+1):take*pool+positions[r]%pool+1;max_count=std::max(max_count,count);
        if(!swa){for(int i=0;i<take*pool;++i)selected[size_t(r)*topk+i]=i;for(int i=take*pool;i<count;++i)selected[size_t(r)*topk+i]=positions[r]-positions[r]%pool+i-take*pool;}}
    const int chunks=(max_count+127)/128;
    std::vector<float> qa(size_t(rows)*heads*latent),q(size_t(rows)*heads*hd),sinks(heads,.03125f);
    for(size_t i=0;i<qa.size();++i)qa[i]=std::sin(float(i)*.017f)*.1f;for(size_t i=0;i<q.size();++i)q[i]=std::cos(float(i)*.019f)*.1f;
    Buffer ds(slots.size()*4),dp(positions.size()*4),dm(map.size()*4),di(selected.size()*4),dqa(qa.size()*4),dq(q.size()*4),dn(sinks.size()*4);
    Buffer stage(size_t(8)*stage_capacity*width*sizeof(T));const size_t partial_bytes=size_t(8)*heads*chunks*(latent+2)*4;
    Buffer direct(partial_bytes),staged(partial_bytes),out_direct(size_t(rows)*heads*latent*4),out_staged(size_t(rows)*heads*latent*4);
    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto copy=[&](void* dst,const void* src,size_t n){ck(cudaMemcpyAsync(dst,src,n,cudaMemcpyHostToDevice,stream));};
    copy(ds.p,slots.data(),slots.size()*4);copy(dp.p,positions.data(),positions.size()*4);copy(dm.p,map.data(),map.size()*4);copy(di.p,selected.data(),selected.size()*4);copy(dqa.p,qa.data(),qa.size()*4);copy(dq.p,q.data(),q.size()*4);copy(dn.p,sinks.data(),sinks.size()*4);
    for(int base=0;base<rows;base+=8){const int take=std::min(8,rows-base);
        K::stage_kv_batch<<<dim3((stage_capacity*width+255)/256,take),256,0,stream>>>(static_cast<const T*>(mapped),stage.as<T>(),dm.as<int32_t>(),di.as<int32_t>(),ds.as<int32_t>(),dp.as<int32_t>(),base,width,topk,pool,window,capacity,stage_capacity,ring);
        for(int rr=0;rr<std::min(2,take);++rr){const int r=base+rr,pos=positions[r],count=swa?std::min(window,pos+1):std::min(pos/pool,topk/pool-1)*pool+pos%pool+1;
            K::stage_kv<<<(count*width+255)/256,256,0,stream>>>(static_cast<const T*>(mapped)+(swa?size_t(slots[r])*ring*width:0),stage.as<T>()+size_t(rr)*stage_capacity*width,dm.as<int32_t>()+size_t(slots[r])*capacity,swa?nullptr:di.as<int32_t>()+size_t(r)*topk,pos,count,width,ring);}
        K::attn_split_batch<<<dim3(heads,chunks,take),256,0,stream>>>(dqa.as<float>(),dq.as<float>(),static_cast<const T*>(mapped),dm.as<int32_t>(),di.as<int32_t>(),dn.as<float>(),direct.as<float>(),ds.as<int32_t>(),dp.as<int32_t>(),base,heads,latent,hd,rot,value,topk,pool,window,capacity,chunks,ring,false,stage_capacity);
        K::attn_merge_batch<<<dim3(heads,take),256,0,stream>>>(direct.as<float>(),out_direct.as<float>(),rows,heads,latent,chunks,base);
        K::attn_split_batch<<<dim3(heads,chunks,take),256,0,stream>>>(dqa.as<float>(),dq.as<float>(),stage.as<T>(),dm.as<int32_t>(),di.as<int32_t>(),dn.as<float>(),staged.as<float>(),ds.as<int32_t>(),dp.as<int32_t>(),base,heads,latent,hd,rot,value,topk,pool,window,capacity,chunks,ring,true,stage_capacity);
        K::attn_merge_batch<<<dim3(heads,take),256,0,stream>>>(staged.as<float>(),out_staged.as<float>(),rows,heads,latent,chunks,base);ck(cudaGetLastError());
        std::vector<T> gathered(size_t(take)*stage_capacity*width);std::vector<uint8_t>a(size_t(take)*heads*chunks*(latent+2)*4),b(a.size());
        ck(cudaMemcpyAsync(gathered.data(),stage.p,gathered.size()*sizeof(T),cudaMemcpyDeviceToHost,stream));ck(cudaMemcpyAsync(a.data(),direct.p,a.size(),cudaMemcpyDeviceToHost,stream));ck(cudaMemcpyAsync(b.data(),staged.p,b.size(),cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
        check(a==b,"selected KV staging changed attention scores/partial sums");
        for(int rr=0;rr<take;++rr){const int r=base+rr,pos=positions[r],count=swa?std::min(window,pos+1):std::min(pos/pool,topk/pool-1)*pool+pos%pool+1;
            for(int j=0;j<count;++j){const int logical=swa?pos-count+1+j:selected[size_t(r)*topk+j],physical=swa?slots[r]*ring+logical%ring:map[size_t(slots[r])*capacity+logical];
                check(std::memcmp(gathered.data()+(size_t(rr)*stage_capacity+j)*width,host+size_t(physical)*width,width*sizeof(T))==0,"selected KV raw bits changed during staging");}}
    }
    std::vector<uint8_t>a(size_t(rows)*heads*latent*4),b(a.size());ck(cudaMemcpyAsync(a.data(),out_direct.p,a.size(),cudaMemcpyDeviceToHost,stream));ck(cudaMemcpyAsync(b.data(),out_staged.p,b.size(),cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));check(a==b,"staging changed final attention values");ck(cudaStreamDestroy(stream));
    std::cout<<"raw KV bits/attention exact dtype_bytes="<<sizeof(T)<<" swa="<<swa<<" window="<<window<<" rows="<<rows<<'\n';
}
}
int main(){try{for(int rows:{1,8,16}){test<half>(false,rows,512);test<float>(false,rows,512);test<half>(true,rows,512);test<float>(true,rows,4);}return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
