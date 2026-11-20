// GPU integration check: zero-budget streaming, LRU caching and all-resident routed kernels
// must produce the same output. No model download or libllama dependency.
#include "strata/hy4/device_experts.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>

using namespace strata::hy4;
namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
void require(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
struct DeviceBuffer {
    void *p = nullptr;
    explicit DeviceBuffer(size_t bytes) { ck(cudaMalloc(&p, bytes)); }
    ~DeviceBuffer() { cudaFree(p); }
    template <class T> T *as() { return static_cast<T *>(p); }
};
struct Fixture {
    Geometry g;
    std::map<std::string, strata::TensorInfo> infos;
    std::map<std::string, std::vector<uint8_t>> bytes;
    std::map<std::string, Tensor> tensors;
    size_t per_expert = 0;
    Fixture(int width, int down_type, bool floating = false, int floating_type = 0) {
        g.layers = 1;
        g.latent = width;
        g.ff = width;
        g.experts = 4;
        g.used = 2;
        g.clamp = {10};
        std::mt19937 rng(19482);
        for (const auto &suffix : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
            const bool down = std::string(suffix) == "ffn_down_exps.weight";
            const std::string name = "blk.0." + std::string(suffix);
            const int type = floating ? floating_type : (down ? down_type : 12);
            auto &info = infos[name];
            info.name = name;
            info.shape = {uint64_t(width), uint64_t(width), 4};
            info.type = type;
            const size_t block = type == 12 ? 144 : type == 14 ? 210 : type == 0 ? 4 : 2;
            const size_t nblocks = floating ? size_t(width) * width * 4 : size_t(width) * width * 4 / 256;
            auto &raw = bytes[name];
            raw.resize(nblocks * block);
            if (floating) {
                const float value =
                    down ? .1f : (std::string(suffix) == "ffn_gate_exps.weight" ? 1.0f : 2.0f);
                for (size_t i = 0; i < nblocks; ++i) {
                    if (type == 0)
                        std::memcpy(raw.data() + i * 4, &value, 4);
                    else {
                        uint16_t v;
                        if (type == 1) {
                            const half h = __float2half(value);
                            std::memcpy(&v, &h, 2);
                        } else {
                            uint32_t bits;
                            std::memcpy(&bits, &value, 4);
                            v = uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
                        }
                        std::memcpy(raw.data() + i * 2, &v, 2);
                    }
                }
            } else
                for (size_t i = 0; i < nblocks; ++i) {
                    auto *at = raw.data() + i * block;
                    for (size_t j = 0; j < block; ++j)
                        at[j] = uint8_t(rng());
                    if (type == 12) {
                        const half d = __float2half(.025f), m = __float2half(.0125f);
                        std::memcpy(at, &d, 2);
                        std::memcpy(at + 2, &m, 2);
                        for (int j = 0; j < 12; ++j)
                            at[4 + j] &= 0x3f;
                    } else {
                        for (int j = 0; j < 16; ++j)
                            at[192 + j] = uint8_t(int(rng() % 16) - 8);
                        const half d = __float2half(.01875f);
                        std::memcpy(at + 208, &d, 2);
                    }
                }
            tensors[name] = Tensor{&info, raw.data()};
            per_expert += raw.size() / 4;
        }
    }
    ExpertExecutor::Lookup lookup() {
        return [this](const std::string &name) -> const Tensor & { return tensors.at(name); };
    }
};

std::vector<float> execute(Fixture &f, ExpertExecutor &executor, int device, int rows, bool ones = false,int layer=0) {
    ck(cudaSetDevice(device));
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    struct Cleanup {
        cudaStream_t stream;
        ~Cleanup() {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    } cleanup{stream};
    const int latent = f.g.latent, used = f.g.used;
    std::vector<float> x(size_t(rows) * latent), weights(size_t(rows) * used), out(x.size());
    std::vector<int32_t> ids(size_t(rows) * used);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = ones ? 1.0f : std::sin(float(i) * .017f) * .4f;
    for (int row = 0; row < rows; ++row)
        for (int k = 0; k < used; ++k) {
            ids[row * used + k] = (row + k) % f.g.experts;
            weights[row * used + k] = k ? .75f : .25f;
        }
    DeviceBuffer dx(x.size() * 4), di(ids.size() * 4), dw(weights.size() * 4), dy(out.size() * 4);
    ck(cudaMemcpyAsync(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice, stream));
    ck(cudaMemcpyAsync(di.p, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice, stream));
    ck(cudaMemcpyAsync(dw.p, weights.data(), weights.size() * 4, cudaMemcpyHostToDevice, stream));
    executor.run(layer, device, dx.as<float>(), di.as<int32_t>(), dw.as<float>(), dy.as<float>(), rows, stream);
    ck(cudaMemcpyAsync(out.data(), dy.p, out.size() * 4, cudaMemcpyDeviceToHost, stream));
    ck(cudaStreamSynchronize(stream));
    for (float v : out)
        require(std::isfinite(v), "nonfinite expert output");
    return out;
}
void same(const std::vector<float> &a, const std::vector<float> &b, const char *name) {
    require(a.size() == b.size(), "expert result shape");
    float error = 0, maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        error = std::max(error, std::abs(a[i] - b[i]));
        maximum = std::max(maximum, std::abs(a[i]));
    }
    std::cout << name << " max_abs=" << error << " normalized=" << error / (1 + maximum) << '\n';
    // The routed kernel preserves the native dot reduction; tiny differences can
    // remain when a native multi-column architecture path is explicitly enabled.
    require(error <= 1e-6f * (1 + maximum), name);
}
float rounded_prefill(float x,int type){return type==30?__bfloat162float(__float2bfloat16_rn(x)):__half2float(__float2half_rn(x));}
void wide_cpu_reference(Fixture& f,const std::vector<float>& actual,int rows){
    if(rows<=8)return;
    auto gate=f.tensors.at("blk.0.ffn_gate_exps.weight").values();auto up=f.tensors.at("blk.0.ffn_up_exps.weight").values();auto down=f.tensors.at("blk.0.ffn_down_exps.weight").values();
    const int gu=f.infos.at("blk.0.ffn_gate_exps.weight").type,dt=f.infos.at("blk.0.ffn_down_exps.weight").type,n=f.g.latent;
    for(float& v:gate)v=rounded_prefill(v,gu);for(float& v:up)v=rounded_prefill(v,gu);for(float& v:down)v=rounded_prefill(v,dt);
    double error=0,maximum=0;
    for(int row:{0,rows/2,rows-1}){std::vector<double> expected(n,0);Vec x(n);for(int i=0;i<n;++i)x[i]=rounded_prefill(std::sin(float(size_t(row)*n+i)*.017f)*.4f,gu);
        for(int k=0;k<f.g.used;++k){const int expert=(row+k)%f.g.experts;Vec h(n);
            for(int j=0;j<n;++j){double g=0,u=0;for(int i=0;i<n;++i){const size_t at=(size_t(expert)*n+j)*n+i;g+=double(gate[at])*x[i];u+=double(up[at])*x[i];}
                const float clipped=std::min(float(g),10.f),clipped_up=std::clamp(float(u),-10.f,10.f);h[j]=rounded_prefill(clipped/(1.f+std::exp(-clipped))*clipped_up,dt);}
            for(int j:{0,1,std::min(17,n-1),n/2,n-1}){double value=0;for(int i=0;i<n;++i)value+=double(down[(size_t(expert)*n+j)*n+i])*h[i];expected[j]+=value*(k?.75:.25);}
        }
        for(int j:{0,1,std::min(17,n-1),n/2,n-1}){maximum=std::max(maximum,std::abs(expected[j]));error=std::max(error,std::abs(double(actual[size_t(row)*n+j])-expected[j]));}
    }
    std::cout<<"prefill independent CPU experts width="<<n<<" rows="<<rows<<" max_abs="<<error<<" normalized="<<error/(1+maximum)<<'\n';
    // Rounding is explicitly represented in the oracle. This covers FP32 dot
    // reduction/SiLU error propagated through the second projection, not routing.
    require(error<=2e-4*(1+maximum),"prefill differs from independently decoded CPU expert reference");
}
void quant_case(int device, int width, int down_type, int rows) {
    Fixture f(width, down_type);
    ExpertExecutor streaming(f.g, f.lookup(), {{device, 0}}, std::max(8,rows), 2);
    const auto expected = execute(f, streaming, device, rows);
    wide_cpu_reference(f,expected,rows);
    require(streaming.stats().cache_bytes == 0, "zero budget allocated a resident cache");
    require(streaming.reserved_bytes(device) ==
                ExpertExecutor::memory_requirements(f.g, f.lookup(),std::max(8,rows)).device_workspace,
            "workspace accounting");
    ExpertExecutor lru(f.g, f.lookup(), {{device, f.per_expert}}, std::max(8,rows), 2);
    same(expected, execute(f, lru, device, rows), "LRU eviction/streaming");
    ExpertExecutor cached(f.g, f.lookup(), {{device, f.per_expert * f.g.experts}}, std::max(8,rows), 2);
    same(expected, execute(f, cached, device, rows), "cold cache/native");
    same(expected, execute(f, cached, device, rows), "warm cache/native");
    require(cached.preload_layer(0, device), "full layer preload");
    const auto before = cached.stats();
    same(expected, execute(f, cached, device, rows), "resident routed/native");
    const auto after = cached.stats();
    require(after.metadata_readbacks == before.metadata_readbacks, "resident path read route IDs to host");
    require(after.resident_calls == before.resident_calls + 1, "resident path not used");
    require(after.hits + after.misses == after.routed_entries, "cache accounting units");
    require(after.cache_bytes <= after.cache_capacity, "cache budget exceeded");
}
void float_clamp_case(int device) {
    Fixture f(32, 0, true);
    ExpertExecutor executor(f.g, f.lookup(), {{device, 0}}, 8, 2);
    const auto out = execute(f, executor, device, 2, true);
    const float expected = (10.0f / (1.0f + std::exp(-10.0f))) * 10.0f * 32.0f * .1f;
    for (float value : out)
        require(std::abs(value - expected) < .002f, "gate must be clipped before SiLU, up symmetrically");
}
void floating_resident_case(int device, int type, int width, int rows) {
    Fixture f(width, 0, true, type);
    ExpertExecutor streaming(f.g, f.lookup(), {{device, 0}}, std::max(8,rows), 2);
    const auto expected = execute(f, streaming, device, rows);
    wide_cpu_reference(f,expected,rows);
    ExpertExecutor resident(f.g, f.lookup(), {{device, f.per_expert * f.g.experts}}, std::max(8,rows), 2);
    require(resident.preload_layer(0, device), "floating full layer preload");
    const auto before = resident.stats();
    same(expected, execute(f, resident, device, rows), "floating resident routed/native");
    require(resident.stats().metadata_readbacks == before.metadata_readbacks,
            "floating resident route readback");
    require(resident.stats().resident_calls == before.resident_calls + 1,
            "floating resident fast path not used");
    require(resident.reserved_bytes(device) ==
                ExpertExecutor::memory_requirements(f.g, f.lookup(),std::max(8,rows)).device_workspace +
                    f.per_expert * f.g.experts,
            "floating workspace accounting");
}
void mixed_arena_case(int device){
    Fixture q4(1536,12),q6(1536,14),bf(1536,0,true,30);Fixture* variants[]{&q4,&q6,&bf};
    Geometry g=q4.g;g.layers=3;g.clamp={10,10,10};std::map<std::string,strata::TensorInfo> infos;std::map<std::string,Tensor> tensors;
    for(int layer=0;layer<3;++layer)for(const auto& [old,t]:variants[layer]->tensors){const auto name="blk."+std::to_string(layer)+"."+old.substr(6);infos[name]=*t.info;infos[name].name=name;tensors[name]=Tensor{&infos[name],t.data};}
    const auto lookup=[&](const std::string& name)->const Tensor&{return tensors.at(name);};
    ExpertExecutor baseline(g,lookup,{{device,0}},31,2);std::vector<std::vector<float>> expected;
    for(int layer=0;layer<3;++layer)expected.push_back(execute(*variants[layer],baseline,device,31,false,layer));
    size_t before,total;ck(cudaMemGetInfo(&before,&total));const size_t cap=2*bf.per_expert+q4.per_expert+256;
    ExpertExecutor cached(g,lookup,{{device,cap}},31,2);size_t initialized;ck(cudaMemGetInfo(&initialized,&total));
    const auto memory=cached.stats();
    require(memory.arena_bytes==cap,"arena reservation is not its full configured capacity");
    require(cached.reserved_bytes(device)==ExpertExecutor::memory_requirements(g,lookup,31).device_workspace+cap,"reserved bytes hid unoccupied arena space");
    require(memory.observed_driver_bytes==before-initialized,"quiescent driver allocation delta accounting");
    const size_t reserved=cached.reserved_bytes(device);
    for(int repeat=0;repeat<3;++repeat)for(int layer:{0,2,1,2,0})same(expected[layer],execute(*variants[layer],cached,device,31,false,layer),"mixed-format arena LRU/coalescing");
    require(cached.preload_layer(0,device),"mixed arena pinned Q4 layer did not fit");
    for(int layer:{2,1,2,0})same(expected[layer],execute(*variants[layer],cached,device,31,false,layer),"mixed arena overlap with pinned prefix");
    same(execute(q4,baseline,device,8,false,0),execute(q4,cached,device,8,false,0),"narrow decode after mixed arena prefill");
    size_t after;ck(cudaMemGetInfo(&after,&total));
    require(cached.reserved_bytes(device)==reserved,"runtime cache grew its arena");
    require(cached.stats().cache_bytes<=cap&&cached.stats().evictions>0,"arena charge or eviction accounting");
    // No per-entry device allocation remains; allow one lazy driver page but not
    // the old2MiB-per-expert growth during cache population and eviction.
    require(after>=initialized||initialized-after<=size_t(2)<<20,"cache entries allocated device storage outside arena");
    std::cout<<"mixed arena requested="<<cap<<" observed_driver="<<memory.observed_driver_bytes<<" fixed_requested="<<reserved-cap<<" runtime_growth="<<(initialized>after?initialized-after:0)<<'\n';
}
void peer_case(const std::vector<int> &devices) {
    Fixture f(256, 14);
    ExpertExecutor baseline(f.g, f.lookup(), {{devices[0], 0}}, 8, 2);
    const auto expected = execute(f, baseline, devices[0], 8);
    ExpertExecutor peer(f.g, f.lookup(), {{devices[0], 0}, {devices[1], f.per_expert * f.g.experts}}, 8, 2);
    same(expected, execute(f, peer, devices[0], 8), "two-card streamed experts");
    require(peer.preload_layer(0, devices[1]), "remote layer preload");
    const auto before = peer.stats();
    same(expected, execute(f, peer, devices[0], 8), "remote resident routed experts");
    require(peer.stats().metadata_readbacks == before.metadata_readbacks, "remote resident route readback");
}
} // namespace
int main(int argc, char **argv) {
    try {
        std::vector<int> devices{0};
        if (argc == 3 && std::string(argv[1]) == "--devices") {
            devices.clear();
            std::istringstream in(argv[2]);
            std::string part;
            while (std::getline(in, part, ','))
                devices.push_back(std::stoi(part));
        }
        require(!devices.empty(), "no test device");
        strata::kernels::native_mmvq_set_multi_exact(true);
        for (int rows : {1, 8}) {
            quant_case(devices[0], 256, 12, rows);
            quant_case(devices[0], 1536, 14, rows);
        }
        for(int rows:{9,15,16,31,128,256,511}){quant_case(devices[0],256,12,rows);quant_case(devices[0],256,14,rows);}
        float_clamp_case(devices[0]);
        for (int type : {0, 1, 30})
            for (int rows : {1, 8})
                floating_resident_case(devices[0], type, 256, rows);
        floating_resident_case(devices[0], 30, 1536, 8);
        floating_resident_case(devices[0],30,1536,128);
        quant_case(devices[0],1536,14,128);
        for(int type:{0,1,30})floating_resident_case(devices[0],type,256,31);
        mixed_arena_case(devices[0]);
        if (devices.size() > 1)
            peer_case(devices);
        else
            std::cout << "two-device peer execution NOT RUN (one selected device)\n";
        std::cout << "Hy4 CUDA expert execution passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
