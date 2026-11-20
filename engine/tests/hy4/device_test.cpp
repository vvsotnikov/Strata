#include "strata/hy4/device.hpp"
#include "strata/quant/fp16.hpp"
#include <map>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace strata::hy4;
namespace {
struct Row {
    int token = 0, slot = 0, next = -1;
    std::array<int32_t, 3> position{-1, -1, -1};
    Vec embedding;
    DeviceInput input() const { return {token, slot, position, next, embedding}; }
    RopePosition rope() const { return {position[0], position[1], position[2]}; }
};
Vec target(Engine &cpu, const Row &row) {
    if (!row.embedding.empty())
        return cpu.forward_embedding(row.embedding, row.rope(), row.next);
    if (row.position[0] < 0)
        return cpu.forward(row.token);
    return cpu.forward_at(row.token, row.rope(), row.next);
}
Vec draft(Engine &cpu, const Row &row, std::span<const float> previous) {
    if (!row.embedding.empty())
        return cpu.draft_forward_embedding(row.embedding, previous, row.rope(), row.next);
    return cpu.draft_forward(row.token, previous, row.rope(), row.next);
}
void compare(const Vec &actual, const Vec &expected, float &maximum, const std::string &stage) {
    if (actual.size() != expected.size())
        throw std::runtime_error(stage + " output width");
    float error = 0;
    for (size_t i = 0; i < actual.size(); ++i)
        error = std::max(error, std::abs(actual[i] - expected[i]));
    maximum = std::max(maximum, error);
    if (error > 1e-4f)
        throw std::runtime_error(stage + " max_abs=" + std::to_string(error));
}
void require(bool value, const char *text) {
    if (!value)
        throw std::runtime_error(text);
}
} // namespace

namespace {
// Independent scalar reference for the standard prefill FP16/BF16 conversion,
// with FP32 accumulation. The ordinary CPU engine separately measures drift
// from the original F32 path; no GPU prefill kernel is reused in this oracle.
class RoundedMatmul final:public Matmul {
public:
    bool enabled=true;
    std::map<std::string,Vec> cache;
    RoundedMatmul():Matmul(1){}
    static float round(float x,int type){
        if(type==30){auto bits=strata::quant::f32_to_bits(x);return strata::quant::f32_from_bits((bits+0x7fff+((bits>>16)&1))&0xffff0000u);}
        return strata::quant::fp16_to_fp32(strata::quant::fp32_to_fp16(x));
    }
    Vec linear(const Tensor& w,std::span<const float> x,int plane=0)override{
        if(!enabled||w.info->name=="output.weight")return Matmul::linear(w,x,plane);
        const int ni=w.info->shape[0],no=w.info->shape[1],type=w.info->type;
        auto at=cache.find(w.info->name);if(at==cache.end()){auto values=w.values();for(auto& v:values)v=round(v,type);at=cache.emplace(w.info->name,std::move(values)).first;}
        Vec input(x.begin(),x.end()),out(no);for(float& v:input)v=round(v,type);
        for(int r=0;r<no;++r){double sum=0;for(int k=0;k<ni;++k)sum+=double(at->second[(size_t(plane)*no+r)*ni+k])*input[k];out[r]=float(sum);}return out;
    }
};
void wide_compare(const Vec& actual,const Vec& expected,double tolerance,const char* stage){
    require(actual.size()==expected.size(),"wide oracle row width");double error=0,maximum=0;
    for(size_t i=0;i<actual.size();++i){require(std::isfinite(actual[i]),"nonfinite wide oracle value");error=std::max(error,std::abs(double(actual[i])-expected[i]));maximum=std::max(maximum,std::abs(double(expected[i])));}
    std::cout<<stage<<" max_abs="<<error<<" normalized="<<error/(1+maximum)<<'\n';
    require(error<=tolerance*(1+maximum),stage);
}
int wide_test(const char* path,bool peer){
    Engine weights(path,1,1024),rounded(path,1,1024),f32(path,1,1024);const auto& g=weights.geometry();
    auto backend=std::make_unique<RoundedMatmul>();auto* convert=backend.get();rounded.set_matmul(std::move(backend));
    DeviceOptions o;o.devices=peer?std::vector<int>{0,1}:std::vector<int>{0};o.dense_bytes.assign(o.devices.size(),16ull<<20);o.expert_bytes.assign(o.devices.size(),0);o.vram_reserve=64ull<<20;
    o.slots=4;o.max_context=1024;o.context_pool=2049;o.prefill_rows=512;o.f32_kv=true;
    auto model=device_model(weights,o);
    for(int count:{8,9,15,16,31,128,256,511}){
        model->reset(0);rounded.reset();f32.reset();convert->enabled=count>8;
        std::vector<DeviceInput> input;for(int i=0;i<count;++i){DeviceInput row;row.token=1+i%(g.vocab-1);input.push_back(row);}
        Vec gpu_hidden;
        const auto traced=model->forward(input,[&](const std::string& stage,int,std::span<const float> values){if(stage=="result_norm")gpu_hidden.assign(values.begin(),values.end());});
        require(gpu_hidden.size()==size_t(count)*g.dim,"missing GPU trunk features for independent MTP oracle");
        model->reset(0);const auto actual=model->prefill(input,model->has_mtp());
        wide_compare(actual,traced.back(),1e-6,"optimized last-row prefill equals all-row target forward");
        Vec expected,original;
        for(int i=0;i<count;++i){expected=rounded.forward(input[i].token);original=f32.forward(input[i].token);
            if(model->has_mtp()){const Vec zero(g.dim,0);const auto previous=i?std::span<const float>(gpu_hidden).subspan(size_t(i-1)*g.dim,g.dim):std::span<const float>(zero);rounded.draft_forward(input[i].token,previous);}}
        // Trunk compares independently rounded/F32 CPU inputs. MTP below instead
        // uses the same actual target features, isolating its cache/shift/math
        // from a separate accumulation error in the CPU trunk approximation.
        wide_compare(actual,expected,1e-4,"wide rounded CPU reference");
        wide_compare(actual,original,2e-3,"wide F32 conversion drift");
        require(model->length(0)==size_t(count),"wide target length");
        if(model->has_mtp()){
            require(model->draft_length(0)==size_t(count),"canonical wide MTP length");
            model->draft_accept(0,count-1);convert->enabled=false;DeviceInput next;next.token=7;
            auto draft=model->draft_forward(std::array{next},false);
            wide_compare(draft[0],rounded.draft_forward(next.token,std::span<const float>(gpu_hidden).last(g.dim)),1e-4,"recursive draft after wide canonical cache");
        }
        std::cout<<"prefill rows="<<count<<" passed\n";
    }
    // Cross-chunk target h_(p-1) must survive a wide target forward before its
    // canonical draft ingestion; a fixed32 feature ring cannot pass this case.
    model->reset(0);rounded.reset();convert->enabled=true;Vec previous_gpu(g.dim,0);
    std::vector<std::vector<DeviceInput>> chunks;std::vector<Vec> expected_chunks;bool cross_chunk_gate=true;
    for(int count:{31,65}){std::vector<DeviceInput> input;const int initial=model->length(0);for(int i=0;i<count;++i){DeviceInput row;row.token=1+(initial+i)%(g.vocab-1);input.push_back(row);}
        std::map<std::pair<int,std::string>,Vec> gpu_trace,cpu_trace;
        Vec hidden;const auto output=model->forward(input,[&](const std::string& stage,int layer,std::span<const float> values){if(stage=="result_norm")hidden.assign(values.begin(),values.end());auto& out=gpu_trace[{layer,stage}];if(stage=="attn_selected")out.insert(out.end(),values.begin(),values.end());else out.assign(values.begin(),values.end());});
        require(hidden.size()==size_t(count)*g.dim,"missing cross-chunk target features");
        Vec expected;for(int i=0;i<count;++i){expected=rounded.forward(input[i].token,[&](const std::string& stage,int layer,std::span<const float> values){auto& out=cpu_trace[{layer,stage}];out.insert(out.end(),values.begin(),values.end());});if(model->has_mtp())rounded.draft_forward(input[i].token,i?std::span<const float>(hidden).subspan(size_t(i-1)*g.dim,g.dim):std::span<const float>(previous_gpu));}
        for(const auto& [key,actual]:gpu_trace){auto at=cpu_trace.find(key);if(at==cpu_trace.end()||at->second.size()!=actual.size())continue;double e=0,m=0;size_t different=0;for(size_t i=0;i<actual.size();++i){e=std::max(e,std::abs(double(actual[i])-at->second[i]));m=std::max(m,std::abs(double(at->second[i])));different+=actual[i]!=at->second[i];}std::cout<<"chunk="<<count<<" layer="<<key.first<<" stage="<<key.second<<" normalized_error="<<e/(1+m)<<" max_abs="<<e<<" different="<<different;
            if(key.second=="attn_projection_input"){size_t rounding=0;double max_rounding=0;for(size_t i=0;i<actual.size();++i){const float a=RoundedMatmul::round(actual[i],0),b=RoundedMatmul::round(at->second[i],0);rounding+=a!=b;max_rounding=std::max(max_rounding,std::abs(double(a)-b));}std::cout<<" fp16_input_disagreements="<<rounding<<" fp16_max_abs="<<max_rounding;}
            std::cout<<'\n';}
        // FP16 conversion is discontinuous: an F32 accumulation difference can
        // straddle a midpoint, then propagate a full FP16 quantum through later
        // layers/chunks. Prove the output projection itself independently on the
        // identical captured GPU operands instead of requiring two differently
        // accumulated end-to-end paths to remain below an arbitrary1e-4.
        for(int layer=0;layer<g.layers;++layer){
            const auto& operand=gpu_trace.at({layer,"attn_projection_input"});
            const auto& gpu_output=gpu_trace.at({layer,"attn_out"});
            const auto& cpu_selected=cpu_trace.at({layer,"attn_selected"});
            require(gpu_trace.at({layer,"attn_selected"})==cpu_selected,"wide CPU/GPU selected indices differ");
            const std::string stem="blk."+std::to_string(layer)+".";
            const auto& w=weights.tensor(stem+"attn_output.weight");const size_t width=w.info->shape[0];Vec oracle;
            const auto scale=weights.tensor(stem+"post_attention_norm.weight").values();
            for(int row=0;row<count;++row){auto result=convert->linear(w,std::span<const float>(operand).subspan(size_t(row)*width,width));rms(result,g.eps,scale);oracle.insert(oracle.end(),result.begin(),result.end());}
            wide_compare(gpu_output,oracle,1e-5,"same-input independent rounded attention projection");
        }
        previous_gpu.assign(hidden.end()-g.dim,hidden.end());
        // One FP16 relative quantum (2^-11) is an explicit empirical regression
        // tolerance for this multi-layer synthetic path, not an arithmetic bound
        // or a replacement for full-model FP32 quality qualification.
        try{wide_compare(output.back(),expected,std::ldexp(1.0,-11),"wide shifted-feature rounded target (FP16 quantum gate)");}catch(const std::runtime_error&){cross_chunk_gate=false;}
        if(model->has_mtp())model->draft_forward(input,true);
        chunks.push_back(input);expected_chunks.push_back(output.back());
    }
    model->reset(0);
    for(size_t i=0;i<chunks.size();++i)wide_compare(model->prefill(chunks[i],model->has_mtp()),expected_chunks[i],1e-6,"optimized shifted-feature chunk boundary");
    require(!model->has_mtp()||model->draft_length(0)==96,"wide cross-chunk MTP cache length");
    if(model->has_mtp()){model->draft_accept(0,95);convert->enabled=false;DeviceInput next;next.token=8;
        wide_compare(model->draft_forward(std::array{next},false)[0],rounded.draft_forward(next.token,previous_gpu),1e-4,"recursive draft after two wide canonical chunks");}
    model.reset();
    // Restore after enough generation to overwrite SWA, raw-pool and feature
    // rings. Both cache precisions must replay bit-for-bit and preserve peers.
    for(bool f32_kv:{true,false})for(bool host_swa:{true,false}){
        o.prefill_rows=32;o.f32_kv=f32_kv;o.host_swa_kv=host_swa;model=device_model(weights,o);
        std::vector<Row> prefix(31);std::vector<DeviceInput> inputs;
        for(size_t i=0;i<prefix.size();++i){auto& row=prefix[i];row.token=1+i%(g.vocab-1);
            if(i>=2&&i<6){row.embedding=weights.tensor("token_embd.weight").values(30+i);for(float& v:row.embedding)v*=1.7f;row.token=-1;row.position={2,2+int(i-2)/2,2+int(i-2)%2};row.next=i==5?4:2;}}
        for(const auto& row:prefix)inputs.push_back(row.input());model->prefill(inputs,model->has_mtp());
        if(model->has_mtp())model->draft_accept(0,30);require(model->checkpoint(0),"native prompt checkpoint capture");
        std::vector<DeviceInput> peer_input(13,DeviceInput{3,1});model->prefill(peer_input,model->has_mtp());if(model->has_mtp())model->draft_accept(1,12);require(model->checkpoint(1),"peer checkpoint capture");
        const auto peer_probe=model->forward(std::array{DeviceInput{4,1}})[0];require(model->restore_checkpoint(1,13),"peer checkpoint restore");
        const size_t free_before=model->free_tokens();
        auto continuation=[&](){std::vector<Vec> result;
            if(model->has_mtp()){result=model->draft_forward(std::array{DeviceInput{7,0}},false);model->draft_truncate(0,31);model->draft_accept(0,30);}
            for(int i=0;i<64;++i){DeviceInput row{int32_t(7+i%(g.vocab-8)),0};const auto target=model->forward(std::array{row});result.push_back(target[0]);
                if(model->has_mtp()){auto draft=model->draft_forward(std::array{row},true);result.push_back(draft[0]);model->draft_accept(0,model->length(0)-1);}}
            return result;};
        const auto first=continuation();require(model->restore_checkpoint(0,31),"long generation checkpoint restore");
        require(model->length(0)==31&&model->length(1)==13&&model->free_tokens()==free_before,"checkpoint corrupted logical pool or peer slot");
        const auto peer_again=model->forward(std::array{DeviceInput{4,1}})[0];require(peer_probe==peer_again,"checkpoint restore touched another slot KV");require(model->restore_checkpoint(1,13),"second peer checkpoint restore");
        const auto second=continuation();require(first==second,"checkpoint target/draft replay was not bit-identical");
        model->truncate(0,30);require(!model->restore_checkpoint(0,31),"rewind freed a still-restorable checkpoint prefix");
        model->reset(1);require(!model->restore_checkpoint(1,13),"reset left a stale checkpoint");
        std::cout<<"native prompt checkpoint exact replay f32_kv="<<f32_kv<<" host_swa="<<host_swa<<'\n';model.reset();
    }
    if(g.draft_layers){
        // A target feature can establish valid recursive carry before the draft
        // has any cache rows. Restoring must recover that carry independently of
        // the draft length, including after a future draft changes it.
        o.prefill_rows=32;model=device_model(weights,o);
        std::vector<DeviceInput> peer_input(13,DeviceInput{3,1});
        model->prefill(peer_input,true);model->draft_accept(1,12);
        require(model->checkpoint(1),"carry-edge peer checkpoint capture");
        const auto peer_future=model->forward(std::array{DeviceInput{4,1}})[0];
        require(model->restore_checkpoint(1,13),"carry-edge peer checkpoint baseline restore");
        model->forward(std::array{DeviceInput{7,0}});
        require(model->draft_length(0)==0,"empty draft baseline unexpectedly populated");
        model->draft_accept(0,0);require(model->checkpoint(0),"empty draft valid-carry checkpoint");
        const auto future=model->draft_forward(std::array{DeviceInput{9,0}},false);
        require(model->restore_checkpoint(0,1),"empty draft valid-carry checkpoint restore");
        require(model->draft_length(0)==0,"empty draft checkpoint restored wrong length");
        require(model->draft_forward(std::array{DeviceInput{9,0}},false)==future,
                "empty draft checkpoint failed to restore recursive carry exactly");

        // Requested destination30 is above the saved20-row prefix, but a
        // 50->30 rewind exceeds batch8 and really frees the entire draft cache.
        // Regrowing identical IDs reacquires the same LIFO pages. A stale
        // checkpoint would pass token/block equality despite their new contents.
        std::vector<DeviceInput> prefix(20,DeviceInput{3,2});model->prefill(prefix,true);
        model->draft_accept(2,19);require(model->checkpoint(2),"draft rewind ABA checkpoint");
        std::vector<DeviceInput> extension(30,DeviceInput{3,2});model->prefill(extension,true);
        require(model->draft_length(2)==50,"draft rewind ABA initial length");
        model->draft_truncate(2,30);require(model->draft_length(2)==0,"large draft rewind did not reset");
        model->draft_accept(2,49);
        for(int i=0;i<20;++i)model->draft_forward(std::array{DeviceInput{3,2}},false);
        require(model->draft_length(2)==20,"draft rewind ABA regrown length");
        require(!model->restore_checkpoint(2,20),"released draft pages revived a stale checkpoint");
        require(model->restore_checkpoint(1,13),"draft edge invalidation touched peer checkpoint");
        require(model->forward(std::array{DeviceInput{4,1}})[0]==peer_future,
                "draft edge restore changed peer-slot KV");
        require(model->restore_checkpoint(1,13),"draft edge peer checkpoint second restore");
        std::cout<<"Empty-draft carry and forced-zero draft rewind checkpoint edges passed\n";
        model.reset();
    }
    // Odd logical capacities and four partial tails consume exactly the requested
    // pool; padding physical cells must never appear as extra logical capacity.
    o.prefill_rows=32;o.context_pool=17;model=device_model(weights,o);
    std::vector<DeviceInput> input;
    for(int slot=0;slot<4;++slot){DeviceInput row;row.token=1;row.slot=slot;input.push_back(row);}
    model->prefill(input,model->has_mtp());require(model->free_tokens()==13,"partial tails disappeared from logical capacity");
    input.assign(13,DeviceInput{2,0});model->prefill(input,model->has_mtp());require(model->free_tokens()==0,"odd logical pool not fully usable");
    const auto before=model->history(0).size();bool failed=false;try{model->forward(std::array{DeviceInput{3,0}});}catch(const std::runtime_error&){failed=true;}
    require(failed&&model->length(0)==before&&model->healthy(),"logical overcommit mutated state");
    model->reset(1);require(model->free_tokens()==1,"partial-tail reset logical accounting");
    model->prefill(std::array{DeviceInput{3,0}},model->has_mtp());require(model->free_tokens()==0,"reclaimed logical token unusable");
    for(int slot=0;slot<4;++slot)model->reset(slot);require(model->free_tokens()==17,"logical pool leak after all resets");
    std::cout<<"Wide native prefill + odd unified pool passed "<<model->stats()<<'\n';require(cross_chunk_gate,"independent cross-chunk rounded reference gate");return 0;
}
}

int main(int argc, char **argv) {
    try {
        if (argc < 2)
            throw std::runtime_error("hy4_device_test TINY_MODEL [--actual|--peer]");
        bool actual = false, peer = false, wide=false;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--actual") actual = true;
            else if (option == "--peer") peer = true;
            else if(option=="--wide")wide=true;
            else throw std::runtime_error("unknown device test option " + option);
        }
        require(!actual || !peer, "peer mode is restricted to the tiny fixture");
        if(wide){require(!actual,"wide test requires the1024-context synthetic fixture");return wide_test(argv[1],peer);}
        Engine source(argv[1], 4, actual ? 8192 : 64);
        const auto &g = source.geometry();
        DeviceOptions options;
        // Ordinals within the explicitly selected CUDA_VISIBLE_DEVICES UUIDs.
        options.devices = peer ? std::vector<int>{0, 1} : std::vector<int>{0};
        options.dense_bytes.assign(options.devices.size(), actual ? 128ull << 20 : 16ull << 20);
        options.expert_bytes.assign(options.devices.size(), actual ? 32ull << 20 : 0);
        if (peer) options.vram_reserve = 64ull << 20;
        options.slots = 4;
        options.max_context = actual ? 8192 : 64;
        options.context_pool = actual ? 8192 : 128;
        options.f32_kv = !actual;
        auto model = device_model(source, options);
        std::vector<std::unique_ptr<Engine>> refs;
        for (int slot = 0; slot < options.slots && !actual; ++slot)
            refs.push_back(std::make_unique<Engine>(argv[1], 2, 64));
        float maximum = 0;
        std::vector<Vec> all;
        for (int phase = 0; phase < (actual ? 2 : 4); ++phase) {
            std::vector<DeviceInput> input;
            for (int row = 0; row < (actual ? 1 : 8); ++row) {
                DeviceInput in;
                in.slot = actual ? 0 : phase % 2;
                in.token = 1 + phase * (actual ? 1 : 8) + row;
                input.push_back(in);
            }
            const auto output = model->forward(input);
            for (size_t row = 0; row < input.size(); ++row) {
                if (!actual)
                    compare(output[row], refs[input[row].slot]->forward(input[row].token), maximum,
                            "batched trunk");
                all.push_back(output[row]);
            }
        }
        if (actual) {
            std::ofstream output("device-actual-logits.f32", std::ios::binary);
            for (const auto &row : all) {
                output.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
                std::cout << "top=" << std::max_element(row.begin(), row.end()) - row.begin() << '\n';
            }
        } else {
            const size_t before = model->length(0), free = model->free_tokens();
            DeviceInput invalid[2];
            invalid[0].token = invalid[1].token = 1;
            invalid[1].rope = {0, -1, 0};
            bool rejected = false;
            try {
                model->forward(invalid);
            } catch (const std::exception &) {
                rejected = true;
            }
            require(rejected && model->length(0) == before && model->free_tokens() == free &&
                        model->healthy(),
                    "position validation mutated state");
            for (int slot = 0; slot < options.slots; ++slot)
                model->reset(slot);
            require(model->free_tokens() == size_t(options.context_pool),
                    "unified pool leak after baseline reset");
            if (model->has_mtp()) {
                auto ingested = device_model(source, options);
                std::vector<std::vector<Row>> history(options.slots);
                std::vector<std::vector<Vec>> features(options.slots);
                Vec zero(g.dim, 0);
                auto rejects_stale_carry = [&](int slot) {
                    DeviceInput in;
                    in.slot = slot;
                    in.token = 1;
                    const size_t before_draft = model->draft_length(slot);
                    bool rejected_carry = false;
                    try {
                        model->draft_forward(std::array{in}, false);
                    } catch (const std::runtime_error &error) {
                        rejected_carry =
                            std::string(error.what()).find("carry is unavailable") != std::string::npos;
                    }
                    require(rejected_carry && model->healthy() && model->draft_length(slot) == before_draft,
                            "recursive MTP consumed stale or uninitialized carry");
                };
                rejects_stale_carry(0);
                auto make = [&](int slot, int position, bool image) {
                    Row row;
                    row.slot = slot;
                    row.token = 1 + (position + slot * 17) % (g.vocab - 1);
                    int p = position < 6 ? position : position - 2;
                    row.position = {p, p, p};
                    row.next = p + 1;
                    if (image && position >= 2 && position < 6) {
                        row.embedding = source.tensor("token_embd.weight").values(30 + position);
                        for (float &value : row.embedding)
                            value *= 1.7f;
                        row.token = -1;
                        row.position = {2, 2 + (position - 2) / 2, 2 + (position - 2) % 2};
                        row.next = position == 5 ? 4 : 2;
                    }
                    return row;
                };
                auto submit_target = [&](const std::vector<Row> &rows) {
                    std::vector<DeviceInput> input;
                    for (const auto &row : rows)
                        input.push_back(row.input());
                    const auto output = model->forward(input);
                    require(output == ingested->forward(input), "canonical ingestion peer changed target logits");
                    for (size_t i = 0; i < rows.size(); ++i) {
                        const auto &row = rows[i];
                        compare(output[i], target(*refs[row.slot], row), maximum, "image/MRoPE trunk");
                        history[row.slot].push_back(row);
                        features[row.slot].push_back(refs[row.slot]->last_hidden());
                    }
                };
                auto canonical_draft = [&](const std::vector<Row> &rows) {
                    std::vector<DeviceInput> input;
                    for (const auto &row : rows)
                        input.push_back(row.input());
                    std::vector<size_t> position(options.slots);
                    for (int slot = 0; slot < options.slots; ++slot)
                        position[slot] = model->draft_length(slot);
                    const auto output = model->draft_forward(input, true);
                    ingested->draft_ingest(input);
                    for (int slot=0;slot<options.slots;++slot) require(model->draft_length(slot)==ingested->draft_length(slot), "headless canonical ingestion cache length");
                    for (size_t i = 0; i < rows.size(); ++i) {
                        const auto &row = rows[i];
                        const size_t p = position[row.slot]++;
                        compare(output[i], draft(*refs[row.slot], row, p ? features[row.slot][p - 1] : zero),
                                maximum, "canonical MTP/image");
                    }
                };
                auto rebuild = [&](int slot) {
                    refs[slot] = std::make_unique<Engine>(argv[1], 2, 64);
                    features[slot].clear();
                    for (const auto &row : history[slot]) {
                        target(*refs[slot], row);
                        features[slot].push_back(refs[slot]->last_hidden());
                    }
                };
                for (int slot = 0; slot < options.slots; ++slot) {
                    refs[slot]->reset();
                    for (int first = 0; first < 12; first += 8) {
                        std::vector<Row> rows;
                        for (int p = first; p < std::min(first + 8, 12); ++p)
                            rows.push_back(make(slot, p, true));
                        submit_target(rows);
                        canonical_draft(rows);
                    }
                }
                // Recursive drafts use accepted target features first, then the previous
                // draft's own post-norm feature; distinct slots execute together.
                for (int step = 0; step < 2; ++step) {
                    std::vector<Row> rows;
                    std::vector<Vec> previous;
                    for (int slot = 0; slot < options.slots; ++slot) {
                        if (!step) {
                            model->draft_accept(slot, 11);
                            ingested->draft_accept(slot, 11);
                        }
                        previous.push_back(step ? refs[slot]->last_draft_hidden() : features[slot][11]);
                        rows.push_back(make(slot, 12 + step, false));
                    }
                    std::vector<DeviceInput> input;
                    for (const auto &row : rows)
                        input.push_back(row.input());
                    const auto output = model->draft_forward(input, false);
                    require(output == ingested->draft_forward(input,false), "headless canonical cache changed recursive logits");
                    for (size_t i = 0; i < rows.size(); ++i)
                        compare(output[i], draft(*refs[i], rows[i], previous[i]), maximum, "recursive MTP");
                }
                // Verify multiple candidates, retain one, discard two, and replay the
                // canonical draft through both a completed pooling group and SWA wrap.
                for (int slot = 0; slot < options.slots; ++slot) {
                    std::vector<Row> verify;
                    for (int p = 12; p < 15; ++p)
                        verify.push_back(make(slot, p, false));
                    submit_target(verify);
                    model->truncate(slot, 13);
                    ingested->truncate(slot, 13);
                    // A feature cached for a discarded target position cannot authorize
                    // canonical MTP execution beyond the new target sequence length.
                    DeviceInput stale;
                    stale.slot = slot;
                    stale.token = 1;
                    bool rejected_feature = false;
                    try {
                        model->draft_forward(std::array{stale}, true);
                    } catch (const std::runtime_error &error) {
                        rejected_feature = std::string(error.what()).find("outside the retained window") !=
                                           std::string::npos;
                    }
                    require(rejected_feature && model->draft_length(slot) == 14 && model->healthy(),
                            "canonical MTP used a discarded target feature");
                    model->draft_truncate(slot, 12);
                    ingested->draft_truncate(slot, 12);
                    rejects_stale_carry(slot);
                    history[slot].resize(13);
                    rebuild(slot);
                    for (int p = 0; p < 12; ++p)
                        draft(*refs[slot], history[slot][p], p ? features[slot][p - 1] : zero);
                    canonical_draft({history[slot][12]});
                    model->draft_accept(slot, 12);
                    ingested->draft_accept(slot, 12);
                    Row continuation = make(slot, 13, false);
                    continuation.token = (continuation.token + 7) % g.vocab;
                    const auto output = model->draft_forward(std::array{continuation.input()}, false);
                    require(output == ingested->draft_forward(std::array{continuation.input()},false), "headless canonical rejection replay changed next draft logits");
                    compare(output[0], draft(*refs[slot], continuation, features[slot][12]), maximum,
                            "MTP after rejection");
                    submit_target({continuation});
                    require(model->length(slot) == 14 && model->draft_length(slot) == 14,
                            "rejection replay length");
                }
                // Reuse every pool slot in a new order; no KV, feature or Engram state
                // from the old image-bearing sessions may affect the replacement.
                for (int slot : {2, 0, 3, 1}) {
                    model->reset(slot);
                    ingested->reset(slot);
                    history[slot].clear();
                    features[slot].clear();
                    refs[slot]->reset();
                    require(model->draft_length(slot) == 0, "slot reset leaked draft history");
                    rejects_stale_carry(slot);
                }
                require(model->free_tokens() == size_t(options.context_pool),
                        "slot reset leaked unified blocks");
                for (int p = 0; p < 8; ++p) {
                    std::vector<Row> rows;
                    for (int slot : {3, 1, 0, 2})
                        rows.push_back(make(slot, p, false));
                    submit_target(rows);
                    canonical_draft(rows);
                }
                std::cout << "GPU MTP/image, recursive features, rejection replay, headless canonical ingestion and slot reuse passed\n";
            }
            for (int slot = 0; slot < options.slots; ++slot)
                model->reset(slot);
            require(model->free_tokens() == size_t(options.context_pool), "final unified pool leak");
        }
        std::cout << "GPU batch/reference devices=" << options.devices.size() << " max_abs=" << maximum
                  << " stats=" << model->stats() << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
