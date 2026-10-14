// Test-only route injection: copied CPU backend named CUDA0, still using ONLY
// CPU device callbacks/buffers. Exercises production merge branch numerically;
// does not register CUDA and does not claim GPU support or performance.
#define H3_CACHE_OFFLOAD_NO_MAIN
#include "h3_cuda_cache_offload_test.cpp"
#include "ggml/src/ggml-backend-impl.h"
int main(){try {
 auto* cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
 ggml_backend mock=*cpu;mock.iface.get_name=[](ggml_backend_t){return "CUDA0-cpu-routing-test-NOT-GPU";};
 {OffloadHarness r(&mock,false);r.execution.cpu_test_only=false;
 sd::Tensor<float> x({2,2,1,2},{.2f,-.3f,.1f,.4f,.5f,-.2f,0.f,0.f});
 sd::Tensor<float> context({8,3}),time({1},{800.f});for(int64_t i=0;i<context.numel();++i)context[i]=.03f*float(i+1);
 sd::Tensor<int32_t> tags({3},{1,2,0}),keyframes({1},{0});
 std::vector<sd::Tensor<float>> videos,audios;
 videos.emplace_back(std::vector<int64_t>{2,2,1,1,1},std::vector<float>{.3f,.1f,-.2f,.4f});
 audios.emplace_back(std::vector<int64_t>{1,2,1,1},std::vector<float>{.2f,-.1f});
 std::vector<MiniMaxH3ReferenceBlock> refs={{MiniMaxH3ReferenceKind::VIDEO_AUDIO,0,0}};
 DiffusionParams p;p.x=&x;p.context=&context;p.timesteps=&time;p.minimax_audio_length=1;
 for(bool mixed:{false,true}) {
  p.ref_latents=mixed?&videos:nullptr;p.minimax_reference_audio_latents=mixed?&audios:nullptr;
  p.minimax_reference_blocks=mixed?&refs:nullptr;p.minimax_text_token_tags=mixed?&tags:nullptr;p.minimax_keyframe_indices=mixed?&keyframes:nullptr;
  for(int branch=0;branch<2;++branch)for(float sigma:{800.f,500.f,200.f}){
   r.reset();time[0]=sigma;context[0]=branch?.3f:-.3f;
   auto expected=r.baseline(p);MiniMaxH3::CacheOffload::Statistics miss,hit;
   auto a=r.cached(p,0,branch,&miss);compare_outputs(expected,a,false,"merged miss mismatch");
   auto b=r.cached(p,1,branch,&hit);compare_outputs(a,b,false,"merged hit mismatch");
   demand(hit.hit && !r.middle_transferred() && !r.middle_computed,"merged hit recomputed/transferred middle");
   demand(miss.probe_segments==1 && miss.remainder_segments==1,"greedy merge was not actually exercised");
  }
 }
 r.fail_after_segment=1;bool threw=false;
 try{r.cached(p,2,true);}catch(const std::exception& e){threw=std::string(e.what())=="injected segment exception";}
 demand(threw,"merged remainder failure not injected");r.restored();
 r.fail_after_segment=-1;MiniMaxH3::CacheOffload::Statistics recovery;r.cached(p,3,true,&recovery);demand(!recovery.hit,"merged failure anchor reused");
 }
 ggml_backend_free(cpu);std::cout<<"PASS CPU-only merge execution comparisons="<<compared_outputs<<" checks="<<offload_checks<<" segments="<<executed_segments<<"\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
