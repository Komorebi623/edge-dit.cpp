#include "dit_models/models/minimax_h3_vdn_reference.hpp"
#include <cassert>
#include <limits>
using namespace MiniMaxH3::VDNReference;
static bool close(float a,float b){return std::abs(a-b)<1e-5f;}
int main(){
    auto ab=frame_statistics({1,0,0,1},{2,3,4,5},{.5f,.5f},2);
    assert(close(ab.first[0],.5f)&&close(ab.first[3],.5f));
    auto f=factor(ab.first,ab.second,{1,1},2,DeltaRule::VdnSolve);
    auto state=apply_state(Matrix(4,0),f);assert(close(state[0],2.f/3)&&close(state[1],4.f/3));
    auto inv=inverse_spd({2,1,1,2},2);assert(close(inv[0],2.f/3)&&close(inv[1],-1.f/3));
    auto s=factor({.5f,0,0,.5f},{1,2,3,4},{1,1},2,DeltaRule::SanaScaled);assert(close(s.transition[0],.75f));
    std::vector<Factor> frames;for(int i=0;i<3;++i)frames.push_back(factor({0},{float(i+1)},{1},1,DeltaRule::VdnSolve));
    auto all=outside_window_states(frames,window_bounds(3,4),true);for(auto x:all)assert(x[0]==0); // no double counting
    auto local=outside_window_states(frames,window_bounds(3,0),true);assert(local[0][0]==5&&local[1][0]==4&&local[2][0]==3);
    assert(readout({2},{3},1)[0]==6);
    bool failed=false;try{inverse_spd({-1},1);}catch(const std::invalid_argument&){failed=true;}assert(failed);
    failed=false;try{frame_statistics({1},{2},{std::numeric_limits<float>::quiet_NaN()},1);}catch(const std::invalid_argument&){failed=true;}assert(failed);
}
