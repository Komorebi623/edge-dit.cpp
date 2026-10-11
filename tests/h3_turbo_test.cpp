#include "dit_models/models/minimax_h3_turbo.hpp"
#include <cassert>
#include <limits>
int main() {
    MiniMaxH3::TurboRecipe r; std::string error;
    assert(!r.valid(&error));
    r = {8,12.f,3.f,1.f,"euler","simple",true}; assert(r.valid(&error));
    r.matching_merged_weights=false; assert(!r.valid(&error));
    r.matching_merged_weights=true; r.steps=20; assert(!r.valid(&error));
    r.steps=4; r.video_shift=std::numeric_limits<float>::quiet_NaN(); assert(!r.valid(&error));
    r.video_shift=6.f; r.cfg_scale=2.f; assert(!r.valid(&error));
    r.cfg_scale=1.f; assert(r.valid(&error));
}
