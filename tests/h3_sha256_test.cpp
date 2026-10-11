#include "dit_models/models/minimax_h3_sha256.hpp"
#include <cassert>
int main(){
    MiniMaxH3::Sha256 a;assert(a.finish()=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    MiniMaxH3::Sha256 b;b.update("abc",3);assert(b.finish()=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    MiniMaxH3::Sha256 c;for(int i=0;i<1000000;++i)c.update("a",1);assert(c.finish()=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}
