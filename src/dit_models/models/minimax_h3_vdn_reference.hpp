#pragma once
// CPU mathematical oracle for OpenVDN's frame-level hybrid attention. These
// functions are test/reference code, NOT a CUDA deployment or trained adapter.
// Match the released vdn_solve recurrence (not a token-wise delta-rule surrogate).
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
namespace MiniMaxH3 { namespace VDNReference {
using Matrix=std::vector<float>; // row-major d x d
inline void require(bool ok,const char* text){if(!ok)throw std::invalid_argument(text);}
inline Matrix mm(const Matrix& a,const Matrix& b,size_t d){
    require(d>0&&a.size()==d*d&&b.size()==d*d,"VDN matrix shape");Matrix c(d*d,0);
    for(size_t i=0;i<d;++i)for(size_t j=0;j<d;++j)for(size_t k=0;k<d;++k)c[i*d+j]+=a[i*d+k]*b[k*d+j];
    return c;
}
inline Matrix inverse_spd(const Matrix& a,size_t d){
    require(d>0&&a.size()==d*d,"VDN SPD shape");Matrix l(d*d,0),inv(d*d,0);
    for(size_t i=0;i<d;++i)for(size_t j=0;j<=i;++j){
        float sum=a[i*d+j];require(std::isfinite(sum)&&std::abs(a[i*d+j]-a[j*d+i])<1e-4f,"VDN nonfinite or asymmetric SPD input");
        for(size_t k=0;k<j;++k)sum-=l[i*d+k]*l[j*d+k];
        if(i==j){require(sum>0,"VDN matrix not positive definite");l[i*d+j]=std::sqrt(sum);}else l[i*d+j]=sum/l[j*d+j];
    }
    for(size_t col=0;col<d;++col){std::vector<float> y(d),x(d);
        for(size_t i=0;i<d;++i){float sum=i==col?1.f:0.f;for(size_t k=0;k<i;++k)sum-=l[i*d+k]*y[k];y[i]=sum/l[i*d+i];}
        for(size_t p=d;p>0;--p){size_t i=p-1;float sum=y[i];for(size_t k=i+1;k<d;++k)sum-=l[k*d+i]*x[k];x[i]=sum/l[i*d+i];}
        for(size_t i=0;i<d;++i)inv[i*d+col]=x[i];
    }return inv;
}
struct Factor {Matrix transition,injection;std::vector<float> alpha;};
enum class DeltaRule {VdnSolve,SanaScaled,VdnScaled};
inline Factor factor(const Matrix& a,const Matrix& b,const std::vector<float>& alpha,size_t tokens,DeltaRule rule){
    size_t d=alpha.size();require(d&&tokens&&a.size()==d*d&&b.size()==d*d,"VDN factor shape");
    for(float v:alpha)require(std::isfinite(v)&&v>0&&v<=1,"VDN alpha outside (0,1]");
    for(float v:b)require(std::isfinite(v),"VDN nonfinite B");
    float c=rule==DeltaRule::VdnSolve?1.f:1.f/std::sqrt(float(tokens));
    Matrix system(d*d);for(size_t i=0;i<d;++i)for(size_t j=0;j<d;++j){
        require(std::isfinite(a[i*d+j]),"VDN nonfinite A");
        system[i*d+j]=(i==j?1.f:0.f)+(rule==DeltaRule::SanaScaled?-1.f:1.f)*c*c*a[i*d+j];
    }
    Matrix inv=rule==DeltaRule::SanaScaled?system:inverse_spd(system,d);
    Matrix transition=inv,injection;
    for(size_t i=0;i<d;++i)for(size_t j=0;j<d;++j)transition[i*d+j]*=alpha[i];
    if(rule==DeltaRule::SanaScaled){injection=b;for(float& v:injection)v*=c;}
    else{injection=mm(b,inv,d);for(float& v:injection)v*=c;}
    return {std::move(transition),std::move(injection),alpha};
}
inline std::pair<Matrix,Matrix> frame_statistics(const std::vector<float>& keys,const std::vector<float>& values,
                                                const std::vector<float>& beta,size_t d){
    size_t n=beta.size();require(d&&n&&keys.size()==n*d&&values.size()==n*d,"VDN frame stats shape");Matrix a(d*d,0),b(d*d,0);
    for(size_t s=0;s<n;++s){require(std::isfinite(beta[s])&&beta[s]>=0,"VDN beta invalid");
        for(size_t i=0;i<d;++i){require(std::isfinite(keys[s*d+i])&&std::isfinite(values[s*d+i]),"VDN stats nonfinite");
            for(size_t j=0;j<d;++j){a[i*d+j]+=keys[s*d+i]*beta[s]*keys[s*d+j];b[i*d+j]+=values[s*d+i]*beta[s]*keys[s*d+j];}}
    }return {a,b};
}
inline Matrix apply_state(const Matrix& state,const Factor& f){size_t d=f.alpha.size();Matrix out=mm(state,f.transition,d);require(out.size()==f.injection.size(),"VDN injection shape");for(size_t i=0;i<out.size();++i)out[i]+=f.injection[i];
    return out;}
struct Window {size_t lo,hi;};
inline std::vector<Window> window_bounds(size_t frames,size_t radius){
    require(frames>0,"VDN zero frames");std::vector<Window> out;
    for(size_t f=0;f<frames;++f)out.push_back({f>radius?f-radius:0,std::min(frames-1,f+std::min(radius,frames-1-f))});
    return out;
}
// Prefix/suffix banks. Each query reads ONLY frames outside its softmax window.
// alpha bridge decays those states through the intervening window to query f.
inline std::vector<Matrix> outside_window_states(const std::vector<Factor>& factors,const std::vector<Window>& windows,bool alpha_bridge){
    size_t n=factors.size();require(n&&windows.size()==n,"VDN scan shape");size_t d=factors[0].alpha.size();
    std::vector<Matrix> prefix(n),suffix(n),out(n,Matrix(d*d,0));Matrix state(d*d,0);
    for(size_t f=0;f<n;++f){require(factors[f].alpha.size()==d,"VDN head dimensions changed");state=apply_state(state,factors[f]);prefix[f]=state;}
    state.assign(d*d,0);for(size_t p=n;p>0;--p){size_t f=p-1;state=apply_state(state,factors[f]);suffix[f]=state;}
    for(size_t f=0;f<n;++f){auto w=windows[f];require(w.lo<=f&&f<=w.hi&&w.hi<n,"VDN invalid local window");
        std::vector<float> left_decay(d,1),right_decay(d,1);
        if(alpha_bridge){for(size_t p=w.lo;p<=f;++p)for(size_t j=0;j<d;++j)left_decay[j]*=factors[p].alpha[j];
            for(size_t p=f;p<=w.hi;++p)for(size_t j=0;j<d;++j)right_decay[j]*=factors[p].alpha[j];}
        for(size_t i=0;i<d;++i)for(size_t j=0;j<d;++j){
            if(w.lo)out[f][i*d+j]+=prefix[w.lo-1][i*d+j]*left_decay[j];
            if(w.hi+1<n)out[f][i*d+j]+=suffix[w.hi+1][i*d+j]*right_decay[j];
        }
    }return out;
}
inline std::vector<float> readout(const Matrix& state,const std::vector<float>& queries,size_t d){
    require(d&&state.size()==d*d&&queries.size()%d==0,"VDN readout shape");std::vector<float> output(queries.size(),0);
    for(size_t t=0;t<queries.size()/d;++t)for(size_t v=0;v<d;++v)for(size_t k=0;k<d;++k)output[t*d+v]+=state[v*d+k]*queries[t*d+k];
    return output;
}
} } // namespace MiniMaxH3::VDNReference
