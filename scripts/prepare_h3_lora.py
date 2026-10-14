#!/usr/bin/env python3
"""CPU-only immutable H3 adapter normalization. Never changes base GGUF bytes.

Native adapters retain all factors. Diffusers Q/K/V factors are combined with
block-diagonal B (not assumed to share A); SwiGLU B halves are swapped once.
Only exact A/B payloads are accepted. VSA/dense-delta files require a different
adapter type, not ignored keys. A separate manifest binds base/adapter/recipe.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def gguf_shapes(path):
    # This normalizer intentionally only accepts the locked full H3 GGUF family.
    with Path(path).open('rb') as f:
        if f.read(4) != b'GGUF': raise ValueError('base must be a GGUF')
        version, count, metadata = struct.unpack('<IQQ', f.read(20))
        if version != 3 or count > 32768 or metadata != 0:
            raise ValueError('unsupported base GGUF header; audit before widening')
        result = {}
        for _ in range(count):
            n, = struct.unpack('<Q', f.read(8))
            if not 0 < n < 1024: raise ValueError('invalid tensor name')
            name = f.read(n).decode(); dims, = struct.unpack('<I', f.read(4))
            if not 1 <= dims <= 4: raise ValueError('invalid tensor dimensions')
            shape = struct.unpack('<'+'Q'*dims, f.read(dims*8))
            f.read(12)
            if name in result: raise ValueError('duplicate base key')
            result[name] = tuple(reversed(shape))
    return result


def normalize(tensors, layout):
    import torch
    pairs = {}
    for key, tensor in tensors.items():
        m = re.fullmatch(r'(.+)\.lora_([AB])(?:\.default)?\.weight', key)
        if not m: raise ValueError('unsupported payload (do not silently omit): '+key)
        stem, side = m.groups()
        for prefix in ('base_model.model.', 'model.diffusion_model.', 'transformer.'):
            if stem.startswith(prefix): stem = stem[len(prefix):]; break
        if side in pairs.setdefault(stem, {}): raise ValueError('duplicate factor')
        if tensor.ndim != 2 or not torch.isfinite(tensor).all(): raise ValueError('invalid factor '+key)
        if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32): raise ValueError('unsupported factor dtype')
        pairs[stem][side] = tensor
    for stem, p in pairs.items():
        if set(p) != {'A','B'} or p['A'].shape[0] != p['B'].shape[1]: raise ValueError('incomplete/mismatched '+stem)
        if p['A'].dtype != p['B'].dtype: raise ValueError('mixed pair dtype')
    if layout == 'native': return pairs
    result = {}; fused = {}
    for stem, p in pairs.items():
        name = re.sub(r'^transformer_blocks\.(\d+)\.', r'blocks.\1.', stem)
        name = re.sub(r'^token_refiner\.refiner_blocks\.(\d+)\.', r'token_refiner.blocks.\1.', name)
        if name == stem and not stem.startswith(('transformer_blocks.', 'token_refiner.refiner_blocks.')):
            raise ValueError('unsupported Diffusers module '+stem)
        qkv = re.fullmatch(r'(.+\.attn)\.to_([qkv])', name)
        if qkv:
            fused.setdefault(qkv[1]+'.qkv_proj', {})[qkv[2]] = p
            continue
        if name.endswith('.attn.to_out.0'): name = name[:-len('to_out.0')]+'out_proj'
        elif name.endswith('.ff.net.0.proj'):
            name = name[:-len('ff.net.0.proj')]+'mlp.fc1'
            b = p['B']; n = b.shape[0]
            if n % 2: raise ValueError('odd SwiGLU width')
            p = {'A': p['A'], 'B': torch.cat((b[n//2:], b[:n//2]), dim=0)}
        elif name.endswith('.ff.net.2'): name = name[:-len('ff.net.2')]+'mlp.fc2'
        else: raise ValueError('unknown Diffusers target '+stem)
        if name in result: raise ValueError('duplicate normalized target')
        result[name] = p
    for name, group in fused.items():
        if set(group) != set('qkv'): raise ValueError('missing Q/K/V factors '+name)
        ps = [group[k] for k in 'qkv']
        if len({p['A'].shape[1] for p in ps}) != 1 or len({p['A'].dtype for p in ps}) != 1:
            raise ValueError('QKV dimensions/dtype mismatch')
        result[name] = {'A': torch.cat([p['A'] for p in ps], dim=0),
                        'B': torch.block_diag(*[p['B'] for p in ps])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--adapter-sha256', required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--base-sha256', required=True)
    parser.add_argument('--layout', choices=('native','diffusers'), required=True)
    parser.add_argument('--coefficient', type=float, required=True, help='published strength*alpha/rank, not guessed')
    parser.add_argument('--steps', type=int, choices=(4,8), required=True)
    parser.add_argument('--video-shift', type=float, required=True)
    parser.add_argument('--audio-shift', type=float, default=3.)
    parser.add_argument('--partition', choices=('fl2va','ref2va'), required=True)
    parser.add_argument('--output', type=Path, required=True, help='NEW directory only')
    a = parser.parse_args()
    if a.output.exists(): raise ValueError('refusing existing output directory')
    if any(not math.isfinite(v) or v <= 0 for v in (a.coefficient,a.video_shift,a.audio_shift)): raise ValueError('invalid coefficient/shift')
    if sha256(a.adapter) != a.adapter_sha256 or sha256(a.base) != a.base_sha256: raise ValueError('full checksum mismatch')
    from safetensors.torch import load_file, save_file
    import torch
    torch.set_num_threads(4)
    source = load_file(str(a.adapter), device='cpu');pairs=normalize(source,a.layout);shapes=gguf_shapes(a.base)
    tensors={}
    for target,p in pairs.items():
        if shapes.get(target+'.weight') != (p['B'].shape[0],p['A'].shape[1]): raise ValueError('base target mismatch '+target)
        if not 0<p['A'].shape[0]<=2048: raise ValueError('rank outside supported bounds')
        for side,t in p.items(): tensors['model.diffusion_model.'+target+'.lora_'+side+'.weight']=t.contiguous()
    a.output.mkdir(parents=True)
    path=a.output/'adapter.safetensors'
    save_file(tensors,str(path),metadata={'format':'edge-dit-h3-dynamic-lora-v1','source_sha256':a.adapter_sha256,'source_layout':a.layout,'conversion':'native identity or exact block-diagonal QKV + SwiGLU B half swap; no weight merge/requantization'})
    manifest={'schema_version':1,'kind':'minimax-h3-dynamic-lora','base_sha256':a.base_sha256,'base_partition':a.partition,'adapter_file':path.name,'adapter_sha256':sha256(path),'source_adapter_sha256':a.adapter_sha256,'coefficient':a.coefficient,'steps':a.steps,'video_shift':a.video_shift,'audio_shift':a.audio_shift,'cfg_scale':1.,'sampler':'euler','scheduler':'simple','pairs':len(pairs),'note':'New model+NFE variant on immutable Q4. Not a same-weights speedup; all other research options disabled initially.'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
