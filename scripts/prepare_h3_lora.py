#!/usr/bin/env python3
"""CPU-only immutable H3 adapter normalization. Never changes base GGUF bytes.

Native adapters retain all factors. Diffusers Q/K/V factors are combined with
block-diagonal B (not assumed to share A); SwiGLU B halves are swapped once.
Native/Diffusers accept only A/B payloads; explicit FastVideo dense layouts also
accept exact parameter deltas. Only explicit fastvideo-vsa accepts the full V1
payload with all 50 trained gates; it does not enable runtime support.
A separate manifest binds base/adapter/recipe; its coefficient scales B@A,
deltas and (VSA only) gates once, without prescaling or changing payload dtypes.
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


def _fastvideo_target(stem):
    """Explicit H3 Diffusers/FastVideo module map, not a generic name rewrite."""
    top = {'proj_in': 'video_patch_proj', 'audio_proj_in': 'audio_patch_proj',
           'context_embedder': 'condition_proj',
           'time_embedder.linear_1': 'time_embedder.proj_in',
           'time_embedder.linear_2': 'time_embedder.proj_out',
           'norm_out.linear': 'final_layer.adaln_proj.linear',
           'proj_out': 'final_layer.video_out', 'audio_proj_out': 'final_layer.audio_out',
           'norm_out.norm': 'final_layer.norm', 'token_refiner.final_norm': 'token_refiner.final_norm'}
    if stem in top: return top[stem], '', False
    m = re.fullmatch(r'(transformer_blocks|token_refiner\.refiner_blocks)\.(0|[1-9]\d*)\.(.+)', stem)
    if not m: raise ValueError('unknown FastVideo module '+stem)
    root, index, tail = m.groups()
    prefix = ('blocks' if root == 'transformer_blocks' else 'token_refiner.blocks')+'.'+index+'.'
    qkv = re.fullmatch(r'attn\.to_([qkv])', tail)
    if qkv: return prefix+'attn.qkv_proj', qkv[1], False
    targets = {'attn.to_out.0': 'attn.out_proj', 'attn.norm_q': 'attn.q_norm',
               'attn.norm_k': 'attn.k_norm', 'ff.net.0.proj': 'mlp.fc1', 'ff.net.2': 'mlp.fc2',
               'adaln_proj.linear': 'adaln_proj.linear', 'norm1': 'norm1', 'norm2': 'norm2'}
    if tail not in targets: raise ValueError('unknown FastVideo module '+stem)
    return prefix+targets[tail], '', tail == 'ff.net.0.proj'


def normalize_fastvideo(tensors, shapes, layout='fastvideo-dense'):
    """Return (flat native tensors, {'pairs': N, 'delta_count': M}).

    shapes maps full native parameters (no model prefix) to PyTorch-order shapes.
    Both explicit FastVideo layouts are dense-only: no replacements or VSA gates.
    Counts describe emitted payloads, after exact QKV fusion. No coefficient is
    baked in: the consumer applies it once to each B@A and each additive delta.
    """
    import torch
    if layout not in ('fastvideo', 'fastvideo-dense'): raise ValueError('not a FastVideo dense layout')
    if not tensors: raise ValueError('empty FastVideo adapter')
    pairs, deltas, result = {}, {}, {}

    def expected(parameter, part):
        shape = tuple(shapes.get(parameter, ()))
        if len(shape) not in (1, 2) or any(n <= 0 for n in shape):
            raise ValueError('missing/invalid base target '+parameter)
        if part:
            if shape[0] % 3: raise ValueError('base QKV width not divisible by three '+parameter)
            shape = (shape[0]//3,)+shape[1:]
        return shape

    def swap_halves(t):
        n = t.shape[0]
        if n % 2: raise ValueError('odd SwiGLU width')
        return torch.cat((t[n//2:], t[:n//2]), dim=0)

    def complete(group, name):
        if set(group) not in ({''}, set('qkv')): raise ValueError('missing Q/K/V payload '+name)
        return [group[k] for k in ('qkv' if 'q' in group else [''])]

    for original, t in tensors.items():
        key = re.sub(r'^(?:base_model\.model\.|model\.diffusion_model\.|transformer\.)', '', original)
        if re.search(r'(?:^|\.)(?:to_gate_compress|gate_compress|vsa_gate)(?:\.|$)', key):
            raise ValueError('VSA gates reserved for separate adapter support: '+original)
        if key.endswith(('.set_weight', '.set_param')):
            raise ValueError('replacement payload unsupported in FastVideo dense layouts: '+original)
        factor = re.fullmatch(r'(.+)\.lora_([AB])(?:\.default)?\.weight', key)
        delta = re.fullmatch(r'(.+)\.(diff_param|diff_b|diff)', key)
        if not factor and not delta: raise ValueError('unsupported FastVideo payload: '+original)
        if t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError('unsupported payload dtype '+original)
        if t.ndim not in (1, 2) or any(n <= 0 for n in t.shape) or not torch.isfinite(t).all():
            raise ValueError('invalid/nonfinite payload '+original)
        if factor:
            stem, side = factor.groups()
            if t.ndim != 2: raise ValueError('factor must be a matrix '+original)
            name, part, swap = _fastvideo_target(stem)
            pair = pairs.setdefault(name, {}).setdefault(part, {})
            if side in pair: raise ValueError('duplicate factor '+original)
            pair[side] = swap_halves(t) if swap and side == 'B' else t
        else:
            stem, kind = delta.groups()
            parameter = 'bias' if kind == 'diff_b' else 'weight'
            if kind == 'diff_param':
                param = re.fullmatch(r'(.+)\.(weight|bias)', stem)
                if not param: raise ValueError('diff_param requires explicit weight/bias '+original)
                stem, parameter = param.groups()
            name, part, swap = _fastvideo_target(stem)
            name += '.'+parameter
            if tuple(t.shape) != expected(name, part): raise ValueError('base delta shape mismatch '+original)
            group = deltas.setdefault(name, {})
            if part in group: raise ValueError('duplicate delta target '+original)
            group[part] = swap_halves(t) if swap else t

    for name, group in pairs.items():
        ps = complete(group, name)
        for part, p in group.items():
            if set(p) != {'A', 'B'}: raise ValueError('incomplete A/B pair '+name)
            a, b = p['A'], p['B']
            if a.dtype != b.dtype: raise ValueError('mixed pair dtype '+name)
            if not 0 < a.shape[0] <= 2048 or a.shape[0] != b.shape[1]:
                raise ValueError('mismatched/rank outside supported bounds '+name)
            if (b.shape[0], a.shape[1]) != expected(name+'.weight', part):
                raise ValueError('base factor shape mismatch '+name)
        if len({p['A'].dtype for p in ps}) != 1: raise ValueError('QKV dtype mismatch '+name)
        if sum(p['A'].shape[0] for p in ps) > 2048: raise ValueError('fused rank outside supported bounds '+name)
        a = ps[0]['A'] if len(ps) == 1 else torch.cat([p['A'] for p in ps], dim=0)
        b = ps[0]['B'] if len(ps) == 1 else torch.block_diag(*[p['B'] for p in ps])
        for side, t in (('A', a), ('B', b)):
            result['model.diffusion_model.'+name+'.lora_'+side+'.weight'] = t.contiguous()
    for name, group in deltas.items():
        ts = complete(group, name)
        if len({t.dtype for t in ts}) != 1: raise ValueError('QKV delta dtype mismatch '+name)
        t = ts[0] if len(ts) == 1 else torch.cat(ts, dim=0)
        result['model.diffusion_model.'+name+'.delta'] = t.contiguous()
    return result, {'pairs': len(pairs), 'delta_count': len(deltas)}


def normalize_fastvideo_vsa(tensors, shapes):
    """Full 50-layer FastH3 V1 only; return native tensors and emitted counts.

    Gate matrices are new parameters, not base deltas. Ordinary conversion is
    exactly normalize_fastvideo, including all trained norm deltas. The locked
    V1 coverage below excludes only parameters unchanged in the source artifact.
    No coefficient is baked in; V1 preparation permits only coefficient 1.
    """
    import torch
    blocks = {int(m[1]) for key in shapes if (m := re.match(r'blocks\.(\d+)\.', key))}
    if blocks != set(range(50)): raise ValueError('FastVideo VSA V1 requires all 50 base blocks')
    ordinary, gates = {}, {}
    for key, t in tensors.items():
        gate = re.fullmatch(r'transformer_blocks\.(0|[1-9]\d*)\.attn\.to_gate_compress\.set_weight', key)
        if not gate:
            if key in ordinary: raise ValueError('duplicate payload '+key)
            ordinary[key] = t
            continue
        layer = int(gate[1])
        name = f'model.diffusion_model.blocks.{layer}.attn.to_gate_compress.weight'
        if name in gates: raise ValueError('duplicate VSA gate '+key)
        if layer not in blocks: raise ValueError('unexpected VSA gate layer '+key)
        if shapes.get(f'blocks.{layer}.attn.qkv_proj.weight') != (21504, 5376):
            raise ValueError('VSA gate/base QKV shape mismatch '+key)
        if tuple(t.shape) != (7168, 5376): raise ValueError('invalid VSA gate shape '+key)
        if t.dtype != torch.bfloat16: raise ValueError('VSA gate must retain BF16 precision '+key)
        if not torch.isfinite(t).all() or not torch.count_nonzero(t):
            raise ValueError('nonfinite or all-zero VSA gate '+key)
        gates[name] = t.contiguous()
    if len(gates) != 50: raise ValueError('missing VSA gates: require all 50 trained matrices')
    result, counts = normalize_fastvideo(ordinary, shapes)
    pair_targets = {f'blocks.{i}.{tail}' for i in range(50) for tail in
                    ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2', 'adaln_proj.linear')}
    pair_targets.update(f'token_refiner.blocks.{i}.{tail}' for i in range(2) for tail in
                        ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2'))
    delta_targets = {f'blocks.{i}.adaln_proj.linear.bias' for i in range(50)}
    delta_targets.update(target+'.'+parameter for target in
                         ('video_patch_proj', 'audio_patch_proj', 'condition_proj',
                          'time_embedder.proj_in', 'time_embedder.proj_out',
                          'final_layer.adaln_proj.linear', 'final_layer.video_out', 'final_layer.audio_out')
                         for parameter in ('weight', 'bias'))
    delta_targets.add('final_layer.norm.weight')
    # Trained norm deltas in the V1 datafree artifact, not all base norm weights.
    for norm, layers in (('norm1', (3, 4, 5, 6, 10, 17, 19)),
                         ('norm2', (4, 6, 7, 8, 12, 13, 15, 17))):
        delta_targets.update(f'blocks.{i}.{norm}.weight' for i in layers)
    expected = {'model.diffusion_model.'+target+'.lora_'+side+'.weight'
                for target in pair_targets for side in 'AB'}
    expected.update('model.diffusion_model.'+target+'.delta' for target in delta_targets)
    if set(result) != expected:
        raise ValueError('incomplete/unexpected FastVideo VSA V1 A/B or delta coverage')
    result.update(gates)
    return result, {**counts, 'gate_count': len(gates)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--adapter-sha256', required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--base-sha256', required=True)
    parser.add_argument('--layout', choices=('native','diffusers','fastvideo','fastvideo-dense','fastvideo-vsa'), required=True)
    parser.add_argument('--coefficient', type=float, required=True, help='published strength*alpha/rank; scales B@A, deltas and VSA gates once; VSA V1 requires 1')
    parser.add_argument('--steps', type=int, choices=(4,8), required=True)
    parser.add_argument('--video-shift', type=float, required=True)
    parser.add_argument('--audio-shift', type=float, default=3.)
    parser.add_argument('--partition', choices=('fl2va','ref2va'), required=True)
    parser.add_argument('--output', type=Path, required=True, help='NEW directory only')
    a = parser.parse_args()
    if a.output.exists(): raise ValueError('refusing existing output directory')
    if any(not math.isfinite(v) or v <= 0 for v in (a.coefficient,a.video_shift,a.audio_shift)): raise ValueError('invalid coefficient/shift')
    vsa = a.layout == 'fastvideo-vsa'
    if vsa and (a.coefficient != 1. or a.steps != 4 or a.video_shift != 12. or a.audio_shift != 3.):
        raise ValueError('FastVideo VSA V1 requires coefficient 1, four steps and video/audio shifts 12/3')
    if sha256(a.adapter) != a.adapter_sha256 or sha256(a.base) != a.base_sha256: raise ValueError('full checksum mismatch')
    from safetensors.torch import load_file, save_file
    import torch
    torch.set_num_threads(4)
    source = load_file(str(a.adapter), device='cpu');shapes=gguf_shapes(a.base)
    fastvideo = a.layout in ('fastvideo', 'fastvideo-dense')
    if vsa:
        tensors, counts = normalize_fastvideo_vsa(source, shapes)
    elif fastvideo:
        tensors, counts = normalize_fastvideo(source, shapes, a.layout)
    else:
        pairs=normalize(source,a.layout);tensors={};counts={'pairs':len(pairs)}
        for target,p in pairs.items():
            if shapes.get(target+'.weight') != (p['B'].shape[0],p['A'].shape[1]): raise ValueError('base target mismatch '+target)
            if not 0<p['A'].shape[0]<=2048: raise ValueError('rank outside supported bounds')
            for side,t in p.items(): tensors['model.diffusion_model.'+target+'.lora_'+side+'.weight']=t.contiguous()
    a.output.mkdir(parents=True)
    path=a.output/'adapter.safetensors'
    metadata = {'format':'edge-dit-h3-dynamic-lora-v1','source_sha256':a.adapter_sha256,'source_layout':a.layout,'conversion':'native identity or exact block-diagonal QKV + SwiGLU B half swap; no weight merge/requantization'}
    if vsa: metadata.update(payload_format='fastvideo-vsa-v1', runtime_supported='false')
    save_file(tensors,str(path),metadata=metadata)
    manifest={'schema_version':1,'kind':'minimax-h3-dynamic-lora','base_sha256':a.base_sha256,'base_partition':a.partition,'adapter_file':path.name,'adapter_sha256':sha256(path),'source_adapter_sha256':a.adapter_sha256,'coefficient':a.coefficient,'steps':a.steps,'video_shift':a.video_shift,'audio_shift':a.audio_shift,'cfg_scale':1.,'sampler':'euler','scheduler':'simple',**counts,'note':'New model+NFE variant on immutable Q4. Not a same-weights speedup; all other research options disabled initially.'}
    if fastvideo:
        manifest.update(payload_format='fastvideo-dense-v1', coefficient_applies_to='lora_B @ lora_A and delta')
    if vsa:
        sigmas = [1., .75, .5, .25, 0.]  # Four forwards, plus terminal zero.
        manifest.update(payload_format='fastvideo-vsa-v1',
                        coefficient_applies_to='lora_B @ lora_A, delta and gate weight',
                        vsa={'tile_tokens': 64, 'sparsity': .9, 'prefix_policy': 'exempt',
                             'gate_activation': 'linear', 'gate_count': counts['gate_count']},
                        scheduler='uniform', sigmas=sigmas,
                        video_sigmas=[a.video_shift*s/(1+(a.video_shift-1)*s) for s in sigmas],
                        audio_sigmas=[a.audio_shift*s/(1+(a.audio_shift-1)*s) for s in sigmas],
                        runtime_supported=False,
                        note='Preparation only; runtime unsupported until separate native CUDA qualification on main.')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
