import importlib.util
import contextlib
import io
import json
import struct
import tempfile
from unittest import mock
from pathlib import Path
import unittest
import torch
p=Path(__file__).resolve().parents[1]/'scripts/prepare_h3_lora.py'
s=importlib.util.spec_from_file_location('prepare_h3_lora',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class Prepare(unittest.TestCase):
    def pair(self,name,a,b):return {name+'.lora_A.default.weight':a,name+'.lora_B.default.weight':b}
    def test_native_no_rounding(self):
        a=torch.randn(4,8).bfloat16();b=torch.randn(9,4).bfloat16();out=m.normalize(self.pair('blocks.0.attn.out_proj',a,b),'native')
        self.assertIs(out['blocks.0.attn.out_proj']['A'],a);self.assertIs(out['blocks.0.attn.out_proj']['B'],b)
    def test_qkv_independent_factors(self):
        torch.manual_seed(41);inp=torch.randn(8,5);d={};expected=[]
        for index,key in enumerate('qkv'):
            a=torch.randn(index+2,8);b=torch.randn(6,index+2);expected.append(b@a@inp);d.update(self.pair('transformer_blocks.0.attn.to_'+key,a,b))
        p=m.normalize(d,'diffusers')['blocks.0.attn.qkv_proj']
        torch.testing.assert_close(p['B']@p['A']@inp,torch.cat(expected,dim=0),rtol=1e-5,atol=1e-5)
    def test_swiglu_swap_b_not_a(self):
        a=torch.arange(12.).reshape(3,4);b=torch.arange(24.).reshape(8,3)
        p=m.normalize(self.pair('transformer_blocks.0.ff.net.0.proj',a,b),'diffusers')['blocks.0.mlp.fc1']
        torch.testing.assert_close(p['B']@p['A'],torch.cat(((b@a)[4:],(b@a)[:4])))
        self.assertIs(p['A'],a)
    def test_reject_unknown_payload(self):
        for key in ['foo.diff','foo.set_weight','foo.weight']:
            with self.assertRaises(ValueError):m.normalize({key:torch.ones(2,2)},'native')
    def test_reject_partial_nonfinite_bad_rank(self):
        for d in [{'x.lora_A.weight':torch.ones(2,3)},self.pair('x',torch.ones(2,3),torch.ones(4,3)),self.pair('x',torch.full((2,3),float('nan')),torch.ones(4,2))]:
            with self.assertRaises(ValueError):m.normalize(d,'native')
    def test_reject_missing_qkv(self):
        with self.assertRaises(ValueError):m.normalize(self.pair('transformer_blocks.0.attn.to_q',torch.ones(2,3),torch.ones(4,2)),'diffusers')
class FastVideo(unittest.TestCase):
    pair = Prepare.pair
    prefix = 'model.diffusion_model.'
    roots = (('transformer_blocks.0', 'blocks.0'),
             ('token_refiner.refiner_blocks.1', 'token_refiner.blocks.1'))
    dtypes = (torch.bfloat16, torch.float16, torch.float32)
    top = {'proj_in': 'video_patch_proj', 'audio_proj_in': 'audio_patch_proj',
           'context_embedder': 'condition_proj', 'time_embedder.linear_1': 'time_embedder.proj_in',
           'time_embedder.linear_2': 'time_embedder.proj_out',
           'norm_out.linear': 'final_layer.adaln_proj.linear',
           'proj_out': 'final_layer.video_out', 'audio_proj_out': 'final_layer.audio_out'}

    def test_all_top_level_factors_weights_biases_and_dtypes(self):
        for dtype in self.dtypes:
            for src, dst in self.top.items():
                with self.subTest(dtype=dtype, module=src):
                    a = torch.arange(8, dtype=dtype).reshape(2, 4)
                    b = torch.arange(6, dtype=dtype).reshape(3, 2)
                    w = torch.arange(12, dtype=dtype).reshape(3, 4)
                    bias = torch.arange(3, dtype=dtype)
                    shapes = {dst+'.weight': (3, 4), dst+'.bias': (3,)}
                    for weight_suffix, bias_suffix in (('.diff', '.diff_b'),
                                                       ('.weight.diff_param', '.bias.diff_param')):
                        source = self.pair(src, a, b)
                        source.update({src+weight_suffix: w, src+bias_suffix: bias})
                        out, counts = m.normalize_fastvideo(source, shapes)
                        self.assertEqual(counts, {'pairs': 1, 'delta_count': 2})
                        for suffix, tensor in (('.lora_A.weight', a), ('.lora_B.weight', b),
                                               ('.weight.delta', w), ('.bias.delta', bias)):
                            self.assertIs(out[self.prefix+dst+suffix], tensor)

    def test_all_norms_and_block_refiner_names(self):
        modules = {'norm_out.norm': 'final_layer.norm',
                   'token_refiner.final_norm': 'token_refiner.final_norm'}
        for src, dst in self.roots:
            for tail, native in (('norm1', 'norm1'), ('norm2', 'norm2'),
                                 ('attn.norm_q', 'attn.q_norm'), ('attn.norm_k', 'attn.k_norm')):
                modules[src+'.'+tail] = dst+'.'+native
        for dtype in self.dtypes:
            for src, dst in modules.items():
                for suffix in ('.diff', '.weight.diff_param'):
                    with self.subTest(module=src, suffix=suffix, dtype=dtype):
                        t = torch.tensor([.125, -2., 3.], dtype=dtype)
                        out, counts = m.normalize_fastvideo({src+suffix: t}, {dst+'.weight': (3,)})
                        self.assertEqual(counts, {'pairs': 0, 'delta_count': 1})
                        self.assertIs(out[self.prefix+dst+'.weight.delta'], t)
        for src, dst in self.roots:
            for tail, native in (('adaln_proj.linear', 'adaln_proj.linear'),
                                 ('attn.to_out.0', 'attn.out_proj'), ('ff.net.2', 'mlp.fc2')):
                source = self.pair(src+'.'+tail, torch.ones(2, 4), torch.ones(3, 2))
                source[src+'.'+tail+'.diff_b'] = torch.ones(3)
                target = dst+'.'+native
                out, counts = m.normalize_fastvideo(source, {target+'.weight': (3, 4), target+'.bias': (3,)})
                self.assertEqual(counts, {'pairs': 1, 'delta_count': 1})
                self.assertEqual(set(out), {self.prefix+target+s for s in
                                           ('.lora_A.weight', '.lora_B.weight', '.bias.delta')})

    def test_split_qkv_exact_independent_ranks_and_dense_deltas(self):
        for src, dst in self.roots:
            for dtype in self.dtypes:
                with self.subTest(root=src, dtype=dtype):
                    source, factors, weights, biases = {}, {}, {}, {}
                    for index, side in enumerate('kvq'):
                        a = torch.arange((index+1)*4, dtype=dtype).reshape(index+1, 4)
                        b = torch.arange(2*(index+1), dtype=dtype).reshape(2, index+1)
                        factors[side] = (a, b)
                        weights[side] = torch.full((2, 4), index+.5, dtype=dtype)
                        biases[side] = torch.full((2,), index-.5, dtype=dtype)
                        stem = src+'.attn.to_'+side
                        source.update(self.pair(stem, a, b))
                        source[stem+'.diff'] = weights[side]
                        source[stem+'.bias.diff_param'] = biases[side]
                    target = dst+'.attn.qkv_proj'
                    out, counts = m.normalize_fastvideo(source, {target+'.weight': (6, 4), target+'.bias': (6,)})
                    self.assertEqual(counts, {'pairs': 1, 'delta_count': 2})
                    a, b = (out[self.prefix+target+'.lora_'+s+'.weight'] for s in 'AB')
                    self.assertEqual(a.dtype, dtype); self.assertEqual(b.dtype, dtype)
                    self.assertTrue(torch.equal(a, torch.cat([factors[k][0] for k in 'qkv'])))
                    self.assertTrue(torch.equal(b, torch.block_diag(*[factors[k][1] for k in 'qkv'])))
                    expected = torch.cat([factors[k][1].float() @ factors[k][0].float() for k in 'qkv'])
                    self.assertTrue(torch.equal(b.float() @ a.float(), expected))
                    for parameter, ts in (('weight', weights), ('bias', biases)):
                        delta = out[self.prefix+target+'.'+parameter+'.delta']
                        self.assertEqual(delta.dtype, dtype)
                        self.assertTrue(torch.equal(delta, torch.cat([ts[k] for k in 'qkv'])))

    def test_swiglu_factors_and_dense_weight_bias_swap_once(self):
        for src, dst in self.roots:
            for dtype in self.dtypes:
                a = torch.arange(8, dtype=dtype).reshape(2, 4)
                b = torch.arange(12, dtype=dtype).reshape(6, 2)
                w = torch.arange(24, dtype=dtype).reshape(6, 4)
                bias = torch.arange(6, dtype=dtype)
                stem, target = src+'.ff.net.0.proj', dst+'.mlp.fc1'
                source = self.pair(stem, a, b)
                source.update({stem+'.diff': w, stem+'.diff_b': bias})
                originals = {k: v.clone() for k, v in source.items()}
                out, _ = m.normalize_fastvideo(source, {target+'.weight': (6, 4), target+'.bias': (6,)})
                self.assertIs(out[self.prefix+target+'.lora_A.weight'], a)
                for suffix, t in (('.lora_B.weight', b), ('.weight.delta', w), ('.bias.delta', bias)):
                    actual = out[self.prefix+target+suffix]
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(torch.equal(actual, torch.cat((t[3:], t[:3]))))
                for key in source: self.assertTrue(torch.equal(source[key], originals[key]))

    def test_explicit_layout_alias_prefixes_and_noncontiguous_inputs(self):
        for layout in ('fastvideo', 'fastvideo-dense'):
            for prefix in ('', 'base_model.model.', 'model.diffusion_model.', 'transformer.'):
                a, b, w = torch.ones(4, 2).t(), torch.ones(2, 3).t(), torch.arange(12.).reshape(4, 3).t()
                source = {prefix+k.replace('.default', ''): v for k, v in self.pair('proj_in', a, b).items()}
                source[prefix+'proj_in.diff'] = w
                out, counts = m.normalize_fastvideo(source, {'video_patch_proj.weight': (3, 4)}, layout)
                self.assertEqual(counts, {'pairs': 1, 'delta_count': 1})
                for t in out.values(): self.assertTrue(t.is_contiguous())
                self.assertTrue(torch.equal(out[self.prefix+'video_patch_proj.weight.delta'], w))
        with self.assertRaises(ValueError): m.normalize_fastvideo({}, {}, 'native')
        with self.assertRaises(ValueError): m.normalize_fastvideo({}, {})

    def test_dense_shape_and_parameter_failures(self):
        cases = [({'proj_in.diff': torch.ones(4, 3)}, {'video_patch_proj.weight': (3, 4)}),
                 ({'proj_in.diff': torch.ones(3, 4)}, {}),
                 ({'proj_in.diff_b': torch.ones(3, 1)}, {'video_patch_proj.bias': (3,)}),
                 ({'proj_in.bias.diff_param': torch.ones(4)}, {'video_patch_proj.bias': (3,)}),
                 ({'norm_out.norm.diff': torch.ones(3, 1)}, {'final_layer.norm.weight': (3,)}),
                 ({'norm_out.norm.diff_b': torch.ones(3)}, {'final_layer.norm.weight': (3,)}),
                 ({'proj_in.diff': torch.ones(3, 4)}, {'video_patch_proj.weight': (3, 4, 1)}),
                 ({'proj_in.diff': torch.ones(0, 4)}, {'video_patch_proj.weight': (0, 4)}),
                 ({'proj_in.diff': torch.ones(1)}, {'video_patch_proj.weight': (-1,)}),
                 ({'proj_in.diff_param': torch.ones(3, 4)}, {'video_patch_proj.weight': (3, 4)}),
                 ({'proj_in.weight.diff': torch.ones(3, 4)}, {'video_patch_proj.weight': (3, 4)})]
        for source, shapes in cases:
            with self.subTest(keys=list(source), shapes=shapes), self.assertRaises(ValueError):
                m.normalize_fastvideo(source, shapes)
        for suffix, shape in (('.diff', (5, 4)), ('.diff_b', (5,))):
            parameter = 'weight' if suffix == '.diff' else 'bias'
            with self.assertRaisesRegex(ValueError, 'odd SwiGLU'):
                m.normalize_fastvideo({'transformer_blocks.0.ff.net.0.proj'+suffix: torch.ones(shape)},
                                      {'blocks.0.mlp.fc1.'+parameter: shape})

    def test_finite_and_dtype_validation_for_factors_and_deltas(self):
        invalid = [torch.full((3, 4), v) for v in (float('nan'), float('inf'), -float('inf'))]
        invalid += [torch.ones(3, 4, dtype=d) for d in (torch.float64, torch.int32, torch.bool)]
        invalid += [torch.ones(()), torch.ones(3, 4, 1), torch.ones(0, 4), torch.ones(3, 0)]
        for t in invalid:
            for source in ({'proj_in.diff': t}, self.pair('proj_in', t, torch.ones(3, 3))):
                with self.subTest(dtype=t.dtype, shape=t.shape), self.assertRaises(ValueError):
                    m.normalize_fastvideo(source, {'video_patch_proj.weight': (3, 4)})
        for key in ('proj_in.diff_b', 'proj_in.bias.diff_param'):
            with self.assertRaises(ValueError):
                m.normalize_fastvideo({key: torch.full((3,), float('nan'))}, {'video_patch_proj.bias': (3,)})

    def test_reject_incomplete_mismatched_and_duplicate_factors(self):
        valid = self.pair('proj_in', torch.ones(2, 4), torch.ones(3, 2))
        cases = [{next(iter(valid)): next(iter(valid.values()))},
                 self.pair('proj_in', torch.ones(2, 4), torch.ones(3, 3)),
                 self.pair('proj_in', torch.ones(2, 4), torch.ones(3, 2).half()),
                 self.pair('proj_in', torch.ones(2, 5), torch.ones(3, 2)),
                 self.pair('proj_in', torch.ones(2049, 4), torch.ones(3, 2049)),
                 {**valid, 'proj_in.lora_A.weight': torch.ones(2, 4)},
                 {**valid, 'transformer.proj_in.lora_B.default.weight': torch.ones(3, 2)}]
        for source in cases:
            with self.subTest(keys=list(source)), self.assertRaises(ValueError):
                m.normalize_fastvideo(source, {'video_patch_proj.weight': (3, 4)})
        with self.assertRaises(ValueError):
            m.normalize_fastvideo(self.pair('norm_out.norm', torch.ones(1, 3), torch.ones(3, 1)),
                                  {'final_layer.norm.weight': (3,)})
        with self.assertRaisesRegex(ValueError, 'odd SwiGLU'):
            m.normalize_fastvideo(self.pair('transformer_blocks.0.ff.net.0.proj', torch.ones(2, 4), torch.ones(5, 2)),
                                  {'blocks.0.mlp.fc1.weight': (5, 4)})

    def test_reject_qkv_partial_shapes_dtype_and_fused_rank(self):
        stem, target = 'transformer_blocks.0.attn.to_', 'blocks.0.attn.qkv_proj.weight'
        good = {}
        for k in 'qkv': good.update(self.pair(stem+k, torch.ones(2, 4), torch.ones(3, 2)))
        cases = [{k: v for k, v in good.items() if '.to_v.' not in k},
                 {**good, stem+'q.lora_A.default.weight': torch.ones(2, 5)},
                 {**good, stem+'q.lora_B.default.weight': torch.ones(2, 2),
                  stem+'v.lora_B.default.weight': torch.ones(4, 2)},
                 {k: v.half() if '.to_k.' in k else v for k, v in good.items()}]
        oversized = {}
        for k in 'qkv': oversized.update(self.pair(stem+k, torch.ones(683, 4), torch.ones(3, 683)))
        cases.append(oversized)
        for source in cases:
            with self.assertRaises(ValueError): m.normalize_fastvideo(source, {target: (9, 4)})
        with self.assertRaises(ValueError): m.normalize_fastvideo(good, {target: (8, 4)})
        dense = {stem+k+'.diff': torch.ones(3, 4) for k in 'qkv'}
        for source in ({stem+'q.diff': torch.ones(3, 4)},
                       {**dense, stem+'k.diff': torch.ones(3, 4).half()},
                       {**dense, stem+'q.diff': torch.ones(2, 4), stem+'v.diff': torch.ones(4, 4)}):
            with self.assertRaises(ValueError): m.normalize_fastvideo(source, {target: (9, 4)})

    def test_reject_duplicate_deltas_unknown_keys_and_vsa_without_partial_adapter(self):
        valid = {'proj_in.diff': torch.ones(3, 4)}
        unknown = ['proj_in.weight.diff_param', 'transformer.proj_in.diff', 'proj_in.alpha',
                   'proj_in.dora_scale', 'proj_in.weight', 'unknown.diff',
                   'transformer_blocks.0.unknown.diff', 'blocks.0.norm1.diff',
                   'transformer_blocks.0.attn.qkv_proj.diff', 'proj_in.set_weight', 'proj_in.weight.set_param']
        for gate in ('to_gate_compress', 'gate_compress', 'vsa_gate'):
            for suffix in ('.set_weight', '.weight.set_param', '.diff', '.weight.diff_param', '.lora_A.weight'):
                unknown.append('transformer_blocks.0.attn.'+gate+suffix)
        for layout in ('fastvideo', 'fastvideo-dense'):
            for key in unknown:
                with self.subTest(key=key, layout=layout), self.assertRaises(ValueError):
                    m.normalize_fastvideo({**valid, key: torch.ones(3, 4)}, {'video_patch_proj.weight': (3, 4)}, layout)
        with self.assertRaises(ValueError):
            m.normalize_fastvideo({'proj_in.diff_b': torch.ones(3), 'proj_in.bias.diff_param': torch.ones(3)},
                                  {'video_patch_proj.bias': (3,)})
        for layout in ('native', 'diffusers'):
            for key in ('proj_in.diff', 'proj_in.diff_b', 'proj_in.weight.diff_param', 'proj_in.set_weight'):
                with self.assertRaises(ValueError): m.normalize({key: torch.ones(3, 4)}, layout)
        with self.assertRaises(ValueError):
            m.normalize(self.pair('transformer_blocks.0.adaln_proj.linear', torch.ones(2, 4), torch.ones(3, 2)), 'diffusers')

    def run_cli(self, root, source, shapes, layout, coefficient=.25):
        from safetensors.torch import save_file
        adapter, base, output = root/'source.safetensors', root/'base.gguf', root/'out'
        save_file(source, str(adapter))
        with base.open('wb') as f:
            f.write(b'GGUF'+struct.pack('<IQQ', 3, len(shapes), 0))
            for name, shape in shapes.items():
                encoded = name.encode()
                f.write(struct.pack('<Q', len(encoded))+encoded+struct.pack('<I', len(shape)))
                f.write(struct.pack('<'+'Q'*len(shape), *reversed(shape))+struct.pack('<IQ', 0, 0))
        original_base, original_adapter = base.read_bytes(), adapter.read_bytes()
        argv = ['prepare_h3_lora', '--adapter', str(adapter), '--adapter-sha256', m.sha256(adapter),
                '--base', str(base), '--base-sha256', m.sha256(base), '--layout', layout,
                '--coefficient', str(coefficient), '--steps', '4', '--video-shift', '3',
                '--partition', 'fl2va', '--output', str(output)]
        with mock.patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()): m.main()
        self.assertEqual(base.read_bytes(), original_base)
        self.assertEqual(adapter.read_bytes(), original_adapter)
        return output, json.loads((output/'manifest.json').read_text())

    def test_cli_dense_manifest_counts_coefficient_and_saved_dtype(self):
        from safetensors.torch import load_file
        a = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        b = torch.arange(6, dtype=torch.bfloat16).reshape(3, 2)
        delta = torch.full((3, 4), .5, dtype=torch.float16)
        source = self.pair('proj_in', a, b)
        source['proj_in.diff'] = delta
        for layout in ('fastvideo', 'fastvideo-dense'):
            with tempfile.TemporaryDirectory() as tmp:
                output, manifest = self.run_cli(Path(tmp), source, {'video_patch_proj.weight': (3, 4)}, layout)
                self.assertEqual(manifest['kind'], 'minimax-h3-dynamic-lora')
                self.assertEqual(manifest['payload_format'], 'fastvideo-dense-v1')
                self.assertEqual((manifest['pairs'], manifest['delta_count']), (1, 1))
                self.assertEqual(manifest['coefficient'], .25)
                self.assertEqual(manifest['coefficient_applies_to'], 'lora_B @ lora_A and delta')
                self.assertEqual(manifest['adapter_sha256'], m.sha256(output/'adapter.safetensors'))
                saved = load_file(str(output/'adapter.safetensors'))
                aa, bb, dd = (saved[self.prefix+'video_patch_proj'+s] for s in
                              ('.lora_A.weight', '.lora_B.weight', '.weight.delta'))
                for actual, original in ((aa, a), (bb, b), (dd, delta)):
                    self.assertEqual(actual.dtype, original.dtype)
                    self.assertTrue(torch.equal(actual, original))
                self.assertTrue(torch.equal(.25*(bb.float() @ aa.float()+dd.float()),
                                            .25*(b.float() @ a.float()+delta.float())))

    def test_cli_legacy_formats_unchanged_and_errors_do_not_write_output(self):
        for layout, stem in (('native', 'blocks.0.attn.out_proj'),
                             ('diffusers', 'transformer_blocks.0.attn.to_out.0')):
            with tempfile.TemporaryDirectory() as tmp:
                _, manifest = self.run_cli(Path(tmp), self.pair(stem, torch.ones(2, 4), torch.ones(3, 2)),
                                           {'blocks.0.attn.out_proj.weight': (3, 4)}, layout)
                self.assertEqual(manifest['pairs'], 1)
                for key in ('payload_format', 'delta_count', 'coefficient_applies_to'):
                    self.assertNotIn(key, manifest)
        for bad in ({'proj_in.diff': torch.ones(4, 3)},
                    {'proj_in.diff': torch.ones(3, 4), 'oops.diff': torch.ones(3, 4)},
                    {'proj_in.diff': torch.ones(3, 4), 'transformer_blocks.0.attn.to_gate_compress.set_weight': torch.ones(3, 4)},
                    {'proj_in.lora_A.weight': torch.ones(2, 4)}):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):
                    self.run_cli(Path(tmp), bad, {'video_patch_proj.weight': (3, 4)}, 'fastvideo-dense')
                self.assertFalse((Path(tmp)/'out').exists())


if __name__=='__main__':unittest.main()
