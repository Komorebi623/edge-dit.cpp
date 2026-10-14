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

    def run_cli(self, root, source, shapes, layout, coefficient=.25, video_shift=3):
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
                '--coefficient', str(coefficient), '--steps', '4', '--video-shift', str(video_shift),
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


class FastVideoVSA(unittest.TestCase):
    prefix = FastVideo.prefix
    gate_key = 'transformer_blocks.0.attn.to_gate_compress.set_weight'

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        # Initialize lazy meta dispatch before patching value-only reductions.
        torch.cat([torch.empty(1, device='meta')])

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def fixture():
        # Shape-only full V1 fixture: no multi-GB allocations or saved payloads.
        source, shapes = {}, {}
        def tensor(shape): return torch.empty(shape, dtype=torch.bfloat16, device='meta')
        def pair(src, dst, shape):
            source[src+'.lora_A.weight'] = tensor((64, shape[1]))
            source[src+'.lora_B.weight'] = tensor((shape[0], 64))
            shapes[dst+'.weight'] = shape
        def delta(src, dst, shape, bias=False):
            source[src+('.diff_b' if bias else '.diff')] = tensor(shape)
            shapes[dst+('.bias' if bias else '.weight')] = shape
        for i in range(50):
            source[f'transformer_blocks.{i}.attn.to_gate_compress.set_weight'] = tensor((7168, 5376))
        for root, native, layers in (('transformer_blocks', 'blocks', 50),
                                     ('token_refiner.refiner_blocks', 'token_refiner.blocks', 2)):
            for i in range(layers):
                src, dst = f'{root}.{i}', f'{native}.{i}'
                for side in 'qkv': pair(src+'.attn.to_'+side, dst+'.attn.qkv_proj', (7168, 5376))
                shapes[dst+'.attn.qkv_proj.weight'] = (21504, 5376)
                for tail, target, shape in (('attn.to_out.0', 'attn.out_proj', (5376, 7168)),
                                            ('ff.net.0.proj', 'mlp.fc1', (28672, 5376)),
                                            ('ff.net.2', 'mlp.fc2', (5376, 14336))):
                    pair(src+'.'+tail, dst+'.'+target, shape)
                if native == 'blocks':
                    pair(src+'.adaln_proj.linear', dst+'.adaln_proj.linear', (96768, 2688))
                    delta(src+'.adaln_proj.linear', dst+'.adaln_proj.linear', (96768,), bias=True)
                    for norm in ('norm1', 'norm2'): shapes[dst+'.'+norm+'.weight'] = (5376,)
        top_shapes = ((5376, 96), (5376, 32), (5376, 5120), (5376, 256),
                      (2688, 5376), (10752, 2688), (96, 5376), (32, 5376))
        for (src, dst), shape in zip(FastVideo.top.items(), top_shapes):
            delta(src, dst, shape)
            delta(src, dst, (shape[0],), bias=True)
        delta('norm_out.norm', 'final_layer.norm', (5376,))
        for norm, layers in (('norm1', (3, 4, 5, 6, 10, 17, 19)),
                             ('norm2', (4, 6, 7, 8, 12, 13, 15, 17))):
            for i in layers: delta(f'transformer_blocks.{i}.{norm}', f'blocks.{i}.{norm}', (5376,))
        return source, shapes

    @staticmethod
    @contextlib.contextmanager
    def header_checks():
        # Meta tensors validate names/shapes/dtypes/fusion, NOT payload values.
        # Real CPU tensors still run both unmocked value checks below.
        finite, nonzero = torch.isfinite, torch.count_nonzero
        with mock.patch.object(torch, 'isfinite', side_effect=lambda t:
                               torch.tensor(True) if t.is_meta else finite(t)) as f, \
             mock.patch.object(torch, 'count_nonzero', side_effect=lambda t:
                               torch.tensor(1) if t.is_meta else nonzero(t)) as n:
            yield f, n

    def test_full_counts_reuses_dense_conversion_and_keeps_gate_bytes(self):
        source, shapes = self.fixture()
        # A mostly-zero, signed trained gate is valid; do not require all entries
        # nonzero, apply sigmoid, round, normalize or prescale it.
        gate = torch.zeros((7168, 5376), dtype=torch.bfloat16)
        gate[0, 0] = -.0004215240478515625
        source[self.gate_key] = gate
        original_keys = set(source)
        with self.header_checks(), mock.patch.object(m, 'normalize_fastvideo', wraps=m.normalize_fastvideo) as dense:
            out, counts = m.normalize_fastvideo_vsa(source, shapes)
        self.assertEqual(counts, {'pairs': 258, 'delta_count': 82, 'gate_count': 50})
        self.assertEqual(len(out), 648)
        dense.assert_called_once()
        ordinary, passed_shapes = dense.call_args.args
        self.assertIs(passed_shapes, shapes)
        self.assertEqual(len(ordinary), 806)
        self.assertFalse(any(k.endswith('.set_weight') for k in ordinary))
        self.assertEqual(set(source), original_keys)
        self.assertIs(out[self.prefix+'blocks.0.attn.to_gate_compress.weight'], gate)
        self.assertEqual(gate[0, 0].item(), -.0004215240478515625)
        self.assertEqual(torch.count_nonzero(gate).item(), 1)
        self.assertTrue(all(t.dtype == torch.bfloat16 for t in out.values()))

    def test_missing_misnamed_and_duplicate_gates(self):
        source, shapes = self.fixture()
        with self.header_checks():
            for layer in (0, 25, 49):
                bad = dict(source); del bad[f'transformer_blocks.{layer}.attn.to_gate_compress.set_weight']
                with self.subTest(missing=layer), self.assertRaisesRegex(ValueError, 'missing VSA gates'):
                    m.normalize_fastvideo_vsa(bad, shapes)
            for key in ('transformer_blocks.00.attn.to_gate_compress.set_weight',
                        'transformer_blocks.50.attn.to_gate_compress.set_weight',
                        'transformer_blocks.0.attn.gate_compress.set_weight',
                        'transformer_blocks.0.attn.vsa_gate.set_weight',
                        'transformer.'+self.gate_key, 'model.diffusion_model.'+self.gate_key,
                        self.gate_key.replace('.set_weight', '.weight.set_param'),
                        self.gate_key.replace('.set_weight', '.diff'),
                        self.gate_key.replace('.set_weight', '.lora_A.weight')):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    m.normalize_fastvideo_vsa({**source, key: source[self.gate_key]}, shapes)
            # A mapping-like input can expose repeated original keys; never keep
            # only the last trained gate. Aliases above are rejected outright.
            repeated = mock.Mock()
            repeated.items.return_value = [*source.items(), (self.gate_key, source[self.gate_key])]
            with self.assertRaisesRegex(ValueError, 'duplicate VSA gate'):
                m.normalize_fastvideo_vsa(repeated, shapes)

    def test_gate_shape_precision_and_base_geometry(self):
        source, shapes = self.fixture()
        with self.header_checks():
            for shape in ((5376, 7168), (7168, 5375), (7168, 5376, 1), (0, 5376), (7168,)):
                with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, 'gate shape'):
                    m.normalize_fastvideo_vsa({**source, self.gate_key: torch.empty(shape, dtype=torch.bfloat16, device='meta')}, shapes)
            for dtype in (torch.float16, torch.float32, torch.float64, torch.int16):
                with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, 'BF16'):
                    m.normalize_fastvideo_vsa({**source, self.gate_key: source[self.gate_key].to(dtype)}, shapes)
            for shape in ((21501, 5376), (21504, 5375), ()):
                with self.subTest(base=shape), self.assertRaisesRegex(ValueError, 'gate/base QKV'):
                    m.normalize_fastvideo_vsa(source, {**shapes, 'blocks.0.attn.qkv_proj.weight': shape})
            for layers in (range(49), range(51), range(1, 51)):
                bad_shapes = {f'blocks.{i}.attn.qkv_proj.weight': (21504, 5376) for i in layers}
                with self.subTest(layers=layers), self.assertRaisesRegex(ValueError, '50 base blocks'):
                    m.normalize_fastvideo_vsa(source, bad_shapes)

    def test_zero_and_nonfinite_gate_rejected_on_cpu_per_layer(self):
        source, shapes = self.fixture()
        # Expanded views keep tests small; checks still inspect the full shape.
        with self.header_checks():
            for layer in (0, 25, 49):
                for value in (0., float('nan'), float('inf'), -float('inf')):
                    gate = torch.tensor(value, dtype=torch.bfloat16).expand(7168, 5376)
                    bad = {**source, f'transformer_blocks.{layer}.attn.to_gate_compress.set_weight': gate}
                    with self.subTest(layer=layer, value=value), self.assertRaisesRegex(ValueError, 'nonfinite or all-zero'):
                        m.normalize_fastvideo_vsa(bad, shapes)

    def test_full_ordinary_payload_required_including_norm_deltas(self):
        source, shapes = self.fixture()
        omissions = [('transformer_blocks.0.attn.to_q.lora_A.weight',),
                     tuple(f'transformer_blocks.0.ff.net.2.lora_{s}.weight' for s in 'AB'),
                     tuple(f'token_refiner.refiner_blocks.1.attn.to_{q}.lora_{s}.weight' for q in 'qkv' for s in 'AB'),
                     ('transformer_blocks.49.adaln_proj.linear.diff_b',), ('proj_in.diff',),
                     ('norm_out.norm.diff',), ('transformer_blocks.3.norm1.diff',),
                     ('transformer_blocks.17.norm2.diff',)]
        with self.header_checks():
            for keys in omissions:
                bad = {k: t for k, t in source.items() if k not in keys}
                with self.subTest(omitted=keys), self.assertRaises(ValueError):
                    m.normalize_fastvideo_vsa(bad, shapes)
            # Counts alone must not allow substituting another norm for a
            # required delta, nor accepting duplicate aliases/replacements.
            bad = dict(source)
            bad['transformer_blocks.0.norm1.diff'] = bad.pop('transformer_blocks.3.norm1.diff')
            with self.assertRaisesRegex(ValueError, 'coverage'): m.normalize_fastvideo_vsa(bad, shapes)
            for key in ('norm_out.norm.weight.diff_param', 'proj_in.set_weight', 'unknown.diff'):
                with self.subTest(extra=key), self.assertRaises(ValueError):
                    m.normalize_fastvideo_vsa({**source, key: source['norm_out.norm.diff']}, shapes)

    def test_dense_gate_rejection_unchanged(self):
        source, shapes = self.fixture()
        for layout in ('fastvideo', 'fastvideo-dense', 'fastvideo-vsa'):
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                m.normalize_fastvideo(source, shapes, layout)

    def test_vsa_manifest_recipe_and_metadata_without_full_save(self):
        from safetensors.torch import save_file
        source, shapes = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def save_header_only(tensors, filename, metadata=None):
                if Path(filename).name == 'source.safetensors':
                    return save_file(tensors, filename, metadata=metadata)
                # Exercise CLI dispatch/manifest, never serialize the meta/full
                # 5GB fixture. Ordinary real save/dtype is covered by old tests.
                self.assertEqual(len(tensors), 648)
                self.assertEqual(metadata['payload_format'], 'fastvideo-vsa-v1')
                self.assertEqual(metadata['runtime_supported'], 'false')
                self.assertTrue(all(t.dtype == torch.bfloat16 for t in tensors.values()))
                Path(filename).write_bytes(b'header-only test placeholder')
            with self.header_checks(), mock.patch('safetensors.torch.load_file', return_value=source), \
                 mock.patch('safetensors.torch.save_file', side_effect=save_header_only) as save:
                # The existing fixture writes a tiny real source and GGUF
                # header; only the full meta output writer is a placeholder.
                output, manifest = FastVideo.run_cli(self, root, {'placeholder': torch.ones(1)}, shapes,
                                                    'fastvideo-vsa', coefficient=1., video_shift=12)
            self.assertEqual(save.call_count, 2)
            self.assertEqual(manifest['payload_format'], 'fastvideo-vsa-v1')
            self.assertEqual((manifest['pairs'], manifest['delta_count'], manifest['gate_count']), (258, 82, 50))
            self.assertEqual(manifest['vsa'], {'tile_tokens': 64, 'sparsity': .9, 'prefix_policy': 'exempt',
                                              'gate_activation': 'linear', 'gate_count': 50})
            self.assertEqual(manifest['coefficient'], 1.)
            self.assertEqual(manifest['coefficient_applies_to'], 'lora_B @ lora_A, delta and gate weight')
            self.assertEqual((manifest['steps'], manifest['video_shift'], manifest['audio_shift']), (4, 12., 3.))
            self.assertEqual((manifest['sampler'], manifest['scheduler']), ('euler', 'uniform'))
            self.assertEqual(manifest['sigmas'], [1., .75, .5, .25, 0.])
            self.assertEqual(manifest['video_sigmas'], [1., 36/37, 12/13, .8, 0.])
            self.assertEqual(manifest['audio_sigmas'], [1., .9, .75, .5, 0.])
            self.assertFalse(manifest['runtime_supported'])
            self.assertIn('native CUDA qualification', manifest['note'])
            self.assertEqual(manifest['adapter_sha256'], m.sha256(output/'adapter.safetensors'))

    def test_invalid_recipe_rejected_before_loading_or_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['prepare_h3_lora', '--adapter', 'unused', '--adapter-sha256', 'unused',
                    '--base', 'unused', '--base-sha256', 'unused', '--layout', 'fastvideo-vsa',
                    '--coefficient', '1', '--steps', '4', '--video-shift', '12', '--audio-shift', '3',
                    '--partition', 'fl2va', '--output', str(Path(tmp)/'out')]
            for flag, value in (('--coefficient', '.25'), ('--coefficient', '2'), ('--coefficient', 'nan'),
                                ('--steps', '8'), ('--video-shift', '3'), ('--audio-shift', '12')):
                bad = argv.copy(); bad[bad.index(flag)+1] = value
                with self.subTest(flag=flag, value=value), mock.patch('sys.argv', bad), \
                     mock.patch.object(m, 'sha256') as checksum, self.assertRaises(ValueError): m.main()
                checksum.assert_not_called()
                self.assertFalse((Path(tmp)/'out').exists())

    def test_real_header_maps_all_856_without_loading_or_saving_payload(self):
        scratch = p.parents[1]/'_scratch'
        root = scratch/'h3-vsa-continuation-20261014'
        adapter, audit_path = root/'vsa-datafree.safetensors', root/'vsa-header-audit.json'
        if not adapter.exists() or not audit_path.exists(): self.skipTest('local VSA artifact/header audit unavailable')
        with adapter.open('rb') as f:
            size, = struct.unpack('<Q', f.read(8))
            self.assertLess(size, 1 << 20)
            header = json.loads(f.read(size))
        metadata = header.pop('__metadata__')
        audit = json.loads(audit_path.read_text())
        self.assertEqual(audit['original_full_sha256'], '42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470')
        self.assertEqual(metadata, audit['metadata'])
        self.assertEqual({k: {'dtype': v['dtype'], 'shape': v['shape']} for k, v in header.items()}, audit['tensors'])
        fixture, shapes = self.fixture()
        self.assertEqual(set(header), set(fixture))
        for k, v in header.items():
            self.assertEqual(v['dtype'], 'BF16')
            self.assertEqual(tuple(v['shape']), tuple(fixture[k].shape))
        base = scratch/'rtx4090-20261013/models/minimax_h3_ref2va-Q4_K_M.gguf'
        if base.exists(): shapes = m.gguf_shapes(base)
        source = {k: torch.empty(v['shape'], dtype=torch.bfloat16, device='meta') for k, v in header.items()}
        with self.header_checks() as (finite, nonzero):
            out, counts = m.normalize_fastvideo_vsa(source, shapes)
        self.assertEqual(len(source), 856)
        self.assertEqual(counts, {'pairs': 258, 'delta_count': 82, 'gate_count': 50})
        self.assertEqual(len(out), 648)
        # Every original tensor reaches validation, gates included. No payload
        # bytes read: this is not a new numerical audit of the real artifact.
        self.assertCountEqual([id(c.args[0]) for c in finite.call_args_list], [id(t) for t in source.values()])
        self.assertEqual(nonzero.call_count, 50)
        for i in range(50):
            self.assertIs(out[self.prefix+f'blocks.{i}.attn.to_gate_compress.weight'],
                          source[f'transformer_blocks.{i}.attn.to_gate_compress.set_weight'])
        self.assertEqual(sum(k.endswith('.delta') for k in out), 82)
        self.assertEqual(sum('.norm' in k and k.endswith('.delta') for k in out), 16)
        self.assertTrue(all(t.dtype == torch.bfloat16 for t in out.values()))

if __name__=='__main__':unittest.main()
