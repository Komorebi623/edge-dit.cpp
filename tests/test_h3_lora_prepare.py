import importlib.util
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
if __name__=='__main__':unittest.main()
