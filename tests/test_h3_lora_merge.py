"""CPU-only tiny checkpoint regression for the historical H3 offline merger."""
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from safetensors.torch import save_file
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("merge_h3",ROOT/"scripts/merge_minimax_h3_lora.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class MergeTest(unittest.TestCase):
    def test_cpu_only_device(self):
        self.assertEqual(str(m.resolve_device("cpu")),"cpu")
        for x in ["cuda","cuda:0","auto"]:
            with self.assertRaises(ValueError):m.resolve_device(x)
    def test_safe_shards(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["../x.safetensors","/x.safetensors","a/b.safetensors","x.bin"]:
                with self.assertRaises(ValueError):m.safe_shard_path(Path(d),name)
    def test_merge_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/"transformer";base.mkdir();out=root/"merged"
            w=torch.arange(12,dtype=torch.float32).reshape(3,4)/10
            a=torch.tensor([[1.,2.,0.,-1.],[0.,1.,1.,0.]])
            b=torch.tensor([[1.,0.],[0.,1.],[1.,-1.]])
            save_file({"blocks.0.attn.to_q.weight":w},base/"s0.safetensors")
            index={"metadata":{},"weight_map":{"blocks.0.attn.to_q.weight":"s0.safetensors"}}
            (base/m.INDEX_NAME).write_text(json.dumps(index))
            (base/"config.json").write_text('{"_class_name":"MiniMaxH3Transformer3DModel"}')
            adapter=root/"lora.safetensors"
            save_file({"blocks.0.attn.to_q"+m.LORA_A_SUFFIX:a,"blocks.0.attn.to_q"+m.LORA_B_SUFFIX:b},adapter,metadata={"alpha":"2","key_format":"minimax-h3-diffusers"})
            args=SimpleNamespace(base=base,lora=adapter,output=out,device="cpu",alpha=None,scale=1.,dry_run=False,resume=False,steps=8,video_shift=12.,audio_shift=3.)
            m.merge(args)
            with safe_open(out/"s0.safetensors",framework="pt",device="cpu") as f:
                torch.testing.assert_close(f.get_tensor("blocks.0.attn.to_q.weight"),w+b@a)
            recipe=json.loads((out/"edge-dit-h3-turbo.json").read_text())
            self.assertEqual(recipe["steps"],8)
            self.assertEqual(recipe["shards"]["s0.safetensors"]["output_sha256"],m.sha256_file(out/"s0.safetensors"))
            args.resume=True;m.merge(args)
            args.scale=.5
            with self.assertRaises(ValueError):m.merge(args)
    def test_reject_nonfinite_and_dtype_overflow(self):
        a=torch.ones(1,2);b=torch.ones(2,1);w=torch.zeros(2,2)
        for bad in [float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                m.merge_weight(w,a*bad,b,1.,1.,torch.device("cpu"))
        with self.assertRaises(ValueError):
            m.merge_weight(torch.zeros(2,2,dtype=torch.float16),a*1e5,b,1.,1.,torch.device("cpu"))
    def test_invalid_pair(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.safetensors";save_file({"x"+m.LORA_A_SUFFIX:torch.ones(2,3)},p,metadata={"alpha":"2"})
            with self.assertRaises(ValueError):m.load_adapter(p,None)
if __name__=="__main__":unittest.main()
