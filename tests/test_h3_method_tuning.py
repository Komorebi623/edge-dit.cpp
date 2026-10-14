import importlib.util
from pathlib import Path
import unittest
import tempfile
spec=importlib.util.spec_from_file_location('h3_tune',Path(__file__).resolve().parents[1]/'scripts/h3_cuda_workflow.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
class MethodTuning(unittest.TestCase):
    def setUp(self):
        self.m=w.resolve_manifest(Path(__file__).resolve().parents[1]/'benchmark/h3_cuda/cache_dit.json')
    def test_cache_threshold_recorded_without_changing_sampling(self):
        before=w.contract(self.m,['t2va'])
        self.m['binaries']['candidate']['tuning_env']={'ED_H3_CACHE_THRESHOLD':'0.12'}
        w.validate_manifest(self.m);e,_=w.isolated_env(self.m,'candidate',{})
        self.assertEqual(e['ED_H3_CACHE_THRESHOLD'],'0.12');self.assertEqual(before,w.contract(self.m,['t2va']))
        b,_=w.isolated_env(self.m,'baseline',{'ED_H3_CACHE_THRESHOLD':'1'})
        self.assertNotIn('ED_H3_CACHE_THRESHOLD',b)
    def test_bad_value_or_unrelated_switch_rejected(self):
        for setting in ({'ED_H3_CACHE_THRESHOLD':'nan'},{'ED_H3_CACHE_THRESHOLD':'-1'},{'ED_H3_CACHE_THRESHOLD':.1},{'ED_H3_SPARSE_KEEP_RATIO':'.5'},{'ED_SAGE_PV_FP8':'1'}):
            self.m['binaries']['candidate']['tuning_env']=setting
            with self.subTest(setting=setting),self.assertRaises(w.WorkflowError):w.validate_manifest(self.m)
    def test_kernel_variant_requires_own_total_switch(self):
        self.m['binaries']['candidate']['optimization_env']['ED_H3_CACHE_DIT']='0'
        self.m['binaries']['candidate']['optimization_env']['ED_H3_SPARSE_ATTN']='1'
        self.m['binaries']['candidate']['tuning_env']={'ED_H3_SUBBLOCK_INDEXED':'1'}
        w.validate_manifest(self.m)
        e,_=w.isolated_env(self.m,'candidate',{})
        self.assertEqual(e['ED_H3_SUBBLOCK_INDEXED'],'1')
        self.m['binaries']['candidate']['tuning_env']={'ED_H3_SUBBLOCK_INDEXED':'.5'}
        with self.assertRaises(w.WorkflowError):w.validate_manifest(self.m)
    def test_old_binary_with_total_switch_but_no_variant_parser_is_rejected(self):
        self.m['binaries']['candidate']['optimization_env']={k:'0' for k in w.OPTIONS}
        self.m['binaries']['candidate']['optimization_env']['ED_H3_SPARSE_ATTN']='1'
        self.m['binaries']['candidate']['tuning_env']={'ED_H3_SUBBLOCK_INDEXED':'1'}
        self.m['method_catalog']['ED_H3_SPARSE_ATTN']['status']='implemented'
        with tempfile.TemporaryDirectory() as d:
            binary=Path(d)/'candidate';binary.write_bytes(b'ED_H3_SPARSE_ATTN')
            values={'source_revision':'candidate','cuda_sage_attention':'OFF','ggml_revision':self.m['submodules']['ggml'],'onednn_revision':self.m['submodules']['onednn']}
            provenance={'binary':str(binary),'binary_sha256':w.sha256(binary),'files':{'h3-build-provenance.txt':{'values':values},'CMakeCache.txt':{'values':{'ED_GGML_CUDA':'ON',**self.m['required_build'],**self.m['method_catalog']['ED_H3_SPARSE_ATTN'].get('required_build',{})}}}}
            with self.assertRaisesRegex(w.WorkflowError,'tuning parser absent'):w.validate_build(self.m,'candidate',provenance)
            binary.write_bytes(b'ED_H3_SPARSE_ATTN ED_H3_SUBBLOCK_INDEXED')
            w.validate_build(self.m,'candidate',provenance)
    def test_baseline_tuning_rejected(self):
        self.m['binaries']['baseline']['tuning_env']={'ED_H3_CACHE_THRESHOLD':'0'}
        with self.assertRaises(w.WorkflowError):w.validate_manifest(self.m)
if __name__=='__main__':unittest.main()
