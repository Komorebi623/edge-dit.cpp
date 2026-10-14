import copy
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('h3_sync',Path(__file__).resolve().parents[1]/'scripts/h3_cuda_workflow.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
class SyncPlan(unittest.TestCase):
    def setUp(self):self.m=w.resolve_manifest(Path(__file__).resolve().parents[1]/'benchmark/h3_cuda/rtx4090_t2va.json')
    def test_control_changes_contract_and_is_same_in_both_arms(self):
        before=copy.deepcopy(w.contract(self.m,['t2va']))
        self.m['runtime_env']={'ED_CUDNN_SDPA_SYNC_F32_SELF_ATTN':'1'}
        w.validate_manifest(self.m)
        self.assertNotEqual(before,w.contract(self.m,['t2va']))
        for phase in ('baseline','candidate'):
            e,removed=w.isolated_env(self.m,phase,{'ED_CUDNN_SDPA_SYNC_F32_SELF_ATTN':'0','ED_SAGE_ATTN':'1'})
            self.assertEqual(e['ED_CUDNN_SDPA_SYNC_F32_SELF_ATTN'],'1');self.assertNotIn('ED_SAGE_ATTN',e)
    def test_inherited_control_cannot_change_old_baseline(self):
        e,_=w.isolated_env(self.m,'baseline',{'ED_CUDNN_SDPA_SYNC_F32_SELF_ATTN':'1'})
        self.assertNotIn('ED_CUDNN_SDPA_SYNC_F32_SELF_ATTN',e)
        self.assertNotIn('runtime_env',w.contract(self.m,['t2va']))
    def test_only_exact_whitelist_allowed(self):
        for value in ({'ED_H3_CACHE_DIT':'1'},{'ED_SAGE_ATTN':'1'},{'ED_CUDNN_SDPA_SYNC_F32_SELF_ATTN':True},{'ED_CUDNN_SDPA_SYNC_F32_SELF_ATTN':'true'},[]):
            self.m['runtime_env']=value
            with self.subTest(value=value),self.assertRaises(w.WorkflowError):w.validate_manifest(self.m)
    def test_explicit_off_distinguishes_contract(self):
        before=copy.deepcopy(w.contract(self.m,['t2va']))
        self.m['runtime_env']={'ED_CUDNN_SDPA_SYNC_F32_SELF_ATTN':'0'}
        self.assertNotEqual(before,w.contract(self.m,['t2va']))
if __name__=='__main__':unittest.main()
