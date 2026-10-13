import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec=importlib.util.spec_from_file_location('h3_build_stability',Path(__file__).resolve().parents[1]/'scripts/h3_cuda_build_config.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SourceStabilityTests(unittest.TestCase):
    def source(self,root):
        (root/'src').mkdir();(root/'include').mkdir()
        (root/'src/model.hpp').write_text('old model')
        (root/'CMakeLists.txt').write_text('old config')

    def test_manifest_detects_source_changes_but_not_docs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.source(root);before=m.source_manifest(root)
            (root/'note.md').write_text('not compiled')
            self.assertEqual(before,m.source_manifest(root))
            (root/'include/public.hpp').write_text('new API')
            self.assertNotEqual(before,m.source_manifest(root))

    def test_compile_time_change_prevents_provenance(self):
        for change in (True,False):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                root=Path(d);self.source(root);out=root/'_scratch/build'
                phase={'phase':'candidate','source':str(root),'destination':str(out/'candidate'),'configure':['fake-configure'],'build':['fake-build'],'sage_attention':'ON'}
                config={'repo_root':str(root),'build_root':str(out),'environment':{'TMPDIR':str(out/'tmp')},'phases':[phase]}
                def run(cmd,**kwargs):
                    if change and cmd==['fake-build']:(root/'src/model.hpp').write_text('new model during compile')
                with mock.patch.object(m,'dependency_state',return_value={'pin':'fixed'}),mock.patch.object(m,'check_cache'),mock.patch.object(m.subprocess,'run',side_effect=run),mock.patch.object(m,'write_provenance') as write:
                    if change:
                        with self.assertRaisesRegex(ValueError,'source changed'):m.build(config,{'pin':'fixed'})
                        write.assert_not_called()
                    else:
                        m.build(config,{'pin':'fixed'});write.assert_called_once()

if __name__=='__main__':unittest.main()
