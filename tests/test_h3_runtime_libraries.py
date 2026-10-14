import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('h3_lib',Path(__file__).resolve().parents[1]/'scripts/h3_cuda_workflow.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
class RuntimeLibraries(unittest.TestCase):
 def test_unchanged_cli_library_replacement_changes_provenance(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'bin').mkdir();exe=root/'bin/ed-cli';exe.write_text('fake nonELF fixture');lib=root/'bin/libggml-cuda.so';lib.write_bytes(b'v1')
   a=w.build_provenance(exe,lock_runtime=True,runtime_env={'LD_LIBRARY_PATH':'a'})
   lib.write_bytes(b'v2');b=w.build_provenance(exe,lock_runtime=True,runtime_env={'LD_LIBRARY_PATH':'a'})
   self.assertEqual(a['binary_sha256'],b['binary_sha256']);self.assertNotEqual(a['runtime_libraries'],b['runtime_libraries'])
   c=w.build_provenance(exe,lock_runtime=True,runtime_env={'LD_LIBRARY_PATH':'b'})
   self.assertNotEqual(b['loader_environment'],c['loader_environment'])
 def test_preload_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'ed-cli';p.write_text('fixture')
   with self.assertRaises(w.WorkflowError):w.build_provenance(p,lock_runtime=True,runtime_env={'LD_PRELOAD':'arbitrary.so'})
 def test_declared_missing_library_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);(r/'bin').mkdir();p=r/'bin/ed-cli';p.write_text('fixture');(r/'CMakeCache.txt').write_text('cudnn_LIBRARY:FILEPATH=/missing/h3-cudnn.so\n')
   with self.assertRaises(w.WorkflowError):w.build_provenance(p,lock_runtime=True,runtime_env={})
if __name__=='__main__':unittest.main()
