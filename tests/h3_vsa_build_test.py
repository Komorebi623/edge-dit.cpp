"""CPU-only configuration test: independent VSA option, composable with SubBlock."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
CUDA=Path('src/core/backend/ggml/cuda')
DISPATCH=Path('third_party/ggml/src/ggml-cuda/ggml-cuda.cu')
class Build(unittest.TestCase):
    def run_case(self,folder,vsa,subblock=False,copies=1,root=ROOT,tamper=False):
        p=folder/'src';p.mkdir();(p/'dummy.cpp').write_text('int unused;\n')
        prior='file(APPEND "${CMAKE_BINARY_DIR}/h3-subblock-generated/ggml-cuda.cu" "\\n// corruption\\n")' if tamper else ''
        sources=' '.join('"'+str(root/DISPATCH)+'"' for _ in range(copies))
        (p/'CMakeLists.txt').write_text(f'''cmake_minimum_required(VERSION 3.18)
project(vsa_build LANGUAGES CXX)
set(ED_GGML_CUDA ON)
set(ED_ENABLE_CUDA_H3_VSA {"ON" if vsa else "OFF"})
set(ED_ENABLE_CUDA_H3_SUBBLOCK {"ON" if subblock else "OFF"})
add_library(ggml-cuda STATIC {sources} dummy.cpp)
add_library(edgedit STATIC dummy.cpp)
include("{root/CUDA/'h3_subblock.cmake'}")
{prior}
include("{root/CUDA/'h3_vsa.cmake'}")
get_target_property(sources ggml-cuda SOURCES)
foreach(s IN LISTS sources)
 if(s MATCHES "[.]cu$")
  set_source_files_properties("${{s}}" PROPERTIES LANGUAGE CXX)
 endif()
endforeach()
file(WRITE "${{CMAKE_BINARY_DIR}}/sources.txt" "${{sources}}")
get_target_property(public ggml-cuda INTERFACE_COMPILE_DEFINITIONS)
file(WRITE "${{CMAKE_BINARY_DIR}}/public.txt" "${{public}}")
''')
        return subprocess.run(['cmake','-S',str(p),'-B',str(folder/'build')],capture_output=True,text=True,timeout=40)
    def test_independent_options_and_hook_composition(self):
        original=(ROOT/DISPATCH).read_bytes()
        for vsa,sub in [(False,False),(False,True),(True,False),(True,True)]:
            with self.subTest(vsa=vsa,subblock=sub),tempfile.TemporaryDirectory() as td:
                p=Path(td);r=self.run_case(p,vsa,sub);self.assertEqual(r.returncode,0,r.stdout+r.stderr)
                generated=p/'build/h3-vsa-generated/ggml-cuda.cu'
                self.assertEqual(generated.exists(),vsa)
                self.assertNotIn('ED_ENABLE_CUDA_H3_VSA',(p/'build/public.txt').read_text())
                if vsa:
                    s=generated.read_text();self.assertEqual(s.count('ed_cuda_h3_vsa_compute(ctx, dst)'),1)
                    self.assertEqual(s.count('ed_cuda_h3_vsa_supported(dev_ctx->device, op)'),1)
                    self.assertEqual(s.count('#include "h3_vsa.h"'),1)
                    self.assertEqual(s.count('ed_cuda_h3_subblock_compute(ctx, dst)'),int(sub))
                    prov=(p/'build/h3-vsa-generated/provenance.txt').read_text()
                    self.assertIn(hashlib.sha256(generated.read_bytes()).hexdigest(),prov)
        self.assertEqual((ROOT/DISPATCH).read_bytes(),original)
    def test_missing_duplicate_dispatcher_refused(self):
        for n in (0,2):
            with self.subTest(n=n),tempfile.TemporaryDirectory() as td:
                r=self.run_case(Path(td),True,copies=n);self.assertNotEqual(r.returncode,0)
                self.assertIn('expected one verified dispatcher',r.stdout+r.stderr)
    def test_prior_generated_tamper_refused(self):
        with tempfile.TemporaryDirectory() as td:
            r=self.run_case(Path(td),True,True,tamper=True);self.assertNotEqual(r.returncode,0)
            self.assertIn('prior generated dispatcher SHA256 mismatch',r.stdout+r.stderr)
    def test_original_tamper_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);root=p/'copy'
            for rel in (CUDA/'h3_subblock.cmake',CUDA/'h3_vsa.cmake',DISPATCH):
                dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
            with (root/DISPATCH).open('a') as f:f.write('// changed')
            r=self.run_case(p,True,root=root);self.assertNotEqual(r.returncode,0)
            self.assertIn('fixed dispatcher SHA256 mismatch',r.stdout+r.stderr)
if __name__=='__main__':unittest.main()
