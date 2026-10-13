#!/usr/bin/env python3
"""CPU-only configure tests for the build-local dispatcher (never builds/runs CUDA)."""
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
REL = Path('src/core/backend/ggml/cuda/h3_subblock.cmake')
DISPATCH = Path('third_party/ggml/src/ggml-cuda/ggml-cuda.cu')
SHA = '1284940278a57c529b88ba4a3b4f49a82cf431fea63848bfb74d9e7f1f6ee265'


class BuildLocalDispatcher(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='h3-subblock-cmake-')
        self.path = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def configure(self, enabled, root=ROOT, copies=1):
        source = self.path / 'harness'
        source.mkdir()
        (source / 'dummy.cpp').write_text('int h3_build_probe;\n')
        originals = ' '.join(f'"{root / DISPATCH}"' for _ in range(copies))
        # The harness maps .cu to CXX *only for configure*. No CUDA compiler,
        # toolkit discovery or GPU initialization; do not use it to compile.
        text = f'''cmake_minimum_required(VERSION 3.18)
project(h3_source_replacement LANGUAGES CXX)
set(ED_GGML_CUDA ON)
set(ED_ENABLE_CUDA_H3_SUBBLOCK {"ON" if enabled else "OFF"})
add_library(ggml-cuda STATIC {originals} dummy.cpp)
add_library(edgedit STATIC dummy.cpp)
include("{root / REL}")
get_target_property(sources ggml-cuda SOURCES)
foreach(s IN LISTS sources)
  if(s MATCHES "[.]cu$")
    set_source_files_properties("${{s}}" PROPERTIES LANGUAGE CXX)
  endif()
endforeach()
file(WRITE "${{CMAKE_BINARY_DIR}}/sources.txt" "${{sources}}")
get_target_property(definitions ggml-cuda COMPILE_DEFINITIONS)
file(WRITE "${{CMAKE_BINARY_DIR}}/definitions.txt" "${{definitions}}")
get_target_property(interface_defs ggml-cuda INTERFACE_COMPILE_DEFINITIONS)
file(WRITE "${{CMAKE_BINARY_DIR}}/interface.txt" "${{interface_defs}}")
'''
        (source / 'CMakeLists.txt').write_text(text)
        return subprocess.run(['cmake', '-S', str(source), '-B', str(self.path / 'build')],
                              text=True, capture_output=True, timeout=60)

    def test_off_does_not_generate_or_replace(self):
        result = self.configure(False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        build = self.path / 'build'
        self.assertFalse((build / 'h3-subblock-generated').exists())
        self.assertIn(str(ROOT / DISPATCH), (build / 'sources.txt').read_text())
        self.assertNotIn('ED_ENABLE_CUDA_H3_SUBBLOCK', (build / 'definitions.txt').read_text())

    def test_on_exactly_three_hooks_and_unchanged_pinned_source(self):
        original = (ROOT / DISPATCH).read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(), SHA)
        result = self.configure(True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        build = self.path / 'build'
        generated = build / 'h3-subblock-generated' / 'ggml-cuda.cu'
        body = generated.read_text()
        self.assertEqual(body.count('#include "h3_subblock.h"'), 1)
        self.assertEqual(body.count('if (ed_cuda_h3_subblock_compute(ctx, dst))'), 1)
        self.assertEqual(body.count('if (ed_cuda_h3_subblock_supported(dev_ctx->device, op))'), 1)
        self.assertEqual(body.count('#ifdef ED_ENABLE_CUDA_H3_SUBBLOCK'), 3)
        sources = (build / 'sources.txt').read_text().split(';')
        self.assertNotIn(str(ROOT / DISPATCH), sources)
        self.assertEqual(sources.count(str(generated)), 1)
        self.assertEqual((ROOT / DISPATCH).read_bytes(), original)
        provenance = (generated.parent / 'provenance.txt').read_text()
        self.assertIn('original_sha256=' + SHA, provenance)
        self.assertIn('generated_sha256=' + hashlib.sha256(generated.read_bytes()).hexdigest(), provenance)
        self.assertNotIn('ED_ENABLE_CUDA_H3_SUBBLOCK', (build / 'interface.txt').read_text())

    def test_modified_source_fails_before_generation(self):
        root = self.path / 'copy'
        for rel in [REL, DISPATCH]:
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, dest)
        with (root / DISPATCH).open('a') as out:
            out.write('\n// mismatch\n')
        result = self.configure(True, root=root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA256 mismatch', result.stdout + result.stderr)
        self.assertFalse((self.path / 'build' / 'h3-subblock-generated').exists())

    def test_missing_target_source_does_not_discard_other_feature(self):
        result = self.configure(True, copies=0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('found 0', result.stdout + result.stderr)

    def test_duplicate_target_source_fails_closed(self):
        result = self.configure(True, copies=2)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('found 2', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
