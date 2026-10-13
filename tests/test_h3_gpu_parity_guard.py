"""Shell admission tests only: fake nvidia-smi and fake test binary, no CUDA."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/test_h3_gpu_parity.sh'
UUID = 'GPU-01234567-89ab-cdef-0123-456789abcdef'


class ParityGuardTests(unittest.TestCase):
    def run_guard(self, scenario='idle', args=None):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            gpu = root/'nvidia-smi'
            gpu.write_text('''#!/usr/bin/env python3
import os,sys
u=os.environ['FAKE_UUID']; mode=os.environ['FAKE_GPU']; arg=' '.join(sys.argv[1:])
if '--query-gpu=' in arg:
 if mode=='bad_memory': print(u+', N/A, 0')
 elif mode=='used': print(u+', 1024, 0')
 elif mode=='busy': print(u+', 3, 99')
 elif mode=='wrong_uuid': print('GPU-ffffffff-ffff-ffff-ffff-ffffffffffff, 3, 0')
 else: print(u+', 3, 0')
elif '--query-compute-apps=' in arg:
 if mode=='compute': print('12345')
elif '-x' in sys.argv:
 if mode=='xml_bad': print('invalid')
 elif mode=='xml_missing': print('<nvidia_smi_log><gpu><uuid>'+u+'</uuid></gpu></nvidia_smi_log>')
 else:
  p='<process_info><pid>1234</pid><type>G</type></process_info>' if mode=='graphics' else ('N/A' if mode=='xml_na' else '')
  print('<nvidia_smi_log><gpu><uuid>'+u+'</uuid><processes>'+p+'</processes></gpu></nvidia_smi_log>')
else: sys.exit(2)
''')
            gpu.chmod(0o755)
            binary=root/'fake-parity'; binary.write_text('#!/bin/sh\necho MOCK_LAUNCHED_$CUDA_VISIBLE_DEVICES\n');binary.chmod(0o755)
            # Avoid a real 1s sleep in each fake telemetry check.
            sleep=root/'sleep';sleep.write_text('#!/bin/sh\nexit 0\n');sleep.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+':'+os.environ['PATH'],FAKE_UUID=UUID,FAKE_GPU=scenario)
            return subprocess.run(['bash',str(SCRIPT),*(args if args is not None else ['--run','--device',UUID,'--binary',str(binary)])],env=env,text=True,capture_output=True)

    def test_default_off(self):
        r=self.run_guard(args=[]);self.assertEqual(r.returncode,2);self.assertNotIn('MOCK_LAUNCHED',r.stdout)

    def test_idle_single_uuid_passes_guard(self):
        r=self.run_guard();self.assertEqual(r.returncode,0,r.stderr);self.assertIn('MOCK_LAUNCHED_'+UUID,r.stdout)

    def test_graphics_and_compute_are_busy(self):
        for scenario in ('graphics','compute','busy','used'):
            with self.subTest(scenario=scenario):
                r=self.run_guard(scenario);self.assertEqual(r.returncode,75,r.stderr);self.assertNotIn('MOCK_LAUNCHED',r.stdout)

    def test_missing_unknown_or_wrong_telemetry_fails_closed(self):
        for scenario in ('xml_bad','xml_missing','xml_na','wrong_uuid','bad_memory'):
            with self.subTest(scenario=scenario):
                r=self.run_guard(scenario);self.assertEqual(r.returncode,75,r.stderr);self.assertNotIn('MOCK_LAUNCHED',r.stdout)

    def test_multiple_devices_rejected(self):
        r=self.run_guard(args=['--run','--device','0,1','--binary','/not-used']);self.assertEqual(r.returncode,2)


if __name__ == '__main__':
    unittest.main()
