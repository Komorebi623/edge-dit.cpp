import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('h3_memory', Path(__file__).resolve().parents[1] / 'scripts/h3_gpu_memory_monitor.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
UUID = 'GPU-01234567-89ab-cdef-0123-456789abcdef'


class MemoryMonitorTests(unittest.TestCase):
    def invoke(self, text):
        return lambda *a, **k: subprocess.CompletedProcess(a, 0, text, '')

    def test_sample(self):
        s = m.sample(UUID, self.invoke(f'{UUID}, 123, 24564, 7\n'))
        self.assertEqual(s['memory_used_mib'], 123)

    def test_telemetry_rejects_invalid(self):
        for text in ('', f'{UUID}, N/A, 24564, 0', f'{UUID}, nan, 24564, 0',
                     f'{UUID}, 25000, 24564, 0', f'{UUID}, 1, 24564, 101',
                     f'{UUID}, -1, 24564, 0', f'{UUID}, 0, 0, 0',
                     f'GPU-other, 1, 24564, 0', f'{UUID}, 1, 24564, 0\n{UUID}, 1, 24564, 0'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                m.sample(UUID, self.invoke(text))

    def test_summary_is_sampled_not_exact(self):
        s = m.summarize([{'elapsed_seconds':0,'memory_used_mib':3},
                         {'elapsed_seconds':0.6,'memory_used_mib':23400}], [], .25)
        self.assertEqual(s['sampled_peak_used_mib'], 23400)
        self.assertEqual(s['max_sample_gap_seconds'], .6)
        self.assertTrue(s['telemetry_complete'])
        self.assertIn('not exact', s['measurement'])

    def test_empty_and_errors_are_not_complete(self):
        self.assertIsNone(m.summarize([], [], .25)['sampled_peak_used_mib'])
        self.assertFalse(m.summarize([], [], .25)['telemetry_complete'])
        self.assertFalse(m.summarize([{'elapsed_seconds':0,'memory_used_mib':1}], ['error'], .25)['telemetry_complete'])

    def test_fail_before_launch_if_no_telemetry(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(m, 'sample', side_effect=ValueError('no GPU')), mock.patch.object(m.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                m.monitor(['not-launched'], UUID, Path(d)/'new')
            run.assert_not_called()

    def test_refuse_existing_output(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileExistsError):
            m.monitor(['not-launched'], UUID, Path(d))

    def test_bad_args(self):
        for uuid, interval, command in [('0',.25,['echo']), (UUID,0,['echo']), (UUID,float('nan'),['echo']), (UUID,.25,[])]:
            with self.assertRaises(ValueError):
                m.monitor(command,uuid,'/unused',interval)


if __name__ == '__main__':
    unittest.main()
