"""CPU-only controller semantics; no GPU polling or long waits."""
import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));import h3_overnight as n
class Queue(unittest.TestCase):
 def tasks(self):return [dict(id='parity',status='pending',timeout_seconds=30),dict(id='vsa',status='pending',depends=['parity'],timeout_seconds=900),dict(id='mixed',status='pending',timeout_seconds=900)]
 def test_dependencies(self):
  t=self.tasks();self.assertEqual([x['id'] for x in n.ready_tasks(t)],['parity','mixed']);t[0]['status']='succeeded';self.assertEqual([x['id'] for x in n.ready_tasks(t)],['vsa','mixed'])
 def test_failed_parity_no_vsa(self):
  t=self.tasks();t[0]['status']='failed';self.assertTrue(n.skip_dependents(t));self.assertEqual(t[1]['status'],'skipped');self.assertEqual(t[2]['status'],'pending')
 def test_never_late_launch(self):self.assertFalse(n.enough_time({'timeout_seconds':900},929));self.assertTrue(n.enough_time({'timeout_seconds':900},930))
 def test_pin_changed(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'code';f.write_text('a');p={'deadline':'2026-10-15T09:00:00+08:00','poll_seconds':180,'input_hashes':{str(f):n.sha(f)},'tasks':[dict(id='x',command=['true'],timeout_seconds=20)]};n.validate_plan(p);f.write_text('b')
   with self.assertRaises(RuntimeError):n.validate_plan(p)
 def test_refuse_signal_other_group(self):
  class Proc:
   pid=12345
   def poll(self):return None
  with patch.object(n.os,'getpgid',return_value=42),patch.object(n.os,'killpg') as kill:
   with self.assertRaises(RuntimeError):n.stop_owned(Proc())
   kill.assert_not_called()
 def test_atomic_status(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'s.json';n.atomic(p,{'status':'pending'});n.atomic(p,{'status':'done'});self.assertEqual(json.loads(p.read_text())['status'],'done')
 def test_duplicate_or_unknown_dependency_rejected(self):
  for tasks in [[dict(id='x',command=['true'],timeout_seconds=1),dict(id='x',command=['true'],timeout_seconds=1)],[dict(id='x',command=['true'],timeout_seconds=1,depends=['missing'])]]:
   with self.assertRaises(AssertionError):n.validate_plan({'deadline':'2026-10-15T09:00:00+08:00','poll_seconds':180,'tasks':tasks,'input_hashes':{}})
 def test_deadline_always_triggers_report_without_gpu(self):
  obj=n.Night.__new__(n.Night);obj.p={'poll_seconds':180};obj.s={'tasks':self.tasks(),'ppt':{}};obj.renderer=type('R',(),{'wait':lambda self,timeout:0})();obj.remaining=lambda:-1;events=[];obj.save=lambda:None;obj.event=lambda label,**k:events.append(label);obj.launch_render=lambda reason:events.append('render:'+reason);obj.gpu_free=lambda:self.fail('GPU check must not run after deadline')
  with tempfile.TemporaryDirectory() as d:
   obj.root=Path(d);obj.run()
  self.assertEqual(obj.s['ppt']['status'],'succeeded');self.assertTrue(all(t['status']=='deadline' for t in obj.s['tasks']));self.assertTrue(any(e.startswith('render:') for e in events))
if __name__=='__main__':unittest.main()
