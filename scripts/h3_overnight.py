#!/usr/bin/env python3
"""Finite, single-GPU overnight queue with a hard reporting deadline.

Never stops external processes. Only explicitly spawned process groups may be
terminated on timeout/deadline/conflict. Busy/unknown telemetry never launches.
Reports use completed measurements only; no automatic parameter tweaks/rebuilds.
"""
import argparse,copy,fcntl,hashlib,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import h3_cuda_workflow as w
TERMINAL={'succeeded','failed','skipped','deadline','timeout','integrity_failed'}
TZ=ZoneInfo('Asia/Shanghai')
def now():return datetime.now(TZ)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(path,data):
 p=Path(path);tmp=p.with_suffix('.writing');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def ready_tasks(tasks):
 by={t['id']:t for t in tasks}
 return [t for t in tasks if t['status']=='pending' and all(by[d]['status']=='succeeded' for d in t.get('depends',[]))]
def skip_dependents(tasks):
 by={t['id']:t for t in tasks};changed=False
 for t in tasks:
  if t['status']=='pending' and any(by[d]['status'] in TERMINAL-{'succeeded'} for d in t.get('depends',[])):
   t.update(status='skipped',reason='前置数值验证未通过，禁止运行模型');changed=True
 return changed
def enough_time(task,seconds):return seconds>=task['timeout_seconds']+30
def stop_owned(proc):
 if proc.poll() is not None:return
 # start_new_session guarantees PGID==PID. Revalidate before any group signal.
 if os.getpgid(proc.pid)!=proc.pid:raise RuntimeError('Refusing signal: child process group ownership changed')
 os.killpg(proc.pid,signal.SIGTERM)
 try:proc.wait(timeout=10)
 except subprocess.TimeoutExpired:
  if os.getpgid(proc.pid)!=proc.pid:raise RuntimeError('Refusing changed child PGID')
  os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
def validate_plan(p):
 deadline=datetime.fromisoformat(p['deadline']);assert deadline.tzinfo
 ids=[t['id'] for t in p['tasks']];assert len(set(ids))==len(ids)
 for t in p['tasks']:
  assert t['timeout_seconds']>0 and t['command'] and all(isinstance(x,str) for x in t['command'])
  assert set(t.get('depends',[]))<=set(ids)-{t['id']}
 assert p['poll_seconds']>=30
 for filename,digest in p['input_hashes'].items():
  if sha(filename)!=digest:raise RuntimeError('Frozen queue input changed: '+filename)
 return deadline
class Night:
 def __init__(self,plan_path):
  self.plan_path=Path(plan_path).resolve();self.p=json.loads(self.plan_path.read_text());self.deadline=validate_plan(self.p)
  self.root=Path(self.p['output_root']);self.root.mkdir(parents=True,exist_ok=True)
  self.state_path=self.root/'status.json'
  if self.state_path.exists():raise RuntimeError('Refusing to restart an existing queue automatically; inspect prior PID/results first')
  self.s={'schema_version':1,'status':'starting','pid':os.getpid(),'started':now().isoformat(),'deadline':self.deadline.isoformat(),'plan':str(self.plan_path),'plan_sha256':sha(self.plan_path),'poll_seconds':self.p['poll_seconds'],'tasks':copy.deepcopy(self.p['tasks']),'last_poll':None,'ppt':{'status':'pending'}}
  for t in self.s['tasks']:t.update(status='pending',attempts=0)
  self.last_event=None;self.renderer=None;self.active_child=None;self.save()
 def save(self):atomic(self.state_path,self.s)
 def event(self,label,**fields):
  self.s['status']=label;self.save()
  key=json.dumps([label,fields],sort_keys=True)
  if key!=self.last_event:
   with (self.root/'events.jsonl').open('a') as f:f.write(json.dumps({'at':now().isoformat(),'event':label,**fields},ensure_ascii=False)+'\n')
   self.last_event=key
 def remaining(self):return (self.deadline-now()).total_seconds()
 def launch_render(self,reason):
  if self.renderer is not None:return
  self.s['ppt']={'status':'starting','reason':reason,'started':now().isoformat()};self.save()
  log=(self.root/'ppt-build.log').open('xb')
  env=dict(os.environ);env['H3_NIGHT_STATUS']=str(self.state_path)
  self.renderer=subprocess.Popen(self.p['render_command'],cwd=self.p['repo_root'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
  log.close();self.s['ppt'].update(status='running',pid=self.renderer.pid);self.save()
 def gpu_free(self):
  try:
   snap=w.gpu_precheck(self.p['gpu']);self.s['last_poll']={'at':now().isoformat(),'free':True,'snapshot':snap};self.save();return True
  except Exception as e:
   self.s['last_poll']={'at':now().isoformat(),'free':False,'reason':str(e)};self.save();return False
 def task(self,t):
  validate_plan(self.p) # all code/config/binary pins before each launch
  t['attempts']+=1;directory=self.root/(t['id']+f'-attempt{t["attempts"]:02d}');directory.mkdir()
  t.update(status='running',started=now().isoformat(),directory=str(directory))
  cmd=[x.replace('{output}',str(directory/'result')).replace('{memory}',str(directory/'memory')) for x in t['command']]
  env=dict(os.environ);env.update(t.get('env',{}));t['executed_command']=cmd;self.event('running',task=t['id'])
  with (directory/'process.log').open('xb') as log:
   proc=subprocess.Popen(cmd,cwd=self.p['repo_root'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   self.active_child=proc;t['pid']=proc.pid;self.save();start=time.monotonic();reason=None
   next_resource_check=start+15
   while proc.poll() is None:
    if self.remaining()<=0:
     reason='deadline';stop_owned(proc);break
    if time.monotonic()-start>t['timeout_seconds']:
     reason='timeout';stop_owned(proc);break
    if time.monotonic()>=next_resource_check:
     # If another user starts using the selected GPU, yield only our own job.
     # Read PIDs only, never inspect or signal unrelated processes.
     try:
      rows=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True).stdout
      for row in rows.splitlines():
       uuid,pid=[x.strip() for x in row.split(',')]
       if uuid!=self.p['gpu']['gpu_uuid']:continue
       try: group=os.getpgid(int(pid))
       except ProcessLookupError: continue
       if group!=proc.pid:reason='resource_conflict';break
     except Exception:reason='telemetry_unavailable'
     if reason:stop_owned(proc);break
     next_resource_check=time.monotonic()+15
    time.sleep(min(3,max(.05,self.remaining())))
   code=proc.wait();self.active_child=None;t.update(returncode=code,ended=now().isoformat(),elapsed_seconds=time.monotonic()-start)
  report=directory/'result/report.json'
  if reason:t.update(status=reason if reason in ('deadline','timeout') else 'failed',reason='自有任务已停止：'+reason+'；未触碰其他进程')
  elif code==0:
   if t['kind']=='model':
    if not report.exists():t.update(status='failed',reason='缺少结果报告')
    else:
     data=json.loads(report.read_text());ok=data.get('status')=='validated_model_variant_single_run'
     t.update(status='succeeded' if ok else 'failed',report=str(report),report_sha256=sha(report))
   else:
    text=(directory/'process.log').read_text(errors='replace')
    t.update(status='succeeded' if t['success_text'] in text else 'failed',log=str(directory/'process.log'))
  else:
   text=(directory/'process.log').read_text(errors='replace');failure=json.loads(report.read_text()).get('error','') if report.exists() else text
   if (code==75 or 'occupied' in failure or '占用' in failure) and t['attempts']<3:
    t.update(status='pending',reason='launch precheck detected resource conflict; no unchecked retry')
   else:t.update(status='failed',reason=failure[-1600:],report=str(report) if report.exists() else None)
  self.event('task_finished',task=t['id'],status=t['status']);skip_dependents(self.s['tasks']);self.save()
 def run(self):
  lock=(self.root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   while True:
    skip_dependents(self.s['tasks']);pending=[t for t in self.s['tasks'] if t['status'] not in TERMINAL]
    if not pending:self.launch_render('所有夜间任务已完成或明确失败');break
    if self.remaining()<=0:
     for t in pending:t.update(status='deadline',reason='截止前无可用资源或剩余时间不足')
     self.save();self.launch_render('北京时间10月15日09:00截止，按已有证据出报告');break
    eligible=[t for t in ready_tasks(self.s['tasks']) if enough_time(t,self.remaining())]
    if not eligible:
     self.event('waiting_deadline',reason='不启动无法在截止前完成的任务');time.sleep(min(self.p['poll_seconds'],max(0,self.remaining())));continue
    if not self.gpu_free():
     self.event('waiting_gpu',reason='GPU忙或遥测不明确；无推理启动');time.sleep(min(self.p['poll_seconds'],max(0,self.remaining())));continue
    self.task(eligible[0])
    # Let occupancy telemetry settle, without weakening the next preflight.
    time.sleep(min(3,max(0,self.remaining())))
   self.event('building_report')
   try:code=self.renderer.wait(timeout=900)
   except subprocess.TimeoutExpired:stop_owned(self.renderer);code=124
   self.s['ppt'].update(status='succeeded' if code==0 else 'failed',returncode=code,ended=now().isoformat())
   self.event('complete' if code==0 else 'report_failed')
  except BaseException as e:
   if self.active_child is not None and self.active_child.poll() is None:stop_owned(self.active_child)
   self.s['error']=str(e);self.event('controller_failed')
   # Preserve completed evidence and attempt a report even on a queue error.
   try:
    self.launch_render('队列异常；只报告已验证数据');code=self.renderer.wait(timeout=900)
    self.s['ppt'].update(status='succeeded' if code==0 else 'failed',returncode=code);self.save()
   except Exception as render_error:self.s['ppt'].update(status='failed',error=str(render_error));self.save()
   raise
  finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
 if a.check_only:
  plan=json.loads(a.plan.read_text());print('VALID',validate_plan(plan).isoformat(),len(plan['tasks']));return
 Night(a.plan).run()
if __name__=='__main__':main()
