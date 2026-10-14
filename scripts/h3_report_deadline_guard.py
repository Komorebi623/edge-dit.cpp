#!/usr/bin/env python3
"""Independent CPU reporting fallback; never starts/stops GPU processes."""
import argparse,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
TZ=ZoneInfo('Asia/Shanghai')
def live(pid):
 try:
  stat=Path(f'/proc/{pid}/stat').read_text();return stat.split(') ')[1][0]!='Z'
 except (OSError,IndexError):return False
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());deadline=datetime.fromisoformat(plan['deadline']);root=Path(plan['output_root']);state=root/'status.json'
 guardian=root/'report-guardian.json';guardian.write_text(json.dumps({'pid':os.getpid(),'deadline':deadline.isoformat(),'status':'watching'},indent=2)+'\n')
 while True:
  now=datetime.now(TZ)
  if state.exists():
   try:s=json.loads(state.read_text())
   except (OSError,ValueError):s={}
  else:s={}
  ppt=s.get('ppt',{});controller_alive=live(s.get('pid',-1))
  if ppt.get('status')=='succeeded':
   guardian.write_text(json.dumps({'pid':os.getpid(),'status':'main_report_verified','ended':now.isoformat()},indent=2)+'\n');return
  if now>=deadline or (s and not controller_alive):break
  time.sleep(min(30,max(.1,(deadline-now).total_seconds())))
 # A legitimate active renderer gets a short grace; if it dies the fallback
 # writes to a DIFFERENT directory and never overwrites partial user artifacts.
 for _ in range(30):
  if not state.exists():break
  try:s=json.loads(state.read_text())
  except ValueError:break
  ppt=s.get('ppt',{})
  if ppt.get('status')=='succeeded':return
  if not live(ppt.get('pid',-1)):break
  time.sleep(10)
 cmd=list(plan['render_command']);cmd[2]=str(Path(cmd[2]).with_name('deadline-fallback-report'))
 with (root/'deadline-fallback.log').open('xb') as log:
  result=subprocess.run(cmd,cwd=plan['repo_root'],stdout=log,stderr=subprocess.STDOUT,timeout=900)
 guardian.write_text(json.dumps({'pid':os.getpid(),'status':'fallback_succeeded' if result.returncode==0 else 'fallback_failed','returncode':result.returncode,'directory':cmd[2],'ended':datetime.now(TZ).isoformat()},indent=2)+'\n')
 if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
