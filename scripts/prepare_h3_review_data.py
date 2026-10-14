#!/usr/bin/env python3
"""Build traceable report inputs from frozen measurements; CPU/media only."""
import argparse,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name):return read(ROOT/'benchmark/h3_cuda'/name)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--night-state',type=Path);p.add_argument('--label',default='阶段实测');a=p.parse_args();out=a.output.resolve();out.parent.mkdir(parents=True,exist_ok=True);assets=out.parent/'assets';assets.mkdir(exist_ok=True)
 four=load('four-methods-results.json');sf=load('subblock-fix-results.json');s25=load('subblock-keep25-results.json');ad=load('adaln-fix-results.json');lo=load('lora-results.json')
 results={}
 baseline=read(ROOT/'_scratch/h3-subblock-fix-20261014/t2va-once/baseline/report.json');b=baseline['modes']['t2va'][0]
 results['baseline_t2va']={'cli_s':b['wall_ms']/1000,'dit_s':b['profile']['summaries'][0]['diffusion_ms']/1000,'n':1,'note':'原Q4；20步；已有优化保留'}
 for key,method in [('cache','cache'),('cache_executor','cache-no-reuse')]:
  x=four['methods'][method];q=x['quality'];s=x['statistics_ms'];results[key]={'cli_s':s['cli_wall_ms']['mean']/1000,'dit_s':s['diffusion_ms']['mean']/1000,'base_cli_s':254.148,'n':3,'ssim':None,'audio_rmse':None,'note':'原20步；3次均值，独立旧实验组'}
  # Use comparison reports for unambiguous quality fields.
  rpt=read(ROOT/four['evidence_root']/str(x['report'])) if not Path(x['report']).is_absolute() else read(x['report'])
  media=next(z for z in rpt['modes']['t2va'] if not z['warmup'])
  results[key]['report']=str(x['report']);results[key]['output']=media['output']
 results['cache'].update(ssim=.97613,audio_rmse=.00604,hits=6,steps=20,front=2,back=2,threshold=.12)
 results['cache_executor'].update(ssim=1.,audio_rmse=0.,threshold=0.)
 for key,x in [('subblock50',sf['cases']['t2va']),('subblock25',s25['cases']['t2va'])]:
  if key=='subblock50':q=x['quality_vs_dense'];cs=x['fixed_cli_ms']/1000;ds=x['fixed_dit_ms']/1000;ssim=q['video']['mean_ssim'];rmse=q['audio']['rmse'];bs=x['baseline_cli_ms']/1000
  else:q=x['quality_vs_dense']['keep25'];cs=x['keep25_cli_seconds'];ds=x['keep25_dit_seconds'];ssim=q['ssim'];rmse=q['audio_rmse'];bs=x['baseline_cli_seconds']
  results[key]={'cli_s':cs,'dit_s':ds,'ssim':ssim,'audio_rmse':rmse,'base_cli_s':bs,'n':1,'note':'仅SubBlock，非视频模态保护；不叠加cache'}
 for key,lk in [('larry4','larry4'),('light4','lightx2v4')]:
  x=lo['cases'][lk];q=x['quality_distance_vs_dense'];results[key]={'cli_s':x['wall_ms']/1000,'dit_s':x['diffusion_ms']/1000,'ssim':q['ssim'],'audio_rmse':q['audio']['rmse'],'base_cli_s':lo['baseline']['wall_ms']/1000,'n':1,'output':x['output'],'note':'新LoRA模型＋4步Euler；不是同权重无损加速'}
 ar=ad['candidate_mixed_completed_measured_runs'];results['adaln']={'cli_s':None,'dit_s':None,'base_cli_s':ad['baseline_mixed_statistics']['ref_mixed']['cli_wall_ms']['mean']/1000,'samples_s':[x['wall_ms']/1000 for x in ar],'dit_samples_s':[x['diffusion_ms']/1000 for x in ar],'n':len(ar),'note':'修复版mixed：两次已完成；重复验收按用户要求停止。每个candidate各匹配一个baseline产物；baseline自身不完全相同。','profile_allocation_s':[80.673,173.569,34.333],'profile_labels':['baseline','旧AdaLN','修复版'],'profile_segments':[91,91,10],'profile_note':'诊断计时，不与正式CLI相混'}
 media={'baseline':b['output'],'cache':results['cache']['output'],'subblock25':s25['cases']['t2va']['output'],'larry4':results['larry4']['output'],'light4':results['light4']['output']}
 pairs={'cache_compare.png':[('Baseline / 20 steps',media['baseline']),('Cache-DiT / 20 steps',media['cache'])],
 'subblock_compare.png':[('Baseline / 20 steps',media['baseline']),('Keep 50%',sf['cases']['t2va']['output']),('Keep 25%',media['subblock25'])],
 'lora_compare.png':[('Baseline / 20 steps',media['baseline']),('Larry / 4 steps',media['larry4']),('LightX2V / 4 steps',media['light4'])]}
 ab=read(ROOT/'_scratch/h3-adaln-fix-20261014/accept-mixed/baseline/report.json');pairs['adaln_compare.png']=[('Mixed baseline',ab['modes']['ref_mixed'][1]['output']),('AdaLN repaired',ar[0]['output'])]
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
 for name,items in pairs.items():
  dest=assets/name
  if dest.exists():continue
  im=Image.new('RGB',(1296,286*len(items)),'white');draw=ImageDraw.Draw(im)
  for row,(label,file) in enumerate(items):
   raw=subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-i',file,'-an','-vf',"select='eq(n,0)+eq(n,61)+eq(n,123)',scale=432:240",'-fps_mode','vfr','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],check=True,stdout=subprocess.PIPE,timeout=60).stdout;n=432*240*3
   if len(raw)!=3*n:raise ValueError('Unexpected frame count')
   draw.text((8,row*286+5),label+'   |   frame 0 / 61 / 123',font=font,fill='#18232B')
   for col in range(3):im.paste(Image.frombytes('RGB',(432,240),raw[col*n:(col+1)*n]),(432*col,row*286+36))
  im.save(dest)
 night=read(a.night_state) if a.night_state and a.night_state.exists() else {'tasks':[],'status':'not_started'}
 for task in night.get('tasks',[]):
  if task.get('status')=='succeeded' and task.get('report'):
   p=Path(task['report']);d=read(p)
   if d.get('status')!='validated_model_variant_single_run':raise ValueError('night succeeded without validated report')
   q=d.get('quality_vs_dense',{});summ=d['profile']['summaries'][0]
   task['result']={'cli_s':d['wall_ms']/1000,'dit_s':summ['diffusion_ms']/1000,'ssim':q.get('video',{}).get('mean_ssim'),'NFE':summ['calls'],'output':d['output'],'note':'单次新模型实验；不与不同模式横比'}
 data={'schema_version':1,'as_of':datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M'),'status_label':a.label,'deadline':'2026-10-15 09:00 Asia/Shanghai','setup':{'gpu':'RTX4090 × 1','geometry':'864×480 · 124帧 · 24fps','sampling':'seed42 · CFG1 · 原20步RES/simple；LoRA4步Euler/simple','precision':'原混合Q4 DiT/Qwen；视频VAE FP16，音频VAE FP32','placement':'DiT/TE/VAE卸载 · 22GiB规划预算 · CPU6线程','baseline':'97cc585，保留原项目已有优化；不把新增开关全关理解为撤销原优化','controls':'同输入/seed/精度/卸载；单项独立；CLI含加载/hash/封装，外层hash和媒体复验不计时'},'results':results,'assets':{name:str(assets/name) for name in pairs},'media':media,'overnight':night,'sources':{name:{'path':str(ROOT/'benchmark/h3_cuda'/name),'sha256':sha(ROOT/'benchmark/h3_cuda'/name)} for name in ['four-methods-results.json','subblock-fix-results.json','subblock-keep25-results.json','adaln-fix-results.json','lora-results.json']},'quality_note':'SSIM衡量与baseline距离，非画质百分比；LoRA改变轨迹。未完成主观听评/高速运动/口型验收。','negative_results':'Sage和新增融合组合仍退化；不纳入有效提速宣传。','report_scope':'独立实验结果不能连乘；各表baseline已注明，单次不冒充统计稳定性。'}
 out.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');print(out)
if __name__=='__main__':main()
