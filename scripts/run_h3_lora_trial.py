#!/usr/bin/env python3
"""Single H3 LoRA generation using existing workflow safety/media helpers.

Reuses a frozen dense baseline; explicitly a changed-model/NFE experiment,
not a same-weights optimization comparison. No warmups, sweeps, or baseline runs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone
import h3_cuda_workflow as w


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--adapter-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=w.MODE_ORDER,help='select one mode from an existing multi-mode functional baseline')
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False);report=a.output/'report.json'
    r={'schema_version':1,'kind':'h3-lora-model-variant-trial','status':'preflight','created_utc':datetime.now(timezone.utc).isoformat(),'baseline_report':str(a.baseline.resolve()),'baseline_report_sha256':w.sha256(a.baseline),'warmup':0,'repeats':1,'comparison_scope':'model + NFE + sampler variant; same original Q4 bytes/input/geometry/seed/placement; no claim of lossless same-weights speedup'}
    def save():report.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n')
    save()
    try:
        base=w.frozen_report(a.baseline);w.require(base['phase']=='baseline','frozen dense baseline required')
        mode_order=base['mode_order'];w.require(a.mode in mode_order if a.mode else len(mode_order)==1,'select one mode present in baseline')
        mode=a.mode or mode_order[0];r['mode']=mode
        manifest=base['manifest'];recipe=w.read_json(a.adapter_manifest)
        r['gpu_initial_precheck']=w.gpu_precheck(manifest['run']);save()
        adapter=a.adapter_manifest.resolve().parent/recipe['adapter_file']
        w.require(recipe['kind']=='minimax-h3-dynamic-lora' and recipe['steps'] in (4,8),'unsupported recipe')
        w.require(w.sha256(adapter)==recipe['adapter_sha256'],'adapter checksum mismatch')
        r['identities']=w.verify_inputs(manifest);r['recipe']=recipe;r['adapter_manifest_sha256']=w.sha256(a.adapter_manifest)
        task=manifest['modes'][mode];w.require(r['identities']['models'][task['dit']]['sha256']==recipe['base_sha256'],'adapter does not match baseline base file')
        w.require(task['dit']==recipe['base_partition'],'FL/Ref partition mismatch')
        r['sampling']={**manifest['sampling'],'steps':recipe['steps'],'cfg_scale':recipe['cfg_scale'],'sampler':'euler','scheduler':'simple'};r['placement']=manifest['placement'];r['prompt']=task['prompt']
        env,removed=w.isolated_env(manifest,'baseline');env.update(ED_H3_TURBO='1',ED_H3_LORA_MANIFEST=str(a.adapter_manifest.resolve()),CUDA_VISIBLE_DEVICES=manifest['run']['gpu_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID')
        if recipe.get('payload_format')=='fastvideo-vsa-v1': env['ED_H3_VSA']='1'
        r['optimization_env']={k:v for k,v in env.items() if k.startswith(('ED_','GGML_'))};r['removed_inherited_keys']=removed
        before=w.build_provenance(a.binary.resolve(),lock_runtime=True,runtime_env=env);r['build']=before
        previous=next(z for z in base['modes'][mode] if not z['warmup']);cmd=list(previous['command']);cmd[0]=str(a.binary.resolve());output=a.output/(mode+'.mp4')
        for key,value in [('--steps',recipe['steps']),('--cfg-scale',recipe['cfg_scale']),('--sampler','euler'),('--scheduler','simple'),('--output',output.resolve())]:cmd[cmd.index(key)+1]=str(value)
        r['command']=cmd;r['gpu_precheck']=w.gpu_precheck(manifest['run']);r['status']='running';save();start=time.perf_counter()
        result=subprocess.run(cmd,env=env,cwd=Path(__file__).resolve().parents[1],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=manifest['run']['timeout_seconds'])
        r['wall_ms']=(time.perf_counter()-start)*1000;r['returncode']=result.returncode
        (a.output/'stdout.log').write_bytes(result.stdout);log=a.output/'stderr.log';log.write_bytes(result.stderr)
        w.require(result.returncode==0,'generation failed; see stderr.log')
        text=result.stderr.decode(errors='replace');w.require('dynamic LoRA loaded:' in text,'runtime adapter not attested')
        if recipe.get('payload_format')=='fastvideo-vsa-v1':
            w.require('VSA trained gates loaded:50 BF16,tile64' in text,'runtime VSA trained gates not attested')
        r['profile']=w.parse_profiles(text,log)
        r['output']=str(output.resolve());r['media']=w.validate_media(output,r['sampling'],manifest['run']['media_timeout_seconds'],a.output)
        r['quality_vs_dense']={'video':w.video_metrics(Path(previous['output']),output,r['sampling']['frames'],manifest['run']['media_timeout_seconds'],a.output),'audio':w.audio_errors(Path(previous['media']['audio_pcm']),Path(r['media']['audio_pcm'])),'note':'Changed distilled model; distance to dense is not a perceptual quality score'}
        after=w.build_provenance(a.binary.resolve(),lock_runtime=True,runtime_env=env);w.require(before==after,'binary/shared library changed during run')
        w.require(w.sha256(a.baseline)==r['baseline_report_sha256'],'baseline report changed')
        w.require(w.sha256(a.adapter_manifest)==r['adapter_manifest_sha256'] and w.sha256(adapter)==recipe['adapter_sha256'],'adapter changed during run')
        r['dense_baseline_wall_ms']=previous['wall_ms'];r['speedup_vs_dense']=previous['wall_ms']/r['wall_ms'];r['status']='validated_model_variant_single_run';save()
        print(json.dumps({'status':r['status'],'seconds':r['wall_ms']/1000,'speedup':r['speedup_vs_dense'],'ssim':r['quality_vs_dense']['video']['mean_ssim'],'report':str(report)},ensure_ascii=False),flush=True)
    except Exception as e:r.update(status='failed',error=str(e));save();raise

if __name__=='__main__':main()
