# H3 / RTX 4090 接续对照

更新：2026-10-13，Asia/Shanghai。目标交付：2026-10-14 晚。接续 `71cc80d6235dfb5c7326cce1613ebda54dd263b4`，不重做已锁定调研。**下表只填本机证据；缺项不补估算加速比。**

## 环境与实验契约

- 个人 `origin` / `codex/minimax-h3-cuda-optimizations`；独立 worktree `_scratch/h3-cuda-continuation`，保留原 main、已有报告及 upstream。
- 8×RTX4090 / driver 580.95.05；推理只用物理 GPU0（UUID 后缀 `894df955a273`）。每次启动复查全部计算/图形进程；只读检查不是资源预留，发现占用则拒绝。
- CPU 2×Xeon Gold6330，112 logical CPUs，RAM 499GiB；构建/推理CPU线程均显式限制6。系统盘初始可用66GiB，数据盘6.5TiB；未删除文件、停止其他进程或修改系统依赖。
- 工具链：CUDA12.2.140 / GCC12.4.0/ cuDNN9.24.0；固定 ggml `f270346d`、oneDNN `5e922403`、frontend `07b368a2`。baseline/candidate 使用相同显式 SM89/toolchain 配置，不复用旧 H200 二进制。
- **五个组件整文件 SHA256 5/5 与原锁一致**。补齐完整 Ref2VA（18,779,848,448 bytes），下载后重新整文件读回校验；保留本机 pruned、不替代精度锁。
- 新 baseline 源 `97cc585`，保留项目已有 CUDA 优化；目标 **864×480 /124帧 /24fps /20步 /seed42 /CFG1 /res_multistep+simple**，`preserve` 精度，DiT/TE/VAE 全卸载，`--max-vram 22`。22GiB 是图规划预算，非硬峰值上限。
- T2VA 与 Ref2VA mixed 各自锁输入、各自 baseline；先一次冷进程容量验证（warmup0/repeat1），再独立 warmup1/repeat3。单次CLI包括加载，不当作load-once稳态。
- T2VA 首轮原创固定 prompt：单镜头山涧、水流、轻风和远处鸟声，便于检查持续运动与自然音频；不是外部权威画质样例。语义盲评与CPU合成wiring数据分开。

## 对照表

| 方法/开关 | 4090 卸载资格/真实状态 | CLI E2E | DiT | 显存采样峰值 | 质量/功能 | 加速比 |
|---|---|---:|---:|---:|---|---:|
| 原有优化 baseline（全部新开关0） | 固定源码 SM89；一次容量验证通过 | 256.37s | 212.91s | 16,035MiB | 124帧/32k立体声音频完整；未语义验收 | 基准 |
| Fused ops | 单次完整T2VA通过，尚未稳定复测 | 299.45s | 256.34s | 16,351MiB | SSIM0.99351 / PSNR48.06dB / audio RMSE0 | 0.856×（慢16.8%） |
| AdaLN | 逐投影CPU传输/切图生命周期回归通过；生产卸载仍拒绝 | — | — | — | tiny H3 24/24逐位一致；未GPU卸载验收 | — |
| Cache-DiT | 继续保留卸载拒绝；中段delta跨切图存活/跳过传输未验收 | — | — | — | 未GPU验证 | — |
| SageAttention | SM89/tag/分段生命周期待验证；不能静默dense回退计成功 | — | — | — | 未GPU验证 | — |
| SubBlock | 主任务有界执行未完成，容量保护保留 | — | — | — | 未GPU验证 | — |
| Turbo | 匹配adapter+独立权重/步数baseline，不能用原Q4少跑步代替 | — | — | — | 部署证据不足 | — |
| VDN | exact solve/short-conv/anchor/完整checkpoint及CUDA部署未完成 | — | — | — | 生产入口拒绝 | — |
| VSA | trained gate/原生tile+compression部署未完成 | — | — | — | 生产入口拒绝 | — |

`—` = 未测，不是0。显存字段采用单卡只读定期采样（标明实际间隔/漏样），不冒充严格硬峰值。

## 本轮增量与复现

1. 新 `benchmark/h3_cuda/rtx4090_t2va.json` 与 `rtx4090_ref_mixed.json`：默认 baseline-only，预算/采样进入A/B契约；单卡UUID需填真实值。T2VA只要求其4个组件，不受未使用Ref2VA缺失阻塞。
2. workflow 增加 `--phase baseline-only`、允许warmup0；不调用candidate/compare、不产生虚假对比。UUID锁定运行同时检查计算与图形进程。
3. `scripts/h3_gpu_memory_monitor.py` 只读采样wrapper；使用workflow原有门禁/超时，不停止或修改任何进程。
4. 原一键CPU回归加入4090配置和显存采样测试；所有新增生产方法仍默认关闭、H3/CUDA限定。

本机未跟踪证据根：研究worktree内 `_scratch/rtx4090-20261013/`，含`model-verification.json`、`gpu-readonly-preflight.json`、`environment.sh`、`t2va-manifest.json`、构建/运行日志。模型、素材、工具、二进制不提交Git。

首轮（依赖/媒体工具准备完、核对配置后）：

```bash
source _scratch/rtx4090-20261013/environment.sh
python3 scripts/h3_cuda_workflow.py workflow \
  --manifest _scratch/rtx4090-20261013/t2va-manifest.json \
  --phase baseline-only --warmup 0 --repeats 1 --dry-run \
  --output-dir _scratch/rtx4090-20261013/t2va-plan
```

删除`--dry-run`之前须通过固定构建/全hash/单卡空闲门禁；输出目录必须全新。后续fused A/B使用`--phase all --warmup 1 --repeats 3`，不跨模式/精度/placement计算加速比。

GPU tiny验证：0号卡52/52数值对照通过（17,252元素），覆盖F32调制/门控、F32/F16 SwiGLU与重放/尾尺寸/视图；无CPU fallback。最大归一化误差0.177624（通过阈值≤1）。不代表完整H3、卸载或新增融合收益。

## 第一轮结果（单次容量，不是稳定收益）

- T2VA baseline：CLI 256.3718s，原生generation total226.286s；conditioning1.387s、DiT212.910s、decode10.613s。全hash/媒体校验不计入CLI时间。
- 单卡显存944次采样，采样最大间隔0.506s、错误0，峰值16,035MiB；不是严格硬峰值。
- 原生日志确认DiT按22GiB规划切为79段，真实CPU→CUDA参数传输；不是CPU推理或常驻模型。VAE tiling实际启用。
- 输出124帧、864×480、24fps，32kHz stereo，解码PCM每声道165,888样本，finite；已抽帧检查水流场景，音频语义/同步未听评。
- CUDA baseline与candidate构建均通过，共享配置SHA256一致，candidate production source构建前后指纹一致；含Sage编译不代表开启Sage。
- CPU CTest12/12、workflow/media53/53（无跳过）、来源18/18、LoRA5/5、build29/29、preflight56/56、cleanup29/29、4090配置14/14、显存7/7、GPU guard5/5、source稳定性2/2。
- AdaLN新helper逐投影真实预填/独立host存储/失败回滚已CPU验收；**未接通生产卸载**，保护条件保留。没有宣称释放投影权重或显存收益。

融合首次单次结果反而慢16.8%，**不宣称加速、不设为默认**。基线与候选均79个DiT图段；显存采样多316MiB。需全关候选控制组区分源码漂移与开关代价，再warmup/repeat复测；SSIM/PSNR不是语义质量验收。

## 混合模态独立 baseline

Ref2VA（图片＋带内嵌音频视频＋额外音频），同几何/步数/精度、全研究开关关闭：

| 次数 | CLI E2E | conditioning | DiT | decode | 显存采样峰值 | 完整媒体 |
|---|---:|---:|---:|---:|---:|---|
| warmup0 / repeat1 | 627.54s | 23.85s | 547.71s | 10.37s | 18,751MiB | 124帧/864×480/24fps、32k立体声、finite |

参考素材从本轮T2VA输出提取，记录父视频hash、转换命令、图片/视频/PCM完整hash；**不是外部真人素材，不当作跨来源语义/人物一致性验收**。与T2VA不计算跨模式加速比。显存1992次采样、最大间隔0.569s、错误0。其他首/末/首尾/视频独立模式仍待生产矩阵验证。

### 当前受控复测

有限串行队列 `_scratch/rtx4090-20261013/run_controlled_series.py`：等待mixed完成→同候选全关control→baseline预热1＋测3→fusion预热1＋测3→严格比较。每次先查单卡无进程，任一步失败即停止后续；状态在`controlled-series-status.json`。原始结果保留，不覆盖第一次单次结果。

融合回退待核对项（非已确定因果）：总开关强制RoPE tail非连续view，可能绕过旧concat连续/vec4路径；CUSTOM与旧MUL+ADD融合重合且可能影响原地buffer复用。两版79段、权重搬运次数/大小相同；优先全关control和重复，再分项诊断，不改量化/任务“改善”数字。
