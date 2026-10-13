# H3：RTX 4090服务器接续计划

更新：2026-10-13。先读`h3-coordination-handoff.md`；分支`codex/minimax-h3-cuda-optimizations`，个人origin为Komorebi623仓库。目标是继承已实现研究和本轮修复，不重新从头调查。

## 1. 实验契约

- 原始优化baseline `97cc585`已有融合和fast paths；保留它们，仅增量开研究方法。
- 新目标：**864×480、124帧、24fps、20步、seed42、CFG1、res_multistep/simple**。先T2VA验证容量和时延，再Ref2VA mixed；两种模式分别锁输入、分别baseline，不跨模式计算加速比。
- 原完整模型混合Q4精度锁保持：两DiT、Qwen、FP16 video VAE、FP32 audio VAE，全部完整SHA256重新核验。不开auto-fit、不转全BF16、不开pruned/LoRA/distilled来缩时。
- 4090需要权重卸载，placement变化另立baseline；源commit、GPU UUID、精度、输入顺序/hash、采样、卸载预算、环境与binary/source hash都记录。

## 2. 已知容量／时延，不混淆实测与估算

本项目`docs/minimax-h3.md`有RTX4090 24GB实测：864×480、**56帧**、20步、Q4、所有组件offload，T2VA进程E2E117s、DiT81.466s、峰值18630MiB；首尾帧E2E140s。另一Q8测试冷进程290s、DiT91s，加载口径与存储不能忽略。

124帧latent17→37，线性算子约2.18×，attention可能约4.74×；据此Q4目标T2VA初步规划：生成约3–7min，冷进程约5–10min，保守12min；mixed仅预留10–15min。**全部为估计，目标124帧尚未实测，不能保证24GB足够，也不能用它填performance_results。** CPU大量RAM、PCIe、存储和offload实现会改变结果。

一次warmup＋三次measurement可先预留20–50min。现有workflow每次启动新CLI进程，不是load-once稳态；不要混用这两类计时。第一轮先一个请求确认容量和正确性，不直接运行全矩阵/组合优化。

## 3. 当前不能直接启用的组合

Cache-DiT/AdaLN/SubBlock/Sage目前拒绝DiT offload/parallel graph cuts及persistent graph reuse；4090搬迁不会自动改变这一点。

- **保留拒绝**，不得只去掉gate便宣称卸载支持。
- SubBlock逐head/Q64展开主任务超过图容量，已加compute/attention前预算拒绝；需原生／批量化有界执行，不简单提高MAX_GRAPH_SIZE。
- VDN exact solve/short-conv/anchor/full checkpoint与VSA trained gate/native tile/compression未完成，生产入口拒绝；Turbo需真实公开匹配adapter，独立baseline。
- fused部分预计较容易保留在offloaded图中，但CUDA数值／支持路径仍须测。

## 4. 新服务器准备（不假定环境）

1. 先核对SSH地址和可用权限；只读查GPU UUID/显存/计算进程、CPU/RAM、磁盘、CUDA/driver/cuDNN、编译器、ffmpeg/ffprobe、Python。GPU有进程或不是独占时不推理、不停止服务。
2. 克隆／更新个人研究分支，记录实际HEAD；固定ggml/oneDNN gitlink，初始化不升级。cuDNN frontend按交接完整commit取本地clean checkout。
3. 找现有五个模型文件，根路径可以变，内容不能变；整文件SHA256必须对精度锁。无共享时迁移已知文件，不自动下载数十GiB未知权重。
4. CPU一键回归`bash scripts/test_h3_cpu.sh`。新机不要盲用旧H200的`g++_old`或container名；选真实兼容编译器。设置TMPDIR为用户可写充足磁盘，不删文件。
5. CUDA构建显式本地设置：`H3_CUDA_ARCHITECTURES=89`、`H3_CUDA_ROOT`、`H3_CUDNN_ROOT`、`H3_CUDNN_FRONTEND_SOURCE`；生成器可Ninja/Unix Makefiles。`build_h3_cuda.sh all --dry-run`→`--validate`→构建，baseline/candidate一致设置、全新_scratch输出；H200旧二进制不搬来执行。

## 5. Phase A：先恢复offloaded baseline

- 从现有manifest派生**独立4090 manifest**：新几何、新binary路径、新素材与hash；placement设置DiT/TE/VAE offload，所有新开关关闭。当前`min_free_mib=110000`是旧H200阈值，**不能沿用到4090**；新阈值和工作区余量按24GB独占容量安排，不虚构峰值。
- CLI复用原有`--type preserve`、`--dit-offload`、`--text-encoder-offload`、`--vae-offload`、`--max-vram`路径，H3固定16×16 VAE tiling。此前24GB记录用22GiB planning budget，仅供起点；先确认最小图段与外部workspace峰值。不要以`--max-vram`当万能硬cap。
- 现有workflow很重：首次做dry-run检查，再安排单次baseline。锁hash不等于run；完整质量素材与合成wiring测试分别记录。
- 测视频124帧/864×480/24fps、全部PCM采样率/声道/长度/finite；保留生成结果、失败退出码/日志与完整配置。完成后才warmup＋repeat和全模态矩阵。

## 6. Phase B：卸载兼容与增量优化

1. **融合**：先实际CUDA parity及日志确认新增调用，不把已有SwiGLU等重复计功。
2. **AdaLN**：固定schedule/分支/投影真实读取，明确参数是在CPU还是GPU及其生命周期；不允许缓存绑定在graph-cut释放的buffer上。投影预填、weights identity、预算、repeat/CFG/reset都做CPU回归和GPU验收，不能沿用resident guard直接放行。
3. **Cache-DiT**：确保front/RF/middle delta在多次graph-cut间独立有效，保持同锚和逐模态guard；命中路径真的跳过中段参数传输/计算。先CPU模拟传输/分段/失败清理，再GPU证明。
4. **Sage**：节点应在单个合规segment CUDA执行，局部buffer和H3SA tag不能因切图失效；不传播给别模型/backend。4090 SM89编译与数值单独验证，禁止退回dense却报告启用成功。
5. 先逐项，验收后组合；SubBlock继续有界工程；VDN/VSA/Turbo仍独立模型部署路线。以上顺序可按审查结果调整，不捏造收益。

交付每项：默认关闭、H3/CUDA隔离、CPU负例、CUDA编译、真实GPU功能／质量／性能／峰值、简洁对照表；没测的字段保持null。

## 7. GPU仍忙时

继续CPU数学与图组件、卸载生命周期/错误回归、VDN/VSA缺口开发。完整CPU生成可独立smoke：保持权重精度、用小合法输入，先测阶段耗时再决定是否跑目标；不把8核synthetic时间或CPU算法收益当CUDA结论，不扩展生产开关到CPU。

## 8. 迁移材料与Mac侧交接

Mac侧只需让新Codex连接目标服务器并阅读Git内交接/状态/精度锁/validation/本计划。实际SSH主机、用户和认证由用户或现有SSH配置提供，不能猜或把凭据写Git；不需要恢复旧聊天。

可另行复制约1MiB旧机`_scratch/h3-server-20261013/fixtures/`与locked manifest，以`--asset-root`重映射；不复制则新fixtures重新锁、重新baseline。模型、二进制、source snapshots、原始日志不在Git；reference-lock支持按固定commit重新获取。
