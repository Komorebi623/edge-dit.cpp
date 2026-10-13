# MiniMax-H3：新服务器接续说明

更新：2026-10-13（Asia/Shanghai）。本文件取代之前的暂停／Mac协调方案；用户选择在另一台服务器上继续开发。代码与实验状态以此文件及验证摘要为准，原始聊天迁移是独立事项。

## 1. 获取同一研究分支

个人远端：`Komorebi623/edge-dit.cpp`；开发分支：`codex/minimax-h3-cuda-optimizations`。原始基线固定 `97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2`。已实现代码检查点 `aa3eaeb3f71bbe7621b7aef7295f360c1119196d`，后续交接文档提交不改变算法。

在新目录克隆该研究分支并初始化子模块；若已有项目，先检查未提交内容，不覆盖、不强制重置。保留 `origin` 为个人推送目标，`thu` 指向 THU-MIG 上游；不合并main、不强推。初次接手记录 `git rev-parse HEAD` 和子模块版本。

固定子模块：ggml `f270346d688c7bec7c56bae2ae73343aba6817ad`；oneDNN `5e9224036021433d2577548ed0539fe9a53256bc`。不要顺手更新依赖后沿用旧baseline数据。

## 2. 用户目标与不可违反的约束

- 融合 SGLang / Cache-DiT、RunningHub / Turbo、FastVideo / VSA、VDN 等H3优化，每种独立总控，默认关闭。
- 新优化仅H3、仅CUDA生效，不改变其他模型／后端；CPU tiny tests是数学验证入口，不是对其他后端启用优化。
- **禁止停止／杀死任何GPU或他人的进程；不删除用户文件，不抢占资源。**先读GPU状态，忙则继续CPU开发，推理等待许可及资源。
- 先固定模型、输入和baseline，再逐开关比较；不能更换量化、pruned、LoRA、步数、solver、placement后仍与旧baseline算加速。
- 可并行委派不同写入范围；只有一个代码集成者。简洁记录环境、基线、算法、结果与故障。

## 3. 当前真实完成状态

已完成并默认关闭：`ED_H3_FUSED_OPS`、`ED_H3_CACHE_DIT`、`ED_H3_ADALN_CACHE`、`ED_H3_SPARSE_ATTN`、`ED_H3_SAGE_ATTN`。

- Cache-DiT：真实前段probe、逐模态RF guard、同锚middle delta、真实后段，CFG分离与请求失效。
- AdaLN：全部固定sigma的真实投影预填、逐分支缓存、角色／schedule／weights身份与预算验证；未释放权重，不承诺显存节省。
- SubBlock：逐head路由与TopK/gather；全部真实Q更新，保护text/audio/reference；初期F32/batch1。
- Sage：H3SA专用tag，既有SAGQ全局开关保持原行为；首末层／refiner保dense，不能与SubBlock一起启用。
- `ED_H3_TURBO`：CPU-only offline LoRA合并、全选中DiT分片SHA256验证及采样recipe入口已实现；没有实际公开adapter验证，需要独立模型实验。
- `ED_H3_VDN` / `ED_H3_VSA`：已做数学参考、部分原生组件及CPU测试；**完整CUDA部署未完成，生产请求拒绝**。VDN还需真实精确solve、short-conv、anchor和完整权重加载；VSA需匹配FastH3 gate及原生CUDA tile/compression，不能把SubBlock冒充VSA。
- TP/Ulysses尚未迁移到H3；torch.compile不是C++下的算法开关。RunningHub私有权重不能复现为公开模型收益。

构建：CPU全工程、固定CUDAbaseline与CUDAcandidate均编译链接通过。软件回归：12/12 CTest、48/48 workflow、18/18 sourcefetch、5/5 tinyLoRA通过。actual tiny H3的24次AdaLN完整图对照逐位一致，融合56次CPU对照最大绝对误差0。**未运行生产H3 GPU推理，无真实GPU质量／性能数据。**见 `benchmark/h3_cuda/validation.json`。

## 4. 必须保持的模型精度与输入

主实验使用现有完整 FL2VA/Ref2VA Q4系列GGUF、Qwen Q4系列GGUF、video VAE FP16、audio VAE FP32。它们实际是混合张量类型，具体文件、大小、整文件SHA256在 `docs/research/h3-precision-lock.json`。不能仅按文件名认为全张量Q4。

原机器模型根：`/mnt/cfs/9n-das-admin/llm_models/MiniMax-H3-GGUF`；容器根：`/models/MiniMax-H3-GGUF`。新服务器优先查共享存储；无共享时只复制对应五个文件并核对完整hash，不重新量化。

固定主任务：Ref2VA mixed、512×512、124帧、24fps、20步、seed42、CFG1、RES multistep/simple。覆盖T2VA、首帧／末帧／首尾、参考图／视频／视频配对音频／混合。另做CFG、重复请求、错误输入与取消／失败清理验证。模态标签、输入顺序、图像几何和音频参数必须保持一致。

## 5. 哪些东西没有随Git推送

Git提供源码、测试、简洁日志、精度锁、示例manifest、官方source lock，不提供：模型权重、旧构建目录／二进制、原始附件、`_scratch`内容、完整私密聊天历史或用户原有未跟踪文件。

若能从旧服务器复制，优先迁移：

- `_scratch/h3-fixtures-20261011/`：约1MiB的固定CPU媒体素材与locked manifest；路径可重映射。
- `_scratch/h3-reference-src/`：9仓库167份精选官方源码；也可用 `scripts/fetch_h3_references.py --fetch` 按固定commit重取，默认dry-run。
- `_scratch/h3-logs/` 与 `_scratch/h3-local-logs/`：软件测试和构建原始日志。
- 固定cuDNN frontend `07b368a2696f5fd0fc2207d4635c51cea4c0bf71` 源码；旧路径 `build-h200-current-20260814/_deps/cudnn_frontend-src`。

不能复制素材时，使用 `fixtures`生成新的合法素材并`lock`，对该新数据集重新建立baseline；它不是原数据集的直接续跑。生产画质另需优质合法真实输入，CPU合成素材仅验证流程。

## 6. 新服务器执行顺序

1. 只读检查硬件、GPU空闲、磁盘、CUDA/cuDNN、编译器、ffmpeg/ffprobe、Python依赖与模型hash；不要启动GPU推理或停止服务。
2. `bash scripts/test_h3_cpu.sh` 做CPU回归（需要CMake/C++工具链和CPU测试用torch/safetensors，媒体测试需ffmpeg/ffprobe）。尚缺依赖时复用现有容器或安装到隔离环境，不覆盖系统。
3. 原开发容器为 `wty-edgedit-dev`，GCC13.3/CUDA12.8/cuDNN，项目挂载 `/workspace/edge-dit.cpp`；新机器不要假设存在同名容器。
4. 重新构建baseline及candidate。**`scripts/build_h3_cuda.sh`当前固定SM90和旧cuDNN源码路径**：H200可设置 `H3_CUDNN_FRONTEND_SOURCE` 到实际位置；其他GPU先将架构参数化，对两个构建用同一目标架构，不盲用旧二进制。manifest的架构要求同步修改并保留独立记录。
5. `python3 scripts/h3_cuda_workflow.py --help`；使用 `--repo-root` / `--model-root` / `--asset-root`重映射路径。先dry-run，再锁hash。输出目录必须新建，不覆盖已有结果。
6. 有可用GPU后，固定同一GPU UUID／精度／placement，先跑mixed baseline，再逐项candidate。逐项计划在 `benchmark/h3_cuda/{fused_ops,cache_dit,adaln_cache,sparse_attn,sage_attn}.json`，初期不要叠加。
7. 当前cache/attention rollout拒绝offload、并行graph cuts、H3 persistent graph reuse、可变adapter等未验证组合。资源不够时不能偷偷改变placement；需独立支持实现及该placement自己的baseline。
8. 比较视频帧数／尺寸／fps、全部PCM长度／声道／采样率／finite、SSIM/PSNR及音频误差。再人工或独立评审prompt、参考身份、动作、语音／音乐、音画同步；数值指标不能代替语义质量。
9. 记录首次prefill、模型加载、各stage、端到端和显存边界；跨不同GPU不直接比较加速比。失败保留参数、stdout/stderr、退出码、commit、全weights/assets/binary hash。
10. GPU仍忙则优先补VDN/VSA完整实现、Turbo真实权重验证及错误路径回归。不能把无权重的小模型通过写成生产优化已完成。

## 7. 原始聊天迁移与Thread Handoff边界

2026-10-13检查：此服务器已经装有Codex CLI `0.162.0-alpha.17.2`，CLI帮助提供 `resume` / `fork`，协议提供 `thread/resume` / `thread/fork`。没有从此客户端导出的schema发现独立 `thread/handoff`，当前聊天也未提供App的`handoff_thread`工具。不能通过安装普通skill直接向当前会话注入App内部工具。

原聊天ID `01a12a7a-c737-7b13-8750-cfc804e14382`；旧服务器rollout路径：
`/export/home/wangtianyang.21/.codex/sessions/2026/10/11/rollout-2026-10-11T18-20-51-01a12a7a-c737-7b13-8750-cfc804e14382.jsonl`。

**该会话标记为paginated history**。只复制此JSONL不保证完整恢复；需要检查分页历史存储／索引／客户端版本及App专用tool记录。官方协议的手工history/path注入标为unstable，不能承诺跨机器App迁移。`migrate-rollouts`用于旧会话到分页格式，不是服务器迁移工具。

若要保留原始对话，优先用App实际可用的原生交接；否则单独授权下准备兼容的私密会话迁移材料并实测CLI续接。**不得把聊天、认证文件、私钥、整个`.codex`数据库推入Git**。新Codex没有原聊天时可读此文件接续，无需用户重新整理研究内容。

## 8. 日志与下一步文档

最先读本文件；后读 `h3-cuda-optimization-status.md`、`h3-development-log.md`、`h3-precision-lock.json`、`minimax-h3-optimization-sources.md`。各算法详细测试和缺口在 `tests/h3-*-progress.md`。

旧日志按时间保存，早期“暂停／workflow尚不存在／仅部分构建／权重hash未完成”是历史记录，**不代表当前状态**。本次交接后旧会话不并行修改同一研究分支。
