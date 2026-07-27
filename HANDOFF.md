# edge-dit.cpp CPU 优化 —— 交接文档（Handoff）

> 面向：接手这台/新机器的 AI 助手。目标：不看历史会话也能立即正确接续 CPU 优化工作。
> 生成日期：2026-07-16，**2026-07-17 更新**（见 §10，§7/§8 已被 07-17 干净重测更新，以 §10 为准）。分支 `cpu-bench`。所有 file:line 均为写文档时的快照，**动手前务必 grep/Read 复核当前代码**。

---

## 0. 一句话背景

**edge-dit.cpp** 是 C/C++ 的 DiT 推理引擎（CLI 名 `ed-cli`），底层张量库是 ggml（submodule）。此前所有优化都在 CUDA 后端；**CPU 后端是新方向**，从 2026-07-06 开始做。核心手段：**接 oneDNN 吃 Intel AMX bf16 matmul**，把 CPU 上的 DiT/VAE 追近 diffusers。

当前工作性质：**性能优化**（不是加功能）。逐个模型对齐 diffusers，逐个板块（DiT / VAE / text encoder）找瓶颈、模仿 diffusers 实现、实测验证。

---

## 1. 工作流（用户明确定义的标准流程，必须照做）

1. **逐步跟踪两个框架**（edge-dit vs diffusers）同一板块的执行流程，打印调试信息、逐步定位，找到瓶颈到底在哪。用户坚信"逐步跟踪一定能找到和 diffusers 的区别"。
2. **模仿 diffusers 的实现** —— 找到区别后直接效仿。
3. **绝不乱猜**：一定做实验求证，逐步定位，打印调试信息验证假设。黑盒猜测和孤立微基准在本项目反复翻车。
4. **除非有大矛盾**（方案冲突、需动大结构、可能损画质）才停下问用户；其他情况直接落地、事后汇报。
5. **自主推进，不要频繁征询**："想好实验就做，自己记录 load 就行。"数据不可信（load 高/波动）时放后台轮询重做，不要拿脏数据下结论，也不要停下来问。

### 实验口径（固定，重测不得改）
- **快速验证**：512×512 / 4 步 / seed 42（快、可复现）。注意 4 步必糊，是伪影不是 bug；**20 步才是正常画质**，别拿性能测试图当画质图给用户。
- **端到端报告**：图 1024×1024 / 视频 512×512×40 帧 / 20 步 / seed 42。**端到端 = 命令→拿到图的完整墙钟**（含加载），edge 用 `/usr/bin/time` 抓，diffusers 用进程墙钟（import→save）。
- **串行，绝不并行**：两框架或多实验并行会互相抢核污染数据。曾犯：并行跑把 load 从 11 冲到 177，数据全废。
- **每次测量记录 load average**（开跑+跑完都记）。load 高或波动大 → 数据不可信，优先用**同二进制 A/B 比值**（抗负载），绝对时间要等 load<12 才准。

### 正确性判据（重要）
- bf16 matmul 在 threadpool 下**非 bit-deterministic**：**md5 会变但 mean/std 对齐即正确**，别拿 md5 当正确性判据。
- bf16 / 卷积路线是"计算方式"改动，理论上不影响数值 → 出图应正确。**若不正确，打印调试信息定位数值 bug**。曾出现过 `ed_v_expf`（AVX-512 exp）返回值被多 ×2 的 bug（见 §6）——数值类 bug 的先例。
- 判正确的方法：和已知 good 图对比 mean/std（差 <1% 视觉无差），并实际看图。调试中的坏图必须说明来源、确认已删、从没进过提交。

---

## 2. 提交纪律（用户红线）

- **不要轻易提交，提交前必须问用户**（比"提交后汇报"更严）。
- 提交前汇报：做了什么、带来什么改动、端到端结果、和 diffusers 各阶段用时对比、出图是否验证正确。
- **提交后主动汇报**：改了什么、效果、出图是否正确。
- commit message **简洁**，**不要加 Co-Authored-By**（用户明确要求）。
- 提交策略：**只提优化 hunk**，profile/调试代码留工作区不提交。
- 真正不可逆/破坏性操作（git push、删分支）仍先确认。**push 一律先问。**

---

## 3. 安全性（多人共用服务器 —— 红线）

- 这台开发机**多人共用**。真红线：**不碰全局共享配置** —— `/etc/profile`、`/etc/bashrc`、`/etc/profile.d/*`、`/etc/ssh/sshd_config` 等所有用户都读的文件绝对不能改。
- **用户自己 home 下的 per-user dotfile 可以改**（`~/.bashrc`、`~/.ssh/config` 等），只影响用户自己。改前先讲清"只影响你、不影响别人"，因有过一次被拒最好取得明确同意。
- 模型/权重目录**只读不删**：用户多次强调"注意不要乱删除"。
- 环境变量、线程数等**优先纯内存/命令行临时生效**，不落盘服务器共享位置。
- SSH 掉线根因是 `/etc/profile` 的 `TMOUT=900`（15 分钟无输入自动 logout）；解法是用户 `~/.bashrc` 末尾 `unset TMOUT`（不改 /etc）。

---

## 4. 路径与环境

| 项 | 路径 |
|---|---|
| 仓库根 | `/export/home/wangtianyang.21/code1/edge-dit.cpp` |
| CPU 构建脚本 | `scripts/build_cpu.sh` → 产物 `build-cpu/bin/ed-cli` |
| oneDNN 构建 | `scripts/build_onednn.sh` → 装到 `third_party/onednn/install`（build_cpu.sh 检测到自动开 `GGML_ONEDNN`） |
| 模型根目录 | `/mnt/cfs/9n-das-admin/llm_models/wty_models/dit_models/` |
| ggml 子模块 | `third_party/ggml/`（fork `Komorebi623/ggml`，分支 `wty-modulation-ggml`）；CPU 优化代码在 `src/ggml-cpu/` |
| 内存/记忆导出 | `memory_export.tar.gz`（14 个 .md，本次会话打的包） |

**模型子目录**（`dit_models/` 下）：
```
Qwen-Image  Qwen-Image-Edit  Wan2.1-T2V-1.3B-Diffusers
flux-dev  flux-kontext-dev  stable-diffusion-3-medium-diffusers
```

### 构建
```bash
# 首次：先建 oneDNN（AMX bf16 的关键，不建则回退慢的 ggml 原生）
scripts/build_onednn.sh
# 再建 CPU（自动检测 oneDNN）
scripts/build_cpu.sh
# 运行时需要 oneDNN 动态库：
export LD_LIBRARY_PATH=$PWD/third_party/onednn/install/lib64:$LD_LIBRARY_PATH
```

### diffusers 参考（docker）
- 容器 `wty-edgedit-dev`；模型挂 `/mnt/cfs/.../llm_models` → `/models`；有 PIL/numpy，diffusers 0.39.0 / torch 2.7.1。
- 对比脚本：`scripts/diffusers_stage_timing_multi.py`（分阶段计时，工作区有未提交改动）。

---

## 5. 关键机制（不懂这些会走弯路）

### AMX 门槛（最核心）
- oneDNN AMX matmul **只拦 `src0->type == GGML_TYPE_BF16` 的大矩阵**（gate 在 `ggml-cpu.c` 附近，约 :1367）。
- **f32/f16 权重 → 不满足门槛 → 回退 ggml 原生**（~2000 GFLOP/s，慢）；bf16 → 走 AMX（~13000–16000 GFLOP/s）。
- **CLI 加 `--type bf16`**：把 native f16/f32 权重转 bf16，才吃得到 AMX。**权重是 f32 的模型（如 WAN）不加这个 flag，DiT 根本没上 AMX。**
- 验证：`Diffusion model weight type stat: ...` 那行——`f32:1067`（没转）vs `f32:342|bf16:725`（转了）。

### profiling
- 环境变量 `GGML_CPU_PROFILE=1`：打印每个 op 的 GFLOP/s、按 weight-type 分桶、per-shape top-N、stage 标注（`[stage=clip/t5/mmdit/vae]`）。**这是判断"某板块有没有吃到 AMX"的第一工具。**

### CFG 口径（做端到端对比时最容易踩的坑）
- edge 默认 `cfg_scale=1.0` → **单前向**（每步 1 次 DiT）。
- diffusers `guidance_scale>1`（SD3=7 / WAN=5）→ **双前向**（cond+uncond）。
- **两边前向次数差 2× = 工作量差 2×**。对比时必须对齐：要么都单前向，要么都双前向（edge 传 `--cfg-scale`）。**初版六模型报告因此得出多个错误结论（见 §7）。**

### 编辑类模型（kontext / qwen-edit）
- DiT 序列 = 生成 latent tokens + 参考图 latent tokens（拼接），token 数约翻倍。
- 参考图编码尺寸决定 token 数。diffusers kontext `_auto_resize` 默认 True → ~1MP 参考图 → 8192 总 token，**和 edge 默认一致**。对比脚本若传 `_auto_resize=False` 会人为缩小 diffusers 工作量（假差距）。

### VAE 注意力（别乱开 flash）
- oneDNN flash kernel 只对 d_head 64/128 安全，**d=512 的单头 VAE mid-block 注意力会 break**（AutoencoderKL heads=1, d_head=512）。给 flux VAE 开 flash 会让它变快但**出图坏掉**，别碰。

---

## 6. 已落地优化（git 已提交，勿重复造轮子）

按 git log（`cpu-bench`）：
- **oneDNN bf16 AMX matmul 接入**：拦截层 `third_party/ggml/src/ggml-cpu/ggml-onednn.cpp`。关键 bug 已修：`dnnl_threadpool_interop_set_max_concurrency(192)`（否则 fallback 48 核，单 matmul 15ms→6.5ms）；权重 pack 两级缓存（按 ptr,n,k 缓存 packed 权重跨 m 复用）。
- **`ed_v_expf` ×2 bug 已修**：AVX-512 exp 曾返回 ~2× 真值，导致 SD3/Qwen flash 乱码。**这是"计算方式不该改数值却改了"的经典先例**，遇到出图错先想到这类。
- **RoPE / concat / attention / modulation 并行化**（去掉 CUDA 时代遗留的单线程 no-op）。
- **channel_rms_norm 多线程化**（`ed_ggml_norm_ext.hpp`）：曾是 1/96 核跑，改成按空间位置切分（Qwen VAE 省 ~2s）。
- **ED_CONV_BF16 默认开**（`ggml_extend.hpp` conv2d/conv3d）：VAE 卷积 kernel F16→BF16 才能上 AMX。
- **VAE encode 走 direct conv**（`auto_encoder_kl.hpp` build_graph）：kontext 参考图编码 18× 加速。
- **conv3d im2col SIMD**（`third_party/ggml/src/ggml-cpu/ops.cpp`：memcpy gather + AVX-512 fp32→bf16/fp16 pack）：WAN 3D VAE。
- **Qwen VAE 全面优化**：5.1s→2.4s。

### 试过但失败/回退（别重蹈覆辙）
- flux VAE 开 flash：出图坏（d=512 超 oneDNN flash 范围）。已回退。
- channel_rms_norm 手动 hoist offset：无加速（编译器已做，真瓶颈是跨通道 strided cache miss）。已回退。
- kontext "把参考图改成输入图尺寸"：错的，edge 8192 token 本就对齐 diffusers 默认。已回退。
- llamafile/tinyBLAS：负优化（本机 AMX 下 f16 已走高优路径，llamafile 强项是量化）。死路。

---

## 7. ⚠️ 报告口径坑（务必知道）

`bench_results/six_model_report/REPORT_AVG.md`：
- 第 17–28 行有 **🔴 口径修正总表**（正确结论）：**edge 在所有 6 个模型上都比 diffusers 慢 1.2×–3.8×，没有任何反超**。差距最大是 WAN(3.79×) 和 SD3(1.55×)。
- **但第 42 行还残留旧的错误结论"SD3 edge 反超 diffusers 快 1.30×"**，与修正表自相矛盾 —— 这是初版 CFG 口径 bug 的假象（edge 单前向 vs diffusers 双前向）。**引用报告数字时以 🔴 修正表为准，第 40–54 行的"关键发现"多处是作废的旧结论。** 建议接手后清理这份报告的自相矛盾。

真实差距（同口径）：
| 模型 | 同口径真实结果 |
|---|---|
| SD3 | edge 慢 1.55×（hidden1536 小 matmul 过切分）|
| Flux / Qwen-Image / Qwen-Edit | 慢 1.29–1.39× |
| Flux-Kontext | 慢 1.22×（8192 token 对齐后）|
| WAN | 慢 3.79×（3D VAE + 长视频序列；需 `--type bf16` 才上 AMX）|

差距根因均为**已知架构级问题**（小 matmul 过切分固定 per-node 开销主导 + 双池线程争用），**不是算子缺失**。

---

## 8. 当前状态 / 下一步

**工作区**（`git status`）：
- `M scripts/diffusers_stage_timing_multi.py`（对比脚本改动，未提交）
- `?? bench_results/`（报告和产物，未纳入版本控制）
- `?? third_party/onednn`（本地构建产物）
- 已确认 `src/` 与 `third_party/ggml` **无未提交的源码改动**（上一轮的 norm hoist 试验已回退，工作树干净）。

**上一轮进行到**：验证 WAN `--type bf16` 出图正确性。
- 已确认：WAN 加 `--type bf16` → DiT MUL_MAT 1999→9091 GFLOP/s，sampling 18.65s→6.03s（3.1×），因 WAN 权重默认 f32 不满足 AMX 门槛。
- 现成产物：`bench_results/report/wan_bf16.avi`（bf16）、`bench_results/report/wan_prof2.avi`（f32）。
- **待办**：抽两 avi 首帧对比像素 mean-diff（edge avi 是 MJPG，用 docker PIL 或按 `\xff\xd8\xff`/`\xff\xd9` JPEG 标记抽帧）。若 rounding-level 差异 → bf16 正确；若坏 → 打印调试定位数值 bug。**先核对两文件是否同 seed/参数、仅权重 dtype 不同**（文件大小 770910 vs 766334B，需确认）。

**再下一步**：
1. 用完全对齐口径（`--type bf16` + 两边 CFG 都 cfg=5.0）重跑 WAN 端到端，拿真实差距，更新报告。
2. 考虑用全部口径修正后重跑六模型对比，出一份可信报告。
3. 清理 `REPORT_AVG.md` 第 40–54 行的自相矛盾旧结论。

**视频查看**：edge 用内部 MJPG avi（无 ffmpeg），播放器显示黑屏是容器问题，帧本身没坏；转 GIF（PIL）即可看。scp 到 Mac：`scp H200:<path> ~/Downloads/`。

---

## 9. 记忆文件索引（`memory_export.tar.gz` 内）

优化脉络（核心，按此顺序读）：
`cpu_optimization_plan` → `cpu_flux_tri_benchmark` → `cpu_onednn_amx_integration` → `cpu_flops_gap_resolved` → `cpu_dit_overhead_breakdown` → `cpu_dit_per_thread_decomposition` → `cpu_endtoend_caliber` → `cpu_sd3_qwen_optimization`

流程/协作/安全：`feedback_optimization_workflow` `collaboration_autonomy` `feedback_shared_server`
项目/硬件：`project_edge_dit_overview` `reference_dev_machine` `user_profile`

**硬件**：Intel Xeon Platinum 8558（Emerald Rapids）双路，192 逻辑核，2 NUMA node。有 **AMX（amx_tile/amx_int8/amx_bf16）** + AVX-512 全家。CPU 线程默认 `n_threads=0` → 运行时取 hardware_concurrency()。GPU 是 H200（此前 GPU 优化基准机）。

---

## 10. 2026-07-17 更新（覆盖 §7/§8 的过时内容，以本节为准）

> 本轮做了：对齐口径全量重测（Wan 视频 + 4 生图/编辑模型）、merge origin/main、conv3d bf16 AMX 优化落地+验证+清理。详细数据见 `bench_results/wan_rerun/OPTIMIZATION_REPORT_2026-07-17.md` 和记忆 `cpu_wan_5model_clean_benchmark`。

### 修正 §4 路径表
- ggml 子模块**实际分支是 `cpu-bench`**（§4 写的 `wty-modulation-ggml` 过时）；fork 仍是 `Komorebi623/ggml`。
- 记忆包现 **16 个 .md**（§4/§9 写的 14 个过时）。
- **两个子模块托管策略**：`ggml → Komorebi623/ggml`（改了源码，含 conv3d，要 push 到 fork）；`onednn → 上游 uxlfoundation/oneDNN`（**没改源码**，v3.7 commit `5e92240`，主仓库只存 gitlink 不 push 内容，新机器从上游自动拉）。

### merge origin/main（已提交 13d38b6）
origin(Komorebi623 fork) main 已和上游 yiming-l21 同步（4483482：cache 重构 + flux/qwen/wan 精简）。SSH 拉取（**主仓库 origin 是 HTTPS 会卡认证，push/fetch 都要临时用 SSH URL**），merge **零冲突**，ggml 指针保住。编译通过，20 步 flux 出图与 merge 前 bit 级一致（零回归）。

### §6 补：conv3d bf16 AMX（oneDNN 原生 primitive，本轮新落地）
- 和 §6 已有的"conv3d im2col SIMD"、"ED_CONV_BF16"**不同层**：这个是在 `ops.cpp` 的 `ggml_compute_forward_conv_3d_impl` 里，bf16 kernel 直接调 oneDNN 原生 AMX 3D 卷积 primitive（`ed_onednn_conv3d_bf16`，实现在 `ggml-onednn.cpp`，按权重指针缓存 Conv3dCache），替代 im2col+GEMM，失败自动 fallback。
- **验证**：Wan VAE decode 默认 2.78s vs `ED_ONEDNN_CONV3D_OFF=1` 3.46s → **快 ~20%**；正确性 max_abs 1e-5~1e-7（bf16 舍入级）。
- **唯一保留的开关**：`ED_ONEDNN_CONV3D_OFF=1`（逃生，回退 im2col）。调试脚手架（CMP/MODE/NOCACHE/SRC_BF16）已清理，勿再引入。
- **状态**：ggml 子模块工作区（07-17），准备提交到 `Komorebi623/ggml` cpu-bench。

### 覆盖 §7/§8 的旧结论（旧的 six_model_report 口径存疑，以下为 07-17 干净重测）
口径：全 bf16 / 严格串行（任意时刻仅一个推理进程）/ 大规模或多轮 / 直接打点（不做减法推算）。
- **Wan 480P/41帧/20步**：DiT 1.29×，**VAE 1.10×（3D VAE 已基本追平，靠 conv3d AMX）**。
- **生图 1024²/20步（2轮平均，edge波动<0.5%）**：
  | 模型 | DiT edge/diff | VAE edge/diff |
  |---|---|---|
  | Flux | 112.8/91.0 (1.24×) | 9.0/1.2 (**7.8×**) |
  | SD3 | 24.6/32.1 (**edge 快 1.30×**) | 2.8/1.2 (2.4×) |
  | Qwen-Edit | 288.7/224.4 (1.29×) | 10.2/2.2 (4.6×) |
  | Flux-Kontext | 273.4/201.3 (1.36×) | 8.9/1.2 (7.4×) |
- **DiT 差距一致 1.24~1.36×**（SD3 因小 matmul 反超）；**生图 2D VAE 是真短板 2.4~7.8×**（≠ Wan 3D VAE 已追平）。§7 那份 `six_model_report/REPORT_AVG.md`（含"WAN 3.79×""SD3 无反超"）是 CFG 口径未完全对齐 + 部分负载污染的旧版，**引用请以本轮报告为准**。

### 本轮方法论教训（补 §1）
- **单跑高负载数据不可信**：20 步大任务单跑曾报 DiT 194s，多轮实测 ~52s（被 load 84~107 污染近 4×）。**必须多轮取平均**。
- **减法推算不可信**：曾把"加载"算成随帧数涨，直接打点后加载恒 ~5s。**改用 CLI steady_clock 四段直接实测**（main.cpp `[ED_WALL]`）。
- **debug 探针看分布不看单点**：`next(parameters())` 取到第一个 f32 层误得"没转 bf16"，看 Counter 全分布才对。

### 当前状态（覆盖 §8）
- 已完成：Wan+5模型对齐重测、merge、conv3d 落地验证清理、报告+迁移文档。
- **下一个优化目标：生图 2D VAE 慢 7-8×**（Flux/SD3/Kontext 通用，非架构死路，最值得先啃）。
- 迁移细节见仓库根 `MIGRATION_2026-07-17.md`。
