# 迁移记录 — 未提交工作状态（2026-07-17）

> 目的：换机器时不丢任何改动。本文件记录**当前工作区所有未提交/游离的改动**及其位置、性质、复原方式。
> 配合 `HANDOFF.md`（工作流/环境/机制）一起看。新机器按本文件逐条复原即可。

分支：`cpu-bench`（已 merge origin/main 到 `13d38b6`）。

---

## 一、改动清单总览

| # | 位置 | 内容 | 性质 | 迁移方式 |
|---|---|---|---|---|
| 1 | `third_party/ggml`（子模块）`src/ggml-cpu/{ops.cpp, ggml-onednn.cpp, ggml-onednn.h}` | **conv3d bf16 AMX 优化**（+151 行，已清理调试代码，已验证） | ✅ 成熟优化 | git commit + push |
| 2 | `examples/cli/main.cpp`（+22 行） | CLI 四段计时桩（load/gen/save/free，`[ED_WALL]` 打印） | 测量工具，仅 video 分支 | git commit 或 patch |
| 3 | `scripts/diffusers_stage_timing_multi.py`（+26 行） | 加 flux-kontext 条目 + kontext 输入图处理 | benchmark 脚本 | git commit |
| 4 | `scripts/diffusers_wan_segwall.py`（新，未跟踪） | Wan 分段实测包装器（import/load/gen/free 四段直接测） | benchmark 脚本 | git add |
| 5 | `scripts/ed_conv3d_check.cpp`（新，未跟踪） | conv3d 独立检查工具 | 调试工具 | 可选提交 |
| 6 | `bench_results/`（63M，未跟踪） | 本轮所有 benchmark 日志 + 报告 | 数据/文档 | **打包，不进 git** |
| 7 | `memory_export.tar.gz`（72K，未跟踪） | 记忆导出 | 数据 | 打包 |
| 8 | `third_party/onednn/install/`（64M） | oneDNN 编译产物 | 构建产物 | **不迁移，新机器重建** |

---

## 二、逐条复原细节

### 1. conv3d bf16 AMX 优化（ggml 子模块，核心）

**这是本轮唯一的实质性能优化，已验证：**
- 编译通过；Wan VAE decode `ED_ONEDNN_CONV3D_OFF=0`(默认开) 2.78s vs `=1`(关) 3.46s → **快 ~20%**
- 正确性：`ED_ONEDNN_CONV3D_CMP` 逐算子对比（该调试开关已在清理时删除），max_abs 1e-5~1e-7（纯 bf16 舍入级），sum 逐位吻合。

**改了什么**：`ggml_compute_forward_conv_3d_impl`（ops.cpp）在 bf16 kernel 时，走 oneDNN 原生 AMX 3D 卷积 primitive（`ed_onednn_conv3d_bf16`，实现在 ggml-onednn.cpp），替代原 im2col+GEMM。primitive 按权重指针缓存（Conv3dCache）。失败自动 fallback。

**运行时开关**（唯一保留的）：`ED_ONEDNN_CONV3D_OFF=1` 强制回退 im2col（逃生用）。

**已清理的调试脚手架**（原本有，现已删，勿再引入）：`ED_ONEDNN_CONV3D_CMP`（逐算子对比打印）、`_MODE`（按形状筛选）、`_NOCACHE`（禁缓存）、`_SRC_BF16`（实验性 bf16 输入布局）+ 辅助函数 `ed_f32_to_bf16_rne`。

**复原**：这份改动在 ggml 子模块工作区未提交。迁移时应 commit 到 `Komorebi623/ggml` 的 cpu-bench 分支并 push，再更新父仓库的子模块指针。**注意子模块指针当前是 `40a044f`，父仓库 cpu-bench 已指向它——commit conv3d 后指针会前进，父仓库要跟着更新。**

### 2. main.cpp 计时桩

`main()` 里加了 `steady_clock` 四段计时（load = ed_create_context；gen = ed_generate_video；save = save_video；free = ed_free_context），退出前打印 `[ED_WALL] load=..ms gen=..ms save=..ms free=..ms`。
**局限**：只加在 `--video` 分支；image 分支（生图模型）没加，所以生图跑 gen/save 显示 0。如需生图墙钟，要在 image 分支同样插桩。

### 3-5. 脚本

- `diffusers_stage_timing_multi.py`：加了 `flux-kontext` 模型条目（FluxKontextPipeline）+ kontext 的输入图处理。**注意**：脚本里模型路径是硬编码的 `/mnt/cfs/9n-das-admin/llm_models/<name>`，容器里需建 symlink 指到实际的 `wty_models/dit_models/<name>`（容器重启会丢，需重建）。
- `diffusers_wan_segwall.py`：Wan 分段实测包装器（新文件）。
- `ed_conv3d_check.cpp`：conv3d 独立检查（新文件）。

### 6. bench_results/（本轮成果，打包迁移）

关键文件：
- `bench_results/wan_rerun/OPTIMIZATION_REPORT_2026-07-17.md` — **本轮优化对比报告**（5 模型、含配置口径）
- `bench_results/wan_rerun/edge_inst_f{9,20,80}.log`、`diff_seg_f{9,20,80}.log` — Wan 256² 分段实测
- `bench_results/wan_480p/`、`bench_results/img_1024/`、`bench_results/conv3d_verify/` — 各模型/conv3d 验证数据

### 8. onednn install/（不迁移）

`third_party/onednn` 已是 git 子模块（指向 v3.7 `5e92240`，提交 26431e6 接入）。`install/`（64M）是本地编译产物，**不进 git**。新机器：`git submodule update --init` 拉源码 → `scripts/build_onednn.sh` 重建 install/。

---

## 三、新机器完整复原步骤

```bash
# 1. clone 带子模块
git clone --recursive git@github.com:Komorebi623/edge-dit.cpp.git
cd edge-dit.cpp && git checkout cpu-bench
git submodule update --init --recursive   # 拉 ggml(含conv3d) + onednn

# 2. 建 oneDNN（AMX bf16 关键）
scripts/build_onednn.sh                    # 生成 third_party/onednn/install/
export LD_LIBRARY_PATH=$PWD/third_party/onednn/install/lib64:$LD_LIBRARY_PATH

# 3. 建 CPU（自动检测 oneDNN）
scripts/build_cpu.sh

# 4. 解包 bench_results（如需历史数据/报告）
#    scp 过来的 bench_results.tar.gz 解到仓库根

# 5. 验证 conv3d 优化生效
#    跑 Wan，对比 ED_ONEDNN_CONV3D_OFF=1 vs 默认，VAE decode 应快 ~20%
```

---

## 四、待办（迁移后接续）

- conv3d 优化：commit 到 ggml 子模块 + push + 更新父仓库指针（本文件写时**尚未 commit**，按用户要求先清理+文档，未提交）
- 生图模型 2D VAE 慢 7-8×（报告结论）：下一个优化目标
- main.cpp image 分支补计时桩（如需生图端到端墙钟）
