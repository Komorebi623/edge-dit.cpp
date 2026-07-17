# 新机器复现指南（2026-07-17）

> 面向新机器上接手的 AI / 用户。目标：把 cpu-bench 的工作状态（含 conv3d bf16 AMX 优化）在新机器完整复现，SSH + 上下游 remote 配置清楚，不误碰上游。
> 配合 `HANDOFF.md`（工作流/机制）、`MIGRATION_2026-07-17.md`（改动清单）一起看。

---

## ⚠️ 三个必须先知道的坑

1. **主仓库 origin 是 HTTPS，会卡认证** → 换成 SSH（见下）。所有 push/fetch 用 SSH。
2. **`.gitmodules` 里 ggml 的 url 指向上游 `yiming-l21/ggml` 的 `edge-cudnn-sdpa` 分支，不是你的 fork**。直接 `clone --recursive` 会拉到上游、**拉不到你的 conv3d 优化**。必须手动把 ggml 子模块的 origin 改成你的 fork 并 checkout `cpu-bench`（见步骤 3）。
3. **onednn 不用改**：它的源码你没动，`.gitmodules` 指向上游 `uxlfoundation/oneDNN` 正确，clone 时自动拉 commit `5e92240`（v3.7）。install/ 是本地产物，新机器重新 build。

---

## 上下游 remote 全表（配完应长这样）

| 仓库 | remote | 地址 | 用途 |
|---|---|---|---|
| 主仓库 | **origin** | `git@github.com:Komorebi623/edge-dit.cpp.git` | 你的 fork（SSH，push 到这）|
| 主仓库 | upstream | `git@github.com:yiming-l21/edge-dit.cpp.git` | 上游（只 fetch，**绝不 push**）|
| ggml | **origin** | `git@github.com:Komorebi623/ggml.git` | 你的 fork（conv3d 在这，SSH）|
| ggml | upstream | `https://github.com/yiming-l21/ggml.git` | 上游（只 fetch）|
| onednn | origin | `https://github.com/uxlfoundation/oneDNN.git` | 第三方上游（只拉，不改）|

工作分支：主仓库 `cpu-bench` @ `9821d43`；ggml `cpu-bench` @ `daff6b07`。

---

## 复现步骤（复制粘贴即可）

### 0. 前置：SSH key 能访问 GitHub
```bash
ssh -T git@github.com   # 应显示 "Hi Komorebi623!"，不行则先在新机器配好 SSH key
```

### 1. 用 SSH clone 主仓库（不加 --recursive，子模块单独处理）
```bash
git clone git@github.com:Komorebi623/edge-dit.cpp.git
cd edge-dit.cpp
git checkout cpu-bench

# 配上游（只 fetch，不 push）
git remote add upstream git@github.com:yiming-l21/edge-dit.cpp.git
git remote set-url --push upstream DISABLED   # 防手滑 push 上游
```

### 2. onednn 子模块（直接 init，上游 url 正确）
```bash
git submodule update --init third_party/onednn   # 自动拉 5e92240 (v3.7)
```

### 3. ggml 子模块（关键：改 origin 到你的 fork，拉 conv3d）
```bash
# .gitmodules 指向上游，先 init（会拉到上游的 edge-cudnn-sdpa，不对）
git submodule update --init third_party/ggml
cd third_party/ggml

# 把 origin 改成你的 fork（SSH），重新拉你的 cpu-bench
git remote set-url origin git@github.com:Komorebi623/ggml.git
git remote add upstream https://github.com/yiming-l21/ggml.git 2>/dev/null || true
git fetch origin cpu-bench
git checkout cpu-bench          # 应到 daff6b07 (conv3d bf16 AMX)
git log --oneline -1            # 确认: daff6b07 perf(cpu): route bf16 CONV_3D through oneDNN...
cd ../..

# 确认主仓库记录的 ggml 指针和子模块一致（都应是 daff6b07）
git ls-files -s third_party/ggml   # 160000 daff6b07...
```

### 4. 建 oneDNN（AMX bf16 的关键，不建则回退慢的 ggml 原生）
```bash
scripts/build_onednn.sh          # 生成 third_party/onednn/install/
export LD_LIBRARY_PATH=$PWD/third_party/onednn/install/lib64:$LD_LIBRARY_PATH
```

### 5. 建 CPU（自动检测 oneDNN）
```bash
scripts/build_cpu.sh             # 产物 build-cpu/bin/ed-cli
```

### 6. 验证 conv3d 优化生效
```bash
M=/mnt/cfs/9n-das-admin/llm_models/wty_models/dit_models/Wan2.1-T2V-1.3B-Diffusers
# 默认（conv3d AMX 开）
./build-cpu/bin/ed-cli --model "$M" --video --type bf16 -W 256 -H 256 --frames 9 --steps 4 \
  --cfg-scale 5.0 -s 42 --prompt 'a cat walking on grass' -o /tmp/on.avi -t 96 2>&1 | grep 'vae decode'
# 关掉对照（应变慢 ~20%）
ED_ONEDNN_CONV3D_OFF=1 ./build-cpu/bin/ed-cli --model "$M" --video --type bf16 -W 256 -H 256 --frames 9 --steps 4 \
  --cfg-scale 5.0 -s 42 --prompt 'a cat walking on grass' -o /tmp/off.avi -t 96 2>&1 | grep 'vae decode'
# 预期: on ~2.8s vs off ~3.5s
```

### 7. 手动搬运（不在 git 里的）
- **记忆**：把 `memory_export.tar.gz`（16 个 .md）scp 到新机器，解到新机器的 auto-memory 目录
  （`~/.claude/projects/<新机器的项目 slug>/memory/`；注意 slug 随仓库绝对路径变，可能不同）。
- **bench_results/**（可选，历史数据/报告）：如需 `OPTIMIZATION_REPORT_2026-07-17.md` 等，scp `bench_results/` 过去。

---

## push 时的正确姿势（避免碰上游）

```bash
# 主仓库（origin 已是 SSH，直接 push 你的 fork）
git push origin cpu-bench

# ggml 子模块
cd third_party/ggml && git push origin cpu-bench && cd ../..

# 若主仓库 origin 仍是 HTTPS（没改成 SSH），临时用 SSH URL：
git push git@github.com:Komorebi623/edge-dit.cpp.git cpu-bench:cpu-bench
```

**红线**：push 一律先问用户；只 push `origin`（Komorebi623），**绝不 push upstream（yiming-l21）**。

---

## 从上游同步（日后 merge 上游更新时）
```bash
git fetch upstream main          # 拉上游主仓库
git merge upstream/main          # 或先 fetch 到 origin/main 再 merge
# ggml 同理：cd third_party/ggml && git fetch upstream && ...
```
本轮已 merge 过一次（主仓库 13d38b6，零冲突）。上游改 cache/模型文件，你的改动主要在 ggml 子模块 + 少量 pipeline，冲突面小。
