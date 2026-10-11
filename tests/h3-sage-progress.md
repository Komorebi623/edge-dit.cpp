# H3-only SageAttention 接入进度

日期：2026-10-11。状态：**H3 helper 和获授权的共享 tag 接入完成；CPU 测试通过；CUDA `.o` 编译未成功；未运行 GPU。**

## 1. 写入范围与交接

本分支工作仅写以下五个文件：

- `src/dit_models/models/minimax_h3_cuda_sage.hpp`：独立 H3 图构建、策略、shape/type guards、V conditioning、输出重排。
- `tests/h3_cuda_sage_test.cpp`：CPU 图、数学、层策略、tag、可选精确 host predicate 测试。
- `tests/h3-sage-progress.md`：本文。
- 经用户后续授权：`src/core/backend/ggml/ed_ggml_sage_attn_ext.hpp`，仅新增 H3SA tag 常量及 validator；原 SAGQ validator 不变。
- 经用户后续授权：`src/core/backend/ggml/cuda/ed_cuda_sage_attn.cu`，仅修改 `ed_cuda_sage_attn_custom_supported` 内 tag/gate 分支及 H3 专属 guards；kernel、scratch、compute 均不改。

**没有修改主 model、shared attention helper、third-party ggml dispatch、CMake、runner capabilities 或其他线程文件。没有启动/执行 GPU，没有停止任何进程。** 编译输出和可选 host harness 位于 `/tmp`。

主线程可以接手上述两个共享文件；它们的修改已完成。主 model 集成、CMake activation、runner capabilities 仍由主线程负责，不能把本文当成已完成端到端部署或 GPU 质量验证。

## 2. 更正“kernel 只支持 64”和 scaling 误评

之前“现有 Sage kernel 只支持 head_dim=64”的判断**不准确**。部分注释仍描述最初 SD3/64 范围，但实际代码：

- `sage_attn_shape_supported` 接受 **64 和 128**。
- `ed_cuda_sage_attn_custom_compute` 分派 `run_sage<float/half, 64/128>`。
- `run_sage` 的实际 `sm_scale` 为 `1/sqrt(HEAD_DIM)`，128 时约 **0.08838835**。

H3 传给 `ggml_ext_attention_ext` 的 `kv_scale=1/128=0.0078125` **不是 softmax/logit scale**。现有 helper 的具体语义：

1. `scale = 1/sqrt(d_head)`。
2. Flash 路径 Q 转 F32，**不缩放 Q**。
3. K 乘 `s=kv_scale`，再转 F16。
4. V 在准备分支中乘 `s`，再转 F16。
5. `ggml_flash_attn_ext(..., scale/s, ...)`，H3 这里传约 **11.313708**。
6. 输出乘 `1/s`，H3 为 **128**。

忽略有限精度时，对实际经过 V conditioning 的常规 H3 输入：

```text
O_flash = (1/s) * softmax((scale/s) * Q * (sK)^T) * (sV)
        = softmax(scale * QK^T) * V
scale   = 1/sqrt(128), NOT 1/128
```

Dense 路径完全不应用 `kv_scale`，同样使用 `1/sqrt(d_head)`。因此把 H3 的 `1/128` 移植成 Sage sm_scale，或将 Q 乘 `1/sqrt(128)` 来“补偿 kernel”，都会错误改变 logits；这次 **没有新增 Q scaling**。

### 必须保留的精确语义 caveat

共享 helper 的 V 缩放不是无条件的：它位于 `v_in->type != F16 || !contiguous` 分支内。**已经是 contiguous F16 的 V 会跳过该分支**，但最终输出反缩放仍发生。CPU 测试已捕捉这种输入的输出约为正常结果的 128 倍。这里仅记录现有行为，**不修改共享 helper，也不把该偶然 dtype/layout 行为复制到 H3 adapter**。

普通 H3 的 QKV-strided V/F32 V 会经过 V conditioning；BF16 低显存输入也会进入准备分支，但本 H3 Sage rollout 明确回退 dense，不将 BF16 当作支持。

## 3. 最小复用方案（已实现）

继续调用现有 Sage INT8-QK + F16-PV custom kernel，**没有另写朴素 attention 冒充 Sage**。

H3 graph adapter：

- 接收 H3 QK norm/partial RoPE 后的 Q/K 和原 token-major V。
- Q/K 保持原数值，不乘 1/128 或 sqrt128 因子。
- 移除原 flash 路径里代数抵消的 **K conditioning**：kernel 的固定 `1/sqrt128` 已是有效 logit scale。
- V **明确乘 1/128**，保留通过 F16 PV/half 输出的幅度 conditioning；kernel 返回后 **乘 128**。
- Q/K/V 均规范化为 contiguous F32。现有 `run_sage<T>` 只从 Q 判断 T，并以同一 T 读取 K/V，而共享 shape predicate 允许混合 F16/F32；H3 不能直接传 mixed dtype。
- 输出 raw `[128, heads*batch, L, 1]` 重排为 `[128*heads, L, batch]`，CPU 测试覆盖 batch=1/2/3。未照搬 shared flash 的 batch=1 `view_3d`。

这只是**实数数学目标一致**，不是与 stock flash bitwise 相同：K 的 F16-conditioning/rounding 不再相同；INT8 QK、F16 PV、half raw 输出均带来误差；V conditioning 能否提供足够 H3 数值精度仍需后续 GPU A/B。CPU 数学测试不能证明真实 Sage kernel 的精度、长序列质量或速度。

## 4. H3SA shared ABI 与 gate

```text
SAGQ = 0x53414751（旧）
H3SA = 0x48335341（新）
userdata low32 = magic
userdata high31 = positive n_head
```

新增 `kSageAttnH3CustomMagic`、`sage_attn_h3_params_valid`。`sage_attn_params_valid` 仍仅接受旧 SAGQ。

`ed_cuda_sage_attn_custom_supported`：

- **H3SA**：tag 本身是显式 request/graph opt-in，**该分支不调用 getenv，不依赖 ED_SAGE_ATTN**。
- H3SA 必须 head_dim=128；Q/K/V **全部 F32**；合法 unit channel stride；Q/K axis3=1；self-attn 序列等长；避免 int/u32 narrowing 与 grid.z 超范围。随后仍经过原 shape/dst/layout guards。
- **旧 SAGQ**：仍必须 `sage_attn_params_valid && sage_attn_enabled`，global-off 仍拒绝；没有收紧原 SAGQ mixed dtype/其他 shape 行为。
- 未知 tag、head<=0、缺 src、非 CUSTOM 等均拒绝。
- third-party dispatch、scratch sizing、compute 继续调用**同一个** `custom_supported`，无需改 third-party。

精确源文本 host-only 差分测试覆盖 **8 个环境值 × 2 个 head_dim × 27 个输入 dtype 组合 × 3 个 tag = 1296 项**。旧 SAGQ/未知 tag 对照修改前 predicate，全部一致；新 H3SA 在所有全局 gate 值下仅接受合法 128/F32 组合。另测坏 stride、非整数 unit stride、坏输出、null、缺源、cross-attn、zero-head。

## 5. 默认、隔离、限制

- 复用已存在的 H3 options parser：**`ED_H3_SAGE_ATTN` 默认 off**，严格 0/1；不解析 `ED_SAGE_ATTN` 作为 H3 开关。
- Helper 不 `setenv/putenv`，不替其他 model 打开 Sage；测试里的环境设置仅限测试进程并恢复。
- 首层/末层 DiT 始终 dense。TokenRefiner 始终 dense。小于 3 层的栈全部 dense。
- 不继承 `ED_SAGE_SKIP_FIRST/LAST`。调用者必须显式给 `scope/layer_idx/total_layers`，不得通过 RoPE 是否存在猜 scope。
- Sage + SubBlock/sparse 组合 Reject，直到验证。`h3_sage_request` 同时合并调用者实际 `subblock_enabled` 和 `options->sparse_attn`。
- CPU/其他 backend 对显式 Sage 请求 Reject；flag off 不改变 baseline。
- Mask、GQA、cross-attn、非 128、BF16/quantized 输入等保留 Dense。F16/F32 图输入可接受，但 custom 节点输入均 F32。
- CUDA grid.z 上限守护 `heads*batch<=65535`；序列避免 ceil-to-128/64 和 u32 stride 溢出。不要求 L 为 64/128/256 的整倍数。
- `ED_SAGE_PV_FP8` 为真时 H3 请求 Reject；本 rollout 未验证 FP8 PV。
- **global-effect 边界**：只隔离 on/off 和层范围；复用的 kernel 仍读取现有 `ED_SAGE_SMOOTH_Q`、`ED_SAGE_NO_SMOOTHK`、fallback/debug 等 tuning。没有声称 H3 内部 kernel 的全部调优参数都独立。H3 helper 不修改这些参数；默认 smooth-Q 对 128 开启，fallback 以实际代码为准（当前默认 off），不能按旧注释推断。

### 防止 CPU stub 静默执行

当前 CPU backend 对 generic CUSTOM 默认宣称支持，与 shared Sage header 中“CPU 永远拒绝”的注释不一致。H3 使用独立的 fail-loud CPU stub（不是 attention 实现），误排到 CPU 则 abort。运行入口先检查真实 CUDA backend，再检查 `ggml_backend_supports_op(backend, raw)`；缺 Sage build/tag 支持返回 **Reject**，而不是输出未计算的 tensor。

CPU 测试从不执行任何 CUSTOM 节点：只执行 V preparation、合成 raw 输出重排、stock dense/CPU flash 图。图计算前逐节点检查无 CUSTOM。

## 6. 主线程集成契约

在主 H3 model 的 `Attention::forward` 中，保留原 dense/flash 调用作为 fallback，不改共享 helper：

```cpp
// q/k 已完成 norm/RoPE；v 尚未做共享 attention 的 preparation。
auto request = MiniMaxH3::h3_sage_request(options, ctx->backend,
    ctx->flash_attn_enabled, scope, layer_idx, total_layers, actual_subblock_enabled);
auto candidate = MiniMaxH3::h3_sage_try_build(ctx->ggml_ctx, ctx->backend,
    q, k, v, heads, request);
if (candidate.route == MiniMaxH3::H3SageRoute::Reject)
    throw std::runtime_error(candidate.reason);
auto out = candidate.output;
if (candidate.route == MiniMaxH3::H3SageRoute::Dense)
    out = /* 原 ggml_ext_attention_ext(... kv_scale=1/128 ...) 调用，参数不变 */;
return out_proj->forward(ctx, out);
```

- TransformerBlock/DiT loop 透传明确的 0-based layer_idx 与 total_layers；TokenRefiner 默认 `TokenRefiner` scope、不传 DiT metadata。
- 单独的 `h3_sage_custom_graph` 是 **CPU-safe inspection/build-only API**，不检查 runtime options/backend；生产必须使用 `h3_sage_try_build`，不能直接计算 graph-only CUSTOM。
- 主线程配置 `ED_ENABLE_CUDA_SAGE_ATTN=ON` 的 backend build，并处理 runner capabilities；本工作没有设置 caps=true。
- rollout 应记录实际 Sage/Dense/Reject 路由，BF16 长序列回退不能被宣称 Sage 已加速。
- 后续 GPU 验证前不要声称质量或性能通过。当前环境 `.o` 编译失败也需要在 activation 前解决。

## 7. 已测试状态与复现

使用现有 **CUDA=OFF 的 `build-merge-cpu`**（只读，不改它的配置）。依赖检查显示测试程序仅链接 ggml/ggml-cpu/ggml-base 和系统 CPU 库，**无 libcuda/libcudart**。CPU backend 显式按名初始化，没有 `load_all()`/GPU discovery。

基础测试命令（在仓库根目录；产物在 `/tmp`）：

```sh
c++ -std=c++17 -O0 -g -DGGML_MAX_NAME=128 \
  -Isrc -Isrc/core -Iinclude -Ithird_party -Ithird_party/ggml/include \
  tests/h3_cuda_sage_test.cpp -Lbuild-merge-cpu/bin \
  -Wl,-rpath,"$PWD/build-merge-cpu/bin" build-merge-cpu/lib/libedgedit.a \
  -lggml -lggml-cpu -lggml-base -pthread -o /tmp/h3_cuda_sage_test
/tmp/h3_cuda_sage_test
```

`GGML_MAX_NAME=128` 必须匹配现有库 ABI。链接现有 CPU `libedgedit.a` 仅用于 shared helper 所需的 `sd_backend_is`；不能自行简化这个函数来假装测试 shared 语义。基础编译不启用可选 exact-source harness 时会明确打印其 SKIP，其余测试正常运行。

本次额外 host-only harness `/tmp/h3-sage-predicate.hpp` 从 CUDA `.cu` 中逐字截取 decoder/stride helper 和当前 `custom_supported`，以及修改前保存的同函数（仅改名 `legacy_sage_supported`），在 namespace 内用普通 C++ 编译。**不加载/链接 CUDA，不启动任何 kernel，不是 CUDA 设备测试。** 可复现生成方式（这里从 git HEAD 取修改前 predicate，需 HEAD 仍是本次修改前版本）：

```sh
python3 - <<'PY'
from pathlib import Path
import subprocess
p = 'src/core/backend/ggml/cuda/ed_cuda_sage_attn.cu'
current = Path(p).read_text()
original = subprocess.check_output(['git', 'show', 'HEAD:' + p], text=True)
def predicate(text):
    a = text.index('bool ed_cuda_sage_attn_custom_supported(')
    b = text.index('\nbool ed_cuda_sage_attn_scratch_sizes_for', a)
    return text[a:b]
a = current.index('struct ggml_custom_op_params_local')
b = current.index('// Compute the flattened dims', a)
legacy = predicate(original).replace('ed_cuda_sage_attn_custom_supported(', 'legacy_sage_supported(', 1)
Path('/tmp/h3-sage-predicate.hpp').write_text(
    '#pragma once\nnamespace h3_sage_host_test {\n'
    'using edgedit::ggml_ext::SageAttnCustomParams;\n' + current[a:b] +
    predicate(current) + '\n' + legacy + '\n}\n')
PY
```

在基础编译命令增加以下参数即可执行精确 predicate 差分：

```sh
'-DED_H3_SAGE_HOST_PREDICATE_HEADER="/tmp/h3-sage-predicate.hpp"'
```

额外独立 header 的 `-Wall -Wextra -Werror` 语法检查通过；共享补丁 `git diff --check` 通过。ASan/UBSan 构建尝试因系统缺 `libasan` / `libubsan` 未能链接，**未执行 sanitizer 测试**，不将其计入 PASS。

本次全部 PASS：

- Policy/global isolation/CPU fail-closed。
- 共享 H3SA / legacy SAGQ userdata ABI。
- Exact-source CUDA **host** predicate 1296 项 matrix + malformed guards。
- H3 shape/type/layout/overflow guards。
- 真正 CUSTOM 图（只构图，不计算）；8 个 F16/F32 组合规范化；strided V conditioning CPU 数学。
- output inverse-scale + batch=1/2/3 CPU 图。
- **实际** `ggml_ext_attention_ext` 的 CPU dense/flash 图。与双精度数学参考最大绝对误差：dense **1.63913e-07**；CPU flash **0.00129761**。错误的 `sm_scale=1/128` 相差 **0.711258**，测试能明确区分。
- 已有 contiguous F16 V conditional-scale caveat 测试（预期约 128 倍，不移植该行为）。

### CUDA 编译尝试（未完成，未执行）

仅尝试将现有 CUDA TU 编译成 `/tmp/h3-ed_cuda_sage_attn.o`，没有链接或运行：

```sh
/usr/local/cuda/bin/nvcc -std=c++17 -O2 -DGGML_MAX_NAME=128 \
  -DED_ENABLE_CUDA_SAGE_ATTN -Isrc -Isrc/core \
  -Isrc/core/backend/ggml/cuda -Isrc/core/backend/ggml/cuda/sage \
  -Ithird_party/ggml/include -Ithird_party/ggml/src \
  -gencode arch=compute_89,code=sm_89 -Xcompiler=-fPIC \
  -c src/core/backend/ggml/cuda/ed_cuda_sage_attn.cu \
  -o /tmp/h3-ed_cuda_sage_attn.o
```

系统 nvcc 13.0.88 可启动，现有 vendor 宏重定义/no-effect warnings 出现后，内部 `/usr/local/cuda/nvvm/bin/cicc` **Permission denied**，退出 126。按审批流程在沙箱外重试后仍相同；**没有生成 `.o`**。未修改编译器权限、没有下载工具链，也没有规避批准范围。

待主线程/有权限环境完成：CUDA TU 编译和链接、主 model 路由集成、真实 backend supports_op 验证、授权后的 GPU 精度/质量/长序列与性能 A/B。这些均不能由当前 CPU PASS 推定通过。
