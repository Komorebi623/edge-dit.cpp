# H3 FastVideo VSA — 独立 CPU reference 进度

日期：2026-10-11。状态：**CPU mathematical reference 已实现并验证；不是 native CUDA，不启用 production。**

本次唯一写入的工作区文件：

- `/export/home/wangtianyang.21/code1/edge-dit.cpp/src/dit_models/models/minimax_h3_vsa_reference.hpp`
- `/export/home/wangtianyang.21/code1/edge-dit.cpp/tests/h3_vsa_test.cpp`
- `/export/home/wangtianyang.21/code1/edge-dit.cpp/tests/h3-vsa-progress.md`

没有修改共享 model、FFN、loader、pipeline、CMake 或 GPU 文件。测试产物只放在 `/tmp/h3-vsa-reference-tests`。不重复主线程的 VDN oracle，不运行 GPU 检测、CUDA、真实模型推理。

## 1. 实际源码与不可混淆的语义

固定上游：`hao-ai-lab/FastVideo`，commit `d5253fd71055927f12aa6c4aa679b93a00238851`。不是声称跟踪当前最新上游。源码根目录：

`/export/home/wangtianyang.21/code1/edge-dit.cpp/_scratch/h3-reference-src/fastvideo/d5253fd71055927f12aa6c4aa679b93a00238851/`

以下文件的 Git blob SHA-1 已与本地 `SNAPSHOT-79b805e17d76d2fa.json` 核对一致：

| 固定源码文件 | Git blob SHA-1 | 依据 |
| --- | --- | --- |
| `fastvideo/attention/backends/video_sparse_attn_h3.py` | `10741e394e10ad523a11c0d1e1321b86c14be4a5` | geometry、pool、mask、forward、compression |
| `fastvideo/attention/backends/video_sparse_attn.py` | `cf77e13d7eaad80ac2dbfd1314860dc8b7aa0e50` | clipped 3D tile、valid prefix、`compute_topk` |
| `fastvideo/models/dits/minimax_h3.py` | `070d13637841c9668a3ae0bf86daf94a6a0574d0` | `to_gate_compress` 无偏置线性投影、零权重分支跳过 |
| `fastvideo/pipelines/basic/minimax_h3/vsa_guard.py` | `5820252acdd3ed65a86d24f15937ad7aec21c209` | 拒绝全零 gate、NVFP4 magnitude 检查 |

### Gate 公式核对结果：H3 不是 sigmoid

需求中提到 learned sigmoid gate，但这份实际 H3 源码没有该操作：

```text
Qbar[t] = sum(Q[t, valid rows]) / valid_count[t]
Kbar[t] = sum(K[t, valid rows]) / valid_count[t]
Vbar[t] = sum(V[t, valid rows]) / valid_count[t]
score[b,h,i,j] = dot(Qbar[b,h,i], Kbar[b,h,j]) / sqrt(head_dim)
compressed[b,h,i] = sum_j softmax(score[b,h,i,:])[j] * Vbar[b,h,j]
gate[b,row,h,d] = sum_c W[h,d,c] * hidden[b,row,c]  # bias=False, NO sigmoid
output[b,row,h,d] = fine[b,row,h,d] + gate[b,row,h,d] * compressed[b,h,tile(row),d]
```

对应 H3 源码：`_pool_tiles` 459–472 行；`_build_block_mask` 474–520 行；score 790–795 行；compression/merge 963–991 行。模型中 gate 定义 274–280 行，实际传入线性输出 407–410 行。

- **零 gate 恰好纯 sparse**；不转换成 sigmoid(0)=0.5。负数和大于 1 的 gate 合法。
- 融合是 additive residual，不是 `(1-g)*fine + g*compressed`。
- compression 对**所有 logical tiles**做 softmax，不限 selected 或 discarded tiles，也不按 tile valid_count 再次加权。
- dense/all-keep 只改变 fine mask；非零 gate 的 compression 仍然相加。因此 active-gate all-keep 不等于 vanilla dense。
- dense prefix queries 的 **fine branch** 永远 dense；其 compression 也正常添加，不能断言 active-gate 的最终 prefix output 等于 ungated dense。
- 本 oracle 接收已经 Q/K norm、RoPE 后的 attention Q/K。没有重复处理 RoPE、QK norm、KV scales，也没有照搬其他 H3 sub-block/LSE router。

## 2. 实现范围

### Geometry、tile/untile 和 padding

- selectable 64 `(4,4,4)`、128 `(4,4,8)`、256 `(4,8,8)`。
- 输入 `Segment::video` 为 token grid；`patch_latent_segments` 单独将 raw latent grid 按正且整除的 patch 转为 token grid。
- 允许 text/audio/dense reference/video regions 任意交错；所有 dense segments 移到 leading prefix，每个 segment 单独切块，不跨边界。
- 各 video region 依原有 region 顺序生成独立 contiguous tile span；最后一个 span 视为 generated video。
- 遵循上游 clipped 3D `(t,h,w)` flatten 后压紧到 tile 前部，不能把短 3D tile 当作带内部 spatial holes 的整块。
- 保存 tile partition、source index、original-to-padded inverse、valid block sizes；检查 injective inverse、segment purity、prefix/span 覆盖、padding、整数溢出。
- **保留每一个原始 token**，输出原始 packed order/length；VSA 选择连接而非删 token/FFN 行。
- odd logical tile-64/128 可选增加一个 zero-valid dummy partner。dummy 仅为传输模拟，不加入 scores/mask/compression，并在 untile 时丢弃；tile-256 或 even tiles 拒绝该参数。
- CPU oracle 显式忽略 invalid slots，包括测试注入的 NaN/Inf padding。上游运行时仍要求 padding buffer 为零；这里不是声明 GPU kernel 接受非零/NaN padding。

### Routing 与计算

- 输入/输出使用一致 `[B,token,H,D]` float vector；明确不是 ggml native layout。
- F32 valid-token mean pooling 和 pooled score，逐 batch/head/query tile 独立选择。
- 默认 exempt 模式：每个 video query 保留所有 dense prefix keys，并**分别**对每一个 video region top-k。
- 上游 keep 数公式：`max(1,min(ceil((1-sparsity)*region_tiles),region_tiles))`。
- generated sparsity + reference keep rate 映射；multi-region 要求 reference keep 或显式 per-region budgets。
- 额外 CPU experiment API 支持 exact `RegionBudget::topk` 或逐 region sparsity；不是新增 production knob。
- 支持单 generated-region compete（budget 为 video top-k + prefix tiles），拒绝 multi-region compete。compete 下 prefix keys 不强制保留，但 prefix queries 仍 dense。
- force-dense schedule / sparsity <= 0 覆盖全部 regions；all-keep mask 退化为 dense。
- fine branch 用 max-subtracted softmax，对 mask 内所有有效 key tokens 做精确数学 attention。
- compression branch 使用同一 pooled score 的全 tiles softmax、有效 pooled V、每 tile 广播。
- gate 无偏置 raw linear、per-token/head/channel additive fusion；不 sigmoid、不 clip，不修改输入/权重/FFN。
- CPU score ties 选择较小 tile ID。上游 torch.topk 不承诺 tie order，不能声称跨实现 tie mask bitwise parity；浮点近似并列也可能改变选择。

### 只读 weight capability guard

- 逐层要求 `transformer_blocks.<layer>.attn.to_gate_compress.weight` 与显式 architecture 的 `[heads*head_dim,hidden_dim]` 匹配。
- base H3 没 gate、覆盖缺失、offload/lazy placeholders：Missing，不能自作主张造 gate/启用 VSA。
- tensor 名称、shape、storage、非有限 float 不匹配：Invalid。
- 所有 gate 为零：ZeroOnly，拒绝 production；允许调用者显式运行纯 sparse CPU 实验。
- 完整且至少一个非零 float gate：FloatReference，仅结构上允许 CPU linear 计算；**非零不证明已训练、来源匹配或生成质量**。
- packed NVFP4 仅检查尺寸和 `byte & 0x77`，两个 nibbles 的 +/-0 都算零；非零 packed gate 标为 PackedInspectionOnly，无 scales/dequantization 就不计算 linear gate。
- 相比上游运行时 guard，本地 guard 故意对缺失/未物化 gate fail closed，要求调用者等待真实 weights；不是假装兼容未加载模型。
- `production_ready`、`production_switch_available`、`native_cuda_implemented` **永远 false**；没有任何接到共享 model 的启用路径。

## 3. 验证结果

独立 C++ 测试共有 **121157 checks**，在下列模式全部通过：

1. C++17 `-O2 -Wall -Wextra -Wpedantic -Werror`。
2. `-O3 -DNDEBUG`：不依赖会被 NDEBUG 删除的 assert。
3. `-D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS`：标准库边界/迭代器检查。
4. `-fsanitize=undefined -fsanitize-undefined-trap-on-error`：无需 libubsan runtime 的 trap 模式。

覆盖：short tail、单 token segments、非 tile 倍数 grid、patch 不整除、regions interleaved、modal/segment pure tiles、inverse/padding、每 head/query/batch 每 region top-k、reference keep、forced prefix、compete、all-keep、dense gate、zero/negative/>1 raw gate、weighted compression、无输入/FFN sentinel mutation、dummy partner、poison padding、invalid/nonfinite/overflow guards，以及 24 个固定 seed 的随机 packed masked/dense oracle case。

对照不是同函数自比：C++ 测试另写 original packed-row long-double softmax oracle 和按 tile coordinate stable-sort 的几何 oracle。FFN 测试用 dummy model sentinel 验证独立 API 不触及 FFN；没有宣称验证共享模型真实 forward。

另已从固定 Python 文件 AST 提取 **原始函数体** `_h3_segment_tile_geometry`、geometry helpers、`_pool_tiles`、`_build_block_mask`，用小型 NumPy CPU tensor adapter 执行，比较测试的 `--dump-oracle` fixtures。不 import torch、不 import FastVideo package、不调用 GPU kernel。three tile sizes 的 geometry/inverse/mask 一致；dense override 通过。

| Source-extracted NumPy CPU 对照项 | 三种 tile 最大绝对差 |
| --- | ---: |
| pooled scores | `9.5367431640625e-7` |
| fine attention | `4.470348358154297e-7` |
| compressed pooled attention | `2.086162567138672e-7` |
| raw-gate final output | `4.76837158203125e-7` |

这是数学公式和 CPU functions 的对照，**不是完整 FastVideo runtime/GPU 输出或 BF16/F16 rounding 对照**。Fixtures 使用分离 score，避免将浮点近并列误当语义错误。

未通过/未能执行的检查（不能记成通过）：

- ASan+UBSan linked 模式编译失败：`cannot find -lasan`，本机链接指向缺失的 runtime。
- Valgrind 启动失败：loader 缺 `strlen` redirection 所需调试符号；未安装/修改系统依赖。
- 已用标准库 checked 模式和 UBSan trap 模式补充，但不等价于 ASan/Valgrind 已通过。

## 4. 最小复现

工作目录为上述仓库根；生成文件仅在 `/tmp`：

```bash
mkdir -p /tmp/h3-vsa-reference-tests
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
  -I /export/home/wangtianyang.21/code1/edge-dit.cpp/src \
  /export/home/wangtianyang.21/code1/edge-dit.cpp/tests/h3_vsa_test.cpp \
  -o /tmp/h3-vsa-reference-tests/h3_vsa_test
/tmp/h3-vsa-reference-tests/h3_vsa_test
/tmp/h3-vsa-reference-tests/h3_vsa_test --dump-oracle
```

checked/UBSan trap 模式分别添加 `-D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS` 或 `-fsanitize=undefined -fsanitize-undefined-trap-on-error`，输出到不同 `/tmp` binary。

下面代码块可单独用 CPU Python/NumPy 执行 source-extracted 对照，不需要 torch。它仅移除函数装饰器/annotations，保留实际原始函数体，用 adapter 提供需要的 tensor primitives：

```python
import ast, json, math, pathlib, subprocess, types
import numpy as np

class CPU(np.ndarray):
    def __new__(cls, values, dtype=None):
        return np.asarray(values, dtype=dtype).view(cls)
    @property
    def device(self): return 'cpu'
    def numel(self): return self.size
    def view(self, *shape): return np.reshape(np.asarray(self), shape).view(CPU)
    def reshape(self, *shape): return np.reshape(np.asarray(self), shape).view(CPU)
    def permute(self, *axes): return np.transpose(np.asarray(self), axes).view(CPU)
    def sum(self, dim=None, dtype=None, **kw):
        return CPU(np.sum(np.asarray(self), axis=dim, dtype=dtype, **kw))
    def __truediv__(self, other):
        result = np.divide(np.asarray(self), np.asarray(other))
        return CPU(result, self.dtype if self.dtype.kind == 'f' else None)
    def topk(self, count, dim=-1):
        idx = np.argsort(-np.asarray(self), axis=dim, kind='stable')
        return types.SimpleNamespace(indices=CPU(np.take(idx, np.arange(count), axis=dim)))
    def scatter_(self, dim, indices, value):
        np.put_along_axis(self, np.asarray(indices), value, axis=dim)
        return self

T = types.SimpleNamespace(
    arange=lambda *a,device=None,dtype=None: CPU(np.arange(*a,dtype=dtype)),
    tensor=lambda x,device=None,dtype=None: CPU(x,dtype),
    full=lambda shape,value,device=None,dtype=None: CPU(np.full(shape,value,dtype=dtype)),
    cat=lambda xs,dim=0: CPU(np.concatenate([np.asarray(x) for x in xs],axis=dim)),
    int=np.int32,long=np.int64,float32=np.float32,bool=np.bool_,
    argsort=lambda x: CPU(np.argsort(x)),unique=lambda x: CPU(np.unique(x)),
    zeros_like=lambda x,dtype=None: CPU(np.zeros_like(np.asarray(x),dtype=dtype)),
    ones_like=lambda x,dtype=None: CPU(np.ones_like(np.asarray(x),dtype=dtype)))
root = pathlib.Path('/export/home/wangtianyang.21/code1/edge-dit.cpp/_scratch/h3-reference-src/fastvideo/d5253fd71055927f12aa6c4aa679b93a00238851')
files = {
    'fastvideo/attention/backends/video_sparse_attn.py': [
        'get_tile_partition_indices','construct_variable_block_sizes','get_non_pad_index','compute_topk'],
    'fastvideo/attention/backends/video_sparse_attn_h3.py': [
        '_validate_h3_segment_geometry','_h3_segment_tile_geometry','_pool_tiles','_build_block_mask']}
class RemoveAnnotations(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        self.generic_visit(node)
        node.decorator_list = []
        node.returns = None
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            arg.annotation = None
        return node
    def visit_AnnAssign(self,node):
        return ast.copy_location(ast.Assign(targets=[node.target],value=node.value),node) if node.value else None
ns = {'torch':T,'math':math,'VSA_TILE_SIZE':(4,4,4),'VSA_H3_TILE_SIZE':(4,8,8),'_TILE_ELEMS':256}
for filename,names in files.items():
    nodes = [n for n in ast.parse((root/filename).read_text()).body
             if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    mod = ast.fix_missing_locations(RemoveAnnotations().visit(ast.Module(body=nodes,type_ignores=[])))
    exec(compile(mod,str(root/filename),'exec'),ns)
fixtures = json.loads(subprocess.check_output(['/tmp/h3-vsa-reference-tests/h3_vsa_test','--dump-oracle']))
maxima = {}
for f in fixtures:
    elems = f['tile_elems']
    shape = {64:(4,4,4),128:(4,4,8),256:(4,8,8)}[elems]
    partition,sizes,inverse,prefix,videos,spans = ns['_h3_segment_tile_geometry'](
        (3,(2,3,9),2,1,(1,2,9)), 'cpu', shape)
    for name,actual in [('sizes',sizes),('partition',partition),('inverse',inverse)]:
        np.testing.assert_array_equal(actual,f[name])
    assert prefix == f['prefix']
    padded = sizes.numel()*elems
    packed = {name:np.array(f[name],dtype=np.float32).reshape(2,f['tokens'],2,3)
              for name in ('q','k','v','gate')}
    tiled = {name:CPU(np.zeros((2,padded,2,3),dtype=np.float32)) for name in packed}
    for name in packed: tiled[name][:,inverse] = packed[name]
    pooled = {name:ns['_pool_tiles'](tiled[name],sizes,elems) for name in ('q','k','v')}
    scores = CPU(np.matmul(pooled['q'],np.asarray(pooled['k']).swapaxes(-2,-1))/math.sqrt(3))
    mask = ns['_build_block_mask'](scores,prefix,.8,True,spans,(.7,.8))
    np.testing.assert_array_equal(np.asarray(mask).ravel(),f['mask'])
    logits = np.asarray(scores)
    prob = np.exp(logits-logits.max(axis=-1,keepdims=True))
    prob /= prob.sum(axis=-1,keepdims=True)
    compressed = np.matmul(prob,pooled['v']).transpose(0,2,1,3)
    row_tiles = np.asarray(inverse)//elems
    token_mask = np.asarray(mask)[:,:,row_tiles[:,None],row_tiles[None,:]]
    qh = packed['q'].transpose(0,2,1,3)
    kh = packed['k'].transpose(0,2,1,3)
    vh = packed['v'].transpose(0,2,1,3)
    logits = np.matmul(qh,kh.swapaxes(-2,-1))/math.sqrt(3)
    logits = np.where(token_mask,logits,-np.inf)
    prob = np.exp(logits-logits.max(axis=-1,keepdims=True))
    prob /= prob.sum(axis=-1,keepdims=True)
    fine = np.matmul(prob,vh).transpose(0,2,1,3)
    output = fine + packed['gate']*compressed[:,row_tiles]
    for name,value in [('scores',scores),('compressed',compressed),('fine',fine),('output',output)]:
        actual = np.asarray(value).ravel()
        expected = np.array(f[name],dtype=np.float32)
        maxima[name] = max(maxima.get(name,0),float(np.max(np.abs(actual-expected))))
        np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=3e-6)
    assert ns['_build_block_mask'](scores,prefix,0,True,spans,(.99,.99)).all()
    print(f'FastVideo source-extracted CPU tile-{elems}: PASS')
print('Maximum absolute discrepancies:',json.dumps(maxima,sort_keys=True))
```

## 5. 剩余 gap（不隐藏）

1. **没有 actual FastH3 weights**：不能证明训练来源、architecture 全模型匹配、真实 checkpoint/LORA gate merge、生成质量。capability report 只是 gate inventory 的只读结构检查，不是完整 checkpoint loader。
2. **没有 native CUDA 或生产集成**：不宣称 FA4/Triton/sm100a 实现、性能收益、GPU parity、backward 或完整 forward。CPU reference 为验证语义而保留可审计计算；fine branch 仍有二次方最坏复杂度，不做速度承诺。
3. NVFP4 仅 code/shape inspection，没有 scale、dequantization、FP4 GEMM；BF16/F16/FP8 数值与 fused addcmul rounding 不模拟。
4. 不处理 distributed sequence parallel、cache/tiling buffer alias、autograd、norm/RoPE、attention 输出投影、FFN、PDD scheduler、真实 reference packing 或 model loader mapping。这些均属于后续明确授权的共享模型集成。
5. 无 gate 时允许显式纯 sparse CPU oracle 是测试能力，不是 base H3 production VSA 许可；全 keep + active gate 也不能绕过 FastH3 权重需求。
6. ASan/Valgrind 环境依赖缺失；仅上述可运行检查通过。

结论：几何、routing、fine/compressed branch、真实 raw gate 数学语义和 fail-closed capability reference 已落地；production/native/真实权重验证均明确未完成、未启用。
