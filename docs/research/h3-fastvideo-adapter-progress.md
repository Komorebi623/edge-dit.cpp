# H3 / FastVideo完整adapter：CPU接续进展

2026-10-14，接续`753adab`。**不是新的GPU性能结果**：服务器8卡持续被他人训练占用，没有运行FastVideo或Ref2VA候选，没有停止/修改他人进程。已验证的Larry4/LightFL4视频结果继续以`h3-lora-results.md`为准，不被本轮未验收路径覆盖。

## 已完成

1. **真实VSA权重到位**：官方ModelScope修订`e0b0689c28d0bb78dec099df1fd31df6bb4c7aa0`，`vsa-datafree/adapter_model.safetensors`，5,339,117,712B，整文件SHA256`42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470`。4并发Range、正常IPv4/TLS，77.38秒完成；直接写一个新文件，不生成重复大分块。实际856个张量全部BF16：362对A/B、82个dense增量、50个`to_gate_compress`矩阵，每个7168×5376。50个门控全部有限且非零，并非只从模型卡推断。
2. **FastVideo dense载荷标准化**：显式`--layout fastvideo-dense`（别名fastvideo），支持完整参数增量、顶层投影、norm/bias；362对分离因子规范化258对＋85delta，共601个BF16张量，所有809源key完整映射、形状/finite核对。QKV用blockdiag，不假设A相同；SwiGLU仅交换输出行。未知key、VSA gate/replacement仍拒绝，旧native/Diffusers入口不改变。
3. **独立dense增量运行组件**：只有manifest明确`payload_format=fastvideo-dense-v1`才放行`.weight.delta/.bias.delta`。矩阵以额外`DeltaW*x`计算，不将Q4还原成整张高精度权重合并；norm/bias先FP32相加后回原运行参数dtype，bias消费需要F32时再扩宽已舍入值。底模与adapter数据不写回；全部是图切分可识别的推理叶子。
4. **CPU与编译**：16/16 CTest、53/53 workflow/media、18/18标准化测试及其他原套件通过；LoRA+dense专项35,622检查、66个CPU图通过，包含拒绝/scale0逐位不变、图中实际覆盖与卸载参数/字节统计、BF16向量舍入。GPU候选只重编2个C++文件，CUDA依赖原样复用；**编译通过不是GPU验证**。

## 明确的数值/范围限制

- `(Q4(W)+DeltaW)x`和`Q4(W+DeltaW)x`不等价。本项目选择保持原Q4不变的独立增量方案，不冒充作者BF16底模输出。
- 少步是新模型＋NFE配方，不能按与原20步输出的SSIM单指标宣布画质优劣。
- norm/bias回dtype发生真实舍入；不能把FP32未舍入sum直接交给消费者后声称与原模型精度相同。
- FastVideo四步公开入口使用5个sigma点（含末尾0）＝4次forward，基础双shift12/3；与V2八步shift10/3的特殊梯度不同。
- VSA是稀疏细分支＋池化压缩分支＋**原始线性门控**，不是sigmoid gate，也不删除Q输出。还需独立tile64原生执行/训练门控接入和CUDA质量性能验证，不能因门控已下载就标为完成。

## 下一步

已准备`normalized-fastvideo-dense4/manifest.json`及冻结构建；有独占GPU后只跑1次T2VA，复用原baseline。同GPU再补已准备LightRef4 mixed，不重新跑baseline。GPU忙时继续独立VSA有界图组件，生产开关保持拒绝未实现配置。

证据：`benchmark/h3_cuda/fastvideo-adapter-progress.json`；根`_scratch/h3-vsa-continuation-20261014/`。本记录仅CPU/权重/编译进展，不虚构GPU速度/输出。

## 独立VSA图组件（不冒充全方法）

新增`minimax_h3_vsa_graph.hpp`以真实三维tile64映射构建池化Q/K/V、相关性、全tile压缩attention、恢复原packed行和原始线性gate合并；fine attention由调用者注入，**没有把它替换成SubBlock，也尚未接入生产VSA开关**。图固定44节点，不随head/tile数展开；内存预算显式、溢出检查，常量绑定所有权跨重放保留。

独立CPU参考对照54,029检查、12个图，最大误差9.54e-7；ragged3D、text/audio/reference、负gate、零gate和padding排除均覆盖。真实量级S15256/T285/H56/D128完成仅元数据建图，保守“所有中间buffer无复用之和”7.203GB，**不是实际峰值显存**。后续还需GPU fine稀疏内核、训练门控模型挂接、完整任务验证。

当前冻结候选为`dense-source-02`/`dense-build-02`，完整源码指纹`d64c45f2ae42ee510cfe73b59ebd68810a531df5e47b443048961d9963ae2e44`。所有src/include/CMake与构建记录重新匹配；仍是CPU/构建完成，GPU未测。22:32复查8卡依旧被他人训练占用，未启动生成。
