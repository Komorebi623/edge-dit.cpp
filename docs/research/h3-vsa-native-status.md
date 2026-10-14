# H3 / FastH3 VSA：独立原生路径，等待GPU验收

日期：2026-10-14（Asia/Shanghai）。**本轮完成代码、模型载荷标准化、CPU回归与CUDA编译，但没有GPU运行结果。** 原Larry4步106.12秒／LightFL4步110.85秒仍是此前已验证结果；不能据本文件宣布VSA提速或画质通过。

## 当前实现

- 独立编译开关`ED_ENABLE_CUDA_H3_VSA=OFF`；与SubBlock可分别编译，均关闭时不生成/替换CUDA dispatcher。共享ggml源码不修改，只从hash核验的原文件生成构建目录副本，能串接已验证SubBlock hook而不丢失它。CPU配置测试覆盖4种开关组合及原文件/前序生成文件破坏拒绝。
- 运行时要求`ED_H3_TURBO=1`＋`ED_H3_VSA=1`＋完整`fastvideo-vsa-v1` manifest，其余新增研究方法关闭。T2VA限定，禁止用FL2VA训练学生处理Ref2VA或关键帧输入；新路径不改变默认H3或其他模型/backend。
- **不是SubBlock算法**：用真实3D tile64（4×4×4）映射，按tile池化Q/K产生F32相关性，每head、每查询tile独立选择视频K tile；text/audio前缀查询完整稠密，所有查询输出均保留，前缀K/V始终保留。
- 新CUDA fine kernel使用BF16 QK／BF16 PV、FP32累积与在线softmax，直接索引入选tile。分批head的K/V pack≤128MiB，原34KiB共享内存上做K/V交错异步搬运；准确处理ragged末tile，不生成全序列稀疏mask，不借全局开关影响普通注意力。
- 原生压缩图复用同一池化分数，计算全tile压缩attention，再恢复packed行，以**训练的原始线性gate**相乘加回fine输出；不是sigmoid。50个门控各7168×5376，BF16保持，完整形状/finite/非零/实际图中使用均检查。
- A/B、82个dense delta和50个新gate共648个规范化张量由同一H3 runner权重池持有并参与图切分上传/预算；base Q4文件不写回/不再量化。模型图固定有界，原有少步Euler/simple双shift12/3契约复用。

## 真实模型载荷与修复

原始VSA adapter `vsa-datafree.safetensors`：5,339,117,712B，SHA256 `42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470`。规范化后5,625,360,384B，SHA256 `b9f3760573c568c8916e2ad62dc517bef185a1f21f8c202a79339ec190d468a6`。原始856个BF16tensor→258对融合A/B＋82delta＋50gate；不忽略未知字段，正常化入口仅精确支持V1 datafree完整payload。完整原始和生成文件hash均记录。

- 90%稀疏率不能先保存为float32再求ceil：10块可能错误保留2块。当前固定V1配额用整数`ceil(video_tiles/10)`，避免“实际多保留一块”；其他实验比例仍独立双精度。
- 不能仅验证50gate＋“提供的参数都被使用”。运行时同时核对**完整258对／82delta的准确key集合**，防止少加载一个训练norm增量仍被接受。
- 原生CUDA片段和整模型代码均编译通过；静态审查未发现其他确定性缺陷。但BF16 kernel的数值、实际图分段buffer峰值、时间/音画输出必须由真实GPU对照验收，编译不能替代。

## 验证与证据

| 范围 | 已完成 | 不能推出什么 |
|---|---|---|
| CPU统一回归 | 18/18 CTest，53/53 workflow/media，27/27 LoRA规范化，4/4 VSA构建配置及原附加套件 | 不证明CUDA输出或速度 |
| VSA fine shape/CPU oracle | 19,821检查、42参考比较；ASan/UBSan检查通过（测试代理记录） | CUDA kernel尚未执行 |
| 完整原生图装配 | 一次真实router+fine+compression图，52节点；精确配额与缺参数拒绝 | 不证明全模型峰值显存 |
| CUDA编译 | 新fine/router/dispatcher、H3模型接入与ed-cli链接完成 | 不证明kernel数值正确 |
| GPU | **未运行：8卡持续被无关训练占用** | 无VSA时延／质量数据 |

构建：`_scratch/h3-vsa-native-20261014/source-02`、`build-02/candidate`，source hash `08aae0256f421ae7ffc70ac9325b3663c6151742f5ab5dd4671a0116db0e2b54`；之后仅CMake新增CPU native测试注册，所有src/include仍与冻结构建一致。CUDA primitive库`kernel01`的所有输入完整hash匹配，原ggml/oneDNN clean。

## 待GPU空闲后的有限步骤

1. 指定同一GPU UUID，已有占用保护门通过后运行`h3_vsa_native_gpu_test --run`。必须成功再跑VSA模型；不能直接跳过tiny数值比较。
2. FastVideo dense4与VSA4分别各1次T2VA，复用已有原20步baseline，区分学生质量与同权重近似误差。不叠Cache/AdaLN/SubBlock/Sage，不预热＋重复三次。
3. 单独完成已下载匹配LightRef4的mixed一次；VSA V1不用于mixed。
4. 验收前不改默认开关、不宣称VSA优于已验证Larry。若GPU数值/显存失败，保存失败证据后修复，不以fallback dense冒充VSA。

当前没有后台GPU任务或抢卡循环。占用已跨多个目标回合确认，不能停止他人训练；三篇分享链接的身份又被微信CAPTCHA/小红书登录拦截，需要标题/正文截图或公开论文地址，不能猜测项目身份。目标没有完成。

机器证据：`benchmark/h3_cuda/vsa-native-progress.json`。此前成功LoRA产物继续在`benchmark/h3_cuda/lora-results.json`。
