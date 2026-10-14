# H3 / Q4：独立LoRA少步部署与结果

日期：2026-10-14。接续`7dfe3e3`。**已实际接通Larry及LightX2V的4步LoRA，原GGUF字节不变、BF16因子独立计算；不是合并后再量化，也不需要ComfyUI。** 默认关闭，只有H3/CUDA显式`ED_H3_TURBO=1`＋`ED_H3_LORA_MANIFEST`生效。

## 固定环境与结果

单RTX4090 GPU0，原混合Q4_K/Q4_0/BF16/F32完整H3、原Qwen与VAE精度preserve，864×480/124帧/24fps、seed42、CFG1、CPU6、DiT/TE/VAE卸载、22GiB规划预算。共同同步cuDNN计划。原baseline`97cc585`保留已有优化，**本轮没有重跑baseline**。LoRA改变模型、步数和采样器，不能包装为同模型无损优化。

| 模型/方案 | DiT实际调用 | CLI秒 | DiT秒 | 生成pipeline秒 | 对原baseline CLI加速 | 采样峰值MiB |
|---|---:|---:|---:|---:|---:|---:|
| 原Q4，无LoRA，RES/simple | 20 | 254.33 | 211.297 | 224.456 | 1.000× | 16035 |
| **Larry v4-step600 EMA，Euler/simple** | **4** | **106.12** | **41.661** | **54.257** | **2.397×，省58.28%** | **16457** |
| LightX2V FL v0.1，Euler/simple | **4** | **110.85** | **43.738** | **56.371** | **2.294×，省56.42%** | **16725** |

每个候选仅1次成功的完整目标生成，无预热/三次重复。CLI含启动、加载、完整base/adapter身份hash和编码封装；外层工作流hash/媒体复验不计入。LoRA CLI有额外身份核验开销，不拿54s pipeline冒充106s总耗时。完整内核依赖未重编，仅增量构建两个C++编译单元；源快照/复用DSO/hash见结果JSON。

## 选择与质量边界

当前山涧静态/小运动场景**优先推荐Larry4步**：adapter较小（779,849,816B，259对），实测CLI/显存略优；三时刻抽帧保留自然流水、树木和远山，未见明显结构损坏。Light4（原文件1,383,677,888B，312对）也完整生成了连贯山涧，但构图不同。**不是跨场景“最好”结论**；未进行高速人物动作/口型/主观听评。作者建议Larry遇4步大运动拖影可用8步，本轮4步已工作所以不追加8步耗时实验。

与原20步dense结果的距离：Larry SSIM0.767912、audioRMSE0.014735/relativeL2 0.9074；Light SSIM0.681445、audioRMSE0.017197/relativeL2 1.0590。学生改变生成轨迹，SSIM不是“画质保留率”；不能按这些数字直接宣布Light画质差。两条视频均864×480/124帧、音频32kHz双声道；解码有限值检查通过。Larry音频peak0.0602/RMS0.00457，Light peak0.0726/RMS0.01283，均非静音、没有绝对值≥1的解码样本；不代替主观听评。

Larry另完成**256×256/22帧首末帧条件**功能验证，4次调用和音频输出通过。这是小尺寸功能任务，不纳入上面性能表。

## 最小且解耦的接入

- 独立`minimax_h3_dynamic_lora.hpp`实现`base_linear(x)+coefficient*B(Ax)`；复用原量化线性层，不复制或重写CUDA矩阵乘。BF16因子不再量化。LoRA与SubBlock/Cache/AdaLN/Sage/VSA组合当前显式拒绝，避免未经验证叠加。
- 因子在H3 runner的同一params_ctx/WEIGHTS buffer内分配，完整参与分段权重大小统计、上传与释放；原模型参数不写回、不落盘覆盖。
- 专用manifest绑定base完整SHA、FL/Ref分区、normalized adapter完整SHA、系数、4/8次调用、双shift及Euler/simple/CFG1。请求不能切换adapter或不匹配采样；FL adapter拒绝Ref参考输入。
- `prepare_h3_lora.py`只做CPU格式标准化。Larry原生布局保留全部259对（含51组AdaLN）；Light分离Q/K/V用A纵向拼接＋B块对角构造，**不假设三个A相同**，SwiGLU只交换B输出行的两半。Light FL v0.1 alpha8/rank128，因此系数0.0625；Larry alpha=rank，因此系数1。
- FastVideo dense4实际文件含724个低秩tensor及**85个dense增量**，不是A/B-only。本轮未忽略这些内容硬加载；这个候选下载/检查完成，但未运行。

## 验证与已修复失败

- CPU统一回归15/15 CTest（包含新LoRA33805检查/20CPU图），53/53 workflow/media，normalizer6/6及原附加套件通过。
- 最终短CUDA：10case、20graph、142278检查通过；覆盖实际量化Q4_0/Q4_K、BF16/F16/F32因子、跨batch/view/tail、系数0与原CUDA图逐字节相同、权重字节不变。与独立同CUDA数学图逐字节校验；理想标量与已有CUDA量化/舍入误差仅作诊断，不通过放宽阈值掩盖适配错误。
- 初版将`ggml_set_param`误作推理参数标记；该训练标记让A/B成为内部计算节点，分段执行时未上传。两次首forward失败保留。修复为与原模型相同的推理叶子后通过；新增图分段回归逐个证明A/B属于INPUT_PARAM，并含旧bug负例。
- Larry成功生成后的首次媒体比较因相对路径失败，仅修复后处理、**没有重跑模型**；原报告与恢复报告均保留。
- 极长序列自动切片QKV尚未支持LoRA，提前明确拒绝而非空指针；当前目标与小首末帧验证不走该限制。

## 下载、复现与尚未集成项

下载障碍已实际解决：该环境IPv6出口出现证书/认证拦截，而官方ModelScope CDN在**IPv4＋正常TLS**下可下载。原始文件全部完整SHA匹配公开元数据；不绕过认证、不改服务器路由/证书。后续大文件用有界分块并发。原HF API IPv4仍超时，因此本轮按已验证ModelScope修订与文件hash记录，不冒称核实了HF全球最新。

复用命令：先`prepare_h3_lora.py --help`生成独立manifest，再用`run_h3_lora_trial.py --baseline <已封存report> --binary <candidate ed-cli> --adapter-manifest <manifest> --output <新目录>`；它复用既有workflow的显卡占用、完整hash、媒体规格与比较工具，仅一次生成。原普通H3不开总开关行为不变。CPU一键回归仍为`bash scripts/test_h3_cpu.sh`。

**VSA没有被假冒为普通LoRA或SubBlock。** 官方VSA adapter确实公开列出，但需要额外trained gate/dense delta/压缩分支。为满足“先用上好LoRA、不要大改、保持解耦”，这版只交付已验证的普通LoRA；VSA/VAE替换保持独立、未宣称已集成。用户三个分享链接已IPv4重试，但分别是微信CAPTCHA/小红书登录门；未拿猜测的文章身份冒充已读，详见`h3-new-links-audit-20261014.md`。

## 证据入口

`benchmark/h3_cuda/lora-results.json`：完整配方、原始/normalized权重SHA、动态库/源快照、成功与失败报告、速度/质量/峰值、产物绝对路径。证据根`_scratch/h3-lora-goal-20261014/`；比较图`quality-preview/larry-light4-vs-baseline.png`；视频`larry4-v2-trial/t2va.mp4`、`light4-v2-trial/t2va.mp4`。原文件/失败日志不删除。

## Ref2VA与当前资源（接续）

LightX2V Ref2VA4步v0.1已用4并发Range下载，完整SHA256`9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99`；按官方Ref4示例默认alpha8/rank128、shift12/3规范化，匹配原Ref2VA Q4完整hash。**mixed未运行**：启动前8张4090均被无关训练占用，安全门拒绝，未停止任何进程。未来只运行一次该candidate并复用已有mixed baseline。报告`light-ref4-trial/report.json`明确记录占用，不含虚构时间/产物。
