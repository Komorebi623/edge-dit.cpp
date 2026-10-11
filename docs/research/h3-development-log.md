# H3 CUDA 开发日志

## 2026-10-11，Asia/Shanghai：暂停点

- 分支：`codex/minimax-h3-cuda-optimizations`；基准原始提交 `97cc585`。
- 用户明确要求暂停开发，先建立 Mac / 另一服务器协调。三个子智能体已停止并关闭；本任务基准编译已单独中断，缓存对象保留。没有停止任何他人任务。
- 8 × H200，每卡显存 143771 MiB，查询时已用约 139218–139220 MiB。未跑 H3 推理、未产生性能数据。
- 当前主机 `A03-R50-I14-90-7823344.JD.LOCAL`；用户 `wangtianyang.21`；项目 `/export/home/wangtianyang.21/code1/edge-dit.cpp`。
- 当前聊天 ID：`01a12a7a-c737-7b13-8750-cfc804e14382`。该 ID 是身份信息，不是可直接访问的远程 API。

## 已完成／未完成

- 已保存：官方源码研究、H3-only CUDA policy、fusion 接线、standalone policy tests、固定 source snapshot 和构建脚本。
- Policy tests：本线程 C++17 `-Wall -Wextra -Werror` 编译并执行通过。
- H3 pipeline CPU syntax-only：子智能体报告退出 0；不是完整链接或数值回归。
- CUDA baseline：配置通过，部分目标已编译；用户要求暂停，编译中断；不视为构建完成。
- 待做：完整 CUDA/CPU 构建、算子数值回归、真实 Cache-DiT/AdaLN/Sage/稀疏/VDN/Turbo、工作流 manifest/媒体校验、所有模态 baseline 与 A/B。
- `ED_H3_FUSED_OPS` 已接线但尚未 CUDA 数值验证；其他优化请求 fail closed，不冒充已实现。
- 没有创建定时任务，没有启动常驻 GPU 等待进程；因用户当前要求暂停，协调建立后再安排。

## 访问故障及处理

- 初始沙箱 Git/网络/driver/Docker权限受限；获准后 GitHub、GPU查询及现有容器可用。
- 网页搜索工具返回空；GitHub通过获准网络读取固定源码。
- 官方 Codex 文档页面读取返回403，无法确认跨 host 聊天/自动化功能的当前支持情况。
- 当前暴露工具不含 create_thread/send_message_to_thread/wait_threads/automation_update，不能声称已创建跨聊天通信或定时任务。
- Hugging Face权重模型卡独立核验未完成；不以README声明代替权重完整性检查。

## 精度决策

- 子任务中曾讨论 pruned BF16→Q8，但没有转换或执行；该建议作废。
- 主实验锁定现有完整 FL2VA/Ref2VA Q4系列文件，Qwen Q4系列文件，视频VAE全F16，音频VAE全F32。
- 已解析实际tensor types：DiT含Q4_K/Q4_0/BF16/F32；Qwen含Q4_K/Q6_K/BF16。不能笼统称全部张量Q4_K_M。
- 详见 `h3-precision-lock.json`。仅存header checksum，完整文件SHA256待执行；禁止未经完整校验跨服务器声称同权重。
- 硬件更换后在新硬件重新跑同配置baseline+candidate，禁止跨卡直接计算加速比。

## 恢复顺序

1. Mac协调者确认两个SSH别名和任务范围，检查GPU/显存/工具/模型可读性，不抢占资源。
2. 建立唯一写入者与共享状态/JSONL日志，必要时同步checkpoint；不推送未验证代码到main。
3. 恢复固定baseline构建，完成实际workflow与fixture manifest，完整权重hash。
4. 获得GPU后baseline→单开关candidate→视频/音频/同步校验→报告；无可用GPU仅开发。
5. 定时监控仅有状态改变、完成、失败或需审批时通知，忙时保持安静；每次实验单独锁与资源再检查。

## 2026-10-11：恢复 CPU-only 开发

- 用户撤销暂停，要求补代码，禁止 GPU 推理和杀进程；本轮执行只有源码读写、下载、CPU编译/微型算法/媒体测试。
- 9官方仓库167份固定源码约2.28MiB已完成hash核验；没有安装上游训练框架或下载新大模型。
- fixed baseline CUDA编译成功；候选CUDA编译进行中；CPU全工程构建已通过，10个CTest已通过，整合runtime额外测试待最终汇总。
- 媒体workflow 47项通过；sourcefetch18项通过；CPU tiny LoRA初始4项通过，安全review后新增非finite/FP16overflow测试待重跑。
- 主精度完全不变；五组件整文件SHA256已完成，fixed素材/模型锁在 `_scratch/h3-fixtures-20261011/locked-manifest.json`。
- H3-only CacheDiT/AdaLN/SubBlock/Sage/Turbo接线已实现，VDN/VSA只完成可审计的数学/原生基础组件，不能视为全模型CUDA部署。
- actual tiny H3集成测试发现import callback缺失，已修；独立review发现Turbo全来源校验、音频坐标重复换算、resetOOM、step越界和LoRA非finite漏洞，均已修代码，回归待最终汇总。
- 暂不创建GPU定时任务或跨机器协调，不运行GPU；方法状态/下一步见 `h3-cuda-optimization-status.md`。

### 最终软件回归

- 纯CPU全工程构建成功；一键脚本实跑：12/12 CTest、48/48 workflow、18/18 sourcefetch、5/5 tinyLoRA通过。
- 实际tiny H3 integrated测试24次fullgraph对照全部bitexact，native/curve分别AdaLN GEMM从5到0。它不是完整H3生产权重画质验证。
- CUDA固定baseline与candidate已链接成功，包括H3SA Sage CUDA源。没有执行CUDA二进制推理，也未查询/停止GPU任务。
- independentreview5项代码问题已修，增加非finite/FP16overflow测试；真实运行发现绑定callback缺失已修，runtimeCTest三次通过。
- 验证摘要 `benchmark/h3_cuda/validation.json`；全部大模型性能/质量为空。VDN/VSA仍存在明确未实现部署项。

- 补充融合算子actual CPU图56次对照最大绝对误差0；最终总计12/12 CTest。所有子智能体已关闭，不留后台推理任务。
- 最终CUDA候选重新链接（含8个独立开关，VDN/VSA显式未支持）；source-tree digest与binary digest进入build provenance，构建期间未运行GPU。
