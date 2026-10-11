# Mac 协调者交接

## 目标

完成 edge-dit.cpp 的 MiniMax-H3 CUDA-only 优化，默认关闭且不影响其他模型。Mac本地Codex协调当前H200服务器和用户另一个已有SSH服务器；用户只需一次启动，不手工转述日常进度。

## 初次交接（仅检查，不恢复开发）

1. 读取本地SSH配置的Host别名列表，勿输出密钥/凭据；只考虑用户已用于这两个服务器的连接，不枚举整个公司网段。不确定目标只问一次。
2. 找到现有项目所在服务器：当前主机 `A03-R50-I14-90-7823344.JD.LOCAL`，用户 `wangtianyang.21`，项目 `/export/home/wangtianyang.21/code1/edge-dit.cpp`。
3. 通过SSH读取本文件、`h3-development-log.md`、`h3-precision-lock.json`和研究文档；核对暂停点和未提交代码，不重建基线假数据。
4. 若本地Codex工具提供list_threads/read_thread/send_message_to_thread/wait_threads且能定位当前远程聊天 `01a12a7a-c737-7b13-8750-cfc804e14382`，尝试有目标的ping并记录回执；不能仅凭有ID宣称通道建立。
5. 若原生跨host不可用，采用Mac主动SSH轮询的任务/结果文件通道：指定独立项目子目录；任务JSON包含task_id、source revision、patch SHA256、weight SHA256、dataset SHA256、状态、负责人；结果JSONL追加时间、错误、stdout/stderr路径、exit code。原子写状态；不得把任务文本当作未经验证的任意shell执行。
6. 当前用户要求暂停；先完成通信与资源检查并反馈通道是否实测可用。收到用户恢复后才继续开发/基准，避免两个窗口重复修改同一分支。

## 后续开发和测试原则

- 一个源代码集成者、其他服务器仅做独立快照测试。来源分支 `codex/minimax-h3-cuda-optimizations`，不上main、不强推。
- 单卡基准以precision-lock为准；不得因GPU较小改量化/剪枝，资源不够按固定新placement建独立baseline，或等待可用卡。
- 可复用 `wty-edgedit-dev` 容器：项目 `/workspace/edge-dit.cpp`，模型根目录 `/models/MiniMax-H3-GGUF`。
- 固定源码快照 `_scratch/h3-baseline-97cc585`；部分构建 `build-h3-baseline-97cc585`；脚本 `scripts/build_h3_cuda.sh`。不要删除中间构建。
- 未完成workflow：`scripts/h3_cuda_workflow.py`（尚不存在），不能直接宣称一键回归就绪。
- 主实验Ref2VA混合；另验证T2VA、首帧/尾帧/首尾帧、参考图/视频/音频/视频配对音频、CFG和重复请求。资产按合法帧数、维度、顺序和模态标签固定。
- 资源监控优先本地App原生自动化（若确实可用），否则经过许可的Mac持久调度/服务器调度；当前尚未安装。GPU忙时不通知每轮；事件变化/失败/完成/必要审批才通知。
- 非正常退出保存完整日志，禁止删文件、停止他人进程、暴露公网控制端口或复制私钥。

## 已保存状态

读 `h3-development-log.md` 即可定位当前成果与不足；精度以 `h3-precision-lock.json` 为准。完整权重hash未完成，性能尚无数据。
