# H3 固定源码下载日志

日期：2026-10-11（Asia/Shanghai）。范围仅研究文档、fetch工具、固定来源锁及本日志；下载物在指定scratch。

## 下载与验证结果

- 策略：公开官方GitHub固定commit的精选源码快照，保留仓库目录；未clone大型工程、未取submodule/LFS、未执行源码或安装依赖。
- 来源树与许可证：从GitHub API获取后编入锁；每份下载按固定树的长度和Git blob SHA1核验，并在回执记SHA256。
- 默认dry-run无网络/无写入；显式fetch已走环境升级审批。网络源只允许固定官方raw GitHub，不带token，不读私有仓库。
- 完整下载：**167文件 / 2,387,085 bytes**；回执status=complete，错误0；9份snapshot描述；含回执/manifest的普通文件总大小2,477,711 bytes（不是模型大小）。
- 回执：`/export/home/wangtianyang.21/code1/edge-dit.cpp/_scratch/h3-reference-src/fetch-receipts/20261011T192009-850d44063a9d4cb6ab4a2f5f2ed34123.json`。
- 锁SHA256：`79b805e17d76d2fa8315c5364c0c0ebd7b382b38fce41e4e1c7654f5cad6ca79`。
- 默认限制：目标树64MiB、单文件2MiB；路径限制、符号链接/二进制/LFS拒绝、原子不覆盖、排他锁与已下载hash复用。
- 下载后默认dry-run：待下载0、既有匹配167。已重新逐文件校验固定长度/blob identity。
- 离线内置unit：**18/18通过**；AST语法解析通过；不导入torch/requests，不执行模型转换。
- HF：本轮先前批准的LightX2V权重metadata请求连接失败；按用户要求不再扩大HF调查，无权重内容/文件header独立验证。

## 固定快照目录

| 项目 | 文件数 | 源文件bytes | 本地绝对目录 |
|---|---:|---:|---|
| sglang | 21 | 572382 | `/export/home/wangtianyang.21/code1/edge-dit.cpp/_scratch/h3-reference-src/sglang/53295c697255a5e8ddb263f98ab7e047e2700ad2` |
| cache-dit | 23 | 222005 | `/export/home/wangtianyang.21/code1/edge-dit.cpp/_scratch/h3-reference-src/cache-dit/a7898aaf1ba2c2aabdd94a9526fb5fb82ddc900e` |
| fastvideo | 16 | 248143 | `/export/home/wangtianyang.21/code1/edge-dit.cpp/_scratch/h3-reference-src/fastvideo/d5253fd71055927f12aa6c4aa679b93a00238851` |
| runninghub | 6 | 199996 | `/export/home/wangtianyang.21/code1/edge-dit.cpp/_scratch/h3-reference-src/runninghub/e8f860bb71bc7793e2614e471bbb7e375a8acde7` |
| vdn | 27 | 221044 | `/export/home/wangtianyang.21/code1/edge-dit.cpp/_scratch/h3-reference-src/vdn/e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b` |
| sageattention | 43 | 472712 | `/export/home/wangtianyang.21/code1/edge-dit.cpp/_scratch/h3-reference-src/sageattention/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` |
| vllm-omni | 9 | 185208 | `/export/home/wangtianyang.21/code1/edge-dit.cpp/_scratch/h3-reference-src/vllm-omni/213a1c835ed595d6a51d4f918e3920f4159dd7ed` |
| lightx2v | 15 | 179614 | `/export/home/wangtianyang.21/code1/edge-dit.cpp/_scratch/h3-reference-src/lightx2v/4fe984c07b1610338e97f8c7a268dceef8fd9966` |
| lightx2v-turbo | 7 | 85981 | `/export/home/wangtianyang.21/code1/edge-dit.cpp/_scratch/h3-reference-src/lightx2v-turbo/02e26d591f7a04d5d1a074c9566d5dd4f22f6225` |

## 使用

```bash
python3 -B /export/home/wangtianyang.21/code1/edge-dit.cpp/scripts/fetch_h3_references.py --dry-run
python3 -B /export/home/wangtianyang.21/code1/edge-dit.cpp/scripts/fetch_h3_references.py --self-test
python3 -B /export/home/wangtianyang.21/code1/edge-dit.cpp/scripts/fetch_h3_references.py --fetch --max-mib 64 --max-file-mib 2
```

- 以上精选目录供本地审读，不是可直接构建的完整upstream仓库。来源扩充需明确编辑锁并复核官方固定commit；工具不自动漂移到main。
- 若目标已有不同内容或symlink，工具失败而非覆盖；失败下载回执为partial，可重跑并校验已有文件。仅本工具临时文件/本次锁由本工具清理，不删除用户源码。
- dry-run仍读取和校验现存本地文件，不会写下载回执。下载回执置于scratch，不污染授权交付文件之外的源码。

## 技术反馈及协作边界

- VSA：全量Q/FFN仍执行；`to_gate_compress`控制池化分支，不是Token重要性分类器。
- VDN：新增分支需要训练权重；frame stats、Cholesky/triangular solve、双向scan、bridge/gate要整体移植，不是无训练换Attention。
- AdaLN/融合为优先无训练路线；SubBlock/Cache-DiT/Sage为无训练近似，需模型质量测试。
- Turbo历史commit `a2cc96b9ebbb94706903094aed2ced2e98796654` 仅静态读取；不是当前分支ancestor，没有cherry-pick。
- 主线程正在恢复CPU-only validated recipe/shard hash、已做H3-only加载路径识别；本worker仅记录conversion路线，不重复修改相关脚本或loader。
- 官方固定Turbo说明存在768p八步版本（6/3 shifts），注意旧文件列表不完整。
- 不给法律结论；保留代码LICENSE和第三方声明，权重许可另行确认。

本轮完成后停止研究/下载，释放并发slot。
