# H3 新增三链接：身份与兼容性有界审计

核验日期：**2026-10-14（Asia/Shanghai）**。范围：仅这三个分享链接的身份、官方论文/代码以及 H3 / GGUF Q4 / RTX 4090 兼容性。**本轮未识别出任何一个链接的真实文章身份；全部保留 unresolved，不按“token-pruning / step-distillation / VAE”的猜测配对项目。**

## 1. IPv4 重试后的结论（更新于 2026-10-14 19:54，Asia/Shanghai）

**按本线程的IPv4诊断结果，已用 `curl -4` 重新请求原始两个微信链接及用户补充的完整小红书分享 URL。三个请求均成功通过正常 TLS 校验；本次不再以旧的默认地址族/TLS 失败作为停止依据。**

| 链接 | IPv4 / curl 结果 | HTTP | 本次收到的内容 | 真正阻塞点 |
|---|---|---:|---|---|
| 微信 1 | `43.137.130.213`；exit 0；TLS verify 0 | 302 | 2,252 B 错误模板；Location 指向同站 `/mp/wappoc_appmsgcaptcha` | 站点 CAPTCHA 重定向 |
| 微信 2 | `43.137.144.91`；exit 0；TLS verify 0 | 302 | 2,252 B 错误模板；Location 指向同站 `/mp/wappoc_appmsgcaptcha` | 站点 CAPTCHA 重定向 |
| 小红书完整分享 URL | `81.69.116.102`；exit 0；TLS verify 0 | 302 | 571 B 重定向模板；Location 指向同站 `/login` | 站点登录重定向 |

- **已使用完整小红书 URL**，包括用户提供的 `xsec_token`、`xsec_source=app_share` 及其余查询参数，未改写/伪造；正文报告不重复分享 token。先前“完整 URL 不可用”的判断仅属于首轮记录，现已解除。
- 未跟随登录/CAPTCHA 重定向、未解验证码、未登录、未使用 Cookie/凭据；也未关闭 TLS 校验。这里止步于实际认证/挑战边界，**不是 IPv6 网络失败**。
- 微信响应模板的“未知错误”“失效的验证页面”**不是文章标题**；小红书的 Found 模板也不是笔记正文。因此仍无真实标题、作者、论文名称或官方仓库出站链接。不能把 302 成功响应写成“文章抓取成功”。
- 请求边界：每次连接超时 8 秒、总超时 25 秒、响应上限 1 MiB，禁用 curl 配置文件读取，强制 IPv4，仅 HTTPS；三个响应正文合计 **5,075 B**。证据保存远端 IPv4、状态码、TLS 结果、响应头与原始小页面。响应头不保存 `Set-Cookie`。
- 重试后的公共索引搜索：精确 ID 搜索只有不相关结果，未采信；三个精确完整路径搜索明确无结果。首轮六次 GitHub 公共 issues/README 精确 ID 搜索均 HTTP 200、`total_count: 0`。它们都**不能证明论文或代码不存在**。
- 原 `urllib` 请求未记录连接 IP，故该历史失败仅作为原始记录保存；本轮已经实证强制 IPv4 可通过 TLS。主线程确认的 ModelScopeCDN 下载改善不在本任务的复核/写入范围。
- **实际身份仍 unresolved**。缺少文章内容锚点时，不用“可能相似”的 LoRA、token 或 VAE 项目顶替官方识别，也不干扰主线程的实现/模型下载。

## 2. 按链接列出的身份与支持状态

| 字段 | 微信 1 | 微信 2 | 小红书 |
|---|---|---|---|
| 用户链接 ID | `2Xz1yhgPcS8ae5N3SPf7Zg` | `7niXqWVig_lKLiUOkDv4pQ` | `6acf632f000000001c01ce24` |
| IPv4 重试的实际访问状态 | TLS 成功；CAPTCHA 302 | TLS 成功；CAPTCHA 302 | 完整分享 URL；TLS 成功；登录 302 |
| 文章标题 / 发布日期 | unresolved / unresolved | unresolved / unresolved | unresolved / unresolved |
| 文章作者 / 发布账号 | unresolved | unresolved | unresolved |
| 论文标题 / 学术作者 | unresolved | unresolved | unresolved |
| 实际方法及创新点 | unresolved；未分类 | unresolved；未分类 | unresolved；未分类 |
| 官方论文 / 项目页 | unresolved | unresolved | unresolved |
| 官方代码 / 固定 commit | unresolved | unresolved | unresolved |
| 所需底模 / 新权重 / 是否重训 | unresolved | unresolved | unresolved |
| 官方 H3 支持 | unresolved | unresolved | unresolved |
| 本项目 GGUF 混合 Q4 支持 | unresolved；未验证 | unresolved；未验证 | unresolved；未验证 |
| 官方 RTX 4090 支持 / 本机实测 | unresolved / 未运行 | unresolved / 未运行 | unresolved / 未运行 |

原始请求地址（小红书仅展示基础路径；实际 IPv4 请求使用用户提供的完整查询参数）：

```text
https://mp.weixin.qq.com/s/2Xz1yhgPcS8ae5N3SPf7Zg
https://mp.weixin.qq.com/s/7niXqWVig_lKLiUOkDv4pQ
https://www.xiaohongshu.com/explore/6acf632f000000001c01ce24
```

**unresolved 是证据不足，不是“不支持”或“没有开源”。**既有 Larry / LightX2V / FastVideo / SGLang 等路线不能作为这三篇文章的身份替代品。

## 3. 兼容性判别边界（本地已知事实，不是三篇文章的内容）

### 3.1 Token routing 不等于删除 token

本地 SubBlock 源码为每个 query chunk 计算输出，TopK 选择的是可见的目标视频 K/V 块；非目标视频的条件段受保护。其 `SparseTokenKind::Query` 注释还明确不是输出行掩码。当前 keep25 记录中的“75%”指目标视频 K 块的名义丢弃比例，不能表述为删除 75% 的视频输出 token。

本地依据：`local-context-excerpts.txt` 中 `minimax_h3_cuda_sparse.hpp:21–29,448–456,532–581` 与 `h3-subblock-keep25.md:7–9`。这只证明现有实现的语义，不证明任一新链接讲 SubBlock。

拿到新方法后，应分别检查：

1. 只是为各 Q 选择 K/V、全体 Q 仍更新，还是实际缩短残差/FFN序列？
2. 若做 token 合并、压缩、删除或冻结，如何恢复位置/输出，哪些 token 仍经过 FFN？
3. 是否有训练后的 router/gate、粗粒度补偿分支或专用学生参数？不能把训练分支当作只换 attention mask。
4. 音频、文本、参考视频和目标视频如何分别处理，是否保持 H3 的混合输入契约？

### 3.2 少步蒸馏不是任意降低 steps

如果文章实际属于少步路线，待核验清单为：明确的底模分区（FL2VA / Ref2VA）、学生或 adapter 文件、rank/alpha/非 A/B 参数、实际 denoiser evaluation 次数、音视频 timestep/sigma/shift、CFG 与采样器。**目前没有这些链接的模型证据，不能给出可加载文件或“4 步即用”的结论。**既有 LoRA 审计仅作背景，不将其支持声明转移给未知方法。

### 3.3 Causal 不自动等于可替换 H3 codec / VAE

本地 H3 视频 VAE 已有 `CausalConv3d` 和时间方向前侧 padding；仅看到论文标题或“causal”标签不能判断它提出的是：VAE 流式解码、denoiser 因果化、状态缓存，还是训练新的 codec。这里也不由单个卷积类推断整个生成管线端到端因果。

本地管线约束 `frames >= 22` 且 `frames % 17 == 5`，并有专门的 latent 帧数映射。现有精度锁记录视频 VAE 为 F16、音频 VAE 为 F32，**不是整条管线都量化成 Q4**。依据见 `local-context-excerpts.txt`。

如果是 decoder 蒸馏/替换，应验证它是否明确接受现有 H3 latent，是否需新 decoder 权重，以及边界/分块/音视频同步是否保持；如果是 encoder+decoder 或 latent 空间替换，则需核对通道、时空压缩、缩放/归一化、分布及配套 denoiser 训练。**只匹配 tensor shape 或都称“causal VAE”不足以证明可互换。**以上为后续审查要求，不是对未知文章的定性。

### 3.4 H3、Q4、4090 分开验收

既有本地目标为单 RTX 4090、864×480 / 124 帧、原混合 Q4 DiT 与原 VAE 精度、offload；见既有 keep25 记录。后续须独立给出：

- **模型支持**：官方是否实际列出 H3 及正确 FL/Ref 任务，而非只在其他视频模型示范。
- **量化/运行时支持**：是否接受本项目 GGUF 的混合 tensor 类型和 ggml/C++ 执行路径；Python 或另一种 4-bit 格式不能替代此项证据。
- **硬件支持**：是否有 4090 所需内核/数据类型与显存方案；多卡、大显存或另一代 GPU 的结果不是本机复现。
- **效果与性能**：身份和静态兼容性确认之后才安排单方法对照。本次没有 GPU 运行、速度估计或质量结论。

## 4. 证据与复核

证据目录：

`/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-goal-20261014/links/`

- `audit-summary.json`：更新后的逐链接身份、IPv4/TLS 结果、CAPTCHA/login 阻塞点；`audit-summary-before-ipv4.json` 保留首轮结论，不能当当前状态。
- `*-ipv4-fetch.json`、`*-ipv4-headers.txt`、`*-ipv4.html`：本轮三次 IPv4 请求的元数据、响应头和错误/跳转模板；不是文章正文。
- `web-tool-ipv4-retry-searches.json`：重试后的公共索引搜索与无关结果排除记录。
- `web-tool-attempts.json`：打开/搜索的空返回观察记录；不是文章抓取件。
- `*-sandbox-fetch.json`、`*-fetch.json`：原始地址、UTC 时间、DNS/TLS 错误、大小上限及不跟随重定向策略。
- `*-github-issues.json`、`*-github-repositories.json`：六份原始公开搜索响应；对应 `*-fetch.json` 保存请求、状态、字节数和 SHA256。
- `local-context-excerpts.txt`、`local-context.json`：只读的本地语义证据、读取时 HEAD、源文件 hash。不是新链接身份来源。
- `evidence-manifest.json`：上述证据文件的大小与 SHA256。

只读上下文文件：

- `/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/docs/research/h3-lora-q4-deployment-audit-20261014.md`
- `/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/docs/research/h3-subblock-keep25.md`
- `/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/docs/research/h3-precision-lock.json`
- `/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/src/dit_models/models/minimax_h3_cuda_sparse.hpp`
- `/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/src/dit_models/components/autoencoders/minimax_h3_vae.hpp`
- `/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/src/dit_models/pipelines/minimax_h3_pipeline.cpp`

本轮只写本报告及上述 `links/` 证据，不修改这些上下文、主线程 LoRA 代码、下载计划、默认参数、模型或依赖。

## 5. 最小解阻输入

每个链接给出任一项即可继续定向核验：**带标题和署名的截图、粘贴文章正文/论文名称、或公开的论文/官方仓库地址**。完整小红书分享 URL 已使用，不再请求重复补充。当前需要的是正文身份信息，不是网络修复、账号、Cookie、访问令牌或认证绕过。

**截至 IPv4 重试：已排除“因旧 TLS 失败而未重试”的问题；现有证据为两个微信 CAPTCHA 重定向及小红书登录重定向。三个真实方法、官方仓库和 H3/Q4/4090 支持仍 unresolved。**
