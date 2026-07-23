# AI技术思想史与数学导论 — 项目交接文档

> 本文件记录项目当前状态、已完成工作、遗留问题和后续建议，供接手同学参考。
> 生成日期：2026-07-22
> 最近更新：2026-07-23（P0 数学勘误 + 仓库卫生 + 方向调整为方案 B）

---

## 一、项目概况

- **内容**：一部覆盖 1936-2024 年 AI 技术脉络的中文长篇著作，共 24 章
- **方向**：方案 B — AI 技术思想史与数学导论（区别于严格通史或教材）
- **规模**：约 124 万字符（含 LaTeX 公式、伪代码、论文引用；已从旧估 230 万下调）
- **存储**：`Wch727/token-burning-work`（GitHub 私有仓库）
- **语言**：中文学术写作风格，引用格式为 `作者（年份）`

---

## 二、已完成工作

### 2.1 初稿生成

- 使用 Claude Code 多 Agent 并行生成 24 章初稿
- 总计消耗约 3M+ tokens
- 生成后通过多轮扩写将全部章节字数提升至 15,000 字以上

### 2.2 结构审查（review-r8.js）

已完成多轮修复，包括：

| 轮次 | 修复内容 | 状态 |
|------|---------|------|
| r1-r7 | 章节编号修复、重复节删除、空节清理 | 已完成 |
| r8 | 检查 `## 第X章` 标题层级、重复 `## 第X节`、`###` 小节层级异常、未闭合 `$` | 已完成 |
| fix-ch02-subsections | ch02 裸编号小节（如 `### 1.1`）改为 `### 第1.1节` 格式 | 已完成 |
| fix-ch17-headers | ch17 被 sed 破坏的小节编号修复（`第2节1节` → `第2.1节`） | 已完成 |
| fix-subsection-numbering | ch05（60处）`第X节Y节`→`第X.Y节`；ch15（38处）`第X节Y`→`第X.Y节` | 已完成 |
| ch01 10.6→10.5 | 小节编号跳号修正 | 已完成 |

**说明（2026-07-22 复核）**：HANDOVER 初版记载的 ch02 空节 6.1 / 重复 8.4、ch01 的 2.2.5.x 层级异常，在当前仓库中已不存在（可能已在此前 commit 中修好）。实际遗留结构问题是 **ch05 / ch15** 的 sed 破坏编号，现已修复。

### 2.3 内容审查（review-r10.js）

已完成多轮修复，包括：

| 轮次 | 修复内容 | 状态 |
|------|---------|------|
| r9 | LaTeX 公式完整性（`\log`、`\max`、`\min`、`\sum` 等反斜杠遗漏） | 已完成 |
| r10 | 引用格式统一为 `作者（年份）`（全角括号）、英文术语翻译、`$` 转义 | 已完成 |
| ch03 | LaTeX 公式和英文术语修复 | 已完成 |
| ch04 | 小节编号格式修复 | 已完成 |

**2026-07-22 全量扫描**：`node scripts/review-r8.js` 与 `node scripts/review-r10.js` 均为 **0 findings**（多轮修复后复跑仍为 0）。

### 2.3b 多轮 Agent 通读审稿闭环（2026-07-22）

| 轮次 | 动作 | 结果 | 状态 |
|------|------|------|------|
| Round 1 | 6 组并行通读 24 章 | ~69 条高置信（事实/结构/LaTeX/矛盾） | 已修 → `b30e564` |
| Round 2 | 3 组复审 | ~23 条残留硬伤（图灵九类/JOHNNIAC、Cover、AlexNet 裁剪、Seq2Seq 优化器、DistilBERT/Chinchilla 等） | 已修 → `0f00970` |
| Round 3 | 3 组终审抽检 | ~15 条零星硬伤（ResNet-1001、词表、GT200、DQN Nature、mT5/PaLM、VideoMAE、Samuel/Pearl 等） | 已修 → `2e24f1b`/`95130d8` |
| Round 4 | 3 组终审抽检 | ~15 条（DL 时间线、Good Regulator、IBM 701 功耗、Krizhevsky 履历、ResNet 2015、GTX580 3GB、LAPGAN 2015、RoBERTa/ALBERT/C4/Codex/FAVOR+ 等） | 已修 → `ab91222` |
| Round 5 | **6 组全量通读** 24 章（非抽检、无条数上限） | ~70+ 条 high（公式/史实/基准/归属） | 已修 → `2e0a5a0` |
| Round 6 | **6 组全量复审** 24 章（非抽检、无条数上限） | ~79 条 high（AlexNet 口径/DBN 拓扑/公式/法规编号/交叉引用等） | 已修 → `c7079f4` |
| Round 7 | **6 组全量复审** 24 章 | ~50+ 条 high（IBM 701、PCD、SORT/ByteTrack、CFG/VAE KL、C4/α_C、Shapley 公理等） | 已修 → `89f0e1f` |
| Round 8 | **6 组全量复审** 24 章 | ~50 条 high（AI 冬天口径、SMO/SAMME、AlexNet 算术、FlashAttention/FIM、3DGS 排序、AI 法案罚款等）；ch17–20 返回 `[]` | 已修 → `0e3ca38` |
| Round 9 | **6 组全量复审** 24 章 | ch01–04 返回 `[]`；其余约 30+ 条 high（DBN 顶层公式、SAMME、AlexNet FLOPs/分组参数、sFID、Toolformer/GPT-J、MATH/SWE-bench 等） | 已修 → `662b25c` |
| Round 10 | **6 组全量复审** 24 章 | ch17–24 返回 `[]`；ch05–16 仍有约 15 条 high（Rosenblatt Award、Schapire D2、Brown 聚类、Sigmoid 区间、MI50/CDNA、Samuel 中心格、MNIST SOTA 等） | 已修 → `7558a71` |
| Round 11 | **6 组全量复审** 24 章 | ch17–20 返回 `[]`；其余约 20 条 high（Codex/GPT-J、GLUE/SuperGLUE 分数、ShuffleNet、BERT GLUE 9 项、BERT-Large TPU、定理编号、IBM 701 产量等） | 已修 → `dff9172` |
| Round 12 | **6 组全量复审** 24 章 | 约 33 条 high（Entscheidungsproblem、DiagonalTM、DBN VC、AlexNet 2010/2012 混用、ViT 87.12%、CFG、GAIL、StarCoder 512、Omohundro 六驱动力等） | 已修 → `7a3ce4d` |
| Round 13 | **6 组全量复审** 24 章 | 约 28 条 high（Fisher 渐近、MEMM 标签偏差、LeNet-5 层数、Viola–Jones 6061、VGG-19 参数、StyleGAN FID、RoBERTa 86.4、BCQ、ToT/YOLOv7/DeepLabv3 等）；ch01 组超时无新报 | 已修 → `9f4f651` |
| Round 14 | **6 组全量复审** 24 章（分两批） | 第一批 → `5f97fdb`；第二批 → `67da055` | 已闭环 |
| Round 15 | **6 组全量复审** 24 章 | ~40+ 条 high | 已修 → `f80984e` |
| Round 16 | **6 组全量复审** 24 章 | ~40+ 条 high（TuringStep 停机语义、Eccles/Minsky 门生、Ashby 书名、SRC/FGCS 预算、VC/GBDT/SOM/GT/GloVe、DBN/ImageNet/p3、LRN/标签平滑/f-div、NSP/NTK/SAC/CQL、StarCoder/Chinchilla/Focal/DP-SGD 等） | **本 commit** |
| 脚本 | review-r8 + review-r10 | 0 findings | 通过 |
| 仓库卫生 | 删除根目录 `ch18-test.md`、`fix_ch24.js` | → `7416068` | 已完成 |

**停止准则**：R16 已修一轮；可再开 **R17** 确认是否趋近 `[]`。

### 2.4 事实核验（Agent 并行 + 原文对照）

| 章节 | 核验内容 | 修复数 | 状态 |
|------|---------|--------|------|
| ch01 图灵与AI的诞生 | IBM 701 租金、MIT AI Lab 成立年份、Williams 管、GPS 时间线等 | 11 处 | 已推送 |
| ch06 概率图模型与贝叶斯革命 | Dirichlet Process 等价描述、Onsager 解、pLSA 命名、CRF 变体等 | 6 处 | 已推送 |
| ch11 深度学习的关键理论突破 | BPTT 公式、谱半径、DBN MNIST 声明、Yee Whye Teh 名字等 | 5 处 | 已推送 |
| ch14 CNN架构的黄金时代 | AlexNet/VGG/GoogLeNet/ResNet/Inception/MobileNet 错误率、参数量、FLOPs、训练超参、集成规模 | 见下 | **已完成** |
| ch19 深度学习的理论难题 | Kawaguchi 结果、Choromanska 猜想、Papyan 作者、VC 维、Deep-IRM 等 | 11 处 | 已推送 |
| ch23 图神经网络与结构数据 | GCN over-smoothing、GIN/WL、EGNN 消息函数、SchNet 年份、AlphaFold 2 等 | 7 处 | 已推送 |

#### ch14 事实核验要点（对照 arXiv 原文）

| 声明 | 修正前 | 修正后（来源） |
|------|--------|----------------|
| ResNet ILSVRC 3.57% | 写成单模型 ResNet-152 | **6 模型集成**（含 2 个 152 层）；单模型 ResNet-152 多尺度 Top-5≈4.49%（He et al. 2016 Table 4/5） |
| 集成规模 | “16 个 ResNet-152” | **6 个不同深度残差网络** |
| ResNet 训练 | 180 epoch，第 30/60/90 衰减 | 约 $60\times 10^4$ 迭代（~120 epoch），平台期 ÷10；尺度增强 $[256,480]$ |
| VGG-16 表 | Top-1 28.1 / Top-5 9.9，FLOPs 30.9G | 单尺度 27.0 / 8.8，FLOPs **15.3G**（ResNet 论文口径） |
| VGG-19 表 | FLOPs 38.7G | **19.6G** |
| ResNet FLOPs | 50: 8.2G；152: 23.0G | **3.8G / 11.3G**（原文 Table 1 乘加） |
| ResNet-50 错误率 | 24.0 / 7.0 | **22.85 / 6.71**（10-crop，Table 3） |
| AlexNet 参数 | 62.3M | **~60M**（论文 / Wikipedia） |
| MobileNet-v1 | Top-1 错误率 36.7%（错） | **~29.4%**（准确率 70.6%，Howard et al. 2017） |
| MobileNet Mult-Adds | 1.1G / 0.6G | **0.57G / ~0.30G** |
| GoogLeNet 6.67% | 笼统写多裁剪 | **7 模型 × 144 裁剪集成**；单模型单裁剪 ~10.07% |
| Inception-v3 | Top-5 4.2% | 单帧约 **21.2 / 5.6**；集成更低 |
| Inception-ResNet vs ResNet 3.57% | 直接比较 | 标明 3.57% 为集成，不宜与单模型比 |

---

## 三、遗留问题（待办清单）

### 3.1 结构 / 审查

| # | 问题 | 状态 |
|---|------|------|
| 1–4 | ch02 空节/重复、ch01 层级、小节编号不一致 | **已关闭**（2026-07-22 复核并修复真实问题 ch05/ch15） |
| 6 | 全量 review-r8 + review-r10 | **已通过**（0 findings） |

### 3.2 内容建议（非阻塞）

| # | 问题 | 说明 |
|---|------|------|
| A | ch14 正文零星 FLOPs 表述 | 第 9 节表格与趋势段已统一；第 2–4 节个别叙述可能仍混用 “FLOPs / Mult-Adds” 口径，通读时可顺手统一 |
| B | ch13 表/附录标题 | `### 表13-x`、`### A.x` 为表格与附录，非标准 `第X.Y节`，属合理例外 |
| C | ch20/ch22 三级编号 | 使用 `### 第X.Y.Z节`（三级），与多数章的 `#### 第X.Y.Z节` 层级习惯略有不同，可择机统一 |
| D | 通读 | 重点看 ch05、ch14、ch15 改动较多的章节，确保上下文连贯 |
| E | 目录 | 每章开头可加 `[[TOC]]`（当前无目录） |

### 3.3 Git

| # | 问题 | 说明 |
|---|------|------|
| 7 | 统一 commit | 已多轮 commit + push |

**近期相关 commit（本机）**：
```
（本轮）fix: R15 full re-read — high-confidence hard errors + HANDOVER
7416068 chore: remove root scratch files ch18-test.md and fix_ch24.js
67da055 fix: R14 full re-read closeout — ch13-20 + residual hard errors + HANDOVER
5f97fdb fix: R14 partial full re-read — ch01-12/21-24 hard errors + HANDOVER
9f4f651 fix: R13 full re-read — high-confidence hard errors + HANDOVER
```

---

## 四、工具与脚本说明

| 脚本 | 用途 | 用法 |
|------|------|------|
| `scripts/review-r8.js` | 结构审查：标题层级、重复节、LaTeX `$` 配对 | `node scripts/review-r8.js` |
| `scripts/review-r10.js` | 内容审查：引用格式、LaTeX 反斜杠、英文术语、`$` 转义 | `node scripts/review-r10.js` |
| `scripts/fix-subsection-numbering.js` | 修复 ch05/ch15 小节编号（支持 `--dry`） | `node scripts/fix-subsection-numbering.js` |
| `scripts/ai_history.js` | 原始生成脚本（已不再使用） | — |
| `scripts/review-workflow.js` | 审查工作流编排 | — |
| `scripts/fix-r*.js` | 各轮修复脚本（历史存档） | — |

**review-r10.js 已知跳过规则**（无需修复）：

- `classification-calibrated` — 特殊术语，不检查反斜杠
- `\log-sum-exp`、`\log-likelihood`、`\log-`、`log-`、`log_{` — 合法 LaTeX
- `\max-`、`max_`、`max=` — 合法用法
- `\min-`、`min_` — 合法用法

---

## 五、Git 工作流

- 主分支：`main`
- 每次修复独立 commit，message 格式：`fix: 简短描述`
- push 前确保 `git add` 了修改的文件
- 仓库地址：`https://github.com/Wch727/token-burning-work.git`

---

## 六、各章基本信息速查

| 编号 | 文件 | 时期 | 字符数 |
|------|------|------|--------|
| ch01 | ch01_图灵与AI的诞生.md | 1936-1950s | 59K |
| ch02 | ch02_感知机与连接主义的兴衰.md | 1950s-1970s | 48K |
| ch03 | ch03_统计学习思想的萌芽.md | 1950s-1970s | 23K |
| ch04 | ch04_第一次AI冬天与专家系统的兴衰.md | 1970s-1980s | 23K |
| ch05 | ch05_统计机器学习的基础奠定.md | 1990s | 85K |
| ch06 | ch06_概率图模型与贝叶斯革命.md | 1990s | 51K |
| ch07 | ch07_神经网络的下沉期.md | 1990s-2000s | 42K |
| ch08 | ch08_语言模型的进化：从n-gram到神经网络.md | 1990s-2000s | 56K |
| ch09 | ch09_深度信念网络与预训练的革命.md | 2000s | 57K |
| ch10 | ch10_GPU计算与大数据.md | 2000s-2010s | 40K |
| ch11 | ch11_深度学习的关键理论突破.md | 2000s | 34K |
| ch12 | ch12_计算机视觉的演进.md | 2000s-2010s | 41K |
| ch13 | ch13_ImageNet与AlexNet.md | 2012 | 57K |
| ch14 | ch14_CNN架构的黄金时代.md | 2014-2016 | 64K |
| ch15 | ch15_Seq2Seq与Transformer.md | 2014-2017 | 48K |
| ch16 | ch16_生成模型的演进.md | 2014-2017 | 57K |
| ch17 | ch17_预训练语言模型.md | 2018-2024 | 57K |
| ch18 | ch18_视觉Transformer与扩散模型.md | 2020-2024 | 71K |
| ch19 | ch19_深度学习的理论难题.md | 2018-2024 | 72K |
| ch20 | ch20_强化学习的复兴.md | 2018-2024 | 41K |
| ch21 | ch21_NLP精细化.md | 2018-2024 | 48K |
| ch22 | ch22_计算机视觉精细化.md | 2018-2024 | 51K |
| ch23 | ch23_图神经网络与结构数据.md | 2018-2024 | 42K |
| ch24 | ch24_AI安全对齐与伦理.md | 2018-2024 | 74K |
| **合计** | **24 章** | | **~124 万字符** |

---

## 七、已知问题与经验教训

### 7.1 技术教训

1. **sed 正则替换风险**：ch17 曾因 sed 替换导致乱码；ch05/ch15 残留 `第X节Y节` / `第X节Y` 模式。教训：sed 前先用 grep 确认匹配范围，替换后用 review-r8 全量扫描。
2. **Agent 超时处理**：ch14 事实核验曾超时。教训：超时后改用 `fetch_url` 直接读 ar5iv/arxiv 原文表格，不要空等。
3. **引用格式**：中文学术引用应为 `作者（年份）`，全角括号。
4. **LaTeX 反斜杠遗漏**：注意排除 `\log-likelihood` 等复合词。
5. **FLOPs 计数口径**：不同论文对 FLOPs / Mult-Adds / 是否含 ×2 约定不一；横向表必须注明来源与口径。

### 7.2 内容教训

1. **事实核验范围**：数值数据、年份、作者、集成 vs 单模型必须逐条对照原文。
2. **不要凭记忆写数值**：尤其是 ImageNet 错误率与 FLOPs。
3. **评估协议**：单裁剪 / 10 裁剪 / 多尺度 / 集成不可混写为同一数字。

---

## 八、P0 数学勘误 + 仓库卫生（2026-07-23）

### 方向调整

从 “AI通史” 调整为 “AI技术思想史与数学导论”（方案 B），降低体裁混杂。

### P0 数学错误修复

| 章节 | 错误 | 修复内容 |
|------|------|---------|
| ch05 sec3.1 | SVM 硬间隔写为”目标严格凸 -> 唯一全局最优” | 分拆为 w* 唯一，b* 未必唯一 |
| ch05 sec3.4.2 | 半正定 Gram -> “严格凸，唯一解” | 修正：半正定只保凸，不保严格凸 |
| ch05 sec10.1 | 对偶”强凸性和唯一解”过强 | 限定额外正定条件 |
| ch14 sec4.4 | “梯度始终以幅度 1 传播””有保证的最小梯度流” | 说明 I+J_F 最小奇异值可能为 0 |
| ch14 sec4.4 | “ReLU 后雅可比特征值通常为负” | 删去断言；校正特征值/奇异值混用 |
| ch16 sec2.3 | “Brouwer 不动点证明纳什均衡存在” | 改为一般参数化 GAN 不自动保证 |
| ch16 sec2.3 | “生成器 Hessian 负定、判别器正定”符号颠倒 | 改为 GDA 稳定性取决于 Hessian 块结构 |
| ch16 sec2.3 | 学习率比值条件写成充分条件 | 加注依赖强假设，非一般充分条件 |
| ch24 sec5.4 | DP-SGD 噪声尺度与平均约定不符 | 注明求和 vs 平均两种约束及对应尺度 |
| ch24 sec6.4 | ReLU 编码 a=hat_a+v -> a 必非负 | 改为 a=hat_a-v |
| ch24 sec6.4 | MILP 目标 max CE < 0 不可能成立 | 改为类别 margin 条件 |
| ch24 sec6.4 | IBP 认证条件正确类”低于”其它类 | 改为正确类下界 > 其它类上界 |

### 仓库卫生

| 项目 | 改动 |
|------|------|
| LICENSE | 新增 MIT 许可证文件 |
| package.json | 新增，含 review 脚本 npm 快捷命令 |
| README | 方向改为方案 B；字数 230 万 -> 124 万；增加已知限制；目录同步 |
| 硬编码路径 | review-r9.js、review-r10.js、auto-review-loop.js 改为 dirname 方式 |
| 历史脚本归档 | 24 个 fix-r*.js、2 个 debug-ch18*.js、8 个根目录 py 脚本 -> archive/migrations/ |
| 陈旧审查脚本 | review-r1 至 r7、review-workflow.js、review-round1-findings.json -> archive/ |
| CI | 当前仅跑 review-r8.js；需后续补 r9+r10 |

---

## 九、后续建议（接手者必读）

### 从本轮教训出发的修正方案

1. **体裁统一**：已改为方案 B（技术思想史+数学导论），但正文中历史引语、完整推导、闭源推测仍互有重叠。后续改章时应在引言明确该章侧重。
2. **引用台账**：多数章无独立参考文献表。建议建立 Claim ID-正文位置-原始来源-页码台账。
3. **闭源系统推测**：Sora、GPT-4 等闭源系统内部细节不应进事实叙述，最多标注”社区推测”。
4. **教学类比 vs 原论文结论**：正文中自行构造的形式化（CoT 动态规划、ResNet 值迭代、注意力头因果解释等）需标注为教学模型或作者类比。

### 优先修复顺序

不要再跑 Agent 全量通读。改为按审稿优先级逐章专家审查：

1. **最高** — ch05（统计学习）、ch14（ResNet/CNN）、ch16（GAN）、ch24（DP/验证）
2. **高** — ch06（图模型）、ch20（RL）、ch23（GNN）
3. **中** — ch19（理论难题）、ch21（NLP）、ch22（CV）
4. **低** — ch01-04（早期史）、ch07-12（先完善其它再改体裁）

### 仓库工程 TODO

- 补跑 review-r9.js 与 review-r10.js（当前 CI 只跑 r8）
- 在干净 Linux 环境验证 CI 可从零运行
- 添加 Markdown 渲染与 LaTeX 全量 CI 检查
- 删除 archive/migrations/ 中的硬编码路径

---

*本文件由 Claude Code 维护。最近更新：2026-07-23（P0 勘误 + 仓库卫生 + 方向调整）。*
