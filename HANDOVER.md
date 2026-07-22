# AI通史：从图灵到2024 — 项目交接文档

> 本文件记录项目当前状态、已完成工作、遗留问题和后续建议，供接手同学参考。
> 生成日期：2026-07-22
> 最近更新：2026-07-22（多轮 Agent 通读审稿 + 修复闭环）

---

## 一、项目概况

- **内容**：一部覆盖 1936-2024 年 AI 发展历程的中文学术著作，共 24 章
- **规模**：约 230 万字（含 LaTeX 公式、伪代码、论文引用）
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
| 脚本 | review-r8 + review-r10 | 0 findings | 通过 |

**停止准则**：R5 起恢复全量通读；R5 后脚本 0 findings。可再开 R6 全量复审确认是否返回 `[]`。

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
| 7 | 统一 commit | 已多轮 commit；本地 `main` 领先 `origin/main`（未 push） |

**近期相关 commit（本机）**：
```
2e0a5a0 fix: R5 full-read review — high-confidence hard errors across 24 chapters
4b92dca docs: HANDOVER — record round-4 agent review completion
ab91222 fix: round-4 spot-check — remaining high-confidence hard errors
95130d8 fix: round-3 ch01/ch03 residual timeline facts + update HANDOVER
2e24f1b fix: round-3 spot-check — residual timeline and architecture facts
0f00970 fix: round-2 re-review — remaining high-confidence factual errors
b30e564 fix: round-1 agent review — factual, structural, latex fixes across 24 chapters
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

| 编号 | 文件 | 时期 | 字数（约） |
|------|------|------|-----------|
| ch01 | ch01_图灵与AI的诞生.md | 1936-1950s | 19K |
| ch02 | ch02_感知机与连接主义的兴衰.md | 1950s-1970s | 26K |
| ch03 | ch03_统计学习思想的萌芽.md | 1950s-1970s | 16K |
| ch04 | ch04_第一次AI冬天与专家系统的兴衰.md | 1970s-1980s | 15K |
| ch05 | ch05_统计机器学习的基础奠定.md | 1990s | 19K |
| ch06 | ch06_概率图模型与贝叶斯革命.md | 1990s | 20K |
| ch07 | ch07_神经网络的下沉期.md | 1990s-2000s | 20K |
| ch08 | ch08_语言模型的进化：从n-gram到神经网络.md | 1990s-2000s | 21K |
| ch09 | ch09_深度信念网络与预训练的革命.md | 2000s | 20K |
| ch10 | ch10_GPU计算与大数据.md | 2000s-2010s | 23K |
| ch11 | ch11_深度学习的关键理论突破.md | 2000s | 16K |
| ch12 | ch12_计算机视觉的演进.md | 2000s-2010s | 19K |
| ch13 | ch13_ImageNet与AlexNet.md | 2012 | 20K |
| ch14 | ch14_CNN架构的黄金时代.md | 2014-2016 | 25K |
| ch15 | ch15_Seq2Seq与Transformer.md | 2014-2017 | 20K |
| ch16 | ch16_生成模型的演进.md | 2014-2017 | 20K |
| ch17 | ch17_预训练语言模型.md | 2018-2024 | 27K |
| ch18 | ch18_视觉Transformer与扩散模型.md | 2020-2024 | 24K |
| ch19 | ch19_深度学习的理论难题.md | 2018-2024 | 32K |
| ch20 | ch20_强化学习的复兴.md | 2018-2024 | 14K |
| ch21 | ch21_NLP精细化.md | 2018-2024 | 18K |
| ch22 | ch22_计算机视觉精细化.md | 2018-2024 | 21K |
| ch23 | ch23_图神经网络与结构数据.md | 2018-2024 | 18K |
| ch24 | ch24_AI安全对齐与伦理.md | 2018-2024 | 35K |

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

## 八、后续建议

1. ~~优先处理结构问题~~ — 已完成
2. ~~ch14 事实核验~~ — 已完成
3. ~~全量 review-r8 + review-r10~~ — 已通过
4. **通读** ch05 / ch14 / ch15，检查改写后的上下文
5. 可选：统一 ch20/ch22 三级标题层级；为各章加 `[[TOC]]`
6. push 到 GitHub（若尚未推送）

---

*本文档由 Claude Code 维护。*
