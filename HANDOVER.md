# AI通史：从图灵到2024 — 项目交接文档

> 本文件记录项目当前状态、已完成工作、遗留问题和后续建议，供接手同学参考。
> 生成日期：2026-07-22

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

### 2.3 内容审查（review-r10.js）

已完成多轮修复，包括：

| 轮次 | 修复内容 | 状态 |
|------|---------|------|
| r9 | LaTeX 公式完整性（`\log`、`\max`、`\min`、`\sum` 等反斜杠遗漏） | 已完成 |
| r10 | 引用格式统一为 `作者（年份）`（全角括号）、英文术语翻译、`$` 转义 | 已完成 |
| ch03 | LaTeX 公式和英文术语修复 | 已完成 |
| ch04 | 小节编号格式修复 | 已完成 |

### 2.4 事实核验（Agent 并行）

对以下章节委派独立 Agent 进行事实核验与修订：

| 章节 | 核验内容 | 修复数 | 状态 |
|------|---------|--------|------|
| ch01 图灵与AI的诞生 | IBM 701 租金、MIT AI Lab 成立年份、Williams 管、GPS 时间线等 | 11 处 | 已推送 |
| ch06 概率图模型与贝叶斯革命 | Dirichlet Process 等价描述、Onsager 解、pLSA 命名、CRF 变体等 | 6 处 | 已推送 |
| ch11 深度学习的关键理论突破 | BPTT 公式、谱半径、DBN MNIST 声明、Yee Whye Teh 名字等 | 5 处 | 已推送 |
| ch14 CNN架构的黄金时代 | AlexNet/VGG/GoogLeNet/ResNet 年份、ImageNet 错误率、参数量、FLOPs | 未完成 | 见下文 |
| ch19 深度学习的理论难题 | Kawaguchi 结果、Choromanska 猜想、Papyan 作者、VC 维、Deep-IRM 等 | 11 处 | 已推送 |
| ch23 图神经网络与结构数据 | GCN over-smoothing、GIN/WL、EGNN 消息函数、SchNet 年份、AlphaFold 2 等 | 7 处 | 已推送 |

**ch14 事实核验未完成原因**：委派的 Agent 长时间无输出（0 bytes），未能返回核验结果。该章节涉及大量数值数据（ImageNet Top-1/Top-5 错误率、参数量、FLOPs、推理延迟），建议接手后重点复核。

---

## 三、遗留问题（待办清单）

### 3.1 高优先级 — 结构修复

| # | 问题 | 文件 | 说明 |
|---|------|------|------|
| 1 | ch02 第6节下有空节"6.1" | `volumes/part1/ch02_感知机与连接主义的兴衰.md` | 需删除或合并到第6节正文 |
| 2 | ch02 第8.4 节重复 | 同上 | 需删除重复节 |
| 3 | ch01 标题层级异常 | `volumes/part1/ch01_图灵与AI的诞生.md` | 2.2.5.x 小节的 heading level 和编号需要调整 |
| 4 | 小节编号格式不一致 | 两个文件（待确认具体是哪些） | 应为 `### 第X.Y节` 格式 |

### 3.2 中优先级 — 事实核验

| # | 问题 | 文件 | 说明 |
|---|------|------|------|
| 5 | ch14 关键数据未核验 | `volumes/part4/ch14_CNN架构的黄金时代.md` | ImageNet 错误率、参数量、FLOPs、训练 epoch 数、学习率等 |

**ch14 建议重点复核的数值**（书中原文，需逐条确认）：

- AlexNet: Top-5 16.4%, 参数 62.3M, FLOPs 1.5G
- VGG-16: Top-1 28.1%, Top-5 9.9%, 参数 138M, FLOPs 30.9G
- VGG-19: Top-1 27.3%, Top-5 9.0%, 参数 144M, FLOPs 38.7G
- GoogLeNet: Top-5 6.67%, 参数 6.6M, FLOPs 3.8G
- ResNet-50: Top-1 24.0%, Top-5 7.0%, 参数 25.6M, FLOPs 8.2G
- ResNet-152: Top-5 3.57%, 参数 60.2M, FLOPs 23.0G
- Inception-v3: Top-5 4.2%, 参数 23.8M, FLOPs 5.7G
- Inception-v4: Top-5 3.0%, 参数 42.7M
- Inception-ResNet-v2: Top-1 19.9%, Top-5 4.9%, 参数 55.8M, FLOPs 13.2G
- MobileNet-v1: Top-1 36.7%, Top-5 15.4%, 参数 4.2M, FLOPs 1.1G
- MobileNet-v2: Top-1 28.0%, Top-5 9.4%, 参数 3.5M, FLOPs 0.6G
- ResNet 训练：180 epoch，第 30/60/90 epoch 衰减 ×0.1
- VGG 训练：370k 次迭代（约 74 epoch），batch size 256
- GoogLeNet 训练：初始学习率 0.2，每 epoch 衰减 ×0.96

### 3.3 低优先级 — 全量审查

| # | 问题 | 说明 |
|---|------|------|
| 6 | 全量 review-r8 + review-r10 | 修复完上述问题后，跑一遍全量脚本确认无新问题 |
| 7 | 统一 commit | 上述全部修复完后统一 commit & push |

---

## 四、工具与脚本说明

| 脚本 | 用途 | 用法 |
|------|------|------|
| `scripts/review-r8.js` | 结构审查：标题层级、重复节、LaTeX `$` 配对 | `node scripts/review-r8.js` |
| `scripts/review-r10.js` | 内容审查：引用格式、LaTeX 反斜杠、英文术语、`$` 转义 | `node scripts/review-r10.js` |
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

**最近的 commit 历史**（截至本文件编写时）：

```
6cbd9eb docs: rewrite README - remove word counts, add chapter index table
1f57ee9 docs: update README with accurate stats and workflow
69eb8d7 fix: ch01 factual corrections (Trenchard More, MIT AI Lab, etc.)
```

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

1. **sed 正则替换风险**：ch17 曾因 sed 替换 `第([0-9]+)节([0-9]+)节` → `第\1.\2节` 导致部分已正确格式的标题被二次修改，产生 `第2.1节.1` 之类的乱码。教训：sed 前先用 grep 确认匹配范围，替换后用 review-r8 全量扫描。

2. **Agent 超时处理**：ch14 事实核验的 Agent 长时间无输出。教训：Agent 超时后应直接改用串行工具（net-tools search_web / fetch_url）逐条核验，不要空等。

3. **引用格式**：中文学术引用应为 `作者（年份）`，全角括号。早期章节混有 `Author et al. (Year)` 格式，需统一。

4. **LaTeX 反斜杠遗漏**：review-r10.js 检查 `\log`、`\max`、`\min`、`\sum` 等命令的反斜杠是否遗漏。注意排除 `\log-likelihood` 等复合词。

### 7.2 内容教训

1. **事实核验范围**：以下类型的内容必须逐条网络核验：
   - 数值数据（错误率、参数量、FLOPs、推理时间）
   - 年份和事件时间线
   - 论文作者和标题
   - 人名翻译（中文译名需确认通用译法）

2. **不要凭记忆写数值**：早期章节中不少 FLOPs、参数量的数字来自训练数据的近似记忆，未必与原始论文一致。

---

## 八、后续建议

1. **优先处理 3.1 中的 4 个结构问题**（ch02 两个重复节 + ch01 标题层级 + 小节编号格式）
2. **ch14 事实核验** — 用 `search_web` 逐条确认 ImageNet 基准表中的数值
3. **全量 review-r8 + review-r10** — 确认 24 章无遗漏问题
4. **通读一遍** — 重点看 ch02、ch14 两个改动较多的章节，确保上下文连贯
5. **考虑加目录** — 每章开头可加 `[[TOC]]` 方便导航（当前无目录）

---

*本文档由 Claude Code 生成，交接给下一位编辑同学。*
