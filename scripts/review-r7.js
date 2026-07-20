export const meta = {
  name: 'review-round-7',
  description: 'Round 7: Verify R6 fixes and find remaining issues in ch01-04, ch09-12, ch21-24',
  phases: [
    { title: 'R7-part1', detail: 'ch01-04' },
    { title: 'R7-part3', detail: 'ch09-12' },
    { title: 'R7-part6', detail: 'ch21-24' },
  ],
}

const CHAPTER_FILES = {
  ch01: 'volumes/part1/ch01_图灵与AI的诞生.md',
  ch02: 'volumes/part1/ch02_感知机与连接主义的兴衰.md',
  ch03: 'volumes/part1/ch03_统计学习思想的萌芽.md',
  ch04: 'volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md',
  ch09: 'volumes/part3/ch09_深度信念网络与预训练的革命.md',
  ch10: 'volumes/part3/ch10_GPU计算与大数据.md',
  ch11: 'volumes/part3/ch11_深度学习的关键理论突破.md',
  ch12: 'volumes/part3/ch12_计算机视觉的演进.md',
  ch21: 'volumes/part6/ch21_NLP精细化.md',
  ch22: 'volumes/part6/ch22_计算机视觉精细化.md',
  ch23: 'volumes/part6/ch23_图神经网络与结构数据.md',
  ch24: 'volumes/part6/ch24_AI安全对齐与伦理.md',
}

const REVIEWERS = [
  { label: 'R7-R1-part1', chapters: ['ch01','ch02','ch03','ch04'] },
  { label: 'R7-R3-part3', chapters: ['ch09','ch10','ch11','ch12'] },
  { label: 'R7-R6-part6', chapters: ['ch21','ch22','ch23','ch24'] },
]

function buildPrompt(chapters) {
  const files = chapters.map(c => '- ' + CHAPTER_FILES[c]).join('\n')
  return `你是中文学术书籍审稿人。用中文审查以下章节，只报告 **[严重]** 错误。

## 审查维度
1. 事实准确性：公式、数值、年份、人名、论文引用是否正确
2. 术语一致性：中文术语优先，英文术语需有标准译名
3. LaTeX公式：语法正确性（括号匹配、命令完整、无多余符号）
4. 章节编号：标题格式 "# 第X章 标题"，节编号连续，算法编号连续无重复
5. 引用格式：统一为 "作者（年份）" 格式
6. 数据一致性：表格与正文一致

注意：以下问题已在上一轮修复，无需重复报告：
- ch01: UTM公式LaTeX, exhaustive→穷举, E[Pe(P)] cleanup, uttered→说出, retroactive→追溯性, 二进制1100→10001, 图灵机二进制加法结果
- ch02: 缺主语补充, Parallel Distributed Processing→并行分布式处理, irony→讽刺, automata→自动机, receptive field→感受野, Novikoff（1962）引用格式, VC引用格式, 标题去（续）, 节编号6.5-6.9→第7-11节, 重复第8/9节删除
- ch03: 量词(一→一个), trained→受过训练的, 感应层→视网膜, recognize-act→识别-执行, firing→发放, uniformly→完全, InfoNCE作者(Oord/Vinyals/Li), 香农1951预测工作补充, Hunt 1956决策树补充
- ch04: boating→划船, recognize-act→识别-执行, 节编号已转为阿拉伯数字(第1节等), MYCIN公式LaTeX cases环境, 交叉引用已修正, 子节编号第4节下第1/2/3节→第4/5/6节
- ch09: 所有节编号第X章→第X节, Boltzmann引用修正, 重复4.5节已删除
- ch10: GK110 CUDA核心(1920→2880), SM→SMX, Fermi FP64比(1/2→1/8), 前置比例(1/3到1/5→G80为1/8, GT200约为1/12)
- ch11: 节编号0./1./2.→第1节/第2节/第3节等
- ch12: HOG单元格(8×16→8×8), Dalal-Trigks→Dalal和Triggs, DoG泰勒公式\\right), AdaBoost公式\\right)
- ch15 Transformer big模型头数(8→16), 节编号(6.6→第1节等), §6.6重复段落已删除, ViT/多模态/LLM内容已保留
- ch17 FFN公式(ReLU→GELU), Scaling Laws, C4数据, sparked→集群, GPT-4 MoE参数, RLHF公式括号, M各自→及其
- ch18 所有节/子节编号已转为"第X节"格式, FFN→MLP, 重复子节已删除, 交叉引用已修正
- ch19 所有节/子节编号已转为"第X节"格式, NOTEA→NOTEARS, 前馈架构→前馈网络, 第第→第(×3), 交叉引用已修正
- ch20 节编号已修正, Decision Transformer年份, DQN归属, RLHF公式括号, UCT(2006), Neuro-DP(1996), D4PG(Barth-Maron), IMPALA(2018)
- ch21 Linformer维度, Katharopoulos等人, Performer softmax方差参数(\\sqrt{d_k}→d_k)
- ch11: 节编号0./1./2.→第1节/第2节/第3节等
- ch22 Mosaic数据增强归属(YOLOv4→YOLOv5), DeepLabv3年份, SSD特征图尺寸(10×10/5×5/3×3/1×1), Focal Loss梯度公式(-α_t(1-p_t)^{γ+1}*p_t), YOLOv5引用格式, smooth_L1 otherwise
- ch23 章节编号, et al.引用, diag→operatorname, 数学环境包裹
- ch24 LaTeX命令(partial, gg, neq, text, frac, hat, eta, sim, 数学环境包裹), partially rational→部分理性, CEV operatorname, RLHF公式\beta/text{KL}, \\text{KL}→\\operatorname{KL}, 公式$$包裹, stray braces已修复

只报告你确信是 [严重] 级别且上一轮未覆盖的新问题。

## 输出格式
对每个 [严重] 错误，严格输出一行JSON：
{"severity":"[严重]","chapter":"文件名","line":"行号","issue":"错误描述","suggested_fix":"修改建议"}

无错误的章节输出："该章节无[严重]错误。"

## 文件列表
${files}`
}

const results = await parallel(
  REVIEWERS.map((r) => async () => {
    const output = await agent(buildPrompt(r.chapters), {
      label: 'R7-' + r.label,
      effort: 'max',
    })
    return { reviewer: r.label, chapters: r.chapters, output }
  })
)

// Parse findings
const allFindings = []
for (const r of results.filter(Boolean)) {
  const lines = (r.output || '').split('\n')
  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed.startsWith('{') && trimmed.includes('severity')) {
      try {
        const obj = JSON.parse(trimmed)
        if (obj.severity === '[严重]') {
          allFindings.push({ ...obj, reviewer: r.reviewer })
        }
      } catch (e) { /* skip */ }
    }
  }
}

const byChapter = {}
for (const f of allFindings) {
  const ch = f.chapter || 'unknown'
  byChapter[ch] = (byChapter[ch] || 0) + 1
}

return {
  totalReviewers: results.length,
  totalFindings: allFindings.length,
  byChapter,
  findings: allFindings,
}
