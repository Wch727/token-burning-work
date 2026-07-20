export const meta = {
  name: 'review-round-2',
  description: 'Round 2: Re-review all 24 chapters to verify fixes and find remaining issues',
  phases: [
    { title: 'R2-part1', detail: 'ch01-04' },
    { title: 'R2-part2', detail: 'ch05-08' },
    { title: 'R2-part3', detail: 'ch09-12' },
    { title: 'R2-part4', detail: 'ch13-16' },
    { title: 'R2-part5', detail: 'ch17-20' },
    { title: 'R2-part6', detail: 'ch21-24' },
  ],
}

const CHAPTER_FILES = {
  ch01: 'volumes/part1/ch01_图灵与AI的诞生.md',
  ch02: 'volumes/part1/ch02_感知机与连接主义的兴衰.md',
  ch03: 'volumes/part1/ch03_统计学习思想的萌芽.md',
  ch04: 'volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md',
  ch05: 'volumes/part2/ch05_统计机器学习的基础奠定.md',
  ch06: 'volumes/part2/ch06_概率图模型与贝叶斯革命.md',
  ch07: 'volumes/part2/ch07_神经网络的下沉期.md',
  ch08: 'volumes/part2/ch08_语言模型的进化：从n-gram到神经网络.md',
  ch09: 'volumes/part3/ch09_深度信念网络与预训练的革命.md',
  ch10: 'volumes/part3/ch10_GPU计算与大数据.md',
  ch11: 'volumes/part3/ch11_深度学习的关键理论突破.md',
  ch12: 'volumes/part3/ch12_计算机视觉的演进.md',
  ch13: 'volumes/part4/ch13_ImageNet与AlexNet.md',
  ch14: 'volumes/part4/ch14_CNN架构的黄金时代.md',
  ch15: 'volumes/part4/ch15_Seq2Seq与Transformer.md',
  ch16: 'volumes/part4/ch16_生成模型的演进.md',
  ch17: 'volumes/part5/ch17_预训练语言模型.md',
  ch18: 'volumes/part5/ch18_视觉Transformer与扩散模型.md',
  ch19: 'volumes/part5/ch19_深度学习的理论难题.md',
  ch20: 'volumes/part5/ch20_强化学习的复兴.md',
  ch21: 'volumes/part6/ch21_NLP精细化.md',
  ch22: 'volumes/part6/ch22_计算机视觉精细化.md',
  ch23: 'volumes/part6/ch23_图神经网络与结构数据.md',
  ch24: 'volumes/part6/ch24_AI安全对齐与伦理.md',
}

const REVIEWERS = [
  { label: 'R2-R1-part1', chapters: ['ch01','ch02','ch03','ch04'] },
  { label: 'R2-R2-part2', chapters: ['ch05','ch06','ch07','ch08'] },
  { label: 'R2-R3-part3', chapters: ['ch09','ch10','ch11','ch12'] },
  { label: 'R2-R4-part4', chapters: ['ch13','ch14','ch15','ch16'] },
  { label: 'R2-R5-part5', chapters: ['ch17','ch18','ch19','ch20'] },
  { label: 'R2-R6-part6', chapters: ['ch21','ch22','ch23','ch24'] },
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
- 章节标题已补全"第X章"编号
- 节标题已统一为"第X节"格式
- 常见的英文术语已翻译（formulation, manipulation, receptive field等）
- ch14 VGG感受野(260→114), GoogLeNet(4a→4d)
- ch15 Transformer big模型头数(8→16)
- ch16 InfoGAN公式(-H(c)→+H(c))
- ch17 FFN公式(ReLU→GELU), Scaling Laws, C4数据
- ch18 EfficientNet准确率(top-1 vs top-5)
- ch20 Decision Transformer年份, DQN归属
- ch21 Linformer维度
- ch23 章节编号, et al.引用
- ch24 LaTeX命令(partial, gg, neq)

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
      label: 'R2-' + r.label,
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
