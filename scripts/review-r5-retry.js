export const meta = {
  name: 'review-r5-retry',
  description: 'Retry R5-part5: Review ch17-20 (stalled in round 1)',
  phases: [
    { title: 'R5-part5', detail: 'ch17-20' },
  ],
}

const CHAPTER_FILES = {
  ch17: 'volumes/part5/ch17_预训练语言模型.md',
  ch18: 'volumes/part5/ch18_视觉Transformer与扩散模型.md',
  ch19: 'volumes/part5/ch19_深度学习的理论难题.md',
  ch20: 'volumes/part5/ch20_强化学习的复兴.md',
}

const REVIEWERS = [
  { label: 'R5-part5', chapters: ['ch17','ch18','ch19','ch20'] },
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

## 输出格式
对每个 [严重] 错误，严格输出一行JSON：
{"severity":"[严重]","chapter":"文件名","line":"行号","issue":"错误描述","suggested_fix":"修改建议"}

无错误的章节输出："该章节无[严重]错误。"

## 文件列表
${files}`
}

const results = await parallel(
  REVIEWERS.map((r, i) => async () => {
    const output = await agent(buildPrompt(r.chapters), {
      label: 'R5-' + r.label,
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
