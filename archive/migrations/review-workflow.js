export const meta = {
  name: 'review-all-chapters',
  description: '多轮审稿闭环：6并行Agent逐章审查，修复[严重]错误，迭代至无问题',
  phases: [
    { title: 'Round 1', detail: '6 Agent并行审阅全部24章' },
    { title: 'Fix Round 1', detail: '主循环修复所有[严重]错误并推送' },
    { title: 'Round 2', detail: '复审，验证修复效果' },
    { title: 'Fix Round 2', detail: '修复剩余[严重]错误' },
    { title: 'Final Round', detail: '终审确认' },
  ],
}

// ===== Chapter assignments (6 reviewers x 4 chapters each) =====
const REVIEWERS = [
  { label: 'R1-part1', chapters: ['ch01', 'ch02', 'ch03', 'ch04'], prompt: (ch) => ch },
  { label: 'R2-part2', chapters: ['ch05', 'ch06', 'ch07', 'ch08'], prompt: (ch) => ch },
  { label: 'R3-part3', chapters: ['ch09', 'ch10', 'ch11', 'ch12'], prompt: (ch) => ch },
  { label: 'R4-part4', chapters: ['ch13', 'ch14', 'ch15', 'ch16'], prompt: (ch) => ch },
  { label: 'R5-part5', chapters: ['ch17', 'ch18', 'ch19', 'ch20'], prompt: (ch) => ch },
  { label: 'R6-part6', chapters: ['ch21', 'ch22', 'ch23', 'ch24'], prompt: (ch) => ch },
]

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

// ===== Build reviewer prompt =====
function buildReviewPrompt(chapters) {
  const files = chapters.map(ch => `- ${CHAPTER_FILES[ch]}`).join('\n')
  return `你是一位中文学术书籍审稿人。请严格审查以下章节文件，只报告 **[严重]** 级别的错误。

审查维度：
1. 事实准确性：公式、数值、年份、人名、论文引用是否正确
2. 术语一致性：是否全文使用中文术语，无残留英文（专有名词如算法名、模型名可保留英文但需附中文解释或标准译名）
3. LaTeX公式：语法是否正确（括号匹配、命令完整、下标上标正确）
4 章节编号：章节标题格式应为 "# 第X章 标题"，节编号连续无跳号，算法编号连续无重复
5. 引用格式：年份+作者/团队，格式统一（如 "LeCun等人（1998）" 或 "Mnih等人（2015）"）
6. 数据一致性：表格内数据与正文描述一致

输出格式（严格遵循）：
对每个发现，输出一行 JSON：
{"severity": "[严重]", "chapter": "文件名", "line": "大概行号", "issue": "错误描述", "suggested_fix": "修改建议"}

只输出 [严重] 级别的错误。如果某章节没有 [严重] 错误，输出 "该章节无[严重]错误。"

审查文件：
${files}`
}

// ===== Run one round of 6 parallel reviewers =====
async function reviewRound(roundNum) {
  log(`\n===== Round ${roundNum}: 6 parallel reviewers =====`)
  const results = await parallel(
    REVIEWERS.map((r, i) => async () => {
      const findings = await agent(buildReviewPrompt(r.chapters), {
        label: `round${roundNum}-${r.label}`,
        effort: 'high',
      })
      return { reviewer: r.label, chapters: r.chapters, findings }
    })
  )
  return results.filter(Boolean)
}

// ===== Collect all findings into structured summary =====
function collectFindings(roundResults) {
  const all = []
  for (const rr of roundResults) {
    if (Array.isArray(rr.findings)) {
      for (const f of rr.findings) {
        if (f.severity === '[严重]') all.push(f)
      }
    }
  }
  return all
}

// ===== Main workflow =====
let allClean = false
let round = 1
const MAX_ROUNDS = 5
const findingsHistory = []

while (!allClean && round <= MAX_ROUNDS) {
  phase(`Round ${round}: 并行审稿`)
  const results = await reviewRound(round)

  phase(`Round ${round}: 汇总结果`)
  const findings = collectFindings(results)
  findingsHistory.push({ round, findings })

  if (findings.length === 0) {
    log(`\n=== Round ${round}: 全部24章零[严重]错误，审稿通过！===`)
    allClean = true
    break
  }

  log(`\n=== Round ${round}: 共发现 ${findings.length} 个[严重]错误 ===`)
  // Group by chapter
  const byChapter = {}
  for (const f of findings) {
    const ch = f.chapter.replace('volumes/', '')
    if (!byChapter[ch]) byChapter[ch] = []
    byChapter[ch].push(f)
  }
  for (const [ch, fs] of Object.entries(byChapter)) {
    log(`  ${ch}: ${fs.length} 个[严重]错误`)
  }

  // The main loop (Claude Code session) will fix these findings
  // between workflow rounds by reading the files and applying edits.
  // For now, return findings to main loop.
  round++
}

// If we exit the loop without allClean, it means we hit MAX_ROUNDS
if (!allClean) {
  log(`\n=== 达到最大轮次 ${MAX_ROUNDS}，仍有未解决的问题 ===`)
}

// Return summary for main loop
return {
  totalRounds: round,
  allClean,
  findingsHistory: findingsHistory.map(h => ({
    round: h.round,
    totalFindings: h.findings.length,
    byChapter: h.findings.reduce((acc, f) => {
      const ch = f.chapter.replace('volumes/', '')
      acc[ch] = (acc[ch] || 0) + 1
      return acc
    }, {}),
    findings: h.findings,
  })),
}
