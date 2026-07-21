const fs = require('fs')
const path = require('path')
const BASE = path.join(process.env.USERPROFILE || process.env.HOME, 'desktop', '数模', 'token-burning-work')

function read(fp) { return fs.readFileSync(path.join(BASE, fp), 'utf8') }

const FILES = [
  'volumes/part1/ch01_图灵与AI的诞生.md',
  'volumes/part1/ch02_感知机与连接主义的兴衰.md',
  'volumes/part1/ch03_统计学习思想的萌芽.md',
  'volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md',
  'volumes/part2/ch05_统计机器学习的基础奠定.md',
  'volumes/part2/ch06_概率图模型与贝叶斯革命.md',
  'volumes/part2/ch07_神经网络的下沉期.md',
  'volumes/part2/ch08_语言模型的进化：从n-gram到神经网络.md',
  'volumes/part3/ch09_深度信念网络与预训练的革命.md',
  'volumes/part3/ch10_GPU计算与大数据.md',
  'volumes/part3/ch11_深度学习的关键理论突破.md',
  'volumes/part3/ch12_计算机视觉的演进.md',
  'volumes/part4/ch13_ImageNet与AlexNet.md',
  'volumes/part4/ch14_CNN架构的黄金时代.md',
  'volumes/part4/ch15_Seq2Seq与Transformer.md',
  'volumes/part4/ch16_生成模型的演进.md',
  'volumes/part5/ch17_预训练语言模型.md',
  'volumes/part5/ch18_视觉Transformer与扩散模型.md',
  'volumes/part5/ch19_深度学习的理论难题.md',
  'volumes/part5/ch20_强化学习的复兴.md',
  'volumes/part6/ch21_NLP精细化.md',
  'volumes/part6/ch22_计算机视觉精细化.md',
  'volumes/part6/ch23_图神经网络与结构数据.md',
  'volumes/part6/ch24_AI安全对齐与伦理.md',
]

const findings = []

FILES.forEach((fp, idx) => {
  const ch = `ch${String(idx + 1).padStart(2, '0')}`
  const text = read(fp)
  const lines = text.split('\n')

  // 1. Chapter header format: must start with "# 第X章"
  const firstNonEmpty = lines.find(l => l.trim().length > 0)
  if (firstNonEmpty && !/^# 第\d+章/.test(firstNonEmpty.trim())) {
    findings.push({ ch, line: lines.indexOf(firstNonEmpty) + 1, issue: `Chapter header format: "${firstNonEmpty.trim().substring(0, 40)}"`, severity: 'high' })
  }

  // 2. Duplicate section numbering (## 第X节)
  const sections = new Set()
  lines.forEach((l, i) => {
    const secM = l.match(/^##\s+第(\d+)节/)
    if (secM) {
      const n = parseInt(secM[1])
      if (sections.has(n)) findings.push({ ch, line: i + 1, issue: `Duplicate section 第${n}节`, severity: 'critical' })
      sections.add(n)
    }
  })

  // 3. Subsection numbering must be hierarchical: ### 第X.Y节 under ## 第X节
  let currentSec = null
  lines.forEach((l, i) => {
    const secM = l.match(/^##\s+第(\d+)节/)
    if (secM) { currentSec = parseInt(secM[1]); return }
    const subM = l.match(/^###\s+第(\d+)\.(\d+)节/)
    if (subM && currentSec !== null) {
      const subSec = parseInt(subM[1])
      if (subSec === 0 && currentSec === 1) return // intro subsections OK
      if (subSec !== currentSec) findings.push({ ch, line: i + 1, issue: `Subsection 第${subSec}.${subM[2]} under 第${currentSec}节`, severity: 'medium' })
    }
  })

  // 4. LaTeX: unmatched $ in inline math (exclude currency and multi-line display math)
  let inDisplayBlock = false
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim()
    if (t === '$' || t === '$$') {
      if (!inDisplayBlock) { inDisplayBlock = true } else { inDisplayBlock = false }
      continue
    }
    if (t.startsWith('$$') && t.endsWith('$$') && t.length > 4) continue
    if (t.startsWith('\\[') || t.startsWith('\\]')) continue
    if (inDisplayBlock) continue
    if (t.startsWith('\\begin{') || t.startsWith('\\end{')) continue
    const dollarCount = (lines[i].match(/\$/g) || []).length
    if (dollarCount % 2 !== 0 && dollarCount > 0) {
      // Exclude currency patterns like $7,500 or $2.40
      const withoutCurrency = lines[i].replace(/\$[\d,]+(\.\d+)?/g, 'CURRENCY')
      const remaining = (withoutCurrency.match(/\$/g) || []).length
      if (remaining % 2 === 0) continue
      findings.push({ ch, line: i + 1, issue: `Unmatched $ in math mode (${dollarCount} dollars)`, severity: 'high' })
    }
  }

  // 5. LaTeX: bare commands outside math mode (not in $$, $, \\[, \\begin, code blocks)
  let inCode = false
  let inDisplay = false
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].trim()
    if (t.startsWith('```')) { inCode = !inCode; continue }
    if (inCode) continue
    if (t.startsWith('$$') || t.startsWith('\\[')) { inDisplay = true; continue }
    if (t.startsWith('$$') || t.startsWith('\\]')) { inDisplay = false; continue }
    if (t.startsWith('\\begin{') || t.startsWith('\\end{')) continue
    if (inDisplay) continue
    if (t.startsWith('|')) continue
    if (t.startsWith('图')) continue
    const l = lines[i]
    if (/\bsum\b/.test(l) && !l.includes('\\sum') && !l.includes('\\text') && !l.includes('sum_{'))
      findings.push({ ch, line: i + 1, issue: 'Missing backslash on sum', severity: 'medium' })
    if (/\bcdot\b/.test(l) && !l.includes('\\cdot') && !l.includes('\\text'))
      findings.push({ ch, line: i + 1, issue: 'Missing backslash on cdot', severity: 'medium' })
    if (/\blog\b/.test(l) && !l.includes('\\log') && !l.includes('\\text'))
      findings.push({ ch, line: i + 1, issue: 'Missing backslash on log', severity: 'medium' })
    if (/\bmax\b/.test(l) && !l.includes('\\max') && !l.includes('\\text') && !l.includes('\\arg\\max') && !l.includes('max_{'))
      findings.push({ ch, line: i + 1, issue: 'Missing backslash on max', severity: 'medium' })
    if (/\bmin\b/.test(l) && !l.includes('\\min') && !l.includes('\\text') && !l.includes('\\arg\\min') && !l.includes('min_{'))
      findings.push({ ch, line: i + 1, issue: 'Missing backslash on min', severity: 'medium' })
  }

  // 6. Known regressions: verify previously fixed terms don't reappear
  lines.forEach((l, i) => {
    if (l.includes('逐例')) findings.push({ ch, line: i + 1, issue: 'Regression: 逐例 (should be 逐点)', severity: 'high' })
    if (l.includes('离线策略')) findings.push({ ch, line: i + 1, issue: 'Regression: 离线策略 (should be 离策略)', severity: 'high' })
  })
})

const seen = new Set()
const unique = findings.filter(f => {
  const key = `${f.ch}:${f.line}:${f.issue.substring(0, 50)}`
  if (seen.has(key)) return false
  seen.add(key)
  return true
})

const order = { critical: 0, high: 1, medium: 2, low: 3 }
unique.sort((a, b) => (order[a.severity] || 9) - (order[b.severity] || 9))

const sevs = {}
unique.forEach(f => { sevs[f.severity] = (sevs[f.severity] || 0) + 1 })
const byChapter = {}
unique.forEach(f => { byChapter[f.ch] = (byChapter[f.ch] || 0) + 1 })

console.log(JSON.stringify({
  total: unique.length,
  chaptersCovered: FILES.length,
  bySeverity: sevs,
  byChapter,
  findings: unique
}, null, 2))
