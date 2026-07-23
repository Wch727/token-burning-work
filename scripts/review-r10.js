const meta = {
  name: 'review-r10',
  description: 'Content format review: citation format, LaTeX backslash, English term checks, verification of previous fixes',
  phases: [{ title: 'Review', detail: 'Regex-based review across all 24 chapters' }]
}

const fs = require('fs')
const path = require('path')
const BASE = path.dirname(__dirname)

function read(fp) { return fs.readFileSync(path.join(BASE, fp), 'utf8') }
function lines(text) { return text.split('\n') }

const ALL_FILES = [
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

const args = globalThis.args || []
const group = args[0] || 'all'

function chLabel(fp) {
  const m = fp.match(/ch(\d+)/)
  return m ? 'ch' + m[1].padStart(2, '0') : 'unknown'
}

function inMathMode(line) {
  let inMath = false
  for (const c of line) {
    if (c === '$') inMath = !inMath
  }
  return inMath
}

const findings = []
const seen = new Set()

function addFinding(ch, line, issue, severity) {
  const key = ch + ':' + line + ':' + issue.substring(0, 50)
  if (seen.has(key)) return
  seen.add(key)
  findings.push({ ch, line, issue, severity })
}

function reviewChapter(fp) {
  const ch = chLabel(fp)
  const ls = lines(read(fp))
  const text = read(fp)

  // 1. Section numbering consistency
  const sections = new Set()
  let currentSec = null
  ls.forEach((l, i) => {
    const secM = l.match(/^##\s+第(\d+)节/)
    if (secM) {
      const n = parseInt(secM[1])
      if (sections.has(n)) addFinding(ch, i + 1, 'Duplicate section 第' + n + '节', 'critical')
      sections.add(n)
      currentSec = n
    }
    const subM = l.match(/^###\s+第(\d+)\.(\d+)节/)
    if (subM && currentSec !== null) {
      const subSec = parseInt(subM[1])
      if (subSec === 0 && currentSec === 1) return
      if (subSec !== currentSec) addFinding(ch, i + 1, 'Subsection 第' + subM[1] + '.' + subM[2] + '节 under 第' + currentSec + '节', 'medium')
    }
  })

  // 2. Verify previous fixes
  if (text.includes('逐例')) {
    ls.forEach((l, i) => {
      if (l.includes('逐例')) addFinding(ch, i + 1, 'Still has 逐例 (should be 逐点)', 'high')
    })
  }
  if (text.includes('离线策略')) {
    ls.forEach((l, i) => {
      if (l.includes('离线策略')) addFinding(ch, i + 1, 'Still has 离线策略 (should be 离策略)', 'high')
    })
  }
  if (ch === 'ch18') {
    const idx = ls.findIndex(l => l.includes('VideoMAE') && l.includes('Feichtenhofer') && l.includes('2023'))
    if (idx >= 0) addFinding(ch, idx + 1, 'VideoMAE citation still has Feichtenhofer 2023', 'high')
    const v2Idx = ls.findIndex(l => l.includes('VideoMAE v2') && !l.includes('Tong et al., 2023') && !l.includes('CVPR'))
    if (v2Idx >= 0) addFinding(ch, v2Idx + 1, 'VideoMAE v2 missing citation to Tong et al. 2023', 'medium')
  }

  // 3. Chinese/English mixing
  const untranslated = [
    ' Faculty ', ' faculty ', 'relegated', 'enthusiasm',
    'landmark', 'breakthrough', 'state-of-the-art', 'benchmark',
    'outperforms', 'outperform', 'significantly',
    'countless', 'calibrated', 'analytically', 'pipelines',
    'insight', 'followed by', 'delineating',
  ]
  ls.forEach((l, i) => {
    if (l.startsWith('#') || l.startsWith('|') || l.startsWith('```') || l.startsWith('$$') || l.startsWith('$') || l.startsWith('图') || /^[\s　]/.test(l)) return
    if (l.includes('classification-calibrated')) return
    untranslated.forEach(term => {
      if (l.includes(term)) addFinding(ch, i + 1, 'Untranslated English: "' + term.trim() + '"', 'medium')
    })
  })

  // 4. LaTeX integrity
  ls.forEach((l, i) => {
    const dollarCount = (l.match(/\$/g) || []).length
    if (dollarCount % 2 !== 0 && dollarCount > 0 && !l.includes('\\$')) {
      addFinding(ch, i + 1, 'Unmatched $ in math mode (' + dollarCount + ' dollars)', 'high')
    }
    if (!inMathMode(l) && !l.startsWith('```') && !l.startsWith('|') && !/^[\s　]/.test(l)) {
      if (/\bsum\b/.test(l) && !l.includes('\\sum') && !l.includes('\\text') && !l.includes('sum_{') && !l.includes('\\log-sum-exp')) addFinding(ch, i + 1, 'Missing backslash on sum', 'medium')
      if (/\bcdot\b/.test(l) && !l.includes('\\cdot') && !l.includes('\\text')) addFinding(ch, i + 1, 'Missing backslash on cdot', 'medium')
      if (/\blog\b/.test(l) && !l.includes('\\log') && !l.includes('\\text') && !l.includes('\\log-') && !l.includes('log-') && !l.includes('log_{')) addFinding(ch, i + 1, 'Missing backslash on log', 'medium')
      if (/\bmax\b/.test(l) && !l.includes('\\max') && !l.includes('\\text') && !l.includes('\\max-') && !l.includes('max_') && !l.includes('max=')) addFinding(ch, i + 1, 'Missing backslash on max', 'medium')
      if (/\bmin\b/.test(l) && !l.includes('\\min') && !l.includes('\\text') && !l.includes('\\min-') && !l.includes('min_')) addFinding(ch, i + 1, 'Missing backslash on min', 'medium')
    }
  })

  // 5. Citation format check
  ls.forEach((l, i) => {
    if (l.startsWith('#') || l.startsWith('|') || l.startsWith('```')) return
    const m = l.match(/[A-Z][a-z]+(?:[\s,]+(?:et\s+al\.?|and|\&))?\s*\(\s*\d{4}\s*\)/)
    if (m && !m[0].startsWith('（') && !l.includes('等') && l.length > 20) {
      addFinding(ch, i + 1, 'Non-Chinese citation: "' + m[0].substring(0, 40) + '"', 'low')
    }
  })

  // 6. Factual checks
  const textLines = text.split('\n')
  textLines.forEach((l, i) => {
    if (l.includes('METEOR') && l.includes('Penalty=0.5')) {
      const chunksMatch = l.match(/chunks\s*=\s*(\w+)/)
      if (chunksMatch && chunksMatch[1] === 'm') {
        // Penalty=0.5 when chunks=m is correct (maximum fragmentation)
      } else if (!l.includes('chunks')) {
        // Line mentions METEOR and Penalty=0.5 together but no chunks context - check if it says chunks=m
        addFinding(ch, i + 1, 'METEOR penalty=0.5 mentioned without chunks=m context - verify correctness', 'low')
      }
    }
  })
}

const groups = {
  'ch01-04': ALL_FILES.filter(f => /ch0[1-4]/.test(f)),
  'ch05-08': ALL_FILES.filter(f => /ch0[5-8]/.test(f)),
  'ch09-12': ALL_FILES.filter(f => /ch(09|10|11|12)/.test(f)),
  'ch13-16': ALL_FILES.filter(f => /ch1[3-6]/.test(f)),
  'ch17-20': ALL_FILES.filter(f => /ch1[7-9]|ch20/.test(f)),
  'ch21-24': ALL_FILES.filter(f => /ch2[1-4]/.test(f)),
}

const targetFiles = groups[group] || ALL_FILES
targetFiles.forEach(reviewChapter)

findings.sort((a, b) => {
  const order = { critical: 0, high: 1, medium: 2, low: 3 }
  return (order[a.severity] || 9) - (order[b.severity] || 9)
})

const sevs = {}
findings.forEach(f => { sevs[f.severity] = (sevs[f.severity] || 0) + 1 })

const byChapter = {}
findings.forEach(f => { byChapter[f.ch] = (byChapter[f.ch] || 0) + 1 })

console.log(JSON.stringify({
  group,
  total: findings.length,
  bySeverity: sevs,
  byChapter,
  findings: findings.slice(0, 100)
}, null, 2))

const blockers = findings.some(f => f.severity === 'critical' || f.severity === 'high')
if (blockers) process.exitCode = 1
