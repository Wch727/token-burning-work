const fs = require('fs')
const path = require('path')
const BASE = path.join(process.env.USERPROFILE || process.env.HOME, 'desktop', '数模', 'token-burning-work')

function read(fp) { return fs.readFileSync(path.join(BASE, fp), 'utf8') }

const FILES = {
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

const findings = []

function check(ch, fn) {
  const text = read(FILES[ch])
  const lines = text.split('\n')
  fn(lines, ch)
}

check('ch01', (lines) => {
  // PSSH should be at 1976 (Newell & Simon), not 1969
  const psshLine = lines.findIndex(l => l.includes('物理符号系统假说') && l.includes('1976'))
  if (psshLine === -1) {
    const wrongYear = lines.findIndex(l => l.includes('物理符号系统假说'))
    if (wrongYear >= 0) findings.push({ ch: 'ch01', line: wrongYear + 1, issue: 'PSSH year may be wrong', severity: 'low' })
  }
})

check('ch02', (lines) => {
  // XOR proof should have: w1 + w2 > 2(-b) > -b
  const xorProof = lines.findIndex(l => l.includes('2(-b)'))
  if (xorProof === -1) findings.push({ ch: 'ch02', line: 220, issue: 'XOR proof line with 2(-b) not found', severity: 'high' })
  const sections = []
  lines.forEach((l, i) => {
    const m = l.match(/^##\s+第(\d+)节/)
    if (m) {
      const num = parseInt(m[1])
      if (sections.includes(num)) findings.push({ ch: 'ch02', line: i + 1, issue: `Duplicate section 第${num}节`, severity: 'critical' })
      sections.push(num)
    }
  })
})

check('ch03', (lines) => {
  const bad = lines.findIndex(l => l.includes('感应层') && !l.includes('视网膜'))
  if (bad >= 0) findings.push({ ch: 'ch03', line: bad + 1, issue: 'Still uses 感应层', severity: 'medium' })
  const huntIdx = lines.findIndex(l => l.includes('Hunt') && l.includes('1956'))
  if (huntIdx >= 0) findings.push({ ch: 'ch03', line: huntIdx + 1, issue: 'Hunt year still 1956', severity: 'high' })
})

check('ch04', (lines) => {
  const facIdx = lines.findIndex(l => /Faculty/.test(l) && !/faculty/.test(l))
  if (facIdx >= 0) findings.push({ ch: 'ch04', line: facIdx + 1, issue: 'Still has untranslated Faculty', severity: 'medium' })
  const thouIdx = lines.findIndex(l => l.includes('数千条') && l.includes('更多'))
  if (thouIdx >= 0) findings.push({ ch: 'ch04', line: thouIdx + 1, issue: 'Still has 数千条...更多', severity: 'low' })
})

check('ch09', (lines) => {
  const dbnIdx = lines.findIndex(l => l.includes('Neural Computation') && l.includes('2006'))
  if (dbnIdx === -1) {
    const sciIdx = lines.findIndex(l => l.includes('《科学》') && l.includes('DBN'))
    if (sciIdx >= 0) findings.push({ ch: 'ch09', line: sciIdx + 1, issue: 'DBN journal still says 科学', severity: 'high' })
  }
  const badSub = lines.findIndex(l => /^###\s+\d+\.\d+/.test(l) && !l.includes('第'))
  if (badSub >= 0) findings.push({ ch: 'ch09', line: badSub + 1, issue: 'Subsection missing 第: ' + lines[badSub].trim().substring(0, 50), severity: 'medium' })
})

check('ch10', (lines) => {
  const kepIdx = lines.findIndex(l => l.includes('Keppler'))
  if (kepIdx >= 0) findings.push({ ch: 'ch10', line: kepIdx + 1, issue: 'Still has Keppler', severity: 'high' })
  const colIdx = lines.findIndex(l => /collectively/i.test(l))
  if (colIdx >= 0) findings.push({ ch: 'ch10', line: colIdx + 1, issue: 'Still has collectively', severity: 'medium' })
  lines.forEach((l, i) => {
    if (l.includes('sum') && !l.includes('\\sum') && !l.includes('\\text') && !l.includes('displaystyle') && !l.includes('sum_{')) {
      findings.push({ ch: 'ch10', line: i + 1, issue: 'Missing backslash on sum', severity: 'medium' })
    }
    if (l.includes('cdot') && !l.includes('\\cdot') && !l.includes('\\text') && !l.includes('\\begin')) {
      findings.push({ ch: 'ch10', line: i + 1, issue: 'Missing backslash on cdot', severity: 'medium' })
    }
  })
})

check('ch11', (lines) => {
  const enthIdx = lines.findIndex(l => /enthusiasm/i.test(l))
  if (enthIdx >= 0) findings.push({ ch: 'ch11', line: enthIdx + 1, issue: 'Still has enthusiasm', severity: 'medium' })
  let currentSec = null
  const subIssues = []
  for (let i = 0; i < lines.length; i++) {
    const secM = lines[i].match(/^##\s+第(\d+)节/)
    if (secM) { currentSec = parseInt(secM[1]); continue }
    const subM = lines[i].match(/^###\s+第(\d+)\.(\d+)节/)
    if (subM && currentSec !== null) {
      const subSec = parseInt(subM[1])
      // Skip 第0.x under 第1节 (intro subsections, intentionally kept)
      if (subSec === 0 && currentSec === 1) continue
      if (subSec !== currentSec) subIssues.push({ ch: 'ch11', line: i + 1, issue: ` subsection 第${subSec}.${subM[2]} under 第${currentSec}节`, severity: 'medium' })
    }
  }
  subIssues.slice(0, 5).forEach(f => findings.push(f))
  if (subIssues.length > 5) findings.push({ ch: 'ch11', line: 0, issue: `... and ${subIssues.length - 5} more subsection issues`, severity: 'low' })
})

check('ch12', (lines) => {
  const dogIdx = lines.findIndex(l => l.includes('k^2') && l.includes('sigma') && l.includes('k-1'))
  if (dogIdx >= 0) findings.push({ ch: 'ch12', line: dogIdx + 1, issue: 'DoG formula still has k-1 denominator', severity: 'high' })
  const c3Idx = lines.findIndex(l => l.includes('1516') && !l.includes('连接表'))
  if (c3Idx >= 0) findings.push({ ch: 'ch12', line: c3Idx + 1, issue: 'C3 param explanation incomplete', severity: 'low' })
})

check('ch21', (lines) => {
  const badText = lines.findIndex(l => /\text\{promising\}/.test(l) && !l.includes('$'))
  if (badText >= 0) findings.push({ ch: 'ch21', line: badText + 1, issue: '\\text{promising} outside math mode', severity: 'medium' })
  const spaceQuote = lines.findIndex(l => /"\s+[a-zA-Z]/.test(l))
  if (spaceQuote >= 0) findings.push({ ch: 'ch21', line: spaceQuote + 1, issue: 'Leading space in quotes', severity: 'low' })
})

check('ch22', (lines) => {
  if (lines[0].includes('第七章') && !lines[0].includes('第22章'))
    findings.push({ ch: 'ch22', line: 1, issue: 'Title still says 第七章', severity: 'critical' })
  const yoloIdx = lines.findIndex(l => l.includes('YOLOv5') && l.includes('Glenn Jocher'))
  if (yoloIdx >= 0) findings.push({ ch: 'ch22', line: yoloIdx + 1, issue: 'YOLOv5 citation has full name', severity: 'medium' })
  const partH = lines.findIndex(l => /^#\s+第.+部分/.test(l))
  if (partH >= 0) findings.push({ ch: 'ch22', line: partH + 1, issue: 'Still has part header', severity: 'medium' })
})

check('ch23', (lines) => {
  lines.forEach((l, i) => {
    if (l.includes('mathbf{') && !l.includes('$') && !l.includes('\\begin') && !l.includes('\\end')) {
      findings.push({ ch: 'ch23', line: i + 1, issue: 'mathbf outside math', severity: 'medium' })
    }
  })
  const hellIdx = lines.findIndex(l => l.includes('Hellinger') && l.includes('accret'))
  if (hellIdx >= 0) findings.push({ ch: 'ch23', line: hellIdx + 1, issue: 'Hellinger accrete mistranslation', severity: 'high' })
})

check('ch24', (lines) => {
  if (lines[0].includes('第九章') && !lines[0].includes('第24章'))
    findings.push({ ch: 'ch24', line: 1, issue: 'Title still says 第九章', severity: 'critical' })
  const boatIdx = lines.findIndex(l => l.includes('boat racing'))
  if (boatIdx >= 0) findings.push({ ch: 'ch24', line: boatIdx + 1, issue: 'Still has boat racing', severity: 'medium' })
  const aixIdx = lines.findIndex(l => l.includes('AIX360') && !l.includes('IBM'))
  if (aixIdx >= 0) findings.push({ ch: 'ch24', line: aixIdx + 1, issue: 'AIX360 missing IBM', severity: 'medium' })
  lines.forEach((l, i) => {
    const t = l.trim()
    if ((l.includes('Delta_i') || l.includes('\\forall'))) {
      if (!t.startsWith('$') && !t.startsWith('$$') && !t.startsWith('\\begin') && !t.startsWith('\\end')) {
        findings.push({ ch: 'ch24', line: i + 1, issue: 'Formula not in math mode', severity: 'medium' })
      }
    }
  })
})

const seen = new Set()
const unique = findings.filter(f => {
  const key = `${f.ch}:${f.line}:${f.issue.substring(0, 40)}`
  if (seen.has(key)) return false
  seen.add(key)
  return true
})

const order = { critical: 0, high: 1, medium: 2, low: 3 }
unique.sort((a, b) => (order[a.severity] || 9) - (order[b.severity] || 9))

const sevs = {}
unique.forEach(f => { sevs[f.severity] = (sevs[f.severity] || 0) + 1 })
console.log(JSON.stringify({ total: unique.length, bySeverity: sevs, findings: unique }, null, 2))
