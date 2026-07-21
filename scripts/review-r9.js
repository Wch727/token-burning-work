const fs = require('fs')
const path = require('path')
const BASE = path.join(process.env.USERPROFILE || process.env.HOME, 'desktop', '数模', 'token-burning-work')

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

const findings = []
const seen = new Set()

function addFinding(ch, line, issue, severity) {
  const key = `${ch}:${line}:${issue.substring(0, 50)}`
  if (seen.has(key)) return
  seen.add(key)
  findings.push({ ch, line, issue, severity })
}

function chLabel(fp) {
  return fp.match(/ch(\d+)/)[1].padStart(2, '0')
}

// ============================================================
// DIMENSION 1: Section numbering consistency (ALL chapters)
// ============================================================
ALL_FILES.forEach(fp => {
  const ch = chLabel(fp)
  const ls = lines(read(fp))
  const sections = new Set()
  const subsections = new Set()

  ls.forEach((l, i) => {
    const secM = l.match(/^##\s+第(\d+)节/)
    if (secM) {
      const n = parseInt(secM[1])
      if (sections.has(n)) addFinding(`ch${ch}`, i + 1, `Duplicate section 第${n}节`, 'critical')
      sections.add(n)
    }

    // Check subsection format consistency
    const subM = l.match(/^###\s+(.+)节\s(.+)/)
    if (subM) {
      const subNum = subM[1]
      // Should use 第X.Y节 format, not just X.Y
      if (!subNum.includes('第')) addFinding(`ch${ch}`, i + 1, `Subsection missing 第 prefix: 第${subNum}节`, 'medium')
    }
  })
})

// ============================================================
// DIMENSION 2: Citation format - Chinese academic standard "作者（年份）"
// ============================================================
ALL_FILES.forEach(fp => {
  const ch = chLabel(fp)
  const ls = lines(read(fp))

  ls.forEach((l, i) => {
    // Check for English-style citations: Author (Year) or Author, (Year) or (Year)
    const badCite = l.match(/[A-Z][a-z]+(?:\s(?:et\s+al\.?|and|\&))?\s*\(\s*\d{4}\s*\)/)
    if (badCite) {
      const match = badCite[0]
      // Skip if already in Chinese format or if it's a math/table context
      if (!match.startsWith('（') && !match.includes('等') && l.length > 10) {
        addFinding(`ch${ch}`, i + 1, `Non-Chinese citation format: "${match}"`, 'medium')
      }
    }

    // Check for "Author, Year" without parentheses
    const bareCite = l.match(/[A-Z][a-z]+,\s*19[6-9]\d|20[0-2]\d/)
    if (bareCite && !l.includes('等') && !l.includes('（') && l.length > 20) {
      addFinding(`ch${ch}`, i + 1, `Citation without Chinese parentheses: "${bareCite[0]}"`, 'low')
    }
  })
})

// ============================================================
// DIMENSION 3: LaTeX math mode integrity
// ============================================================
ALL_FILES.forEach(fp => {
  const ch = chLabel(fp)
  const ls = lines(read(fp))

  ls.forEach((l, i) => {
    // Check for unmatched $ in a line (odd number of $)
    const dollarCount = (l.match(/\$/g) || []).length
    if (dollarCount % 2 !== 0 && dollarCount > 0 && !l.includes('\\$')) {
      addFinding(`ch${ch}`, i + 1, `Unmatched $ in math mode (${dollarCount} dollars)`, 'high')
    }

    // Check for \text{} outside math mode (not between $ signs)
    const textMatches = l.match(/\\text\{[^}]+\}/g)
    if (textMatches) {
      const inMath = isBetweenMathMode(l)
      if (!inMath) {
        textMatches.forEach(t => addFinding(`ch${ch}`, i + 1, `\\text{} outside math mode: ${t.substring(0, 40)}`, 'medium'))
      }
    }

    // Check for common LaTeX commands without backslash
    const badLatex = []
    if (/\bsum\b/.test(l) && !l.includes('$') && !l.includes('\\sum') && !l.includes('\\text') && !l.includes('displaystyle') && !l.includes('sum_{')) badLatex.push('sum')
    if (/\bcdot\b/.test(l) && !l.includes('$') && !l.includes('\\cdot') && !l.includes('\\text')) badLatex.push('cdot')
    if (/\bfrac\b/.test(l) && !l.includes('$') && !l.includes('\\frac') && !l.includes('\\text')) badLatex.push('frac')
    if (/\blog\b/.test(l) && !l.includes('$') && !l.includes('\\log') && !l.includes('\\text')) badLatex.push('log')
    if (/\bmax\b/.test(l) && !l.includes('$') && !l.includes('\\max') && !l.includes('\\text')) badLatex.push('max')
    if (/\bmin\b/.test(l) && !l.includes('$') && !l.includes('\\min') && !l.includes('\\text')) badLatex.push('min')
    if (badLatex.length > 0) addFinding(`ch${ch}`, i + 1, `Missing backslash on LaTeX: ${badLatex.join(', ')}`, 'medium')
  })
})

function isBetweenMathMode(line) {
  let inMath = false
  for (const c of line) {
    if (c === '$') inMath = !inMath
  }
  return inMath
}

// ============================================================
// DIMENSION 4: English/Chinese mixing in running text
// ============================================================
ALL_FILES.forEach(fp => {
  const ch = chLabel(fp)
  const ls = lines(read(fp))

  ls.forEach((l, i) => {
    // Skip math mode lines, table rows, code blocks
    if (l.startsWith('$$') || l.startsWith('$') || l.startsWith('|') || l.startsWith('```') || l.startsWith('#')) return
    if (l.startsWith('图') && l.includes('|')) return

    // Check for standalone English words that should be translated
    // Common untranslated terms in Chinese AI history context
    const untranslated = [
      ' Faculty ', ' faculty ', 'relegated', 'enthusiasm',
      'landmark', 'breakthrough', 'state-of-the-art', 'benchmark',
      'outperforms', 'outperform', 'significantly',
    ]
    untranslated.forEach(term => {
      if (l.includes(term)) {
        addFinding(`ch${ch}`, i + 1, `Untranslated English term: "${term.trim()}"`, 'medium')
      }
    })
  })
})

// ============================================================
// DIMENSION 5: Table consistency
// ============================================================
ALL_FILES.forEach(fp => {
  const ch = chLabel(fp)
  const ls = lines(read(fp))

  ls.forEach((l, i) => {
    // Check table rows have consistent column count
    if (l.startsWith('|') && !l.startsWith('|---') && !l.startsWith('| 年')) {
      const cols = l.split('|').filter(s => s.trim().length > 0).length
      if (cols < 3) addFinding(`ch${ch}`, i + 1, `Table row has only ${cols} columns`, 'low')
    }
  })
})

// ============================================================
// DIMENSION 6: Factual claims needing verification (sample)
// ============================================================
// Check specific known facts that should be correct
ALL_FILES.forEach(fp => {
  const ch = chLabel(fp)
  const text = read(fp)

  // Perceptron: Rosenblatt 1958, not 1957
  if (text.includes('感知机') && text.includes('1957')) {
    const idx = text.indexOf('感知机') > -1 && text.indexOf('1957')
    if (idx > -1) {
      const l = lines(text)[Math.floor(idx / 1)]
      addFinding(`ch${ch}`, 0, 'Check: Perceptron year (should be 1958)', 'info')
    }
  }
})

// Deduplicate and sort
findings.sort((a, b) => {
  const order = { critical: 0, high: 1, medium: 2, low: 3, info: 4 }
  return (order[a.severity] || 9) - (order[b.severity] || 9)
})

const sevs = {}
findings.forEach(f => { sevs[f.severity] = (sevs[f.severity] || 0) + 1 })

console.log(JSON.stringify({
  total: findings.length,
  byChapter: Object.fromEntries(
    [...new Set(findings.map(f => f.ch))].sort().map(ch => [ch, findings.filter(f => f.ch === ch).length])
  ),
  bySeverity: sevs,
  findings: findings.slice(0, 80)
}, null, 2))
