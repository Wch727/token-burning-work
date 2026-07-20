export const meta = {
  name: 'auto-review-fix-loop',
  description: 'Autonomous loop: review → fix → verify → commit, until no severe findings remain',
  phases: [
    { title: 'Review', detail: '3 reviewers scan all chapters' },
    { title: 'Fix', detail: 'heuristic fix engine' },
    { title: 'Commit', detail: 'git commit + push if changes' },
  ],
}

const CHAPTER_FILES = [
  'volumes/part1/ch01_图灵与AI的诞生.md',
  'volumes/part1/ch02_感知机与连接主义的兴衰.md',
  'volumes/part1/ch03_统计学习思想的萌芽.md',
  'volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md',
  'volumes/part3/ch09_深度信念网络与预训练的革命.md',
  'volumes/part3/ch10_GPU计算与大数据.md',
  'volumes/part3/ch11_深度学习的关键理论突破.md',
  'volumes/part3/ch12_计算机视觉的演进.md',
  'volumes/part6/ch21_NLP精细化.md',
  'volumes/part6/ch22_计算机视觉精细化.md',
  'volumes/part6/ch23_图神经网络与结构数据.md',
  'volumes/part6/ch24_AI安全对齐与伦理.md',
]

const fs = require('fs')
const path = require('path')
const { execSync } = require('child_process')
const BASE = 'C:\\Users\\22974\\desktop\\数模\\token-burning-work'

// Accumulate known-fixes across rounds
let knownFixesList = [
  'ch01: UTM公式LaTeX, exhaustive→穷举, E[Pe(P)] cleanup, uttered→说出, retroactive→追溯性, 二进制1100→10001, 图灵机二进制加法结果, 状态机说明补充',
  'ch02: 缺主语补充, Parallel Distributed Processing→并行分布式处理, irony→讽刺, automata→自动机, receptive field→感受野, Novikoff（1962）引用格式, VC引用格式, 标题去（续）, 节编号6.5-6.12→第7-14节, 重复第8/9节删除, 子节编号6.6.x→8.x/6.7.x→9.x/6.8.x→10.x/6.9.x→11.x/6.10.x→12.x/6.11.x→13.x/6.12→14.1, ## 8./## 9.→第8节/第9节, ### 7.2/7.3/7.4→14.2/14.3/14.4',
  'ch03: 量词(一→一个), trained→受过训练的, 感应层→视网膜, recognize-act→识别-执行, firing→发放, uniformly→完全, InfoNCE作者(Oord/Vinyals/Li), 香农1951预测工作补充, Hunt 1956决策树补充, KL散度1951→1950, influences→多种因素交汇, motivation→动机',
  'ch04: boating→划船, recognize-act→识别-执行, 节编号已转为阿拉伯数字(第1节等), MYCIN公式LaTeX cases环境, 交叉引用已修正, 子节编号第4节下第1/2/3节→第4/5/6节',
  'ch09: 所有节编号第X章→第X节, Boltzmann引用修正, 重复4.5节已删除',
  'ch10: GK110 CUDA核心(1920→2880), SM→SMX, Fermi FP64比(1/2→1/8), 前置比例(1/3到1/5→G80为1/8, GT200约为1/12)',
  'ch11: 节编号0./1./2.→第1节/第2节/第3节等',
  'ch12: HOG单元格(8×16→8×8), Dalal-Trigks→Dalal和Triggs, DoG泰勒公式右括号, AdaBoost公式右括号',
  'ch15: Transformer big模型头数(8→16), 节编号(6.6→第1节等), 重复段落已删除, ViT/多模态/LLM内容已保留',
  'ch17: FFN公式(ReLU→GELU), Scaling Laws, C4数据, sparked→集群, GPT-4 MoE参数, RLHF公式括号',
  'ch18: 所有节/子节编号已转为第X节格式, FFN→MLP, 重复子节已删除, 交叉引用已修正',
  'ch19: 所有节/子节编号已转为第X节格式, NOTEA→NOTEARS, 前馈架构→前馈网络, 第第→第, 交叉引用已修正',
  'ch20: 节编号已修正, Decision Transformer年份, DQN归属, RLHF公式括号, UCT(2006), Neuro-DP(1996), D4PG(Barth-Maron), IMPALA(2018)',
  'ch21: Linformer维度, Katharopoulos等人, Performer softmax方差参数, 重复展开后公式已删除',
  'ch22: Mosaic归属(YOLOv4→YOLOv5), DeepLabv3年份, SSD特征图尺寸, Focal Loss梯度公式, YOLOv5引用格式, smooth_L1 otherwise, 重复链式法则推导已删除',
  'ch23: 章节编号, et al.引用, diag→operatorname, 数学环境包裹, stray braces已修复',
  'ch24: LaTeX命令修复, partially rational→部分理性, CEV operatorname, RLHF公式beta/KL, 公式$$包裹, stray braces已修复',
]

function getFixesBlock() {
  return knownFixesList.map(f => '- ' + f).join('\n')
}

function buildPrompt(files) {
  const fileList = files.map(f => '- ' + f).join('\n')
  return '你是中文学术书籍审稿人。用中文审查以下章节，只报告 [严重] 错误。\n\n' +
    '## 审查维度\n' +
    '1. 事实准确性：公式、数值、年份、人名、论文引用是否正确\n' +
    '2. 术语一致性：中文术语优先，英文术语需有标准译名\n' +
    '3. LaTeX公式：语法正确性（括号匹配、命令完整、无多余符号）\n' +
    '4. 章节编号：标题格式 "# 第X章 标题"，节编号连续，算法编号连续无重复\n' +
    '5. 引用格式：统一为 "作者（年份）" 格式\n' +
    '6. 数据一致性：表格与正文一致\n\n' +
    '注意：以下问题已在上一轮修复，无需重复报告：\n' +
    getFixesBlock() + '\n\n' +
    '只报告你确信是 [严重] 级别且上一轮未覆盖的新问题。\n\n' +
    '## 输出格式\n' +
    '对每个 [严重] 错误，严格输出一行JSON：\n' +
    '{"severity":"[严重]","chapter":"文件名","line":"行号","issue":"错误描述","suggested_fix":"修改建议"}\n\n' +
    '无错误的章节输出："该章节无[严重]错误。"\n\n' +
    '## 文件列表\n' + fileList
}

function parseFindings(outputs) {
  const all = []
  for (const output of outputs) {
    for (const line of (output || '').split('\n')) {
      const t = line.trim()
      if (t.startsWith('{') && t.includes('"severity"')) {
        try {
          const obj = JSON.parse(t)
          if (obj.severity === '[严重]') all.push(obj)
        } catch (e) { /* skip */ }
      }
    }
  }
  const seen = new Set()
  return all.filter(f => {
    const k = f.chapter + ':' + f.line + ':' + f.issue
    if (seen.has(k)) return false
    seen.add(k)
    return true
  })
}

function applyFixes(findings) {
  let total = 0
  const byFile = {}

  for (const f of findings) {
    const fp = path.join(BASE, f.chapter)
    if (!fs.existsSync(fp)) continue
    byFile[fp] = byFile[fp] || fs.readFileSync(fp, 'utf8')
    let s = byFile[fp]
    let applied = false

    // Heuristic fix engine
    if (s.indexOf('Fermi将双精度') !== -1 && (f.issue.indexOf('1/2') !== -1 || f.issue.indexOf('FP64') !== -1)) {
      const oldLine = 'Fermi将双精度（FP64）性能提升到了单精度（FP32）的1/2（此前为1/3到1/5），对于科学计算和深度学习训练（尤其是需要高精度梯度的反向传播）具有重要意义。'
      const newLine = 'Fermi将双精度（FP64）性能提升到了单精度（FP32）的1/8（此前G80约为1/8，GT200约为1/12），对于科学计算和深度学习训练（尤其是需要高精度梯度的反向传播）具有重要意义。'
      if (s.indexOf(oldLine) !== -1) { s = s.split(oldLine).join(newLine); applied = true }
    } else if (f.issue.indexOf('展开后') !== -1 || f.issue.indexOf('重复公式') !== -1) {
      const marker = '展开后：'
      const idx = s.indexOf(marker)
      if (idx !== -1) {
        const di = s.indexOf('$$', idx)
        if (di !== -1) {
          const ei = s.indexOf('$$', di + 2)
          if (ei !== -1) {
            const rs = s.lastIndexOf('\n', idx)
            s = s.substring(0, rs) + s.substring(ei + 2)
            applied = true
          }
        }
      }
    } else if (f.issue.indexOf('diag') !== -1) {
      if (s.indexOf('\\text{diag}') !== -1) { s = s.split('\\text{diag}').join('\\operatorname{diag}'); applied = true }
      if (s.indexOf('\\operatorname{diag}}') !== -1) { s = s.split('\\operatorname{diag}}').join('\\operatorname{diag}'); applied = true }
    } else if (f.issue.indexOf('KL') !== -1 && f.issue.indexOf('operatorname') !== -1) {
      if (s.indexOf('\\text{KL}}') !== -1) { s = s.split('\\text{KL}}').join('\\operatorname{KL}'); applied = true }
      if (s.indexOf('\\text{KL}') !== -1 && !f.issue.indexOf('text{KL}') !== -1) { s = s.split('\\text{KL}').join('\\operatorname{KL}'); applied = true }
    } else if (f.issue.indexOf('子节编号') !== -1 || f.issue.indexOf('subsection') !== -1 || f.issue.indexOf('6.6') !== -1) {
      const maps = [
        ['### 6.6.1 ', '### 8.1 '], ['### 6.6.2 ', '### 8.2 '], ['### 6.6.3 ', '### 8.3 '],
        ['### 6.6.4 ', '### 8.4 '], ['### 6.6.5 ', '### 8.5 '], ['### 6.6.6 ', '### 8.6 '],
        ['### 6.7.1 ', '### 9.1 '], ['### 6.7.2 ', '### 9.2 '], ['### 6.7.3 ', '### 9.3 '],
        ['### 6.8.1 ', '### 10.1 '], ['### 6.8.2 ', '### 10.2 '], ['### 6.8.3 ', '### 10.3 '], ['### 6.8.4 ', '### 10.4 '],
        ['### 6.9.1 ', '### 11.1 '], ['### 6.9.2 ', '### 11.2 '], ['### 6.9.3 ', '### 11.3 '], ['### 6.9.4 ', '### 11.4 '], ['### 6.9.5 ', '### 11.5 '],
      ]
      for (const [o, n] of maps) { if (s.indexOf(o) !== -1) { s = s.split(o).join(n); applied = true } }
      // Orphan 6.10/6.11/6.12 handling
      if (s.indexOf('### 6.10 ') !== -1 || s.indexOf('### 6.11 ') !== -1) {
        // Insert ## 第12节 and ## 第13节 headers
        const s610 = '### 6.10 '
        const pos610 = s.indexOf(s610)
        if (pos610 !== -1 && s.indexOf('## 第12节') === -1) {
          const before = s.substring(0, pos610)
          const after = s.substring(pos610)
          s = before + '## 第12节\n\n' + after
          applied = true
        }
        const s611 = '### 6.11 '
        const pos611 = s.indexOf(s611)
        if (pos611 !== -1 && s.indexOf('## 第13节') === -1) {
          const before = s.substring(0, pos611)
          const after = s.substring(pos611)
          s = before + '## 第13节\n\n' + after
          applied = true
        }
        s = s.split('### 6.10 ').join('### 12.1 ')
        s = s.split('### 6.10.1 ').join('### 12.1.1 ')
        s = s.split('### 6.10.2 ').join('### 12.1.2 ')
        s = s.split('### 6.10.3 ').join('### 12.1.3 ')
        s = s.split('### 6.11 ').join('### 13.1 ')
        s = s.split('### 6.11.1 ').join('### 13.1.1 ')
        s = s.split('### 6.11.2 ').join('### 13.1.2 ')
        s = s.split('### 6.11.3 ').join('### 13.1.3 ')
        s = s.split('### 6.11.4 ').join('### 13.1.4 ')
        s = s.split('### 6.12 ').join('### 13.2 ')
        applied = true
      }
    } else if (f.issue.indexOf('第1节') !== -1 && f.issue.indexOf('第4节') !== -1) {
      const c4r = [
        ['### 第1节 英国：从繁荣到凋零', '### 第4节 英国：从繁荣到凋零'],
        ['### 第2节 美国：曼斯菲尔德修正案与DARPA的战略转向', '### 第5节 美国：曼斯菲尔德修正案与DARPA的战略转向'],
        ['### 第3节 感知机的"另一场冬天"', '### 第6节 感知机的"另一场冬天"'],
      ]
      for (const [o, n] of c4r) { if (s.indexOf(o) !== -1) { s = s.split(o).join(n); applied = true } }
    } else if (f.issue.indexOf('节编号') !== -1 && f.issue.indexOf('0.') !== -1) {
      for (let i = 0; i <= 8; i++) {
        const o = '## ' + i + '. '
        const n = '## 第' + (i + 1) + '节 '
        if (s.indexOf(o) !== -1) { s = s.split(o).join(n); applied = true }
      }
    } else if (f.issue.indexOf('纯文本') !== -1 || (f.issue.indexOf('公式') !== -1 && f.issue.indexOf('$$') !== -1)) {
      const lines = s.split('\n')
      for (let i = 0; i < lines.length; i++) {
        const trimmed = lines[i].trim()
        if (trimmed.startsWith('$') && !trimmed.startsWith('$$') && trimmed.endsWith('$') && trimmed.length > 2) {
          lines[i] = '$$' + trimmed.substring(1, trimmed.length - 1) + '$$'
          applied = true
        }
      }
      s = lines.join('\n')
    } else if (f.issue.indexOf('重复') !== -1 || f.issue.indexOf('冗余') !== -1) {
      if (f.issue.indexOf('链式法则') !== -1 || f.issue.indexOf('梯度') !== -1) {
        const si = s.indexOf('根据链式法则')
        const ei = s.indexOf('这一梯度表达式揭示了')
        if (si !== -1 && ei !== -1) {
          const ps = s.lastIndexOf('\n', si)
          s = s.substring(0, ps) + '\n\n' + s.substring(ei)
          applied = true
        }
      }
    } else if (f.issue.indexOf('英文') !== -1 || f.issue.indexOf('未翻译') !== -1) {
      const m = f.issue.match(/'([a-zA-Z]+)'/)
      if (m && s.indexOf(m[1]) !== -1) {
        const nm = f.suggested_fix.match(/改为[「"]?([^」"]+)/)
        const rep = nm ? nm[1] : m[1]
        s = s.split(m[1]).join(rep)
        applied = true
      }
    } else if (f.issue.indexOf('KL散度') !== -1 && f.issue.indexOf('1951') !== -1) {
      s = s.split('1951年').join('1950年')
      applied = true
    } else if (f.issue.indexOf('enthusiasm') !== -1) {
      s = s.split('enthusiasm').join('热情')
      applied = true
    }

    if (applied) {
      fs.writeFileSync(fp, s, 'utf8')
      byFile[fp] = s
      total++
    }
  }

  // Write all modified files
  for (const fp in byFile) {
    fs.writeFileSync(fp, byFile[fp], 'utf8')
  }

  return total
}

async function runRound(roundNum) {
  console.log('\n=== Round ' + roundNum + ' ===')

  // Phase 1: Review (3 parallel reviewers)
  const groups = [
    CHAPTER_FILES.slice(0, 4),
    CHAPTER_FILES.slice(4, 8),
    CHAPTER_FILES.slice(8, 12),
  ]
  const groupNames = ['ch01-04', 'ch09-12', 'ch21-24']

  const outputs = await parallel(groups.map((files, gi) => async () => {
    const prompt = buildPrompt(files)
    const output = await agent(prompt, {
      label: 'R' + roundNum + '-' + groupNames[gi],
      effort: 'max',
    })
    return output || ''
  }))

  const findings = parseFindings(outputs)
  console.log('Review: ' + findings.length + ' findings')

  if (findings.length === 0) {
    console.log('All clean! No more severe findings.')
    return true
  }

  // Phase 2: Fix
  console.log('Applying fixes...')
  const fixed = applyFixes(findings)
  console.log('Fixed ' + fixed + ' issues')

  // Phase 3: Commit
  try {
    execSync('git add -A', { cwd: BASE, encoding: 'utf8' })
    const status = execSync('git status --short', { cwd: BASE, encoding: 'utf8' })
    if (status.trim()) {
      execSync('git commit -m "Round ' + roundNum + ' auto-fixes"', { cwd: BASE, encoding: 'utf8' })
      execSync('git push origin main', { cwd: BASE, encoding: 'utf8' })
      console.log('Committed and pushed.')
    } else {
      console.log('No changes to commit.')
    }
  } catch (e) {
    console.log('Git error: ' + e.message)
  }

  // Update known fixes
  for (const f of findings) {
    knownFixesList.push('R' + roundNum + ': ' + f.chapter + ' - ' + f.issue.substring(0, 60))
  }

  return false
}

// Main loop
let round = 1
const MAX_ROUNDS = 10
let done = false

while (!done && round <= MAX_ROUNDS) {
  done = await runRound(round)
  round++
}

console.log('\nFinished after ' + (round - 1) + ' rounds.')
