// Fix R7 review findings v2 - comprehensive fixes for all 24 findings
const fs = require('fs');
const path = require('path');

const BASE = 'C:\\Users\\22974\\desktop\\数模\\token-burning-work';
let total = 0;

function fix(file, oldStr, newStr, label) {
  const fp = path.join(BASE, file);
  let s = fs.readFileSync(fp, 'utf8');
  if (s.includes(oldStr)) {
    s = s.split(oldStr).join(newStr);
    fs.writeFileSync(fp, s, 'utf8');
    console.log(`[FIXED] ${label}`);
    total++;
  } else {
    console.log(`[SKIP] ${label} - pattern not found`);
  }
}

// ===== ch01: PSSH year 1969→1976 =====
fix('volumes/part1/ch01_图灵与AI的诞生.md',
  '| 1969 | PSSH（物理符号系统假说）提出 |',
  '| 1976 | PSSH（物理符号系统假说）提出 |',
  'ch01: PSSH year 1969→1976');

// ===== ch02: XOR inequality fix =====
fix('volumes/part1/ch02_感知机与连接主义的兴衰.md',
  '但由(1)有b < 0，所以-b > 0。而w1 > 0, w2 > 0意味着w1 + w2 > 0。因此w1 + w2 > 0 > -b，即w1 + w2 > -b，这与w1 + w2 < -b矛盾。',
  '但由(1)有b < 0，所以-b > 0。而w1 > 0, w2 > 0意味着w1 + w2 > 0。由(2)(3)得w1 > -b, w2 > -b，故w1 + w2 > 2(-b) > -b（因-b > 0），这与(4)中w1 + w2 < -b矛盾。',
  'ch02: XOR inequality chain');

// ===== ch02: duplicate section numbers =====
{
  const fp = path.join(BASE, 'volumes/part1/ch02_感知机与连接主义的兴衰.md');
  let s = fs.readFileSync(fp, 'utf8');
  // Fix second occurrence of ## 第8节 → ## 第15节
  const pos8 = [];
  let idx = 0;
  while (true) {
    idx = s.indexOf('## 第8节 ', idx);
    if (idx === -1) break;
    pos8.push(idx);
    idx++;
  }
  if (pos8.length >= 2) {
    const before = s.substring(0, pos8[1]);
    const after = s.substring(pos8[1]);
    const newAfter = after.replace('## 第8节 ', '## 第15节 ');
    s = before + newAfter;
    fs.writeFileSync(fp, s, 'utf8');
    console.log('[FIXED] ch02: second ## 第8节 → ## 第15节');
    total++;
  } else {
    console.log('[SKIP] ch02: ## 第8节 duplicates not found (found ' + pos8.length + ')');
  }

  // Fix second occurrence of ## 第9节 → ## 第16节
  const pos9 = [];
  idx = 0;
  while (true) {
    idx = s.indexOf('## 第9节 ', idx);
    if (idx === -1) break;
    pos9.push(idx);
    idx++;
  }
  if (pos9.length >= 2) {
    const before = s.substring(0, pos9[1]);
    const after = s.substring(pos9[1]);
    const newAfter = after.replace('## 第9节 ', '## 第16节 ');
    s = before + newAfter;
    fs.writeFileSync(fp, s, 'utf8');
    console.log('[FIXED] ch02: second ## 第9节 → ## 第16节');
    total++;
  } else {
    console.log('[SKIP] ch02: ## 第9节 duplicates not found (found ' + pos9.length + ')');
  }
}

// ===== ch03: duplicate text fix =====
fix('volumes/part1/ch03_统计学习思想的萌芽.md',
  '多种 多种因素交汇 的交汇',
  '多种因素的交汇',
  'ch03: duplicate text');

// ===== ch04: missing period after 工作 =====
fix('volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md',
  '情况下工作1974年，DARPA',
  '情况下工作。1974年，DARPA',
  'ch04: missing period after 工作');

// ===== ch04: subsection numbering under 第4节 =====
{
  const fp = path.join(BASE, 'volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md');
  let s = fs.readFileSync(fp, 'utf8');
  let c = 0;

  // ### 第4节 英国 → ### 第4.1节 英国 (under ## 第4节)
  if (s.includes('### 第4节 英国：从繁荣到凋零')) {
    s = s.replace('### 第4节 英国：从繁荣到凋零', '### 第4.1节 英国：从繁荣到凋零');
    c++;
  }
  // ### 第5节 美国 → ### 第4.2节 美国
  if (s.includes('### 第5节 美国：曼斯菲尔德修正案与DARPA的战略转向')) {
    s = s.replace('### 第5节 美国：曼斯菲尔德修正案与DARPA的战略转向', '### 第4.2节 美国：曼斯菲尔德修正案与DARPA的战略转向');
    c++;
  }
  // ### 第6节 感知机的"另一场冬天" → ### 第4.3节
  if (s.includes('### 第6节 感知机的"另一场冬天"')) {
    s = s.replace('### 第6节 感知机的"另一场冬天"', '### 第4.3节 感知机的"另一场冬天"');
    c++;
  }

  if (c > 0) { fs.writeFileSync(fp, s, 'utf8'); console.log(`[FIXED] ch04: subsection numbering under 第4节 (${c})`); total += c; }
  else console.log('[SKIP] ch04: subsection numbering under 第4节');
}

// ===== ch04: mixed numbering (Arabic → 第X节 format) =====
{
  const fp = path.join(BASE, 'volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md');
  let s = fs.readFileSync(fp, 'utf8');
  let c = 0;

  const numMaps = [
    ['### 5.1 DENDRAL：第一个专家系统的诞生', '### 第5.1节 DENDRAL：第一个专家系统的诞生'],
    ['### 5.2 MYCIN：不确定性推理的里程碑', '### 第5.2节 MYCIN：不确定性推理的里程碑'],
    ['### 5.3 XCON（R1）：从实验室到装配线的巨大飞跃', '### 第5.3节 XCON（R1）：从实验室到装配线的巨大飞跃'],
    ['### 6.1 知识表示：三种主要范式', '### 第6.1节 知识表示：三种主要范式'],
    ['### 6.2 知识获取瓶颈：专家系统未能解决的核心难题', '### 第6.2节 知识获取瓶颈：专家系统未能解决的核心难题'],
    ['### 7.1 黄金时代：从学术实验到产业支柱', '### 第7.1节 黄金时代：从学术实验到产业支柱'],
    ['### 7.2 阿喀琉斯之踵：专家系统的根本局限', '### 第7.2节 阿喀琉斯之踵：专家系统的根本局限'],
  ];

  for (const [old, neu] of numMaps) {
    if (s.includes(old) && !s.includes(neu)) {
      s = s.split(old).join(neu);
      c++;
    }
  }

  if (c > 0) { fs.writeFileSync(fp, s, 'utf8'); console.log(`[FIXED] ch04: mixed numbering (${c})`); total += c; }
  else console.log('[SKIP] ch04: mixed numbering');
}

// ===== ch09: DBN journal fix =====
fix('volumes/part3/ch09_深度信念网络与预训练的革命.md',
  '在《科学》杂志上发表了一篇里程碑式的论文《A Fast Learning Algorithm for Deep Belief Nets》',
  '在《Neural Computation》期刊上发表了题为《A Fast Learning Algorithm for Deep Belief Nets》的论文',
  'ch09: DBN journal Science→Neural Computation');

// ===== ch11: English translations =====
{
  const fp = path.join(BASE, 'volumes/part3/ch11_深度学习的关键理论突破.md');
  let s = fs.readFileSync(fp, 'utf8');
  let c = 0;

  // enthusiasm → 热情
  if (s.includes('enthusiasm 很快遭遇了严峻的现实')) {
    s = s.replace('enthusiasm 很快遭遇了严峻的现实', '热情很快遭遇了严峻的现实');
    c++;
  }
  // relegated to → 被视为
  if (s.includes('SGD被 relegated to 大规模机器学习的工程折中')) {
    s = s.replace('被 relegated to 大规模机器学习的工程折中', '被视为大规模机器学习的工程折中');
    c++;
  }

  if (c > 0) { fs.writeFileSync(fp, s, 'utf8'); console.log(`[FIXED] ch11: English translations (${c})`); total += c; }
  else console.log('[SKIP] ch11: English translations');
}

// ===== ch11: citation format fixes (13 instances) =====
{
  const fp = path.join(BASE, 'volumes/part3/ch11_深度学习的关键理论突破.md');
  let s = fs.readFileSync(fp, 'utf8');
  let c = 0;

  // Pattern: "XXX等人YYYY年" → "XXX等人（YYYY年）"
  // Pattern: "XXX和XXX YYYY年" → "XXX和XXX（YYYY年）"
  // Pattern: "XXX, XXX和XXX YYYY年" → "XXX, XXX和XXX（YYYY年）"

  const citeFixes = [
    // "LeCun等人1998年" → "LeCun等人（1998年）"
    [/LeCun等人1998年/g, 'LeCun等人（1998年）'],
    // "Dauphin等人2014年" → "Dauphin等人（2014年）"
    [/Dauphin等人2014年/g, 'Dauphin等人（2014年）'],
    // "Erhan等人2010年" → "Erhan等人（2010年）"
    [/Erhan等人2010年/g, 'Erhan等人（2010年）'],
    // "Glorot和Bengio 2010年" → "Glorot和Bengio（2010年）"
    [/Glorot和Bengio 2010年/g, 'Glorot和Bengio（2010年）'],
    // "Pearlmutter 1994年" → "Pearlmutter（1994年）"
    [/Pearlmutter 1994年/g, 'Pearlmutter（1994年）'],
    // "Mandt等人2017年" → "Mandt等人（2017年）"
    [/Mandt等人2017年/g, 'Mandt等人（2017年）'],
    // "Keskar等人2017年" → "Keskar等人（2017年）"
    [/Keskar等人2017年/g, 'Keskar等人（2017年）'],
    // "Sutskever等人2013年" → "Sutskever等人（2013年）"
    [/Sutskever等人2013年/g, 'Sutskever等人（2013年）'],
    // "Hardt, Recht和Singer 2016年" → "Hardt, Recht和Singer（2016年）"
    [/Hardt, Recht和Singer 2016年/g, 'Hardt, Recht和Singer（2016年）'],
    // "Nair和Hinton 2010" → "Nair和Hinton（2010）"
    [/Nair和Hinton 2010/g, 'Nair和Hinton（2010）'],
    // "Glorot和Bengio 2011" → "Glorot和Bengio（2011）"
    [/Glorot和Bordes和Bengio 2011/g, 'Glorot、Bordes和Bengio（2011）'],
    [/Glorot和Bengio 2011/g, 'Glorot和Bengio（2011）'],
  ];

  for (const [pattern, replacement] of citeFixes) {
    const before = s;
    s = s.replace(pattern, replacement);
    if (s !== before) {
      c++;
      console.log(`  citation: ${pattern} → ${replacement}`);
    }
  }

  if (c > 0) { fs.writeFileSync(fp, s, 'utf8'); console.log(`[FIXED] ch11: citation format (${c})`); total += c; }
  else console.log('[SKIP] ch11: citation format');
}

// ===== ch23: mathbf m→M =====
fix('volumes/part6/ch23_图神经网络与结构数据.md',
  String.fromCharCode(92) + 'mathbf{m}^{\\text{edge}}',
  String.fromCharCode(92) + 'mathbf{M}^{\\text{edge}}',
  'ch23: mathbf{m}→mathbf{M}');

// ===== ch23: |mathbf → \mathbf =====
fix('volumes/part6/ch23_图神经网络与结构数据.md',
  '|mathbf{h}_{' + String.fromCharCode(92) + 'varnothing}^{(k)}',
  String.fromCharCode(92) + 'mathbf{h}_{' + String.fromCharCode(92) + 'varnothing}^{(k)}',
  'ch23: |mathbf→mathbf');

// ===== ch24: chi-squared fix =====
fix('volumes/part6/ch24_AI安全对齐与伦理.md',
  '阈值设为$t^{(1)} = 0.9$的$\\chi^2$逆累积分布函数值',
  '阈值设为$t^{(1)} = \\chi^{-2}_{0.9}(4) \\approx 7.78$（即自由度为4的$\\chi^2$分布的第90百分位数）',
  'ch24: chi-squared threshold');

// ===== ch24: IBM AIX360 fix =====
fix('volumes/part6/ch24_AI安全对齐与伦理.md',
  'IBM的AI Explainability 360工具包和Microsoft的InterpretML工具包',
  'IBM的AI Explainability 360（AIX360）工具包和Microsoft的InterpretML工具包',
  'ch24: IBM AIX360 naming');

console.log(`\nTotal fixes applied: ${total}`);
