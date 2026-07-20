// Single-pass comprehensive fix for ch18 and ch19 subsection numbering + cross-references
const fs = require('fs');
const path = require('path');

const base = 'C:/Users/22974/desktop/数模/token-burning-work/volumes';

// ============================================================
// ch18: Single-pass section/subsection renumbering
// ============================================================
let ch18 = fs.readFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), 'utf-8');
const lines18 = ch18.split('\n');
const out18 = [];

// Section mapping: old section number → new 第X节 + subsection count tracker
// Original: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10
// New: 第1节 ... 第10节
const sectionOrder = ['6.1','6.2','6.3','6.4','6.5','6.6','6.7','6.8','6.9','6.10'];
let currentSecIdx = -1;
let subCounter = 0;

for (const line of lines18) {
  // Section header: ## 6.X ...
  const secM = line.match(/^## (6\.[0-9]+) (.+)/);
  if (secM) {
    const oldNum = secM[1];
    const idx = sectionOrder.indexOf(oldNum);
    if (idx >= 0) {
      currentSecIdx = idx;
      subCounter = 0;
      out18.push('## 第' + (idx + 1) + '节 ' + secM[2]);
      continue;
    }
  }

  // Subsection header: ### 6.X.Y or ### 6.X.Y.Z
  const subM = line.match(/^### (6\.[0-9]+\.[0-9]+(?:\.[0-9]+)?) (.+)/);
  if (subM && currentSecIdx >= 0) {
    subCounter++;
    out18.push('### 第' + subCounter + '节 ' + subM[2]);
    continue;
  }

  out18.push(line);
}

ch18 = out18.join('\n');

// Fix body cross-references in ch18
// Subsection refs: 6.2.1节 → 第1节 (within section 第2节)
// Use a mapping approach
// 6.2.X节 → "第X节" (section 第2节的第X个子节)
ch18 = ch18.replace(/6\.2\.1节/g, '第1节');
ch18 = ch18.replace(/6\.2\.2节/g, '第2节');
ch18 = ch18.replace(/6\.2\.3节/g, '第3节');
ch18 = ch18.replace(/6\.2\.4节/g, '第4节');
ch18 = ch18.replace(/6\.2\.5节/g, '第5节');
ch18 = ch18.replace(/6\.2\.6节/g, '第6节');

// Section-level refs
ch18 = ch18.replace(/6\.1节/g, '第1节');
ch18 = ch18.replace(/6\.2节/g, '第2节');
ch18 = ch18.replace(/6\.3节/g, '第3节');
ch18 = ch18.replace(/6\.4节/g, '第4节');
ch18 = ch18.replace(/6\.5节/g, '第5节');
ch18 = ch18.replace(/6\.6节/g, '第6节');
ch18 = ch18.replace(/6\.7节/g, '第7节');
ch18 = ch18.replace(/6\.8节/g, '第8节');
ch18 = ch18.replace(/6\.9节/g, '第9节');
ch18 = ch18.replace(/6\.10节/g, '第10节');

// Also fix refs with 第 prefix: 第6.2节 → 第2节
ch18 = ch18.replace(/第6\.\d+节/g, (m) => {
  const num = m.match(/第6\.(\d+)节/)[1];
  return '第' + num + '节';
});

fs.writeFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), ch18, 'utf-8');
console.log('ch18 fully fixed (single-pass)');

// ============================================================
// ch19: Single-pass section/subsection renumbering
// ============================================================
let ch19 = fs.readFileSync(path.join(base, 'part5/ch19_深度学习的理论难题.md'), 'utf-8');
const lines19 = ch19.split('\n');
const out19 = [];

const secOrder19 = ['3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10'];
let curSec19 = -1;
let subCtr19 = 0;

for (const line of lines19) {
  const secM = line.match(/^## (3\.[0-9]+) (.+)/);
  if (secM) {
    const idx = secOrder19.indexOf(secM[1]);
    if (idx >= 0) {
      curSec19 = idx;
      subCtr19 = 0;
      out19.push('## 第' + (idx + 1) + '节 ' + secM[2]);
      continue;
    }
  }

  const subM = line.match(/^### (3\.[0-9]+\.[0-9]+(?:\.[0-9]+)?) (.+)/);
  if (subM && curSec19 >= 0) {
    subCtr19++;
    out19.push('### 第' + subCtr19 + '节 ' + subM[2]);
    continue;
  }

  out19.push(line);
}

ch19 = out19.join('\n');

// Fix NOTEARS double-replacement
ch19 = ch19.replace(/NOTEARSRS/g, 'NOTEARS');

// Fix body cross-references in ch19
// Section-level: 3.X节 → 第X节
for (let i = 1; i <= 10; i++) {
  ch19 = ch19.replace(new RegExp('3\\.' + i + '节', 'g'), '第' + i + '节');
  ch19 = ch19.replace(new RegExp('第3\\.' + i + '节', 'g'), '第' + i + '节');
}

// Subsection-level: 3.A.B节 → 第B节
ch19 = ch19.replace(/3\.(\d+)\.(\d+)节/g, (m, sec, sub) => '第' + sub + '节');

// Range refs: 3.7.2-3.7.4节 → 第2–第4节
ch19 = ch19.replace(/3\.(\d+)\.(\d+)-3\.(\d+)\.(\d+)节/g, (m, s1, sub1, s2, sub2) => {
  if (s1 === s2) return '第' + sub1 + '–' + '第' + sub2 + '节';
  return m;
});

fs.writeFileSync(path.join(base, 'part5/ch19_深度学习的理论难题.md'), ch19, 'utf-8');
console.log('ch19 fully fixed (single-pass)');

console.log('All comprehensive fixes applied!');
