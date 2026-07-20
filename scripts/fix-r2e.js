// Fixed ch18/ch19 single-pass renumbering with multiline flag
const fs = require('fs');
const path = require('path');

const base = 'C:/Users/22974/desktop/数模/token-burning-work/volumes';

// ============================================================
// ch18
// ============================================================
let ch18 = fs.readFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), 'utf-8');
const lines18 = ch18.split('\n');
const out18 = [];

const sectionOrder = ['6.1','6.2','6.3','6.4','6.5','6.6','6.7','6.8','6.9','6.10'];
let curSec = -1, subCtr = 0;

for (const line of lines18) {
  // Section header (multiline-safe: line-by-line processing, no ^ needed)
  const secM = line.match(/^## (6\.[0-9]+) (.+)/);
  if (secM) {
    const idx = sectionOrder.indexOf(secM[1]);
    if (idx >= 0) {
      curSec = idx;
      subCtr = 0;
      out18.push('## 第' + (idx+1) + '节 ' + secM[2]);
      continue;
    }
  }
  // Subsection header
  const subM = line.match(/^### (6\.[0-9]+\.[0-9]+(?:\.[0-9]+)?) (.+)/);
  if (subM && curSec >= 0) {
    subCtr++;
    out18.push('### 第' + subCtr + '节 ' + subM[2]);
    continue;
  }
  out18.push(line);
}
ch18 = out18.join('\n');

// Body cross-references in ch18
// Subsection refs: 6.2.1节 → 第1节, etc.
ch18 = ch18.replace(/6\.2\.1节/g, '第1节');
ch18 = ch18.replace(/6\.2\.2节/g, '第2节');
ch18 = ch18.replace(/6\.2\.3节/g, '第3节');
ch18 = ch18.replace(/6\.2\.4节/g, '第4节');
ch18 = ch18.replace(/6\.2\.5节/g, '第5节');
ch18 = ch18.replace(/6\.2\.6节/g, '第6节');
// Section refs
['1','2','3','4','5','6','7','8','9','10'].forEach(n => {
  ch18 = ch18.replace(new RegExp('6\\.' + n + '节', 'g'), '第' + n + '节');
  ch18 = ch18.replace(new RegExp('第6\\.' + n + '节', 'g'), '第' + n + '节');
});

fs.writeFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), ch18, 'utf-8');
console.log('ch18 fixed');

// ============================================================
// ch19
// ============================================================
let ch19 = fs.readFileSync(path.join(base, 'part5/ch19_深度学习的理论难题.md'), 'utf-8');
const lines19 = ch19.split('\n');
const out19 = [];

const secOrder19 = ['3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10'];
let c19 = -1, s19 = 0;

for (const line of lines19) {
  const secM = line.match(/^## (3\.[0-9]+) (.+)/);
  if (secM) {
    const idx = secOrder19.indexOf(secM[1]);
    if (idx >= 0) {
      c19 = idx; s19 = 0;
      out19.push('## 第' + (idx+1) + '节 ' + secM[2]);
      continue;
    }
  }
  const subM = line.match(/^### (3\.[0-9]+\.[0-9]+(?:\.[0-9]+)?) (.+)/);
  if (subM && c19 >= 0) {
    s19++;
    out19.push('### 第' + s19 + '节 ' + subM[2]);
    continue;
  }
  out19.push(line);
}
ch19 = out19.join('\n');

// Fix NOTEARS double-replacement
ch19 = ch19.replace(/NOTEARSRS/g, 'NOTEARS');

// Body cross-references
for (let i = 1; i <= 10; i++) {
  ch19 = ch19.replace(new RegExp('3\\.' + i + '节', 'g'), '第' + i + '节');
  ch19 = ch19.replace(new RegExp('第3\\.' + i + '节', 'g'), '第' + i + '节');
}
// Subsection refs: 3.A.B节 → 第B节
ch19 = ch19.replace(/3\.(\d+)\.(\d+)节/g, (m, sec, sub) => '第' + sub + '节');
// Range refs
ch19 = ch19.replace(/3\.(\d+)\.(\d+)-3\.(\d+)\.(\d+)节/g, (m, s1, sub1, s2, sub2) => {
  return s1 === s2 ? '第' + sub1 + '–' + '第' + sub2 + '节' : m;
});

fs.writeFileSync(path.join(base, 'part5/ch19_深度学习的理论难题.md'), ch19, 'utf-8');
console.log('ch19 fixed');

console.log('All fixes applied!');
