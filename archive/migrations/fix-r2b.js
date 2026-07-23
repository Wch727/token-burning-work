// Fix ch18 subsection numbering (6.10.x → 6.11.x) and ch19 issues
const fs = require('fs');
const path = require('path');

const base = 'C:/Users/22974/desktop/数模/token-burning-work/volumes';

// ============ ch18: fix 6.10.x subsections → 6.11.x ============
let ch18 = fs.readFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), 'utf-8');
const lines18 = ch18.split('\n');
const result18 = [];
for (const line of lines18) {
  // Match subsection headers: ### 6.10.X
  const subMatch = line.match(/^### (6\.10\.[0-9]+)\s(.+)/);
  if (subMatch) {
    result18.push('### 6.11.' + subMatch[1].split('.')[2] + ' ' + subMatch[2]);
    continue;
  }
  result18.push(line);
}
ch18 = result18.join('\n');
fs.writeFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), ch18, 'utf-8');
console.log('ch18 subsection 6.10.x → 6.11.x fixed');

// ============ ch19: fix NOTEARSRS → NOTEARS ============
let ch19 = fs.readFileSync(path.join(base, 'part5/ch19_深度学习的理论难题.md'), 'utf-8');
ch19 = ch19.replace(/NOTEARSRS/g, 'NOTEARS');
fs.writeFileSync(path.join(base, 'part5/ch19_深度学习的理论难题.md'), ch19, 'utf-8');
console.log('ch19 NOTEARSRS → NOTEARS fixed');

// ============ ch19: fix subsection numbering ============
// Subsections like ### 3.2.1 should become ### 1 (strip parent section prefix)
ch19 = fs.readFileSync(path.join(base, 'part5/ch19_深度学习的理论难题.md'), 'utf-8');
const lines19 = ch19.split('\n');
const result19 = [];
for (const line of lines19) {
  // Match subsection headers: ### 3.X.Y or ### 3.X.Y.Z
  const subMatch = line.match(/^### (3\.[0-9]+\.[0-9]+(?:\.[0-9]+)?)\s(.+)/);
  if (subMatch) {
    const parts = subMatch[1].split('.');
    // Strip the "3." prefix, keep only the local subsection number
    // e.g., 3.2.1 → 1, 3.2.1.1 → 1.1
    const localNum = parts.slice(2).join('.');
    result19.push('### ' + localNum + ' ' + subMatch[2]);
    continue;
  }
  result19.push(line);
}
ch19 = result19.join('\n');
fs.writeFileSync(path.join(base, 'part5/ch19_深度学习的理论难题.md'), ch19, 'utf-8');
console.log('ch19 subsection numbering fixed');

console.log('All additional fixes applied!');
