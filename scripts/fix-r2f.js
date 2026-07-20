// Final fix: ch18 subsection recovery + cross-refs, ch19 range refs
const fs = require('fs');
const path = require('path');

const base = 'C:/Users/22974/desktop/数模/token-burning-work/volumes';

// ============================================================
// ch18: Recover corrupted subsection headers + fix body refs
// ============================================================
let ch18 = fs.readFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), 'utf-8');

// Fix corrupted subsection headers: "### 第X节.Y" → "### 6.X.Y"
// The previous script's body-text replacements (6.2节 → 第2节) corrupted headers
// like "### 6.2.1" → "### 第2节.1"
const subRecover = [
  ['### 第1节.1', '### 6.2.1'], ['### 第1节.2', '### 6.2.2'], ['### 第1节.3', '### 6.2.3'],
  ['### 第1节.4', '### 6.2.4'], ['### 第1节.5', '### 6.2.5'], ['### 第1节.6', '### 6.2.6'],
  ['### 第2节.1', '### 6.3.1'], ['### 第2节.2', '### 6.3.2'],
  ['### 第3节.1', '### 6.4.1'], ['### 第3节.2', '### 6.4.2'], ['### 第3节.3', '### 6.4.3'],
  ['### 第4节.1', '### 6.5.1'], ['### 第4节.2', '### 6.5.2'],
  ['### 第5节.1', '### 6.6.1'], ['### 第5节.2', '### 6.6.2'], ['### 第5节.3', '### 6.6.3'],
  ['### 第5节.4', '### 6.6.4'], ['### 第5节.5', '### 6.6.5'],
  ['### 第6节.1', '### 6.7.1'], ['### 第6节.2', '### 6.7.2'], ['### 第6节.3', '### 6.7.3'],
  ['### 第6节.4', '### 6.7.4'], ['### 第6节.5', '### 6.7.5'],
  ['### 第7节.1', '### 6.8.1'], ['### 第7节.2', '### 6.8.2'], ['### 第7节.3', '### 6.8.3'],
  ['### 第8节.1', '### 6.9.1'], ['### 第8节.2', '### 6.9.2'], ['### 第8节.3', '### 6.9.3'],
];
for (const [wrong, correct] of subRecover) {
  ch18 = ch18.replaceAll(wrong, correct);
}

// Also recover 6.11.X → 6.9.X (evaluation subsections were wrongly renumbered)
// These are under section 6.9 图像生成评估指标
const evalRecover = [
  ['### 6.11.1', '### 6.9.1'], ['### 6.11.2', '### 6.9.2'], ['### 6.11.3', '### 6.9.3'],
  ['### 6.11.4', '### 6.9.4'], ['### 6.11.5', '### 6.9.5'],
  ['### 6.11.6', '### 6.10.1'], ['### 6.11.7', '### 6.10.2'],
];
for (const [wrong, correct] of evalRecover) {
  ch18 = ch18.replaceAll(wrong, correct);
}

// Verify recovery
const remainingCorrupt = ch18.match(/### 第[0-9]+节\.[0-9]+/g);
if (remainingCorrupt) {
  console.log('WARNING: Still corrupted:', remainingCorrupt.length, 'headers');
} else {
  console.log('ch18 subsection headers recovered');
}

// Now fix body cross-references with word-boundary aware patterns
// Use negative lookahead to avoid matching subsection numbers
// "6.2节" should NOT match "6.2.1" or "6.2.2"
// Pattern: 6.X节(?![0-9]) — section ref not followed by another digit
ch18 = ch18.replace(/6\.(\d)节(?!\d)/g, '第$1节');

// For two-digit sections: 6.10节
ch18 = ch18.replace(/6\.10节(?!\d)/g, '第10节');

// Fix subsection body refs: 6.2.1节 → 第1节 (within section 第2节)
ch18 = ch18.replace(/6\.2\.1节/g, '第1节');
ch18 = ch18.replace(/6\.2\.2节/g, '第2节');
ch18 = ch18.replace(/6\.2\.3节/g, '第3节');
ch18 = ch18.replace(/6\.2\.4节/g, '第4节');
ch18 = ch18.replace(/6\.2\.5节/g, '第5节');
ch18 = ch18.replace(/6\.2\.6节/g, '第6节');

// Fix "第6.X节" → "第X节" in body
ch18 = ch18.replace(/第6\.(\d)节(?!\d)/g, '第$1节');
ch18 = ch18.replace(/第6\.10节(?!\d)/g, '第10节');

fs.writeFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), ch18, 'utf-8');
console.log('ch18 body cross-refs fixed');

// ============================================================
// ch19: Fix range refs that weren't matched
// ============================================================
let ch19 = fs.readFileSync(path.join(base, 'part5/ch19_深度学习的理论难题.md'), 'utf-8');

// Fix the range ref at line 715: "3.7.2-3.7.4节" → "第2–第4节"
// The previous range regex didn't match because it required spaces around "-"
// Also need to handle the case where first half was already partially replaced
// Current state: "3.7.2-第4节" (from earlier replacement of "3.7.4节")
ch19 = ch19.replace(/3\.7\.2-第4节/g, '第2–第4节');

// Verify
const test = ch19.match(/3\.7\.2-3\.7\.4节/g);
console.log('Remaining 3.7.2-3.7.4 refs:', test ? test.length : 0);

fs.writeFileSync(path.join(base, 'part5/ch19_深度学习的理论难题.md'), ch19, 'utf-8');
console.log('ch19 range ref fixed');

console.log('All final fixes applied!');
