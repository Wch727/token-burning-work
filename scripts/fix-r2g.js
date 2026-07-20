// Fix ch18 subsections and body cross-references
const fs = require('fs');
const path = require('path');

const base = 'C:/Users/22974/desktop/数模/token-burning-work/volumes';
let ch18 = fs.readFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), 'utf-8');

// 1. Fix subsection headers: direct old→new mapping
// Section 第2节 (was 6.2) - 6 subsections
ch18 = ch18.replaceAll('### 6.2.1 ', '### 第1节 ');
ch18 = ch18.replaceAll('### 6.2.2 ', '### 第2节 ');
ch18 = ch18.replaceAll('### 6.2.3 ', '### 第3节 ');
ch18 = ch18.replaceAll('### 6.2.4 ', '### 第4节 ');
ch18 = ch18.replaceAll('### 6.2.5 ', '### 第5节 ');
ch18 = ch18.replaceAll('### 6.2.6 ', '### 第6节 ');

// Section 第3节 (was 6.3) - 2 subsections
ch18 = ch18.replaceAll('### 6.3.1 ', '### 第1节 ');
ch18 = ch18.replaceAll('### 6.3.2 ', '### 第2节 ');

// Section 第4节 (was 6.4) - 3 subsections
ch18 = ch18.replaceAll('### 6.4.1 ', '### 第1节 ');
ch18 = ch18.replaceAll('### 6.4.2 ', '### 第2节 ');
ch18 = ch18.replaceAll('### 6.4.3 ', '### 第3节 ');

// Section 第5节 (was 6.5) - 3 subsections (added 6.5.3)
ch18 = ch18.replaceAll('### 6.5.1 ', '### 第1节 ');
ch18 = ch18.replaceAll('### 6.5.2 ', '### 第2节 ');
ch18 = ch18.replaceAll('### 6.5.3 ', '### 第3节 ');

// Section 第6节 (was 6.6) - 5 subsections
ch18 = ch18.replaceAll('### 6.6.1 ', '### 第1节 ');
ch18 = ch18.replaceAll('### 6.6.2 ', '### 第2节 ');
ch18 = ch18.replaceAll('### 6.6.3 ', '### 第3节 ');
ch18 = ch18.replaceAll('### 6.6.4 ', '### 第4节 ');
ch18 = ch18.replaceAll('### 6.6.5 ', '### 第5节 ');

// Section 第7节 (was 6.7) - 5 subsections
ch18 = ch18.replaceAll('### 6.7.1 ', '### 第1节 ');
ch18 = ch18.replaceAll('### 6.7.2 ', '### 第2节 ');
ch18 = ch18.replaceAll('### 6.7.3 ', '### 第3节 ');
ch18 = ch18.replaceAll('### 6.7.4 ', '### 第4节 ');
ch18 = ch18.replaceAll('### 6.7.5 ', '### 第5节 ');

// Section 第8节 (was 6.8) - 5 subsections
ch18 = ch18.replaceAll('### 6.8.1 ', '### 第1节 ');
ch18 = ch18.replaceAll('### 6.8.2 ', '### 第2节 ');
ch18 = ch18.replaceAll('### 6.8.3 ', '### 第3节 ');
ch18 = ch18.replaceAll('### 6.8.4 ', '### 第4节 ');
ch18 = ch18.replaceAll('### 6.8.5 ', '### 第5节 ');

// Section 第9节 (was 6.9) - 7 subsections (evaluation metrics, originally 6.9.X AND 6.10.X)
ch18 = ch18.replaceAll('### 6.9.1 ', '### 第1节 ');
ch18 = ch18.replaceAll('### 6.9.2 ', '### 第2节 ');
ch18 = ch18.replaceAll('### 6.9.3 ', '### 第3节 ');
ch18 = ch18.replaceAll('### 6.9.4 ', '### 第4节 ');
ch18 = ch18.replaceAll('### 6.9.5 ', '### 第5节 ');
// Evaluation subsections that were wrongly numbered 6.11.X or 6.10.X
ch18 = ch18.replaceAll('### 6.10.1 ', '### 第6节 ');
ch18 = ch18.replaceAll('### 6.10.2 ', '### 第7节 ');

// 2. Fix section header for 总结与展望
ch18 = ch18.replaceAll('## 6.10 ', '## 第10节 ');

// 3. Fix body cross-references using word-boundary-aware patterns
// Section refs: 6.X节 → 第X节 (avoid matching subsection numbers like 6.2.1)
// Use negative lookahead: 6.X节 NOT followed by a digit
ch18 = ch18.replace(/6\.(\d)节(?!\d)/g, '第$1节');
ch18 = ch18.replace(/6\.10节(?!\d)/g, '第10节');

// Subsection refs for 6.2.X (most commonly referenced)
ch18 = ch18.replace(/6\.2\.1节/g, '第1节');
ch18 = ch18.replace(/6\.2\.2节/g, '第2节');

// Fix "第6.X节" → "第X节" in body
ch18 = ch18.replace(/第6\.(\d)节(?!\d)/g, '第$1节');
ch18 = ch18.replace(/第6\.10节(?!\d)/g, '第10节');

// Verify no remaining old-format subsection headers
const remaining = ch18.match(/^### 6\.[0-9]+\.[0-9]+ /gm);
if (remaining) {
  console.log('WARNING: Remaining old-format subsections:', remaining.length);
  remaining.forEach(l => console.log('  ' + l));
} else {
  console.log('All ch18 subsections converted');
}

fs.writeFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), ch18, 'utf-8');
console.log('ch18 saved');
