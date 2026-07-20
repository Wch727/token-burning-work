// Minimal test: single-line replacement
const fs = require('fs');
const path = require('path');

const base = 'C:/Users/22974/desktop/数模/token-burning-work/volumes';
let ch18 = fs.readFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), 'utf-8');

// Test: does the simple replace work on a known subsection?
const testLine = '### 6.2.1 图像块（Patch）的数学表示';
const m = testLine.match(/^### (6\.[0-9]+\.[0-9]+(?:\.[0-9]+)?) (.+)/);
console.log('Test match:', m ? m[1] + ' | ' + m[2] : 'NO MATCH');

// Now test the full replacement
let count = 0;
ch18 = ch18.replace(/^### (6\.[0-9]+\.[0-9]+(?:\.[0-9]+)?) (.+)/gm, (match, num, title) => {
  count++;
  return '### REPLACED ' + num + ' ' + title;
});
console.log('Replacements made:', count);

// Write to temp file for inspection
fs.writeFileSync(path.join(base, '..', 'ch18-test.md'), ch18, 'utf-8');
console.log('Written to ch18-test.md');
console.log('First 30 lines:');
ch18.split('\n').slice(0, 30).forEach((l, i) => console.log((i+1) + ': ' + l));
