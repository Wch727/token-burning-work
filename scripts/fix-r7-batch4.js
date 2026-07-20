// Fix R7 batch 4: ch23 \text{} -> \operatorname{} + ch24 formula fixes
const fs = require('fs');
const path = require('path');
const base = 'C:\\Users\\22974\\desktop\\数模\\token-burning-work';

// === ch23: \text{diag} -> \operatorname{diag} (all occurrences) ===
let s23 = fs.readFileSync(path.join(base, 'volumes/part6/ch23_图神经网络与结构数据.md'), 'utf8');
let c23 = 0;
// Use indexOf to find all \text{diag} occurrences
let idx = 0;
while ((idx = s23.indexOf(String.fromCharCode(92) + 'text{diag}', idx)) !== -1) {
  c23++;
  s23 = s23.substring(0, idx) + String.fromCharCode(92) + 'operatorname{diag}' + s23.substring(idx + String.fromCharCode(92).length + 9);
}
fs.writeFileSync(path.join(base, 'volumes/part6/ch23_图神经网络与结构数据.md'), s23);
console.log(`[FIXED] ch23: ${c23} \\text{diag} -> \\operatorname{diag}`);

// === ch24: \text{KL} -> \operatorname{KL} ===
let s24 = fs.readFileSync(path.join(base, 'volumes/part6/ch24_AI安全对齐与伦理.md'), 'utf8');
let c24 = 0;
idx = 0;
while ((idx = s24.indexOf(String.fromCharCode(92) + 'text{KL}', idx)) !== -1) {
  c24++;
  s24 = s24.substring(0, idx) + String.fromCharCode(92) + 'operatorname{KL}' + s24.substring(idx + String.fromCharCode(92).length + 7);
}
fs.writeFileSync(path.join(base, 'volumes/part6/ch24_AI安全对齐与伦理.md'), s24);
console.log(`[FIXED] ch24: ${c24} \\text{KL} -> \\operatorname{KL}`);
