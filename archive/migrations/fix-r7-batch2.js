// Fix R7 batch 2: ch10 Fermi FP64 facts + ch11 section numbering
const fs = require('fs');
const path = require('path');
const base = 'C:\\Users\\22974\\desktop\\数模\\token-burning-work';

// === ch10: Fermi FP64/FP32 ratio ===
let s = fs.readFileSync(path.join(base, 'volumes/part3/ch10_GPU计算与大数据.md'), 'utf8');
const oldLine = 'Fermi将双精度（FP64）性能提升到了单精度（FP32）的1/2（此前为1/3到1/5），对于科学计算和深度学习训练（尤其是需要高精度梯度的反向传播）具有重要意义。';
const newLine = 'Fermi将双精度（FP64）性能提升到了单精度（FP32）的1/8（此前G80约为1/8，GT200约为1/12），对于科学计算和深度学习训练（尤其是需要高精度梯度的反向传播）具有重要意义。';
if (s.includes(oldLine)) {
  s = s.split(oldLine).join(newLine);
  fs.writeFileSync(path.join(base, 'volumes/part3/ch10_GPU计算与大数据.md'), s);
  console.log('[FIXED] ch10: Fermi FP64/FP32 ratio (1/2->1/8, pre-Fermi corrected)');
} else {
  console.log('[SKIP]  ch10: Fermi FP64 text not found');
}

// === ch11: Section numbering ===
let s11 = fs.readFileSync(path.join(base, 'volumes/part3/ch11_深度学习的关键理论突破.md'), 'utf8');
let c11 = 0;
const r11 = [
  ['## 0. ', '## 第1节 '], ['## 1. ', '## 第2节 '],
  ['## 2. ', '## 第3节 '], ['## 3. ', '## 第4节 '],
  ['## 4. ', '## 第5节 '], ['## 5. ', '## 第6节 '],
  ['## 6. ', '## 第7节 '], ['## 7. ', '## 第8节 '],
  ['## 8. ', '## 第9节 '],
];
for (const [old, neu] of r11) {
  if (s11.includes(old)) { s11 = s11.split(old).join(neu); c11++; }
}
fs.writeFileSync(path.join(base, 'volumes/part3/ch11_深度学习的关键理论突破.md'), s11);
console.log(`[FIXED] ch11: ${c11} section renumberings`);
