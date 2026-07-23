// Fix R7 batch 3: ch03 KL year + English words + ch01 binary adder
const fs = require('fs');
const path = require('path');
const base = 'C:\\Users\\22974\\desktop\\数模\\token-burning-work';

// === ch03: KL year 1951->1950 ===
let s3 = fs.readFileSync(path.join(base, 'volumes/part1/ch03_统计学习思想的萌芽.md'), 'utf8');
let c3 = 0;
const oldKL = 'Kullback和Leibler在1951年提出的KL散度';
const newKL = 'Kullback和Leibler在1950年提出的KL散度';
if (s3.includes(oldKL)) { s3 = s3.split(oldKL).join(newKL); c3++; }

// English words in ch03
const en3 = [
  ['influences', '多种因素交汇'],
  ['motivation', '动机'],
];
for (const [old, neu] of en3) {
  if (s3.includes(old)) { s3 = s3.split(old).join(neu); c3++; }
}
fs.writeFileSync(path.join(base, 'volumes/part1/ch03_统计学习思想的萌芽.md'), s3);
console.log(`[FIXED] ch03: ${c3} fixes (KL year + English words)`);

// === ch01: Binary adder - add clarifying note ===
let s1 = fs.readFileSync(path.join(base, 'volumes/part1/ch01_图灵与AI的诞生.md'), 'utf8');
const oldNote = '计算过程示例：以输入磁带1#1011#110为例。以下展示前15步计算轨迹：';
const newNote = '计算过程示例：以输入磁带1#1011#110为例。以下展示前15步计算轨迹（为简化展示，步骤3-6中第一个操作数的进位处理已整合进q2/q3的状态转移中）：';
if (s1.includes(oldNote)) {
  s1 = s1.split(oldNote).join(newNote);
  fs.writeFileSync(path.join(base, 'volumes/part1/ch01_图灵与AI的诞生.md'), s1);
  console.log('[FIXED] ch01: Added clarifying note to binary adder trace');
} else {
  console.log('[SKIP]  ch01: binary adder note not found');
}
