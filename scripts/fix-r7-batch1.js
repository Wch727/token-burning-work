// Fix R7 batch 1: ch02 subsection numbering + ch04 subsection numbering
const fs = require('fs');
const path = require('path');
const base = 'C:\\Users\\22974\\desktop\\数模\\token-burning-work';

// === ch02: Renumber all 6.x.y subsections ===
let s = fs.readFileSync(path.join(base, 'volumes/part1/ch02_感知机与连接主义的兴衰.md'), 'utf8');
let changes = 0;

const renum = [
  ['### 6.6.1 ', '### 8.1 '], ['### 6.6.2 ', '### 8.2 '],
  ['### 6.6.3 ', '### 8.3 '], ['### 6.6.4 ', '### 8.4 '],
  ['### 6.6.5 ', '### 8.5 '], ['### 6.6.6 ', '### 8.6 '],
  ['### 6.7.1 ', '### 9.1 '], ['### 6.7.2 ', '### 9.2 '],
  ['### 6.7.3 ', '### 9.3 '],
  ['### 6.8.1 ', '### 10.1 '], ['### 6.8.2 ', '### 10.2 '],
  ['### 6.8.3 ', '### 10.3 '], ['### 6.8.4 ', '### 10.4 '],
  ['### 6.9.1 ', '### 11.1 '], ['### 6.9.2 ', '### 11.2 '],
  ['### 6.9.3 ', '### 11.3 '], ['### 6.9.4 ', '### 11.4 '],
  ['### 6.9.5 ', '### 11.5 '],
  ['### 6.10.1 ', '### 12.1 '], ['### 6.10.2 ', '### 12.2 '],
  ['### 6.10.3 ', '### 12.3 '],
  ['### 6.11.1 ', '### 13.1 '], ['### 6.11.2 ', '### 13.2 '],
  ['### 6.11.3 ', '### 13.3 '], ['### 6.11.4 ', '### 13.4 '],
  ['### 6.12 ', '### 14.1 '],
];
for (const [old, neu] of renum) {
  if (s.includes(old)) { s = s.split(old).join(neu); changes++; }
}
fs.writeFileSync(path.join(base, 'volumes/part1/ch02_感知机与连接主义的兴衰.md'), s);
console.log(`[FIXED] ch02: ${changes} subsection renumberings`);

// === ch04: Fix subsections under section 4, 6 ===
let s4 = fs.readFileSync(path.join(base, 'volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md'), 'utf8');
let c4 = 0;
const c4r = [
  ['### 第1节 英国：从繁荣到凋零', '### 第4节 英国：从繁荣到凋零'],
  ['### 第2节 美国：曼斯菲尔德修正案与DARPA的战略转向', '### 第5节 美国：曼斯菲尔德修正案与DARPA的战略转向'],
  ['### 第3节 感知机的"另一场冬天"', '### 第6节 感知机的"另一场冬天"'],
];
for (const [old, neu] of c4r) {
  if (s4.includes(old)) { s4 = s4.split(old).join(neu); c4++; }
}
fs.writeFileSync(path.join(base, 'volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md'), s4);
console.log(`[FIXED] ch04: ${c4} subsection renumberings`);
