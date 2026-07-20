// Fix remaining ch02 issues
const fs = require('fs');
const path = require('path');
const base = 'C:\\Users\\22974\\desktop\\数模\\token-burning-work';
const f = 'volumes/part1/ch02_感知机与连接主义的兴衰.md';
let s = fs.readFileSync(path.join(base, f), 'utf8');
let c = 0;

// Fix ## 8. and ## 9. section headers
const fixes = [
  ['## 8. ', '## 第8节 '],
  ['## 9. ', '## 第9节 '],
  ['### 7.2 ', '### 14.2 '],
  ['### 7.3 ', '### 14.3 '],
  ['### 7.4 ', '### 14.4 '],
];
for (const [old, neu] of fixes) {
  if (s.includes(old)) { s = s.split(old).join(neu); c++; }
}
fs.writeFileSync(path.join(base, f), s);
console.log(`[FIXED] ch02: ${c} remaining fixes applied`);
