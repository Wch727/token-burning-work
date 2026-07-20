// Fix R7 batch 6: ch24 formulas wrapped in $$ + ch03/ch04 English words
const fs = require('fs');
const path = require('path');
const base = 'C:\\Users\\22974\\desktop\\数模\\token-burning-work';

// === ch24: Wrap alignment loss formulas in $$ ===
let s24 = fs.readFileSync(path.join(base, 'volumes/part6/ch24_AI安全对齐与伦理.md'), 'utf8');
let c24 = 0;

// Fix 1: alignment loss formula (line 27) - currently: \min_{\pi} \mathbb{E}... without $$
const alignOld = String.fromCharCode(92) + 'min_{\pi} ' + String.fromCharCode(92) + 'mathbb{E}_{' + String.fromCharCode(92) + 'tau ' + String.fromCharCode(92) + 'sim ' + String.fromCharCode(92) + 'pi} [ (U_H(' + String.fromCharCode(92) + 'tau) - U_{AI}(' + String.fromCharCode(92) + 'tau))^2 ]';
// Simpler approach: find the line and check if it starts with $ without $$
const lines24 = s24.split('\n');
let modified = false;
for (let i = 0; i < lines24.length; i++) {
  const line = lines24[i];
  // Find lines with LaTeX formulas not in $$ blocks
  if (line.includes(String.fromCharCode(92) + 'min_{\pi}') && !line.trim().startsWith('$$')) {
    const trimmed = line.trim();
    if (trimmed.startsWith('$') && !trimmed.startsWith('$$')) {
      // Single $ line - wrap in $$
      lines24[i] = '$$' + trimmed.substring(1) + '$$';
      c24++;
      modified = true;
    }
  }
}
if (modified) {
  s24 = lines24.join('\n');
}

// Fix 2: Alignment Gap formula
const gapOld = String.fromCharCode(92) + 'text{Alignment Gap} = ' + String.fromCharCode(92) + 'mathbb{E}_{s ' + String.fromCharCode(92) + 'sim D_{' + String.fromCharCode(92) + 'text{test}}} [ ' + String.fromCharCode(92) + 'max_a U_H(s,a) - ' + String.fromCharCode(92) + 'max_a U_{AI}(s,a) ]';
const gapIdx = s24.indexOf(String.fromCharCode(92) + 'text{Alignment Gap}');
if (gapIdx !== -1) {
  // Check if already in $$
  const before = s24.substring(Math.max(0, gapIdx - 5), gapIdx);
  if (!before.includes('$$')) {
    // Find the line start
    const lineStart = s24.lastIndexOf('\n', gapIdx) + 1;
    const lineEnd = s24.indexOf('\n', gapIdx);
    const line = s24.substring(lineStart, lineEnd !== -1 ? lineEnd : s24.length);
    const trimmed = line.trim();
    if (trimmed.startsWith('$') && !trimmed.startsWith('$$')) {
      s24 = s24.substring(0, lineStart) + '$$' + trimmed.substring(1) + '$$' + s24.substring(lineEnd !== -1 ? lineEnd : s24.length);
      c24++;
    }
  }
}

fs.writeFileSync(path.join(base, 'volumes/part6/ch24_AI安全对齐与伦理.md'), s24);
console.log(`[FIXED] ch24: ${c24} formulas wrapped in $$`);

// === ch03: remaining English words ===
let s3 = fs.readFileSync(path.join(base, 'volumes/part1/ch03_统计学习思想的萌芽.md'), 'utf8');
let c3 = 0;
const en3 = [['influences', '多种因素交汇'], ['motivation', '动机']];
for (const [old, neu] of en3) {
  if (s3.includes(old)) { s3 = s3.split(old).join(neu); c3++; }
}
fs.writeFileSync(path.join(base, 'volumes/part1/ch03_统计学习思想的萌芽.md'), s3);
if (c3 > 0) console.log(`[FIXED] ch03: ${c3} English words translated`);

// === ch04: "enthusiasm" ===
let s4 = fs.readFileSync(path.join(base, 'volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md'), 'utf8');
if (s4.includes('enthusiasm')) {
  s4 = s4.split('enthusiasm').join('热情');
  fs.writeFileSync(path.join(base, 'volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md'), s4);
  console.log('[FIXED] ch04: enthusiasm -> 热情');
} else {
  console.log('[SKIP]  ch04: enthusiasm not found');
}
