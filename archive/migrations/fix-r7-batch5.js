// Fix R7 batch 5: ch21 duplicate formula + ch22 Focal Loss duplicate derivation
const fs = require('fs');
const path = require('path');
const base = 'C:\\Users\\22974\\desktop\\数模\\token-burning-work';

// === ch21: Remove duplicate "展开后" formula ===
let s21 = fs.readFileSync(path.join(base, 'volumes/part6/ch21_NLP精细化.md'), 'utf8');
const marker21 = '展开后：';
const idx21 = s21.indexOf(marker21);
if (idx21 !== -1) {
  // Find the $$ block after "展开后："
  const dollarIdx = s21.indexOf('$$', idx21);
  if (dollarIdx !== -1) {
    const endDollar = s21.indexOf('$$', dollarIdx + 2);
    if (endDollar !== -1) {
      // Remove from "展开后：" through the closing $$
      const removeStart = s21.lastIndexOf('\n', idx21);
      const removeEnd = endDollar + 2;
      s21 = s21.substring(0, removeStart) + s21.substring(removeEnd);
      fs.writeFileSync(path.join(base, 'volumes/part6/ch21_NLP精细化.md'), s21);
      console.log('[FIXED] ch21: Removed duplicate "展开后" formula block');
    } else {
      console.log('[SKIP]  ch21: closing $$ not found');
    }
  } else {
    console.log('[SKIP]  ch21: $$ not found after marker');
  }
} else {
  console.log('[SKIP]  ch21: "展开后" marker not found');
}

// === ch22: Remove duplicate Focal Loss gradient derivation ===
let s22 = fs.readFileSync(path.join(base, 'volumes/part6/ch22_计算机视觉精细化.md'), 'utf8');
const dupStart = '根据链式法则，Focal Loss对logit z的梯度为：';
const dupEnd = '这一梯度表达式揭示了Focal Loss的关键特性：';
const si = s22.indexOf(dupStart);
const ei = s22.indexOf(dupEnd);
if (si !== -1 && ei !== -1) {
  // Remove from "根据链式法则" paragraph start through the formula before "这一梯度表达式"
  const paraStart = s22.lastIndexOf('\n', si);
  s22 = s22.substring(0, paraStart) + '\n\n' + s22.substring(ei);
  fs.writeFileSync(path.join(base, 'volumes/part6/ch22_计算机视觉精细化.md'), s22);
  console.log('[FIXED] ch22: Removed duplicate gradient derivation (chain rule paragraph)');
} else {
  console.log('[SKIP]  ch22: duplicate derivation markers not found (si=' + si + ', ei=' + ei + ')');
}
