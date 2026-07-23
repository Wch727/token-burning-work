// Debug ch18 subsection replacement
const fs = require('fs');
const path = require('path');

const base = 'C:/Users/22974/desktop/数模/token-burning-work/volumes';
let ch18 = fs.readFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), 'utf-8');
const lines = ch18.split('\n');

let matchCount = 0;
let noMatchCount = 0;
const noMatchLines = [];

for (let i = 0; i < lines.length; i++) {
  const line = lines[i];
  const subM = line.match(/^### (6\.[0-9]+\.[0-9]+(?:\.[0-9]+)?) (.+)/);
  if (subM) {
    matchCount++;
    console.log('MATCH line ' + (i+1) + ': ' + subM[1] + ' | ' + subM[2].substring(0, 30));
  } else if (line.startsWith('###')) {
    noMatchCount++;
    noMatchLines.push((i+1) + ': ' + line.substring(0, 60));
  }
}

console.log('\nTotal matches: ' + matchCount);
console.log('Total ### lines NOT matched: ' + noMatchCount);
if (noMatchLines.length > 0) {
  console.log('\nUnmatched ### lines:');
  noMatchLines.forEach(l => console.log('  ' + l));
}
