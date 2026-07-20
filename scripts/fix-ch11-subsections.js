// Fix ch11 subsection numbering to be hierarchical
const fs = require('fs');
const path = require('path');
const BASE = 'C:\\Users\\22974\\desktop\\数模\\token-burning-work';

const fp = path.join(BASE, 'volumes/part3/ch11_深度学习的关键理论突破.md');
let s = fs.readFileSync(fp, 'utf8');
const lines = s.split('\n');
let changed = 0;
let currentSection = null;

for (let i = 0; i < lines.length; i++) {
  // Track current section
  const secMatch = lines[i].match(/^##\s+第([0-9]+)节\s/);
  if (secMatch) {
    currentSection = parseInt(secMatch[1]);
    continue;
  }

  // Check subsection numbering
  const subMatch = lines[i].match(/^###\s+第([0-9]+)\.([0-9]+)节\s(.*)/);
  if (subMatch && currentSection !== null) {
    const subSec = parseInt(subMatch[1]);
    const subSub = parseInt(subMatch[2]);
    const title = subMatch[3];

    // Skip 第0.x under 第1节 (intro subsections)
    if (subSec === 0 && currentSection === 1) continue;

    if (subSec !== currentSection) {
      // Renumber to hierarchical: ### 第X.Y节 title
      lines[i] = `### 第${currentSection}.${subSub}节 ${title}`;
      changed++;
    }
  }
}

if (changed > 0) {
  fs.writeFileSync(fp, lines.join('\n'), 'utf8');
  console.log(`[FIXED] ch11: ${changed} subsections renumbered to hierarchical`);
} else {
  console.log('[SKIP] ch11: no mismatched subsections');
}
