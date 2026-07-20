// Fix subsection numbering in ch09, ch11, ch22
const fs = require('fs');
const path = require('path');
const BASE = 'C:\\Users\\22974\\desktop\\数模\\token-burning-work';
let total = 0;

function fixChapter(file, label) {
  const fp = path.join(BASE, file);
  let s = fs.readFileSync(fp, 'utf8');
  const lines = s.split('\n');
  let changed = 0;

  for (let i = 0; i < lines.length; i++) {
    // Match ### X.Y  format (e.g., ### 1.1, ### 10.5, ### 0.1)
    const m = lines[i].match(/^(###\s+)(\d+)\.(\d+)(\s+.*)/);
    if (m && !lines[i].includes('第')) {
      const prefix = m[1]; // "### "
      const x = m[2]; // X
      const y = m[3]; // Y
      const rest = m[4]; // rest of line
      lines[i] = `${prefix}第${x}.${y}节${rest}`;
      changed++;
    }
  }

  if (changed > 0) {
    fs.writeFileSync(fp, lines.join('\n'), 'utf8');
    console.log(`[FIXED] ${label}: ${changed} subsections renumbered`);
    total += changed;
  } else {
    console.log(`[SKIP] ${label}: no subsections to renumber`);
  }
}

fixChapter('volumes/part3/ch09_深度信念网络与预训练的革命.md', 'ch09');
fixChapter('volumes/part3/ch11_深度学习的关键理论突破.md', 'ch11');

// ch22: fix # 第X部分 → ## 第X节
{
  const fp = path.join(BASE, 'volumes/part6/ch22_计算机视觉精细化.md');
  let s = fs.readFileSync(fp, 'utf8');
  const lines = s.split('\n');
  let changed = 0;

  for (let i = 0; i < lines.length; i++) {
    // Match # 第X部分 → ## 第X节
    const m = lines[i].match(/^(#)\s+(第[一二三四五六七八九十]+部分.*)/);
    if (m) {
      const partNum = lines[i].match(/第([一二三四五六七八九十]+)部分/);
      if (partNum) {
        const numMap = {'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10};
        const n = numMap[partNum[1]] || partNum[1];
        const title = lines[i].replace(/^#\s+第[一二三四五六七八九十]+部分[：:]\s*/, '');
        lines[i] = `## 第${n}节 ${title}`;
        changed++;
      }
    }
    // Also fix # 第六部分：视频生成 style
    const m2 = lines[i].match(/^#\s+(第[一二三四五六七八九十]+部分[：:].*)/);
    if (m2 && !lines[i].includes('##')) {
      const partNum2 = lines[i].match(/第([一二三四五六七八九十]+)部分/);
      if (partNum2) {
        const numMap = {'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10};
        const n = numMap[partNum2[1]] || partNum2[1];
        const title = lines[i].replace(/^#\s+第[一二三四五六七八九十]+部分[：:]\s*/, '');
        lines[i] = `## 第${n}节 ${title}`;
        changed++;
      }
    }
  }

  if (changed > 0) {
    fs.writeFileSync(fp, lines.join('\n'), 'utf8');
    console.log(`[FIXED] ch22: ${changed} part headers → section headers`);
    total += changed;
  } else {
    console.log(`[SKIP] ch22: no part headers found`);
  }
}

console.log(`\nTotal fixes: ${total}`);
