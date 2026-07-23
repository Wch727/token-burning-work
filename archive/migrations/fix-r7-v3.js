// Fix R7 v2 remaining real issues
const fs = require('fs');
const path = require('path');
const BASE = 'C:\\Users\\22974\\desktop\\数模\\token-burning-work';
let total = 0;

function fix(file, oldStr, newStr, label) {
  const fp = path.join(BASE, file);
  let s = fs.readFileSync(fp, 'utf8');
  if (s.includes(oldStr)) {
    s = s.split(oldStr).join(newStr);
    fs.writeFileSync(fp, s, 'utf8');
    console.log(`[FIXED] ${label}`);
    total++;
  } else {
    console.log(`[SKIP] ${label} - not found`);
  }
}

// ===== ch10 line 71: "Ke普勒" → "Kepler" =====
fix('volumes/part3/ch10_GPU计算与大数据.md',
  'Ke普勒还引入了SMX',
  'Kepler还引入了SMX',
  'ch10: Ke普勒→Kepler');

// ===== ch10 line 117: "collectively" → "共同" =====
fix('volumes/part3/ch10_GPU计算与大数据.md',
  ' collectively 执行同一个内核函数',
  ' 共同执行同一个内核函数',
  'ch10: collectively→共同');

// ===== ch10 line 177: Python-style code → LaTeX =====
fix('volumes/part3/ch10_GPU计算与大数据.md',
  'C[i][j] = Σ(k=0 to K-1) A[i][k] × B[k][j]',
  '$$C_{ij} = \sum_{k=0}^{K-1} A_{ik} \cdot B_{kj}$$',
  'ch10: Python code→LaTeX formula');

// ===== ch04 line 120: "Faculty医师" → "主治医师" =====
fix('volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md',
  '与5名Faculty医师的评分',
  '与5名主治医师的评分',
  'ch04: Faculty→主治医师');

// ===== ch03 line 128: "感应层" → "视网膜" (terminology unification) =====
fix('volumes/part1/ch03_统计学习思想的萌芽.md',
  '作为"感应层"（retina）',
  '作为"视网膜"（retina）',
  'ch03: 感应层→视网膜');

console.log(`\nTotal fixes: ${total}`);
