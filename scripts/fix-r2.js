// Fix ch18 section numbering and ch19 section numbering + NOTEA typo
const fs = require('fs');
const path = require('path');

const base = 'C:/Users/22974/desktop/数模/token-burning-work/volumes';

// ============ ch18 fixes ============
let ch18 = fs.readFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), 'utf-8');

// Fix duplicate subsection numbering:
// Line 73: "### 6.2.2 位置编码" should be "### 6.2.3"
// Line 96: "### 6.2.3 纯Transformer" should be "### 6.2.4"
ch18 = ch18.replace(
  /^### 6\.2\.2 位置编码的设计与理论意义/,
  '### 6.2.3 位置编码的设计与理论意义'
);
ch18 = ch18.replace(
  /^### 6\.2\.3 纯Transformer在CV上的可行性证明/,
  '### 6.2.4 纯Transformer在CV上的可行性证明'
);
// Line 130: "### 6.2.4" → "### 6.2.5"
ch18 = ch18.replace(
  /^### 6\.2\.4 ViT与CNN的系统性对比分析/,
  '### 6.2.5 ViT与CNN的系统性对比分析'
);

// Now fix all subsections from 6.2.5 onward (6.2.5 → 6.2.6, 6.3→6.4, ..., 6.10→6.11)
// Pattern: ### 6.N.M ... → increment N by 1 (for N >= 3)
// Actually, let me be more precise. The sections that need incrementing:
// 6.2.5, 6.2.6, 6.3.x, 6.4.x, 6.5.x, 6.6.x, 6.7.x, 6.8.x, 6.9.x, 6.10
// All sections from 6.2.5 onward need their major version incremented by 1

const lines = ch18.split('\n');
const result = [];
for (const line of lines) {
  // Match subsection headers: ### 6.X.Y or ### 6.X.Y.Z
  const subMatch = line.match(/^### (6\.(?:[3-9]|10)(?:\.[0-9]+)+)\s(.+)/);
  if (subMatch) {
    const num = subMatch[1]; // e.g., "6.3.1"
    const parts = num.split('.');
    parts[1] = String(parseInt(parts[1]) + 1);
    result.push('### ' + parts.join('.') + ' ' + subMatch[2]);
    continue;
  }
  // Match section headers: ## 6.X
  const secMatch = line.match(/^## (6\.[0-9]+)\s(.+)/);
  if (secMatch) {
    const num = secMatch[1]; // e.g., "6.2"
    const parts = num.split('.');
    parts[1] = String(parseInt(parts[1]) + 1);
    result.push('## 第' + parts[1] + '节 ' + secMatch[2]);
    continue;
  }
  result.push(line);
}
ch18 = result.join('\n');

fs.writeFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), ch18, 'utf-8');
console.log('ch18 fixed');

// ============ ch19 fixes ============
let ch19 = fs.readFileSync(path.join(base, 'part6/ch23_图神经网络与结构数据.md'), 'utf-8');
// Wait, that's ch23. Let me check the correct path.

// Actually let me re-read the ch19 file path
// from the script: ch19 = 'volumes/part5/ch19_深度学习的理论难题.md'
ch19 = fs.readFileSync(path.join(base, 'part5/ch19_深度学习的理论难题.md'), 'utf-8');

const lines19 = ch19.split('\n');
const result19 = [];
for (const line of lines19) {
  // Match section headers: ## 3.X → ## 第X节
  const secMatch = line.match(/^## (3\.[0-9]+)\s(.+)/);
  if (secMatch) {
    const num = secMatch[1];
    const parts = num.split('.');
    result19.push('## 第' + parts[1] + '节 ' + secMatch[2]);
    continue;
  }
  // Fix NOTEA → NOTEARS
  if (line.includes('NOTEA')) {
    result19.push(line.replace(/NOTEA/g, 'NOTEARS'));
    continue;
  }
  result19.push(line);
}
ch19 = result19.join('\n');
fs.writeFileSync(path.join(base, 'part5/ch19_深度学习的理论难题.md'), ch19, 'utf-8');
console.log('ch19 fixed');

// ============ ch22 fixes ============
let ch22 = fs.readFileSync(path.join(base, 'part6/ch22_计算机视觉精细化.md'), 'utf-8');

// Fix Mosaic attribution: move from YOLOv4 to YOLOv5
// Current YOLOv4 text includes Mosaic, YOLOv5 text doesn't mention Mosaic
// Move Mosaic description from YOLOv4 to YOLOv5

// First, remove Mosaic from YOLOv4 description
ch22 = ch22.replace(
  /（自对抗训练SAT的组合）/,
  '（自对抗训练SAT）等训练技巧'
);

// Add Mosaic to YOLOv5 description
ch22 = ch22.replace(
  /YOLOv5引入了Mosaic数据增强（每次将4张图像拼接为一张）以丰富训练样本的多样性，并在YOLOv4的基础上进一步优化了工程架构/,
  'YOLOv5在YOLOv4的基础上进一步优化了工程架构'
);
// Add Mosaic mention before Focus layer
ch22 = ch22.replace(
  /采用Focus层进行初始下采样/,
  '引入了Mosaic数据增强（每次将4张图像拼接为一张）以丰富训练样本的多样性，并采用Focus层进行初始下采样'
);

fs.writeFileSync(path.join(base, 'part6/ch22_计算机视觉精细化.md'), ch22, 'utf-8');
console.log('ch22 fixed');

console.log('All Round 2 fixes applied!');
