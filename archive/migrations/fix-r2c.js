// Comprehensive fix for ch18 and ch19:
// - ch18: Renumber all subsections to 第X节 format, fix cross-references
// - ch19: Renumber subsections, fix cross-references, fix NOTEARS typo
const fs = require('fs');
const path = require('path');

const base = 'C:/Users/22974/desktop/数模/token-burning-work/volumes';

// ============================================================
// ch18: Complete subsection renumbering
// ============================================================
// Old structure:
//   6.1 → 第1节
//   6.2 has 6 subsections (6.2.1 to 6.2.6)
//   6.3 → 第2节 (1 subsection)
//   6.4 → 第3节 (3 subsections)
//   6.5 → 第4节 (2 subsections)
//   6.6 → 第5节 (5 subsections)
//   6.7 → 第6节 (5 subsections)
//   6.8 → 第7节 (5 subsections)
//   6.9 → 第8节 (3 subsections)
//   6.10 → 第9节 (no subsections of its own)
//   6.10.1 to 6.10.7 → evaluation subsections → 第9节's subsections
//
// Wait, let me re-examine the actual structure more carefully.
// From the grep output:
//   Line 3: ## 6.1 → ## 第1节
//   Line 21: ## 6.2 → ## 第2节
//     Line 23: ### 6.2.1 图像块
//     Line 47: ### 6.2.2 多头自注意力
//     Line 73: ### 6.2.3 位置编码 (was 6.2.2, fixed to 6.2.3)
//     Line 96: ### 6.2.4 纯Transformer (was 6.2.3, fixed to 6.2.4)
//     Line 130: ### 6.2.5 自注意力可视化 (was 6.2.4, fixed to 6.2.5)
//     Line 140: ### 6.2.6 ViT与CNN对比 (was 6.2.5, fixed to 6.2.6)
//   Line 170: ## 6.3 DeiT → ## 第3节
//     Line 172: ### 6.3.1 训练策略
//     Line 186: ### 6.3.2 知识蒸馏
//   Line 208: ## 6.4 Swin → ## 第4节
//     Line 210: ### 6.4.1 层次化特征图
//     Line 231: ### 6.4.2 移位窗口注意力
//     Line 257: ### 6.4.3 多尺度建模
//   Line 273: ## 6.5 MAE → ## 第5节
//     Line 275: ### 6.5.1 编码器-解码器
//     Line 305: ### 6.5.2 高比例掩码策略
//   Line 325: ## 6.6 扩散模型 → ## 第6节
//     Line 329: ### 6.6.1 前向扩散过程
//     Line 370: ### 6.6.2 反向去噪过程
//     Line 440: ### 6.6.3 DDIM
//     Line 456: ### 6.6.4 条件生成
//     Line 492: ### 6.6.5 Score Matching与SDE
//   Line 540: ## 6.7 DALL-E → ## 第7节
//     Line 540: ### 6.7.1 DALL-E系列
//     Line 558: ### 6.7.2 Stable Diffusion
//     Line 612: ### 6.7.3 Midjourney
//     Line 628: ### 6.7.4 Google Imagen
//     Line 646: ### 6.7.5 ControlNet
//   Line 672: ## 6.8 视频理解 → ## 第8节
//     Line 676: ### 6.8.1 TimeSformer
//     Line 710: ### 6.8.2 VideoMAE
//     Line 730: ### 6.8.3 Sora
//   Line 748: ## 6.9 图像生成评估 → ## 第9节
//     Line 752: ### 6.11.1 IS (was 6.10.1, fixed to 6.11.1)
//     Line 775: ### 6.11.2 FID
//     Line 797: ### 6.11.3 Precision
//     Line 819: ### 6.11.4 sFID
//     Line 831: ### 6.11.5 CLIP Score
//     Line 866: ### 6.11.6 生成模型的系统比较 (was 6.10.6)
//     Line 884: ### 6.11.7 视频生成模型的评估 (was 6.10.7)
//   Line ???: ## 6.10 总结与展望 → ## 第10节
//
// Wait, the evaluation metrics were under ## 6.9, not ## 6.10.
// Let me re-examine:
// Line 748: ## 6.9 图像生成评估指标 → ## 第8节
// Its subsections (IS, FID, etc.) have numbering 6.11.X
// This is wrong! They should be under 第8节, numbered 第1节 to 第7节.
//
// And section 6.10 总结与展望 → ## 第9节
//
// Wait no. Let me re-trace:
// After cascading:
// 6.1 → 第1节
// 6.2 → 第2节 (6 subsections)
// 6.3 → 第3节 (2 subsections)
// 6.4 → 第4节 (3 subsections)
// 5 → 第5节 (2)
// 6 → 第6节 (5)
// 7 → 第7节 (5)
// 8 → 第8节 (3)
// 9 → 第9节 (7 evaluation subsections)
// 10 → 第10节 (总结)
//
// So the evaluation sections were originally 6.10.X but should now be 6.9.X (section 6.9 with 7 subsections)
// And 6.9 (Sora) → becomes section 第8节 with 3 subsections
// 6.8 (视频理解) → becomes section 第7节 with 3 subsections
//
// Let me re-trace from the grep output more carefully:
// The subsections under ## 6.9:
//   ### 6.9.1 TimeSformer (line 676)
//   ### 6.9.2 VideoMAE (line 710)
//   ### 6.9.3 Sora (line 730)
// These should become under 第7节: 第1节, 第2节, 第3节
//
// The evaluation sections were originally:
//   ## 6.10 总结与展望? No wait...
//
// Actually wait. Let me look at the structure again from the grep:
// Line 672: ## 6.8 视频理解
// Line 676: ### 6.8.1 TimeSformer
// Line 710: ### 6.8.2 VideoMAE
// Line 730: ### 6.8.3 Sora
// Line 748: ## 6.9 图像生成评估指标
// Line 752: ### 6.11.1 IS
//
// So the evaluation sections (IS, FID, etc.) are under ## 6.9!
// After cascading, section 6.9 becomes 第8节 (since 6.8→7, 6.9→8)
// And its subsections should be numbered 第1节 to 第7节
//
// But wait, where is 6.10? Let me check:
// From my earlier read, line 904 shows:
// ## 6.10 总结与展望
// ## 参考文献
//
// So:
// 6.1 → 第1节 (引言)
// 6.2 → 第2节 (ViT, 6 subsections)
// 6.3 → 第3节 (DeiT, 2 subsections)
// 6.4 → 第4节 (Swin, 3 subsections)
// 6.5 → 第5节 (MAE, 2 subsections)
// 6.6 → 第6节 (扩散模型, 5 subsections)
// 6.7 → 第7节 (DALL-E/SD/MJ/Imagen/ControlNet, 5 subsections)
// 6.8 → 第8节 (视频理解, 3 subsections)
// 6.9 → 第9节 (评估指标, 7 subsections)
// 6.10 → 第10节 (总结)
//
// The current fix-r2.js script changed:
// - ## 6.1 → ## 第1节 ✓
// - ## 6.2 → ## 第2节 ✓
// - ### 6.2.x subsections: kept 6.2.x format ✓
// - ## 6.3 → ## 第3节 ✓
// - ### 6.3.x → ### 7.x (incremented major from 3 to 4)... WAIT
//
// Let me re-check my fix-r2.js script logic:
// For subsections matching `6.(3-9|10).X.Y`:
//   parts[1]++ so 6.3.1 → 6.4.1
// For sections matching `6.X`:
//   parts[1]++ so 6.3 → 第4节
//
// But 6.3 subsections like 6.3.1 don't match the subsection regex!
// The subsection regex is: `^### (6\.(?:[3-9]|10)(?:\.[0-9]+)+)\s(.+)`
// This matches 6.3.1, 6.3.2, etc. And then it does `parts[1]++`.
// So 6.3.1 → parts = ["6","3","1"] → parts[1]++ → ["6","4","1"] → "6.4.1"
//
// And the section regex: `^## (6\.[0-9]+)\s(.+)`
// Matches "6.3" → parts = ["6","3"] → parts[1]++ → ["6","4"] → "第4节"
//
// So subsections 6.3.1 → 6.4.1, and section 6.3 → 第4节
// The subsection format is still "6.X.Y" not "第X节"
//
// But the advisor said ALL section titles should use 第X节 format!
// Including subsections.
//
// OK so now I need to ALSO convert subsections to 第X节 format.
//
// The new structure should be:
// ## 第1节 引言
// ## 第2节 Vision Transformer
//   ### 第1节 图像块（Patch）的数学表示
//   ### 第2节 多头自注意力的数学机制
//   ### 第3节 位置编码的设计与理论意义
//   ### 第4节 纯Transformer在CV上的可行性证明
//   ### 第5节 自注意力可视化与可解释性分析
//   ### 第6节 ViT与CNN的系统性对比分析
// ## 第3节 DeiT
//   ### 第1节 训练策略与数据增强
//   ### 第2节 知识蒸馏
// ## 第4节 Swin Transformer
//   ### 第1节 层次化特征图构建
//   ### 第2节 移位窗口注意力机制
//   ### 第3节 多尺度建模与语义融合
// ...

// OK, so the task is:
// For ch18:
// 1. Build a mapping from old section/subsection to new 第X节 format
// 2. Apply to all headers
// 3. Fix cross-references in body text

// For ch19:
// 1. Build mapping from old section/subsection to new 第X节 format
// 2. Apply to all headers
// 3. Fix NOTEARS typo (the NOTEA→NOTEARS created NOTEARSRS)
// 4. Fix cross-references in body text

// Let me write a comprehensive script for both.

// First, let me define the mapping for ch18 sections:
// Section numbering:
// 6.1 → 第1节
// 6.2 → 第2节
// 6.3 → 第3节
// 6.4 → 第4节
// 6.5 → 第5节
// 6.6 → 第6节
// 6.7 → 第7节
// 6.8 → 第8节
// 6.9 → 第9节
// 6.10 → 第10节

// Subsection numbering (within each section, use local index):
// Section 第1节 (was 6.1): no subsections
// Section 第2节 (was 6.2): 6 subsections
//   6.2.1 → 第1节
//   6.2.2 → 第2节
//   6.2.3 → 第3节
//   6.2.4 → 第4节
//   6.2.5 → 第5节
//   6.2.6 → 第6节
// Section 第3节 (was 6.3): 2 subsections
//   6.3.1 → 第1节
//   6.3.2 → 第2节
// Section 第4节 (was 6.4): 3 subsections
//   6.4.1 → 第1节
//   6.4.2 → 第2节
//   6.4.3 → 第3节
// Section 第5节 (was 6.5): 2 subsections
//   6.5.1 → 第1节
//   6.5.2 → 第2节
// Section 第6节 (was 6.6): 5 subsections
//   6.6.1 → 第1节
//   6.6.2 → 第2节
//   6.6.3 → 第3节
//   6.6.4 → 第4节
//   6.6.5 → 第5节
// Section 第7节 (was 6.7): 5 subsections
//   6.7.1 → 第1节
//   6.7.2 → 第2节
//   6.7.3 → 第3节
//   6.7.4 → 第4节
//   6.7.5 → 第5节
// Section 第8节 (was 6.8): 3 subsections
//   6.8.1 → 第1节
//   6.8.2 → 第2节
//   6.8.3 → 第3节
// Section 第9节 (was 6.9): 7 subsections
//   6.9.1 → 第1节
//   6.9.2 → 第2节
//   6.9.3 → 第3节
//   6.10.1 → 第4节
//   6.10.2 → 第5节
//   6.10.3 → 第6节
//   6.10.4 → 第7节... wait, there are only 7 subsections total.
//
// Let me re-count:
// Under ## 6.9 图像生成评估指标:
//   ### 6.9.1 IS (line 752)
//   ### 6.11.2 FID (line 775) - was 6.10.2, fixed to 6.11.2
//   ### 6.11.3 Precision (line 797)
//   ### 6.11.4 sFID (line 819)
//   ### 6.11.5 CLIP Score (line 831)
//   ### 6.11.6 生成模型的系统比较 (line 866)
//   ### 6.11.7 视频生成模型的评估 (line 884)
//
// So the evaluation subsections that were 6.10.X became 6.11.X.
// Wait, that doesn't make sense. Let me re-check.
//
// Original numbering before any fix:
// ## 6.9 图像生成评估指标
//   ### 6.9.1 IS
//   ### 6.9.2 FID
//   ### 6.9.3 Precision
//   ### 6.9.4 sFID
//   ### 6.9.5 CLIP Score
// ## 6.10 总结与展望
// But wait, there were also:
//   ### 6.10.1 Inception Score
//   ### 6.10.2 FID
//   etc.
//
// Hmm, I'm confused. Let me look at the actual content again.
//
// From the grep output:
// Line 748: ## 6.9 图像生成评估指标
// Line 752: ### 6.11.1 Inception Score（IS）
//
// So after my fix, section 6.9 → 第8节, and its first subsection is 6.11.1.
// But 6.11.1 means "section 6.11, subsection 1" which doesn't make sense anymore.
//
// OH I see the issue now. The fix-r2.js script changed:
// - Section headers: ## 6.X → ## 第(X+1)节
// - Subsection headers: ### 6.N.M → ### 6.(N+1).M (for N>=3)
//
// So 6.10.1 → 6.11.1 (N=10, N+1=11)
// And ## 6.9 → ## 第9节
//
// So we have ## 第9节 with ### 6.11.1 subsections. This is inconsistent.
// The 6.11.X should be renumbered to be sequential within 第9节.
//
// The correct numbering should be:
// ## 第9节 图像生成评估指标
//   ### 第1节 Inception Score (IS)
//   ### 第2节 Fréchet Inception Distance (FID)
//   ### 第3节 Precision与Recall
//   ### 第4节 Sliced Fréchet Inception Distance (sFID)
//   ### 第5节 CLIP Score
//   ### 第6节 生成模型的系统比较
//   ### 第7节 视频生成模型的评估
//
// ## 第10节 总结与展望

// OK so the complete fix for ch18 is:
// For section headers: ## 6.X → ## 第X节
// For subsection headers:
//   - Sections 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8: use local numbering 第1节, 第2节, ...
//   - Section 6.9: has 7 subsections → 第1节 to 第7节
// For body cross-references: update old numbers to new

// For ch19:
// Section headers: ## 3.X → ## 第X节 (already done)
// Subsection headers: ### 3.X.Y → ### 第Y节 (strip 3.X prefix)
// Body cross-references: update old numbers to new

// Let me write a clean, comprehensive script.

// ============ ch18 ============
let ch18 = fs.readFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), 'utf-8');

// First, fix section headers
ch18 = ch18.replace(/^## 6\.1 /g, '## 第1节 ');
ch18 = ch18.replace(/^## 6\.2 /g, '## 第2节 ');
ch18 = ch18.replace(/^## 6\.3 /g, '## 第3节 ');
ch18 = ch18.replace(/^## 6\.4 /g, '## 第4节 ');
ch18 = ch18.replace(/^## 6\.5 /g, '## 第5节 ');
ch18 = ch18.replace(/^## 6\.6 /g, '## 第6节 ');
ch18 = ch18.replace(/^## 6\.7 /g, '## 第7节 ');
ch18 = ch18.replace(/^## 6\.8 /g, '## 第8节 ');
ch18 = ch18.replace(/^## 6\.9 /g, '## 第9节 ');
ch18 = ch18.replace(/^## 6\.10 /g, '## 第10节 ');

// Now fix subsection headers
// Section 第2节 (was 6.2) subsections:
ch18 = ch18.replace(/^### 6\.2\.1 /g, '### 第1节 ');
ch18 = ch18.replace(/^### 6\.2\.2 /g, '### 第2节 ');
ch18 = ch18.replace(/^### 6\.2\.3 /g, '### 第3节 ');
ch18 = ch18.replace(/^### 6\.2\.4 /g, '### 第4节 ');
ch18 = ch18.replace(/^### 6\.2\.5 /g, '### 第5节 ');
ch18 = ch18.replace(/^### 6\.2\.6 /g, '### 第6节 ');

// Section 第3节 (was 6.3) subsections:
ch18 = ch18.replace(/^### 6\.3\.1 /g, '### 第1节 ');
ch18 = ch18.replace(/^### 6\.3\.2 /g, '### 第2节 ');

// Section 第4节 (was 6.4) subsections:
ch18 = ch18.replace(/^### 6\.4\.1 /g, '### 第1节 ');
ch18 = ch18.replace(/^### 6\.4\.2 /g, '### 第2节 ');
ch18 = ch18.replace(/^### 6\.4\.3 /g, '### 第3节 ');

// Section 第5节 (was 6.5) subsections:
ch18 = ch18.replace(/^### 6\.5\.1 /g, '### 第1节 ');
ch18 = ch18.replace(/^### 6\.5\.2 /g, '### 第2节 ');

// Section 第6节 (was 6.6) subsections:
ch18 = ch18.replace(/^### 6\.6\.1 /g, '### 第1节 ');
ch18 = ch18.replace(/^### 6\.6\.2 /g, '### 第2节 ');
ch18 = ch18.replace(/^### 6\.6\.3 /g, '### 第3节 ');
ch18 = ch18.replace(/^### 6\.6\.4 /g, '### 第4节 ');
ch18 = ch18.replace(/^### 6\.6\.5 /g, '### 第5节 ');

// Section 第7节 (was 6.7) subsections:
ch18 = ch18.replace(/^### 6\.7\.1 /g, '### 第1节 ');
ch18 = ch18.replace(/^### 6\.7\.2 /g, '### 第2节 ');
ch18 = ch18.replace(/^### 6\.7\.3 /g, '### 第3节 ');
ch18 = ch18.replace(/^### 6\.7\.4 /g, '### 第4节 ');
ch18 = ch18.replace(/^### 6\.7\.5 /g, '### 第5节 ');

// Section 第8节 (was 6.8) subsections:
ch18 = ch18.replace(/^### 6\.8\.1 /g, '### 第1节 ');
ch18 = ch18.replace(/^### 6\.8\.2 /g, '### 第2节 ');
ch18 = ch18.replace(/^### 6\.8\.3 /g, '### 第3节 ');

// Section 第9节 (was 6.9, evaluation metrics) subsections:
// These were originally 6.9.1 to 6.9.5, then 6.10.1 to 6.10.7
// After cascading: 6.9.1-6.9.5 → 6.10.1-6.10.5 → fixed to 6.11.1-6.11.5
// And 6.10.1-6.10.7 → 6.11.1-6.11.7
// Total 7 subsections under 第9节
ch18 = ch18.replace(/^### 6\.9\.1 /g, '### 第1节 ');
ch18 = ch18.replace(/^### 6\.11\.2 /g, '### 第2节 ');
ch18 = ch18.replace(/^### 6\.11\.3 /g, '### 第3节 ');
ch18 = ch18.replace(/^### 6\.11\.4 /g, '### 第4节 ');
ch18 = ch18.replace(/^### 6\.11\.5 /g, '### 第5节 ');
ch18 = ch18.replace(/^### 6\.11\.6 /g, '### 第6节 ');
ch18 = ch18.replace(/^### 6\.11\.7 /g, '### 第7节 ');

// Also handle the remaining 6.10.X references that might exist
ch18 = ch18.replace(/^### 6\.10\.1 /g, '### 第4节 ');
ch18 = ch18.replace(/^### 6\.10\.2 /g, '### 第5节 ');
ch18 = ch18.replace(/^### 6\.10\.3 /g, '### 第6节 ');
ch18 = ch18.replace(/^### 6\.10\.4 /g, '### 第7节 ');
ch18 = ch18.replace(/^### 6\.10\.5 /g, '### 第8节 ');
ch18 = ch18.replace(/^### 6\.10\.6 /g, '### 第9节 ');
ch18 = ch18.replace(/^### 6\.10\.7 /g, '### 第10节 ');

// Fix body cross-references in ch18
// "详见6.2.2节" → still valid (### 第2节 is the second subsection of 第2节)
// Actually wait, after renumbering ### 6.2.2 → ### 第2节, the text "6.2.2节" should become "第2节"
// But "6.2.1节" should become "第1节"
ch18 = ch18.replace(/6\.2\.1节/g, '第1节');
ch18 = ch18.replace(/6\.2\.2节/g, '第2节');
ch18 = ch18.replace(/6\.2\.3节/g, '第3节');
ch18 = ch18.replace(/6\.2\.4节/g, '第4节');
ch18 = ch18.replace(/6\.2\.5节/g, '第5节');
ch18 = ch18.replace(/6\.2\.6节/g, '第6节');

// Cross-references to section-level (6.X → 第X节)
ch18 = ch18.replace(/第6\.1节/g, '第1节');
ch18 = ch18.replace(/第6\.2节/g, '第2节');
ch18 = ch18.replace(/第6\.3节/g, '第3节');
ch18 = ch18.replace(/第6\.4节/g, '第4节');
ch18 = ch18.replace(/第6\.5节/g, '第5节');
ch18 = ch18.replace(/第6\.6节/g, '第6节');
ch18 = ch18.replace(/第6\.7节/g, '第7节');
ch18 = ch18.replace(/第6\.8节/g, '第8节');
ch18 = ch18.replace(/第6\.9节/g, '第9节');
ch18 = ch18.replace(/第6\.10节/g, '第10节');

// Also handle references like "6.3节" without "第" prefix
ch18 = ch18.replace(/6\.1节/g, '第1节');
ch18 = ch18.replace(/6\.2节/g, '第2节');
ch18 = ch18.replace(/6\.3节/g, '第3节');
ch18 = ch18.replace(/6\.4节/g, '第4节');
ch18 = ch18.replace(/6\.5节/g, '第5节');
ch18 = ch18.replace(/6\.6节/g, '第6节');
ch18 = ch18.replace(/6\.7节/g, '第7节');
ch18 = ch18.replace(/6\.8节/g, '第8节');
ch18 = ch18.replace(/6\.9节/g, '第9节');
ch18 = ch18.replace(/6\.10节/g, '第10节');

fs.writeFileSync(path.join(base, 'part5/ch18_视觉Transformer与扩散模型.md'), ch18, 'utf-8');
console.log('ch18 fully fixed');

// ============ ch19 ============
let ch19 = fs.readFileSync(path.join(base, 'part5/ch19_深度学习的理论难题.md'), 'utf-8');

// Fix section headers
ch19 = ch19.replace(/^## 3\.1 /g, '## 第1节 ');
ch19 = ch19.replace(/^## 3\.2 /g, '## 第2节 ');
ch19 = ch19.replace(/^## 3\.3 /g, '## 第3节 ');
ch19 = ch19.replace(/^## 3\.4 /g, '## 第4节 ');
ch19 = ch19.replace(/^## 3\.5 /g, '## 第5节 ');
ch19 = ch19.replace(/^## 3\.6 /g, '## 第6节 ');
ch19 = ch19.replace(/^## 3\.7 /g, '## 第7节 ');
ch19 = ch19.replace(/^## 3\.8 /g, '## 第8节 ');
ch19 = ch19.replace(/^## 3\.9 /g, '## 第9节 ');
ch19 = ch19.replace(/^## 3\.10 /g, '## 第10节 ');

// Fix subsection headers (strip 3.X prefix, keep local number)
// Section 第1节 (was 3.1): no subsections
// Section 第2节 (was 3.2): has subsections
ch19 = ch19.replace(/^### 3\.2\.1 /g, '### 第1节 ');
ch19 = ch19.replace(/^### 3\.2\.2 /g, '### 第2节 ');
// Section 第3节 (was 3.3): has subsections
ch19 = ch19.replace(/^### 3\.3\.1 /g, '### 第1节 ');
ch19 = ch19.replace(/^### 3\.3\.2 /g, '### 第2节 ');
// Section 第4节 (was 3.4): has subsections
ch19 = ch19.replace(/^### 3\.4\.1 /g, '### 第1节 ');
ch19 = ch19.replace(/^### 3\.4\.2 /g, '### 第2节 ');
ch19 = ch19.replace(/^### 3\.4\.3 /g, '### 第3节 ');
// Section 第5节 (was 3.5): has subsections
ch19 = ch19.replace(/^### 3\.5\.1 /g, '### 第1节 ');
ch19 = ch19.replace(/^### 3\.5\.2 /g, '### 第2节 ');
ch19 = ch19.replace(/^### 3\.5\.3 /g, '### 第3节 ');
// Section 第6节 (was 3.6): has subsections
ch19 = ch19.replace(/^### 3\.6\.1 /g, '### 第1节 ');
ch19 = ch19.replace(/^### 3\.6\.2 /g, '### 第2节 ');
// Section 第7节 (was 3.7): has subsections
ch19 = ch19.replace(/^### 3\.7\.1 /g, '### 第1节 ');
ch19 = ch19.replace(/^### 3\.7\.2 /g, '### 第2节 ');
ch19 = ch19.replace(/^### 3\.7\.3 /g, '### 第3节 ');
ch19 = ch19.replace(/^### 3\.7\.4 /g, '### 第4节 ');
ch19 = ch19.replace(/^### 3\.7\.5 /g, '### 第5节 ');
ch19 = ch19.replace(/^### 3\.7\.6 /g, '### 第6节 ');
ch19 = ch19.replace(/^### 3\.7\.7 /g, '### 第7节 ');
ch19 = ch19.replace(/^### 3\.7\.8 /g, '### 第8节 ');
ch19 = ch19.replace(/^### 3\.7\.9 /g, '### 第9节 ');
ch19 = ch19.replace(/^### 3\.7\.10 /g, '### 第10节 ');
// Section 第8节 (was 3.8): has subsections
ch19 = ch19.replace(/^### 3\.8\.1 /g, '### 第1节 ');
ch19 = ch19.replace(/^### 3\.8\.2 /g, '### 第2节 ');
ch19 = ch19.replace(/^### 3\.8\.3 /g, '### 第3节 ');
// Section 第9节 (was 3.9): has subsections
ch19 = ch19.replace(/^### 3\.9\.1 /g, '### 第1节 ');
ch19 = ch19.replace(/^### 3\.9\.2 /g, '### 第2节 ');
ch19 = ch19.replace(/^### 3\.9\.3 /g, '### 第3节 ');
ch19 = ch19.replace(/^### 3\.9\.4 /g, '### 第4节 ');
ch19 = ch19.replace(/^### 3\.9\.5 /g, '### 第5节 ');
ch19 = ch19.replace(/^### 3\.9\.6 /g, '### 第6节 ');
ch19 = ch19.replace(/^### 3\.9\.7 /g, '### 第7节 ');
ch19 = ch19.replace(/^### 3\.9\.8 /g, '### 第8节 ');
// Section 第10节 (was 3.10): has subsections
ch19 = ch19.replace(/^### 3\.10\.1 /g, '### 第1节 ');
ch19 = ch19.replace(/^### 3\.10\.2 /g, '### 第2节 ');

// Fix NOTEARSRS → NOTEARS (double-replacement artifact)
ch19 = ch19.replace(/NOTEARSRS/g, 'NOTEARS');

// Fix body cross-references in ch19
// These patterns appear in the body text: 3.X.Y节, 第3.X节, 3.X节, etc.
// Section-level references (3.X → 第X节)
ch19 = ch19.replace(/第3\.1节/g, '第1节');
ch19 = ch19.replace(/第3\.2节/g, '第2节');
ch19 = ch19.replace(/第3\.3节/g, '第3节');
ch19 = ch19.replace(/第3\.4节/g, '第4节');
ch19 = ch19.replace(/第3\.5节/g, '第5节');
ch19 = ch19.replace(/第3\.6节/g, '第6节');
ch19 = ch19.replace(/第3\.7节/g, '第7节');
ch19 = ch19.replace(/第3\.8节/g, '第8节');
ch19 = ch19.replace(/第3\.9节/g, '第9节');
ch19 = ch19.replace(/第3\.10节/g, '第10节');

// Subsection references: 3.X.Y → local number Y
// But we need context - "3.2.1节" should become "第1节" (1st sub of section 2)
// "3.5.2节" should become "第2节" (2nd sub of section 5)
// Since each subsection's local number matches its suffix, this works:
ch19 = ch19.replace(/3\.2\.1节/g, '第1节');
ch19 = ch19.replace(/3\.2\.8节/g, '第8节'); // Wait, does 3.2 have 8 subsections?
// Let me check: from the grep, "3.2.8" appears in line 414 and 532
// "3.2.8节从平均场理论的角度初步介绍了深度网络中的信号传播"
// Hmm, 3.2.8 means the 8th subsection of section 3.2.
// But I only listed up to 3.2.2 above. Let me be more comprehensive.

// Actually, let me use a more systematic approach for subsection references
// The pattern is: 3.A.B → B-th subsection of section A
// After renumbering: it becomes "第B节" (within section "第A节")
// So 3.2.1 → 第1节, 3.2.8 → 第8节, etc.

// Let me use regex for this:
ch19 = ch19.replace(/3\.(\d+)\.(\d+)节/g, (match, section, sub) => {
  return '第' + sub + '节';
});

// Handle range references like "3.7.2-3.7.4节"
ch19 = ch19.replace(/3\.(\d+)\.(\d+)-3\.(\d+)\.(\d+)节/g, (match, s1, sub1, s2, sub2) => {
  if (s1 === s2) {
    return '第' + sub1 + '–' + '第' + sub2 + '节';
  }
  return match; // Can't simplify cross-section ranges
});

// Also handle "3.X节" (section-level without 第 prefix)
ch19 = ch19.replace(/3\.1节/g, '第1节');
ch19 = ch19.replace(/3\.2节/g, '第2节');
ch19 = ch19.replace(/3\.3节/g, '第3节');
ch19 = ch19.replace(/3\.4节/g, '第4节');
ch19 = ch19.replace(/3\.5节/g, '第5节');
ch19 = ch19.replace(/3\.6节/g, '第6节');
ch19 = ch19.replace(/3\.7节/g, '第7节');
ch19 = ch19.replace(/3\.8节/g, '第8节');
ch19 = ch19.replace(/3\.9节/g, '第9节');
ch19 = ch19.replace(/3\.10节/g, '第10节');

fs.writeFileSync(path.join(base, 'part5/ch19_深度学习的理论难题.md'), ch19, 'utf-8');
console.log('ch19 fully fixed');

console.log('All comprehensive fixes applied!');
