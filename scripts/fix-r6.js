const fs = require('fs');
const path = require('path');
const base = 'C:\\Users\\22974\\desktop\\数模\\token-burning-work';

let fixes = 0;
function fix(file, desc, fn) {
  const fp = path.join(base, file);
  const orig = fs.readFileSync(fp, 'utf8');
  const modified = fn(orig);
  if (modified !== orig) {
    fs.writeFileSync(fp, modified, 'utf8');
    fixes++;
    console.log('[FIXED] ' + file + ': ' + desc);
  } else {
    console.log('[SKIP]  ' + file + ': ' + desc);
  }
}

// ============================================================
// ch01: Binary addition result (line 173)
// 1011 + 110 = 10001, not 1010 (= 10)
// ============================================================
fix('volumes/part1/ch01_图灵与AI的诞生.md', '二进制加法结果修正',
  function(s) {
    return s.replace(
      '最终磁带上留下和#1010，即1100₂。',
      '最终磁带上留下和#10001，即10001₂。'
    );
  }
);

// ============================================================
// ch02: Title (remove 续), section numbering, duplicates
// ============================================================
fix('volumes/part1/ch02_感知机与连接主义的兴衰.md', '章节标题去（续）',
  function(s) {
    return s.replace(
      '# 感知机与连接主义的兴衰（续）',
      '# 第2章 感知机与连接主义的兴衰'
    );
  }
);

fix('volumes/part1/ch02_感知机与连接主义的兴衰.md', '节编号6.5->第7节',
  function(s) {
    return s.replace('## 6.5 鼎盛时期的隐忧（续）', '## 第7节 鼎盛时期的隐忧');
  }
);

fix('volumes/part1/ch02_感知机与连接主义的兴衰.md', '节编号6.6->第8节',
  function(s) {
    return s.replace('## 6.6 Novikoff收敛定理的完整证明过程', '## 第8节 Novikoff收敛定理的完整证明过程');
  }
);

fix('volumes/part1/ch02_感知机与连接主义的兴衰.md', '节编号6.7->第9节',
  function(s) {
    return s.replace('## 6.7 Cover定理：线性判别函数的容量与样本复杂度', '## 第9节 Cover定理：线性判别函数的容量与样本复杂度');
  }
);

fix('volumes/part1/ch02_感知机与连接主义的兴衰.md', '节编号6.8->第10节',
  function(s) {
    return s.replace('## 6.8 XOR问题：连接主义的"阿喀琉斯之踵"', '## 第10节 XOR问题：连接主义的"阿喀琉斯之踵"');
  }
);

fix('volumes/part1/ch02_感知机与连接主义的兴衰.md', '节编号6.9->第11节',
  function(s) {
    return s.replace('## 6.9 Minsky和Papert《Perceptrons》详细解读', '## 第11节 Minsky和Papert《Perceptrons》详细解读');
  }
);

// Remove duplicate ## 8 and ## 9 sections (the second occurrence)
fix('volumes/part1/ch02_感知机与连接主义的兴衰.md', '删除重复的第8节',
  function(s) {
    var dup = '## 8. 历史人物：Rosenblatt的早逝与Minsky的学术影响';
    var first = s.indexOf(dup);
    var second = s.indexOf(dup, first + 1);
    if (second >= 0) {
      var end = second;
      var next = s.indexOf('\n## ', second + 1);
      if (next >= 0) end = next;
      return s.substring(0, second) + s.substring(end);
    }
    return s;
  }
);

fix('volumes/part1/ch02_感知机与连接主义的兴衰.md', '删除重复的第9节',
  function(s) {
    var dup = '## 9. 结论：摆锤的摆动（扩展版）';
    var first = s.indexOf(dup);
    var second = s.indexOf(dup, first + 1);
    if (second >= 0) {
      var end = second;
      var next = s.indexOf('\n## ', second + 1);
      if (next >= 0) end = next;
      return s.substring(0, second) + s.substring(end);
    }
    return s;
  }
);

// ============================================================
// ch03: InfoNCE authors, Shannon year, Hunt decision trees
// ============================================================
fix('volumes/part1/ch03_统计学习思想的萌芽.md', 'InfoNCE作者补全',
  function(s) {
    return s.replace(
      'Oord等人（2018）',
      'Oord、Li和Vinyals（2018）'
    );
  }
);

fix('volumes/part1/ch03_统计学习思想的萌芽.md', '香农1950年论文修正',
  function(s) {
    return s.replace(
      '这一概念的确切形式化归功于香农本人在1950年关于"预测与熵"的未发表工作，以及后来费诺（Fano）在信息论中的编码定理。',
      '这一概念的确切形式化归功于香农本人，以及后来费诺（Fano）在信息论中的编码定理。'
    );
  }
);

fix('volumes/part1/ch03_统计学习思想的萌芽.md', '补充Shannon 1951预测工作',
  function(s) {
    return s.replace(
      '信息论与当代表示学习之间的最新交汇',
      '香农关于英语信息熵的预测工作（"Prediction and Entropy of Printed English"）发表于1951年，为后来的信息瓶颈方法奠定了基础。\n\n信息论与当代表示学习之间的最新交汇'
    );
  }
);

fix('volumes/part1/ch03_统计学习思想的萌芽.md', '补充Hunt等1956年决策树工作',
  function(s) {
    return s.replace(
      '正是这一洞见，直接导致了ID3算法（Quinlan, 1986）的诞生。',
      '在Hunt等人（1956年）的工作中，决策树的概念就已经被提出——他们的CLS（Concept Learning System）是最早的决策树学习算法之一。正是这一洞见，直接导致了ID3算法（Quinlan, 1986）的诞生。'
    );
  }
);

// ============================================================
// ch04: explicitly -> 显式地
// ============================================================
fix('volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md', 'explicitly翻译',
  function(s) {
    return s.replace(
      '每一条规则都必须由人类explicitly编码',
      '每一条规则都必须由人类显式地编码'
    );
  }
);

// ============================================================
// ch09: All section headings 第X章 -> 第X节
// ============================================================
fix('volumes/part3/ch09_深度信念网络与预训练的革命.md', '节编号第X章->第X节',
  function(s) {
    return s.replace(/## 第(\d+)章 /g, '## 第$1节 ');
  }
);

// ============================================================
// ch22: Focal Loss gradient formula, smooth_L1, citation format
// ============================================================
fix('volumes/part6/ch22_计算机视觉精细化.md', 'Focal Loss梯度公式修正',
  function(s) {
    return s.replace(
      '$$\\frac{\\partial \\text{FL}}{\\partial z} = -\\alpha_t (1 - p_t)^{\\gamma - 1} \\left(\\gamma p_t \\log(p_t) + (1 - p_t)\\right)$$',
      '$$\\frac{\\partial \\text{FL}}{\\partial z} = -\\alpha_t (1 - p_t)^{\\gamma} \\frac{\\partial p_t}{\\partial z}$$'
    );
  }
);

fix('volumes/part6/ch22_计算机视觉精细化.md', 'smooth_L1 otherwise LaTeX',
  function(s) {
    return s.replace(
      '|x| - 0.5 & \\text{otherwise} \\end{cases}',
      '|x| - 0.5 & \\text{otherwise} \\end{cases}'
    );
  }
);

// Check if \\text{otherwise} is already correct (with backslash)
var s22check = fs.readFileSync(path.join(base, 'volumes/part6/ch22_计算机视觉精细化.md'), 'utf8');
if (s22check.includes('|x| - 0.5 & otherwise')) {
  // Re-fix with proper backslash
  fs.writeFileSync(path.join(base, 'volumes/part6/ch22_计算机视觉精细化.md'),
    s22check.replace('|x| - 0.5 & otherwise', '|x| - 0.5 & \\text{otherwise}'), 'utf8');
  console.log('[FIXED] ch22: smooth_L1 otherwise LaTeX (retry)');
  fixes++;
} else if (s22check.includes('|x| - 0.5 & \\text{otherwise}')) {
  console.log('[SKIP]  ch22: smooth_L1 otherwise already correct');
}

fix('volumes/part6/ch22_计算机视觉精细化.md', 'YOLOv5引用格式修正',
  function(s) {
    return s.replace(
      'YOLOv5（Glenn Jocher, 2020）',
      'YOLOv5（Glenn Jocher，2020）'
    );
  }
);

// ============================================================
// ch23: Math environment wrapping, \text commands
// ============================================================
fix('volumes/part6/ch23_图神经网络与结构数据.md', '邻接矩阵定义加数学环境',
  function(s) {
    return s.replace(
      '定义其邻接矩阵 $\\mathbf{A} \\in \\mathbb{R}^{N \\times N}$，度矩阵 $\\mathbf{D} = \\text{diag}(d_1, d_2, \\ldots, d_N)$',
      '定义其邻接矩阵 $\\mathbf{A} \\in \\mathbb{R}^{N \\times N}$，度矩阵 $\\mathbf{D} = \\operatorname{diag}(d_1, d_2, \\ldots, d_N)$'
    );
  }
);

fix('volumes/part6/ch23_图神经网络与结构数据.md', 'Chebyshev otherwise LaTeX',
  function(s) {
    return s.replace(
      '\\text{otherwise}',
      '\\text{otherwise}'
    );
  }
);

// Check if \text{otherwise} is already correct
var s23check = fs.readFileSync(path.join(base, 'volumes/part6/ch23_图神经网络与结构数据.md'), 'utf8');
if (s23check.includes('\\text{otherwise}')) {
  console.log('[SKIP]  ch23: Chebyshev otherwise already correct');
} else if (s23check.includes('otherwise') && !s23check.includes('\\text{otherwise}')) {
  // Need to fix - find the raw otherwise and wrap it
  fs.writeFileSync(path.join(base, 'volumes/part6/ch23_图神经网络与结构数据.md'),
    s23check.replace('otherwise', '\\text{otherwise}'), 'utf8');
  console.log('[FIXED] ch23: Chebyshev otherwise LaTeX (retry)');
  fixes++;
}

fix('volumes/part6/ch23_图神经网络与结构数据.md', 'CEV公式数学环境',
  function(s) {
    return s.replace(
      'rho_{\\text{CEV}} = \\text{CoherentExtrapolation}(V_1, V_2, \\ldots, V_n)',
      '$\\rho_{\\text{CEV}} = \\operatorname{CoherentExtrapolation}(V_1, V_2, \\ldots, V_n)$'
    );
  }
);

// ============================================================
// ch24: partially rational, CEV formula, RLHF formula
// ============================================================
fix('volumes/part6/ch24_AI安全对齐与伦理.md', 'partially rational翻译',
  function(s) {
    return s.replace(
      '如果一个AI系统是" partially rational"（部分理性）',
      '如果一个AI系统是"部分理性"（partially rational）'
    );
  }
);

fix('volumes/part6/ch24_AI安全对齐与伦理.md', 'CEV公式数学环境',
  function(s) {
    return s.replace(
      '$V_{CEV} = \\text{CoherentExtrapolation}(V_1, V_2, \\ldots, V_n)$',
      '$$V_{CEV} = \\operatorname{CoherentExtrapolation}(V_1, V_2, \\ldots, V_n)$$'
    );
  }
);

// Check if the CEV formula is already in math mode
var s24check = fs.readFileSync(path.join(base, 'volumes/part6/ch24_AI安全对齐与伦理.md'), 'utf8');
if (s24check.includes('$V_{CEV}')) {
  // Already in inline math, but might use text instead of operatorname
  console.log('[INFO] ch24: CEV formula check');
}

fix('volumes/part6/ch24_AI安全对齐与伦理.md', 'RLHF公式LaTeX修正',
  function(s) {
    return s.replace(
      '$\\mathcal{L}_{\\text{RLHF}}(\\theta) = \\mathbb{E}_{x \\sim \\mathcal{D}, y \\sim \\pi_{\\theta}(\\cdot|x)} [ R_{\\phi}(x,y) - \\beta \\text{KL}(\\pi_{\\theta}(\\cdot|x) \\parallel \\pi_{\\text{ref}}(\\cdot|x)) ]$',
      '$$\\mathcal{L}_{\\text{RLHF}}(\\theta) = \\mathbb{E}_{x \\sim \\mathcal{D}, y \\sim \\pi_{\\theta}(\\cdot|x)} \\left[ R_{\\phi}(x,y) - \\beta \\text{KL}\\Big(\\pi_{\\theta}(\\cdot|x) \\|\\pi_{\\text{ref}}(\\cdot|x) \\Big) \\right]$$'
    );
  }
);

// Also fix beta and KL in the explanatory text after the formula
fix('volumes/part6/ch24_AI安全对齐与伦理.md', 'RLHF正文beta和KL修正',
  function(s) {
    return s.replace(
      '其中KL散度项防止策略偏离参考模型 pi_{ref} 过远，beta 是正则化系数',
      '其中KL散度项防止策略偏离参考模型 $\\pi_{\\text{ref}}$ 过远，$\\beta$ 是正则化系数'
    );
  }
);

console.log('\nTotal fixes applied: ' + fixes);
