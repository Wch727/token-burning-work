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
// ch07: 3 factual errors
// ============================================================

// Fix 1: SVM history (line 315)
fix('volumes/part2/ch07_神经网络的下沉期.md', 'SVM历史修正',
  function(s) {
    return s.replace(
      '1995年，Cortes和Vapnik提出了标准SVM算法。SVM的核心优势在于：',
      '1992年，Boser、Guyon和Vapnik提出了使用核技巧的SVM算法；1995年，Cortes和Vapnik进一步引入软间隔，扩展了SVM的适用范围。SVM的核心优势在于：'
    );
  }
);

// Fix 2: Mixture of Experts attribution (line 495)
fix('volumes/part2/ch07_神经网络的下沉期.md', '混合专家模型归属修正',
  function(s) {
    return s.replace(
      '辛顿与Michael I. Jordan合作，在1994年提出了混合专家模型，将条件概率建模为多个"专家"网络的加权组合。这一模型不仅具有理论优雅性，还为后来深度学习中的注意力机制（Attention Mechanism）和门控机制（Gating Mechanism）提供了重要思想资源。',
      '辛顿与Michael I. Jordan等人于1991年共同提出了混合专家模型（Jacobs et al., 1991），将条件概率建模为多个"专家"网络的加权组合。这一模型不仅具有理论优雅性，还为后来深度学习中的注意力机制（Attention Mechanism）和门控机制（Gating Mechanism）提供了重要思想资源。'
    );
  }
);

// Fix 3: LeCun PhD year (line 207)
fix('volumes/part2/ch07_神经网络的下沉期.md', 'LeCun博士年份修正',
  function(s) {
    return s.replace(
      '1988年，一位年轻的法国研究者加入了美国AT&T贝尔实验室（Bell Labs）。他就是杨立昆（Yann LeCun），当时刚从 Pierre和Marie Curie大学获得博士学位。',
      '1988年，一位年轻的法国研究者加入了美国AT&T贝尔实验室（Bell Labs）。他就是杨立昆（Yann LeCun），此前已在Pierre和Marie Curie大学获得博士学位（1987）。'
    );
  }
);

// ============================================================
// ch08: Word2Vec gradient direction (line 462)
// ============================================================
fix('volumes/part2/ch08_语言模型的进化：从n-gram到神经网络.md', 'NEG梯度方向修正',
  function(s) {
    return s.replace(
      '注意梯度的不对称性：正样本将 $\\mathbf{v}_c$ 向 $-\\mathbf{v}_w$ 方向推动（当 $\\sigma(\\mathbf{v}_w^\\top \\mathbf{v}_c)$ 接近1时，梯度趋近于0，表示"已足够接近"；当 $\\sigma(\\mathbf{v}_w^\\top \\mathbf{v}_c)$ 接近0时，梯度趋近于 $-\\mathbf{v}_w$，表示需要大幅靠近）；负样本将 $\\mathbf{v}_{w_k}$ 向 $\\mathbf{v}_w$ 方向推动（当 $\\sigma(\\mathbf{v}_w^\\top \\mathbf{v}_{w_k})$ 接近1时，梯度趋近于 $\\mathbf{v}_w$，表示需要大幅远离；当接近0时，梯度趋近于0，表示"已足够远离"）。',
      '注意梯度的不对称性：正样本将 $\\mathbf{v}_c$ 向 $\\mathbf{v}_w$ 方向推动（当 $\\sigma(\\mathbf{v}_w^\\top \\mathbf{v}_c)$ 接近1时，梯度趋近于0，表示"已足够接近"；当 $\\sigma(\\mathbf{v}_w^\\top \\mathbf{v}_c)$ 接近0时，梯度趋近于 $\\mathbf{v}_w$，表示需要大幅靠近）；负样本将 $\\mathbf{v}_{w_k}$ 向 $-\\mathbf{v}_w$ 方向推动（当 $\\sigma(\\mathbf{v}_w^\\top \\mathbf{v}_{w_k})$ 接近1时，梯度趋近于 $-\\mathbf{v}_w$，表示需要大幅远离；当接近0时，梯度趋近于0，表示"已足够远离"）。'
    );
  }
);

// ============================================================
// ch13: section numbering 5.x -> 13.x
// ============================================================
fix('volumes/part4/ch13_ImageNet与AlexNet.md', '节编号5.x->13.x',
  function(s) {
    // Replace all ## 5.x and ### 5.x.x patterns
    // Use regex to match section headings with 5.x numbering
    var r = s.replace(/^## 5\.(\d)/gm, function(m, n) { return '## 第' + n + '节'; });
    r = r.replace(/^### 5\.(\d)\.(\d)/gm, function(m, n1, n2) { return '### 第' + n1 + '.' + n2 + '节'; });
    // Also replace body cross-references like "5.1节" -> "13.1节" etc.
    // Be careful not to match version numbers or other unrelated "5." patterns
    r = r.replace(/(?<![第\d])5\.(\d)(?![.\d])/g, function(m, n, offset, str) {
      // Only replace if it looks like a section reference (preceded by space/punctuation)
      var before = str[offset - 1];
      if (before === ' ' || before === '章' || before === '节' || before === '、' || before === '（') {
        return '13.' + n;
      }
      return m;
    });
    return r;
  }
);

// ============================================================
// ch14: VGG conv1 params (line 61)
// ============================================================
fix('volumes/part4/ch14_CNN架构的黄金时代.md', 'VGG conv1参数数量修正',
  function(s) {
    return s.replace(
      'conv1_1和conv1_2两个卷积层各使用64个3×3滤波器，共约1,792个参数（含偏置），',
      'conv1_1和conv1_2两个卷积层各使用64个3×3滤波器，每层约1,792个参数（含偏置），两层共计约3,584个参数，'
    );
  }
);

// Fix: Inception 3a compression ratio (line 148-149)
fix('volumes/part4/ch14_CNN架构的黄金时代.md', 'Inception 3a压缩比修正',
  function(s) {
    return s.replace(
      '整个Inception 3a模块的参数量仅为约15.9万，而如果不使用降维，仅路径2的5×5卷积就需要192×5×5×64=307,200个参数——降维将参数量压缩了约7.7倍。',
      '整个Inception 3a模块的参数量仅为约5.3万，而如果不使用降维，仅路径2的5×5卷积就需要192×5×5×64=307,200个参数——降维将参数量压缩了约5.8倍。'
    );
  }
);

// Fix: Auxiliary classifier position (line 170)
fix('volumes/part4/ch14_CNN架构的黄金时代.md', '辅助分类器位置修正',
  function(s) {
    return s.replace(
      '这两个辅助分类器分别插入在Inception 3a和4d之后',
      '这两个辅助分类器分别插入在Inception 4a和4d之后'
    );
  }
);

// ============================================================
// ch16: InfoGAN formula sign (line 325) and duplicate gradient vanishing
// ============================================================

// Fix 1: InfoGAN formula - change -λ to +λ to match stated objective
fix('volumes/part4/ch16_生成模型的演进.md', 'InfoGAN公式符号修正',
  function(s) {
    return s.replace(
      '$$\min_{G} \max_{D} V_{\\text{InfoGAN}}(G, D, Q) - \\lambda I(\\mathbf{c}; G(\\mathbf{z}, \\mathbf{c})),$$',
      '$$\min_{G} \max_{D} V_{\\text{InfoGAN}}(G, D, Q) + \\lambda I(\\mathbf{c}; G(\\mathbf{z}, \\mathbf{c})),$$'
    );
  }
);

// Fix 2: Remove duplicate "梯度消失" entry (lines 682-684)
fix('volumes/part4/ch16_生成模型的演进.md', '删除重复的梯度消失条目',
  function(s) {
    var dup = '- **梯度消失（Vanishing Gradients）。** 在训练初期，生成器质量很差，判别器能够以高置信度将假样本识别为假，导致传递给生成器的梯度非常小。当生成器的输出几乎完全落在判别器的"拒绝区域"时，梯度信号会趋近于零，使得生成器的参数更新几乎停滞。';
    var idx = s.indexOf(dup);
    if (idx >= 0) {
      // Check if this is the second occurrence (first should be in section 3.4.4)
      var firstIdx = s.indexOf(dup);
      var secondIdx = s.indexOf(dup, firstIdx + 1);
      if (secondIdx >= 0) {
        return s.substring(0, secondIdx) + s.substring(secondIdx + dup.length);
      }
    }
    return s;
  }
);

// ============================================================
// ch18: Stable Diffusion attribution (line 670)
// ============================================================
fix('volumes/part5/ch18_视觉Transformer与扩散模型.md', 'Stable Diffusion开发主体修正',
  function(s) {
    return s.replace(
      'Stable Diffusion是LAION团队（Stability AI）于2022年8月发布的开源文本到图像生成系统',
      'Stable Diffusion是Stability AI于2022年8月发布的开源文本到图像生成系统，其训练使用了LAION开发的LAION-5B大规模数据集'
    );
  }
);

console.log('\nTotal fixes applied: ' + fixes);
