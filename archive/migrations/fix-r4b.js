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
// ch12: AdaBoost formula missing \right)
// Use regular string to avoid template literal backtick issues
// ============================================================
fix('volumes/part3/ch12_计算机视觉的演进.md', 'AdaBoost missing \\right)',
  function(s) {
    var oldStr = 'H(x) = \\text{sign}\\left(\\sum_{t=1}^T \\alpha_t h_t(x)\\right`';
    var newStr = 'H(x) = \\text{sign}\\left(\\sum_{t=1}^T \\alpha_t h_t(x)\\right)';
    return s.replace(oldStr, newStr);
  }
);

// ============================================================
// ch15: remove duplicate Transformer limitations paragraph
// ============================================================
fix('volumes/part4/ch15_Seq2Seq与Transformer.md', 'remove duplicate Transformer limitations',
  function(s) {
    var dup = '尽管Transformer在众多任务上展现了卓越的性能，但从理论角度而言，它并非没有局限性。自注意力的排列不变性是一把双刃剑：虽然它使模型能够捕获任意位置之间的关系，但也意味着模型缺乏内置的归纳偏置来偏好局部性或层次结构。对于许多具有强烈局部结构的任务（如语音信号处理、时间序列预测），这种缺乏偏置的设计可能导致样本复杂度的增加——即模型需要看到更多的训练样本才能学习到人类认为"显然"的局部模式。此外，自注意力的二次方复杂度在序列长度极大时（如处理64K token的长文档）会成为计算和内存的双重瓶颈。虽然线性注意力和稀疏注意力等变体在一定程度上缓解了这一问题，但这些改进往往以牺牲全注意力的表达精度为代价。从计算复杂性理论的角度，目前尚未有严格的证明表明自注意力机制在表达能力上严格优于具有足够深度的RNN，尽管经验证据强烈支持前者在大多数NLP任务上的优越性。最后，Transformer的梯度流动虽然优于RNN，但在极深模型（如超过100层）上仍然面临优化挑战——残差连接和层归一化虽然提供了梯度高速公路，但注意力层本身的 conditioning number 可能在深层训练中恶化，导致数值不稳定。';
    var idx = s.indexOf(dup);
    if (idx >= 0) {
      return s.substring(0, idx) + s.substring(idx + dup.length);
    }
    return s;
  }
);

// ============================================================
// ch21: Performer softmax variance parameter
// ============================================================
fix('volumes/part6/ch21_NLP精细化.md', 'Performer softmax sqrt(d_k)->d_k',
  function(s) {
    var r = s;
    // Line 114: main decomposition
    r = r.replace(
      '\\exp\\left(\\frac{q_i^\\top k_j}{\\sqrt{d_k}}\\right) = \\exp\\left(\\frac{\\|q_i\\|^2 + \\|k_j\\|^2}{2\\sqrt{d_k}}\\right) \\cdot \\exp\\left(-\\frac{\\|q_i - k_j\\|^2}{2\\sqrt{d_k}}\\right)',
      '\\exp\\left(\\frac{q_i^\\top k_j}{d_k}\\right) = \\exp\\left(\\frac{\\|q_i\\|^2 + \\|k_j\\|^2}{2d_k}\\right) \\cdot \\exp\\left(-\\frac{\\|q_i - k_j\\|^2}{2d_k}\\right)'
    );
    // Line 117: first term
    r = r.replace(
      '\\exp\\left(\\frac{\\|q_i\\|^2 + \\|k_j\\|^2}{2\\sqrt{d_k}}\\right)',
      '\\exp\\left(\\frac{\\|q_i\\|^2 + \\|k_j\\|^2}{2d_k}\\right)'
    );
    r = r.replace(
      '\\exp\\left(\\frac{\\|q_i\\|^2}{2\\sqrt{d_k}}\\right)',
      '\\exp\\left(\\frac{\\|q_i\\|^2}{2d_k}\\right)'
    );
    r = r.replace(
      '\\exp\\left(\\frac{\\|k_j\\|^2}{2\\sqrt{d_k}}\\right)',
      '\\exp\\left(\\frac{\\|k_j\\|^2}{2d_k}\\right)'
    );
    // Line 119: variance definition
    r = r.replace('\\sigma^2 = \\sqrt{d_k}', '\\sigma^2 = d_k');
    // Line 124: softmax feature mapping
    r = r.replace(
      '\\exp\\left(\\frac{\\|x\\|^2}{2\\sqrt{d_k}}\\right)',
      '\\exp\\left(\\frac{\\|x\\|^2}{2d_k}\\right)'
    );
    return r;
  }
);

// ============================================================
// ch22: SSD downsampling ratios
// ============================================================
fix('volumes/part6/ch22_计算机视觉精细化.md', 'SSD downsampling 4,6,8,10 -> 8,16,32,64',
  function(s) {
    return s.replace(
      'conv8_2、conv9_2、conv10_2、conv11_2四个特征图分别进行4倍、6倍、8倍和10倍的尺寸递减',
      'conv8_2、conv9_2、conv10_2、conv11_2四个特征图分别进行8倍、16倍、32倍和64倍的尺寸递减'
    );
  }
);

console.log('\nTotal fixes applied: ' + fixes);
