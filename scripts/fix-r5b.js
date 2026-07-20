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
// ch07: Mixture of Experts (line 495) - NOT YET FIXED
// ============================================================
fix('volumes/part2/ch07_神经网络的下沉期.md', '混合专家模型归属修正',
  function(s) {
    return s.replace(
      '辛顿与Michael I. Jordan合作，在1994年提出了混合专家模型，将条件概率建模为多个"专家"网络的加权组合。这一模型不仅具有理论优雅性，还为后来深度学习中的注意力机制（Attention Mechanism）和门控机制（Gating Mechanism）提供了重要思想资源。',
      '辛顿与Michael I. Jordan等人于1991年共同提出了混合专家模型（Jacobs et al., 1991），将条件概率建模为多个"专家"网络的加权组合。这一模型不仅具有理论优雅性，还为后来深度学习中的注意力机制（Attention Mechanism）和门控机制（Gating Mechanism）提供了重要思想资源。'
    );
  }
);

// ============================================================
// ch08: Word2Vec gradient direction (line 462) - NOT YET FIXED
// ============================================================
fix('volumes/part2/ch08_语言模型的进化：从n-gram到神经网络.md', 'NEG梯度方向修正',
  function(s) {
    // Use char codes to avoid shell escaping issues
    var D = String.fromCharCode(36); // $
    var B = String.fromCharCode(92); // \
    var neg = String.fromCharCode(45); // -
    var vw = D + B + "mathbf{v}" + String.fromCharCode(95) + "w" + D;
    var neg_vw = D + neg + B + "mathbf{v}" + String.fromCharCode(95) + "w" + D;

    // 正样本: v_c 向 -v_w -> v_w
    var old1 = "正样本将 " + D + B + "mathbf{v}" + String.fromCharCode(95) + "c" + D + " 向 " + neg_vw + " 方向推动";
    var new1 = "正样本将 " + D + B + "mathbf{v}" + String.fromCharCode(95) + "c" + D + " 向 " + vw + " 方向推动";
    s = s.replace(old1, new1);

    // First "梯度趋近于 -v_w" (for v_c) -> "梯度趋近于 v_w"
    // Find the first occurrence after the positive sample text
    var idx = s.indexOf("正样本将 " + D + B + "mathbf{v}" + String.fromCharCode(95) + "c");
    if (idx >= 0) {
      var parenthetical = s.substring(idx, s.indexOf("；负样本", idx));
      var first_grad = parenthetical.indexOf("梯度趋近于 " + neg_vw);
      if (first_grad >= 0) {
        var abs_pos = idx + first_grad;
        s = s.substring(0, abs_pos) + "梯度趋近于 " + vw + s.substring(abs_pos + ("梯度趋近于 " + neg_vw).length);
      }
    }

    // 负样本: v_wk 向 v_w -> -v_w
    var old2 = "向 " + vw + " 方向推动（当";
    var new2 = "向 " + neg_vw + " 方向推动（当";
    s = s.replace(old2, new2);

    // "梯度趋近于 v_w，表示需要大幅远离" -> "梯度趋近于 -v_w"
    var old3 = "梯度趋近于 " + vw + "，表示需要大幅远离";
    var new3 = "梯度趋近于 " + neg_vw + "，表示需要大幅远离";
    s = s.replace(old3, new3);

    return s;
  }
);

// ============================================================
// ch14: Inception 3a compression ratio - NOT YET FIXED
// ============================================================
fix('volumes/part4/ch14_CNN架构的黄金时代.md', 'Inception 3a压缩比修正',
  function(s) {
    // The actual text says: "压缩比约为7.7倍"
    // Need to find the correct compression ratio
    // 1x1 conv: 192*1*1*32 = 6,144 params + 32 biases = 6,176
    // 5x5 conv: 192*5*5*64 = 307,200 params + 64 biases = 307,264
    // Total with inception: 6,176 + 307,264 + 32 + 64 = 313,536 (wait, pool path too)
    // Actually: 1x1 path: 192*1*1*32+32 = 6,176
    // 5x5 path: 192*5*5*64+64 = 307,264
    // pool path: 32*1*1*64+64 = 2,080
    // Total: 6,176 + 307,264 + 2,080 = 315,520
    // Without reduction: 192*5*5*64+64 = 307,264 (just the 5x5 path)
    // Hmm, the compression ratio depends on what you compare
    // Reviewer said: actual ratio is ~5.8x, not 7.7x
    // Let me just fix the ratio number
    return s.replace(
      '压缩比约为7.7倍',
      '压缩比约为5.8倍'
    );
  }
);

// ============================================================
// ch16: InfoGAN formula sign (line 325) - NOT YET FIXED
// ============================================================
fix('volumes/part4/ch16_生成模型的演进.md', 'InfoGAN公式符号修正',
  function(s) {
    // Find the actual formula
    var idx = s.indexOf("V_{\\text{InfoGAN}}");
    if (idx >= 0) {
      var line = s.substring(idx, idx + 100);
      console.log("  ch16 InfoGAN formula found: " + line.substring(0, 60));
    }
    // The file has: V_{InfoGAN}(G, D, Q) - \lambda I(...)
    // Change - to +
    return s.replace(
      "V_{\\text{InfoGAN}}(G, D, Q) - \\lambda I(",
      "V_{\\text{InfoGAN}}(G, D, Q) + \\lambda I("
    );
  }
);

// ============================================================
// ch16: Remove duplicate gradient vanishing entry
// ============================================================
fix('volumes/part4/ch16_生成模型的演进.md', '删除重复的梯度消失条目',
  function(s) {
    // Find all occurrences of the gradient vanishing bullet
    var bullet = '**梯度消失（Vanishing Gradients）。**';
    var count = (s.match(new RegExp(bullet.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length;
    console.log("  ch16 gradient vanishing count: " + count);
    // If 2 occurrences, remove the second one
    if (count >= 2) {
      var first = s.indexOf(bullet);
      var second = s.indexOf(bullet, first + 1);
      // Find the end of the second bullet (next bullet or section end)
      var end = second;
      // Find the next bullet or heading after second
      var next = s.indexOf('\n- ', second + 1);
      var nextHead = s.indexOf('\n### ', second + 1);
      var nextHash = s.indexOf('\n## ', second + 1);
      var ends = [next, nextHead, nextHash].filter(function(e) { return e > 0; });
      if (ends.length > 0) {
        end = Math.min.apply(null, ends);
      } else {
        end = s.length;
      }
      var removed = s.substring(second, end);
      console.log("  Removing: " + removed.substring(0, 80) + "...");
      return s.substring(0, second) + s.substring(end);
    }
    return s;
  }
);

// ============================================================
// ch18: Stable Diffusion attribution (line 670) - NOT YET FIXED
// ============================================================
fix('volumes/part5/ch18_视觉Transformer与扩散模型.md', 'Stable Diffusion开发主体修正',
  function(s) {
    return s.replace(
      'Stable Diffusion是LAION团队（Stability AI）于2022年8月发布的开源文本到图像生成系统',
      'Stable Diffusion是Stability AI于2022年8月发布的开源文本到图像生成系统，其训练使用了LAION开发的LAION-5B大规模数据集'
    );
  }
);

// ============================================================
// ch20: 4 factual errors - NOT YET FIXED
// ============================================================
fix('volumes/part5/ch20_强化学习的复兴.md', 'UCT年份修正',
  function(s) {
    return s.replace(
      'Kocsis和Szepesvári在2000年提出了UCT',
      'Kocsis和Szepesvári在2006年提出了UCT'
    );
  }
);

fix('volumes/part5/ch20_强化学习的复兴.md', 'Neuro-Dynamic Programming年份修正',
  function(s) {
    return s.replace(
      'Bertsekas和Tsitsiklis于1999年出版了《Neuro-Dynamic Programming》',
      'Bertsekas和Tsitsiklis于1996年出版了《Neuro-Dynamic Programming》'
    );
  }
);

fix('volumes/part5/ch20_强化学习的复兴.md', 'D4PG作者归属修正',
  function(s) {
    return s.replace(
      'D4PG（Distributed DDPG）由Heess等人（2018）提出',
      'D4PG（Distributed Distributional Deterministic Policy Gradients）由Barth-Maron等人（2018）提出'
    );
  }
);

fix('volumes/part5/ch20_强化学习的复兴.md', 'IMPALA年份修正',
  function(s) {
    return s.replace(
      'Espeholt等人在2019年提出了IMPALA',
      'Espeholt等人在2018年提出了IMPALA'
    );
  }
);

console.log('\nTotal fixes applied: ' + fixes);
