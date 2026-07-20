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
    console.log(`[FIXED] ${file}: ${desc}`);
  } else {
    console.log(`[SKIP]  ${file}: ${desc} (pattern not found)`);
  }
}

// ============================================================
// ch01: binary addition result
// ============================================================
fix('volumes/part1/ch01_图灵与AI的诞生.md', 'binary 1010 -> 1100',
  s => s.replace('即1010₂', '即1100₂')
);

// ============================================================
// ch02: citations
// ============================================================
fix('volumes/part1/ch02_感知机与连接主义的兴衰.md', 'Novikoff citation format',
  s => s.replace('Novikoff 1962', 'Novikoff（1962）')
);
fix('volumes/part1/ch02_感知机与连接主义的兴衰.md', 'learning rate decay untranslated heading',
  s => s.replace('**学习率衰减（learning rate decay）**', '**学习率衰减**')
);
fix('volumes/part1/ch02_感知机与连接主义的兴衰.md', 'VC citation format',
  s => s.replace(
    '需要等到Vapnik和Chervonenkis的VC维理论（1971年提出，1995年系统阐述）才得到完整的发展',
    '需要等到Vapnik和Chervonenkis（1971）的VC维理论才得到完整的发展'
  )
);

// ============================================================
// ch04: title + section renumbering + MYCIN formula
// ============================================================
fix('volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md', 'chapter title 第四章->第4章',
  s => s.replace('# 第四章 第一次AI冬天与专家系统的兴衰', '# 第4章 第一次AI冬天与专家系统的兴衰')
);

// Section headers: Chinese numerals -> 第X节
const chineseToNum = {
  '一、': '1节', '二、': '2节', '三、': '3节', '四、': '4节',
  '五、': '5节', '六、': '6节', '七、': '7节'
};
fix('volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md', 'section headers renumber',
  s => {
    let r = s;
    // Main sections: ## 一、 -> ## 第1节
    for (const [cn, num] of Object.entries(chineseToNum)) {
      r = r.replace(new RegExp(`## ${cn}`, 'g'), `## 第${num} `);
    }
    // Sub sections: ### 4.1 -> ### 第1节, ### 4.2 -> ### 第2节, ### 4.3 -> ### 第3节
    r = r.replace(/### 4\.1 /g, '### 第1节 ');
    r = r.replace(/### 4\.2 /g, '### 第2节 ');
    r = r.replace(/### 4\.3 /g, '### 第3节 ');
    // Body cross-refs: 第一节 -> 第1节, 第二节 -> 第2节, etc.
    r = r.replace(/第([一二三四五六七])节/g, (m, cn) => {
      const map = { '一': '1', '二': '2', '三': '3', '四': '4', '五': '5', '六': '6', '七': '7' };
      return `第${map[cn]}节`;
    });
    return r;
  }
);

// MYCIN formula -> LaTeX
fix('volumes/part1/ch04_第一次AI冬天与专家系统的兴衰.md', 'MYCIN formula LaTeX',
  s => s.replace(
    `CF(x, y) = \n- X + Y − XY，当 X > 0 且 Y > 0（两个正向证据）\n- X + Y + XY，当 X < 0 且 Y < 0（两个负向证据）\n- (X + Y) / (1 − min(|X|, |Y|))，当 X 与 Y 异号（正反证据相互抵消）`,
    `$$
\\begin{cases}
X + Y - XY, & \\text{当 } X > 0 \\text{ 且 } Y > 0 \\text{（两个正向证据）} \\\\
X + Y + XY, & \\text{当 } X < 0 \\text{ 且 } Y < 0 \\text{（两个负向证据）} \\\\
\\frac{X + Y}{1 - \\min(|X|, |Y|)}, & \\text{当 } X \\text{ 与 } Y \\text{ 异号（正反证据相互抵消）}
\\end{cases}
$$`
  )
);

// ============================================================
// ch11: AdaBoost formula missing \right)
// ============================================================
fix('volumes/part3/ch11_深度学习的关键理论突破.md', 'AdaBoost missing \\right)',
  s => s.replace(
    `H(x) = \\text{sign}\\left(\\sum_{t=1}^T \\alpha_t h_t(x)`,
    `H(x) = \\text{sign}\\left(\\sum_{t=1}^T \\alpha_t h_t(x)\\right)`
  )
);

// ============================================================
// ch12: DoG Taylor + SVM formulas
// ============================================================
fix('volumes/part3/ch12_计算机视觉的演进.md', 'DoG Taylor missing \\right)',
  s => s.replace(
    `D(\\mathbf{x}) \\approx D + \\frac{\\partial D^T}{\\partial \\mathbf{x}}\\mathbf{x} + \\frac{1}{2}\\mathbf{x}^T \\frac{\\partial^2 D}{\\partial \\mathbf{x}^2} \\mathbf{x}`,
    `D(\\mathbf{x}) \\approx D + \\frac{\\partial D}{\\partial \\mathbf{x}}^T \\mathbf{x} + \\frac{1}{2}\\mathbf{x}^T \\frac{\\partial^2 D}{\\partial \\mathbf{x}^2} \\mathbf{x} \\right)`
  )
);

fix('volumes/part3/ch12_计算机视觉的演进.md', 'SVM missing \\right)',
  s => s.replace(
    `f(\\mathbf{x}) = \\text{sign}\\left(\\sum_{i=1}^{N} \\alpha_i y_i K(\\mathbf{x}_i, \\mathbf{x}) + b`,
    `f(\\mathbf{x}) = \\text{sign}\\left(\\sum_{i=1}^{N} \\alpha_i y_i K(\\mathbf{x}_i, \\mathbf{x}) + b\\right)`
  )
);

// ============================================================
// ch13: Softmax sign error
// ============================================================
fix('volumes/part4/ch13_ImageNet与AlexNet.md', 'Softmax sign error',
  s => s.replace(
    `= -y_c + \\hat{y}_c + \\sum_{k \\neq c} (-y_k) \\left[0 - \\hat{y}_c\\right]`,
    `= -y_c + \\hat{y}_c + \\sum_{k \\neq c} y_k \\hat{y}_c`
  )
);

// ============================================================
// ch15: DeepMind year + duplicate text
// ============================================================
fix('volumes/part4/ch15_Seq2Seq与Transformer.md', 'DeepMind year 2010->2011',
  s => s.replace('DeepMind成立于2010年', 'DeepMind成立于2011年')
);

fix('volumes/part4/ch15_Seq2Seq与Transformer.md', 'duplicate Transformer limitations paragraph',
  s => {
    // The duplicate is at line 866-967 area. The first occurrence is at 864-866.
    // The second occurrence (lines 866-967) is the duplicate that should be removed.
    // Actually, let me re-examine: lines 864-866 is a SHORT summary, lines 866-967 is a LONG paragraph.
    // The review says "duplicate Transformer text at line 966".
    // Looking at the structure, the long paragraph at 866-967 starts with "尽管Transformer..." and
    // duplicates content from the section at 864-866.
    // The simplest fix: the paragraph at 866-967 is the duplicate of the shorter one at 864-866.
    // Let me check: line 864 says "尽管Transformer在众多任务上展现了卓越的性能..." and
    // line 866 says the same thing. So 866 is the duplicate start.
    // Actually, looking more carefully:
    // Line 864: (end of §2.1) blank + section end
    // Lines 864-866: "尽管Transformer...数值不稳定。" (3 sentences)
    // Lines 866-967: "尽管Transformer...数值不稳定。" (same 3 sentences + much more)
    // The longer one at 866 is the duplicate that extends into §6.6 content.
    // Fix: remove the duplicated first 3 sentences from lines 866-967.
    // Actually the simplest interpretation: the paragraph at 866-967 is a verbatim repeat
    // of the one at 864-866, so remove lines 866-967.
    // But that would remove a lot of content. Let me look at the actual content more carefully.
    // Line 966 content: "Transformer的理论局限性" section which discusses theoretical limits.
    // This is NOT identical to lines 864-866 which are about practical limitations.
    // The REVIEW says "duplicate Transformer text at line 966".
    // Let me check if there's actual text duplication.
    return s; // Will handle with more targeted approach
  }
);

console.log(`\nTotal fixes applied: ${fixes}`);
