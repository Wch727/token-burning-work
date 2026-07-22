/**
 * Fix broken subsection numbering patterns left by bad sed replacements.
 *
 * ch05:  第X节Y节  -> 第X.Y节
 *        第X.Y节Title (no space) -> 第X.Y节 Title
 * ch15:  第X节Y    -> 第X.Y节
 *        第X节Y.Z  -> 第X.Y.Z节
 */
const fs = require('fs')
const path = require('path')

const BASE = path.dirname(__dirname)

function fixFile(relPath, transforms, dryRun = false) {
  const fp = path.join(BASE, relPath)
  const orig = fs.readFileSync(fp, 'utf8')
  const lines = orig.split('\n')
  const changes = []

  const next = lines.map((line, idx) => {
    // Only touch markdown headings
    if (!/^#{1,4}\s+/.test(line)) return line
    let out = line
    for (const { re, repl } of transforms) {
      out = out.replace(re, repl)
    }
    if (out !== line) {
      changes.push({ line: idx + 1, from: line, to: out })
    }
    return out
  })

  console.log(`\n=== ${relPath}: ${changes.length} heading changes ===`)
  for (const c of changes) {
    console.log(`L${c.line}: ${c.from}`)
    console.log(`     -> ${c.to}`)
  }

  if (!dryRun && changes.length > 0) {
    fs.writeFileSync(fp, next.join('\n'), 'utf8')
    console.log(`Wrote ${relPath}`)
  }
  return changes
}

// ch05 patterns
const ch05Transforms = [
  // ### 第1节1节 Title  ->  ### 第1.1节 Title
  { re: /(#{1,4}\s+)第(\d+)节(\d+)节(\s*)/g, repl: '$1第$2.$3节$4' },
  // ### 第3.4节核技巧  ->  ### 第3.4节 核技巧  (missing space after 节)
  { re: /(#{1,4}\s+第\d+(?:\.\d+)+节)([^\s\d.#])/g, repl: '$1 $2' },
]

// ch15 patterns — order matters: longer form first
const ch15Transforms = [
  // #### 第5节4.1 Title  ->  #### 第5.4.1节 Title
  { re: /(#{1,4}\s+)第(\d+)节(\d+)\.(\d+)(\s)/g, repl: '$1第$2.$3.$4节$5' },
  // ### 第2节1 Title  ->  ### 第2.1节 Title
  { re: /(#{1,4}\s+)第(\d+)节(\d+)(\s)/g, repl: '$1第$2.$3节$4' },
]

const dry = process.argv.includes('--dry')
fixFile('volumes/part2/ch05_统计机器学习的基础奠定.md', ch05Transforms, dry)
fixFile('volumes/part4/ch15_Seq2Seq与Transformer.md', ch15Transforms, dry)
