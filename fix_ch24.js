const fs = require('fs');
const path = 'C:/Users/22974/desktop/数模/token-burning-work/volumes/part6/ch24_AI安全对齐与伦理.md';

let content = fs.readFileSync(path, 'utf8');
// Normalize line endings to \n
content = content.replace(/\r\n/g, '\n');
let lines = content.split('\n');

// Fix line 27 (index 26): add $ delimiters, \pi, \tau, \sim
lines[26] = '$\min_{\\pi} E_{\\tau \\sim \\pi} [ (U_H(\\tau) - U_{AI}(\\tau))^2 ]$';

// Fix line 31 (index 30): gamma in [0,1) -> \gamma \in [0,1) in math mode
lines[30] = lines[30].replace('gamma in [0,1)', '$\\gamma \\in [0,1)$');

// Fix line 33 (index 32): wrap in $$...$$ and fix sim, D_test, U_AI
lines[32] = '$$\\text{Alignment Gap} = E_{s \\sim D_{\\text{test}}} [ \\max_a U_H(s,a) - \\max_a U_{AI}(s,a) ]$$';

// Fix line 45 (index 44): add $ delimiters and \text{argmax}
lines[44] = '$\\text{argmax}_a \\tilde{U}_{AI}(s,a) \\neq \\text{argmax}_a U_H(s,a)$';

// Fix line 55 (index 54): hat{r}_t -> \hat{r}_t, eta_t -> \eta_t, add $ delimiters, add \text{true}
lines[54] = '$\\hat{r}_t = f(r_t^{\\text{true}}, \\eta_t)$';

// Fix line 57 (index 56): wrap eta_t and hat{r}_t in math mode
lines[56] = lines[56].replace(/eta_t/g, '$\\eta_t$').replace('hat{r}_t', '$\\hat{r}_t$');

// Fix line 59 (index 58): wrap in $$...$$
lines[58] = '$$' + lines[58] + '$$';

// Fix line 102 (index 101): add $ delimiters, \text{CoherentExtrapolation}, \ldots
lines[101] = '$V_{CEV} = \\text{CoherentExtrapolation}(V_1, V_2, \\ldots, V_n)$';

// Fix line 116 (index 115): full formula fix with math mode and proper LaTeX
lines[115] = '$\\mathcal{L}_{\\text{RLHF}}(\\theta) = \\mathbb{E}_{x \\sim \\mathcal{D}, y \\sim \\pi_{\\theta}(\\cdot|x)} [ R_{\\phi}(x,y) - \\beta \\text{KL}(\\pi_{\\theta}(\\cdot|x) \\parallel \\pi_{\\text{ref}}(\\cdot|x)) ]$';

fs.writeFileSync(path, lines.join('\n'), 'utf8');
console.log('All ch24 fixes applied.');
