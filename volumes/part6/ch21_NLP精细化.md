# 第21章 NLP的精细化：Transformer之后的语言理解

> 本章系统梳理2017年Transformer架构提出以来，自然语言处理领域在效率优化、推理增强、多语言扩展、检索融合、智能体化以及评估体系方面取得的系统性进展。我们将从数学原理出发，深入探讨每一项技术背后的核心思想与形式化推导。

---

## 第1节 高效Transformer：从O(n²)到近线性复杂度

标准Transformer的自注意力机制在处理长度为n的序列时，时间和空间复杂度均为O(n²d)，这在处理长文档、基因组序列等超长输入时成为根本性瓶颈。自2019年起，学术界提出了三类具有代表性的高效注意力方案：基于低秩近似的Linformer、基于核函数随机特征的Performer，以及基于硬件感知IO优化的FlashAttention。本节将给出每一方案的完整数学推导。

### 第1.1节 标准自注意力的复杂度瓶颈

给定输入矩阵$X \in \mathbb{R}^{n \times d}$，其中n为序列长度，d为隐藏维度。标准缩放点积注意力（Scaled Dot-Product Attention）定义为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中$Q = XW_Q, K = XW_K, V = XW_V$为查询、键、值矩阵。计算注意力权重矩阵$A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times n}$需要$O(n^2d)$的时间和空间。对于n=4096（常见大模型上下文窗口），若d=768，仅存储注意力矩阵就需要约$4096^2 \times 4 \text{ bytes} \approx 67 \text{ MB}$；当n扩展到128K时，仅注意力矩阵就需要超过67GB的显存，这远超当前GPU的显存容量。

高效Transformer的核心目标是在保持注意力表达能力的前提下，将复杂度降至$O(n \cdot \text{poly}(\log n))$或$O(nk)$（其中k为远小于n的秩参数）。

### 第1.2节 Linformer：低秩键值矩阵近似

**核心思想。** Linformer（Wang et al., 2020）的出发点是：实际应用中，注意力矩阵$A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)$通常具有低秩结构。更精确地说，键矩阵$K \in \mathbb{R}^{n \times d_k}$和值矩阵$V \in \mathbb{R}^{n \times d_v}$可以分别通过秩为k的投影矩阵映射到低维子空间，且k远小于n（通常k = 256或128），同时保持输出质量接近全注意力。

**数学推导。** Linformer 的关键是沿序列长度维度 $n$ 做低秩投影，而非沿特征维度。记投影矩阵 $E, F \in \mathbb{R}^{n \times k}$（$k \ll n$），键值矩阵 $K, V \in \mathbb{R}^{n \times d}$。先将键、值投影到长度为 $k$ 的序列：

$$
\bar{K} = E^\top K, \quad \bar{V} = F^\top V
$$

其中 $\bar{K}, \bar{V} \in \mathbb{R}^{k \times d}$。近似注意力输出为：

$$
\tilde{O} = \mathrm{softmax}\left(\frac{Q\bar{K}^\top}{\sqrt{d}}\right)\bar{V}
$$

此处 $Q\bar{K}^\top \in \mathbb{R}^{n \times k}$，不再构造完整的 $n \times n$ 注意力矩阵。

**方案一：核心采样（Core Sampling）。** 直接从 $n$ 个键/值位置中随机采样 $k$ 个，即 $E, F$ 为选择矩阵。这相当于对序列长度做下采样，时间复杂度为 $O(nkd)$。

**方案二：可学习线性投影。** 将 $E, F$ 参数化为可训练矩阵，在预训练中与网络其余参数联合优化；也可采用固定卷积核将长度为 $n$ 的序列投影到长度为 $k$ 的序列。

**多头扩展。** 对于 $h$ 个注意力头，Linformer 可为每个头独立构造投影矩阵，也可采用"共享头投影"（head-wise shared projection）——所有头共享同一组 $E$ 和 $F$，在参数效率与表达力之间取得平衡。

**复杂度分析。** 计算 $\bar{K}=E^\top K$ 与 $\bar{V}=F^\top V$ 的代价为 $O(nkd)$；随后 $Q\bar{K}^\top$ 为 $O(nkd)$，$\mathrm{softmax}(\cdot)\bar{V}$ 亦为 $O(nkd)$。因此整体注意力复杂度为 $O(nkd)$，显存占用从 $O(n^2)$ 降至 $O(nk)$。在 $n=4096, k=256, d=512$ 的设置下，相对标准注意力可获得数倍加速。

### 第1.3节 Performer：随机特征核估计

**核心思想。** Performer（Choromanski et al., 2020）从核方法的角度重新审视注意力机制。标准softmax注意力可以写成：

$$
\text{Attention}(Q, K, V) = D^{-1}\hat{A}V
$$

其中$\hat{A}_{ij} = \exp\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right)$为未归一化的注意力分数，$D_i = \sum_{j=1}^n \hat{A}_{ij}$为分母（归一化因子）。

关键在于，softmax核函数$k(x, y) = \exp(x^\top y)$是一个正定核函数（Mercer核），可以表示为某个特征映射$\phi: \mathbb{R}^d \to \mathbb{R}^m$的内积：

$$
\exp(x^\top y) = \langle \phi(x), \phi(y) \rangle
$$

Performer的核心贡献是构造了一个显式的随机特征映射$\phi$，使得：

$$
\exp\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right) \approx \phi(q_i)^\top \phi(k_j)
$$

从而将$O(n^2)$的注意力矩阵计算转化为$O(nm)$的特征映射内积计算，其中m为随机特征的维度（通常m = 256或512）。

**FAVOR+ 的数学要点。** 经典随机 Fourier 特征（RFF，Rahimi & Recht, 2007）用 $\cos/\sin$ 映射近似平移不变核，但用于 softmax 注意力时可能出现负特征、方差偏大。Performer 的 **FAVOR+**（Fast Attention Via positive Orthogonal Random features）采用**正正交随机特征**，以保证非负近似并降低方差。典型形式可写为

$$
\phi(x)\ \propto\ \exp\!\left(-\frac{\|x\|^2}{2}\right)\exp(Wx)
$$

（再对随机投影方向做正交化），其中 $W$ 的行取自高斯再正交化。这与 $\phi(x)=[\cos(w^\top x+b),\sin(\cdot)]$ 的 RFF 不同：FAVOR+ 强调 **正特征 + 正交化**，从而更稳定地近似 softmax 核并支持线性复杂度注意力。

**softmax注意力的分解。** 为了处理softmax的归一化分母，Performer将softmax分解为两个核函数的差：

$$
\exp\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right) = \exp\left(\frac{\|\hat{q}_i\|^2 + \|\hat{k}_j\|^2}{2}\right) \cdot \exp\left(-\frac{\|\hat{q}_i - \hat{k}_j\|^2}{2}\right)
$$

其中缩放后的向量定义为$\hat{q}_i = q_i / d_k^{1/4}$、$\hat{k}_j = k_j / d_k^{1/4}$，从而$\hat{q}_i^\top \hat{k}_j = q_i^\top k_j / \sqrt{d_k}$，与标准注意力中的$\sqrt{d_k}$缩放保持一致。

第一项$\exp\left(\frac{\|\hat{q}_i\|^2 + \|\hat{k}_j\|^2}{2}\right)$可以分解为$\exp\left(\frac{\|\hat{q}_i\|^2}{2}\right) \cdot \exp\left(\frac{\|\hat{k}_j\|^2}{2}\right)$，这是两个向量的外积形式，可以直接通过特征映射计算。

第二项$\exp\left(-\frac{\|\hat{q}_i - \hat{k}_j\|^2}{2}\right)$是高斯核$k(x, y) = \exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)$的形式（此处$\sigma^2 = 1$）。这同样可以通过RFF进行近似。

将两部分结合，定义：

$$
\phi_{\text{softmax}}(x) = \exp\left(\frac{\|\hat{x}\|^2}{2}\right) \cdot \tilde{\phi}(\hat{x}) \in \mathbb{R}^{2m}
$$

（其中$\hat{x} = x / d_k^{1/4}$）
则近似注意力为：

$$
\tilde{O}_i = \frac{\phi_{\text{softmax}}(q_i)^\top \left( \sum_{j=1}^n \phi_{\text{softmax}}(k_j) v_j^\top \right)}{ \phi_{\text{softmax}}(q_i)^\top \left( \sum_{j=1}^n \phi_{\text{softmax}}(k_j) \cdot \mathbf{1} \right) + \epsilon}
$$

定义前缀和矩阵：

$$
S_K = \sum_{j=1}^n \phi_{\text{softmax}}(k_j) v_j^\top \in \mathbb{R}^{2m \times d_v}
$$

$$
S_{K1} = \sum_{j=1}^n \phi_{\text{softmax}}(k_j) \in \mathbb{R}^{2m}
$$

则每个输出$\tilde{O}_i$可以在$O(md_v)$时间内计算，整体复杂度为$O(nmd_v + nmd_k) = O(nmd)$。当$m \ll n$时（如m=256, n=4096），这是显著的加速。

**线性注意力与核函数的关系。** Katharopoulos等人（2020）的线性注意力可以视为Performer的特例。当使用线性核$k(x, y) = x^\top y$时，有：

$$
\text{Attention}_{\text{linear}}(Q, K, V) = \frac{Q(K^\top V)}{Q(K^\top \mathbf{1})}
$$

这避免了显式计算$n \times n$的注意力矩阵，复杂度直接降至$O(n d^2)$。然而，线性核的建模能力有限，FAVOR+通过随机特征映射在保持线性复杂度的同时恢复了softmax注意力的表达能力。

### 第1.4节 FlashAttention：硬件感知的IO优化

**核心思想。** Linformer和Performer从算法层面降低了注意力计算的渐近复杂度，但它们忽略了现代GPU/SRAM/DRAM层次化存储体系的实际特性。FlashAttention（Dao et al., 2022）认识到：GPU中的HBM（High Bandwidth Memory）访问是主要的性能瓶颈，而SRAM（Shared Memory）虽然速度快但容量有限（如A100 GPU的SRAM为192KB per SM，而HBM为80GB）。

标准注意力在HBM中的读写操作如下：
1. 从HBM读取$Q, K, V$（各$O(nd)$）
2. 计算$S = QK^\top/\sqrt{d_k}$（$O(n^2d)$，结果写回HBM）
3. 计算$P = \text{softmax}(S)$（$O(n^2)$，结果写回HBM）
4. 计算$O = PV$（$O(n^2d)$，结果写回HBM）

总HBM访问量为$O(n^2)$量级（中间注意力矩阵的读写占主导）。FlashAttention的目标是通过分块（tiling）技术，将HBM访问量降至$O(N^2 d^2 / M)$，其中$M$为SRAM容量——当$M$足够大时，显著低于朴素实现的$O(N^2)$次中间矩阵读写。

**矩阵分块与在线Softmax。** FlashAttention将$Q, K, V$沿序列长度维度分割为大小为$B_r \times d$的块（row blocks），其中$B_r$的选择使得每个块可以放入SRAM。对于A100 GPU，通常设置$B_r = 64$或128。

将Q, K, V分别分割为$t = \lceil n/B_r \rceil$个块：

$$
Q = [Q_1; Q_2; \ldots; Q_t], \quad K = [K_1; K_2; \ldots; K_t], \quad V = [V_1; V_2; \ldots; V_t]
$$

其中$Q_i, K_i, V_i \in \mathbb{R}^{B_r \times d}$（最后一个块可能更小）。

FlashAttention的核心算法循环如下（伪代码）：

```
输入: Q ∈ R^{n×d_k}, K ∈ R^{n×d_k}, V ∈ R^{n×d_v}
输出: O ∈ R^{n×d_v}

将Q, K, V沿n维度分割为t个块
初始化 O = 0 ∈ R^{n×d_v}
初始化 l = 0 ∈ R^n (行和，用于在线softmax)
初始化 m = -∞ ∈ R^n (行最大值，用于在线softmax)

for i = 1 to t do  // 遍历Q的块
    Qi = Q[i]  // B_r × d_k
    
    // 将Qi再分割为更小的列块以适配SRAM
    for j = 1 to t do  // 遍历K,V的块
        Kj = K[j]  // B_r × d_k
        Vj = V[j]  // B_r × d_v
        
        // 在SRAM中计算当前块的注意力
        Sij = Qi @ Kj^T / sqrt(d_k)  // B_r × B_r
        
        // 在线softmax更新
        m_hat = max(m, rowmax(Sij))
        exp_diff = exp(m - m_hat)
        Pij = exp(Sij - m_hat)  // B_r × B_r
        
        l_new = exp_diff * l + rowsum(Pij)
        Pij = Pij / l_new  // 归一化
        
        // 更新输出
        O = O * (l / l_new) * exp_diff + Pij @ Vj
        l = l_new
        m = m_hat
    end for
end for

return O
```

**在线Softmax的数学推导。** 标准softmax的计算需要知道整个序列的最大值才能进行数值稳定的指数运算。FlashAttention（Dao et al., 2022）采用了增量式归一化策略（online softmax），允许在不知道全局最大值的情况下逐块计算softmax。

对于任意行i，令$m_i^{(j)} = \max(m_i, s_{i1}, \ldots, s_{ij})$为前j个分数中的最大值，$l_i^{(j)} = \sum_{k=1}^j \exp(s_{ik} - m_i^{(j)})$为部分归一化常数。当处理新的块时：

$$
m_i^{(j+1)} = \max(m_i^{(j)}, \max(S_{i, j+1}))
$$

$$
l_i^{(j+1)} = e^{m_i^{(j)} - m_i^{(j+1)}} l_i^{(j)} + \sum_{k=j+1}^{j+B_r} \exp(s_{ik} - m_i^{(j+1)})
$$

$$
\tilde{O}_i^{(j+1)} = \frac{l_i^{(j)}}{l_i^{(j+1)}} e^{m_i^{(j)} - m_i^{(j+1)}} \tilde{O}_i^{(j)} + \frac{1}{l_i^{(j+1)}} \sum_{k=j+1}^{j+B_r} \exp(s_{ik} - m_i^{(j+1)}) v_k
$$

这保证了在增量更新过程中，输出$\tilde{O}_i$始终等于使用全局softmax的精确结果（在浮点精度内）。

**IO复杂度分析。** 设SRAM容量为$S$。选择$B_r = \min(d, \sqrt{S})$使得每个$Q_i$或$K_i$块可以放入SRAM。每个块的$S_{ij} = Q_i K_j^\top$在SRAM中计算，不需要将完整的$n \times n$矩阵写入HBM。每个输入元素从HBM读取的次数为$O(1)$（确切地说，$O(\sqrt{S}/d)$次），因此总的HBM访问量为$O(n^2 d / \sqrt{S}) = O(n^2)$除以SRAM大小的平方根。

在A100 GPU上，FlashAttention-1相比标准注意力实现了2-4倍的加速；FlashAttention-2（Dao, 2023）通过优化并行化策略（减少线程块间的非对称同步、改进工作分区）进一步将加速比提升至4-8倍；FlashAttention-3（Shah et al., 2024）针对Hopper架构的TMA（Tensor Memory Accelerator）和FP8Tensor Core进行了深度优化，在H100上达到了接近理论FLOPS峰值的利用率。

### 第1.5节 高效注意力机制的比较与展望

三种高效注意力机制从不同角度解决O(n²)瓶颈：

| 方法 | 核心思路 | 近似类型 | 理论复杂度 | 实际加速比 |
|------|----------|----------|------------|------------|
| Linformer | 低秩投影 | 结构化解码 | O(nk) | 2-5× |
| Performer | 随机核特征 | 核方法解码 | O(nm) | 2-4× |
| FlashAttention | IO优化 | 精确计算 | O(n²)/√S | 4-8× |

值得注意的是，这三条路线并非互斥。FlashAttention-2可以嵌入Performer的随机特征思想；Hyena（Poli et al., 2023）等后续工作则以长卷积与数据依赖门控替代稠密注意力，在超长序列上追求亚二次复杂度与硬件友好实现。与之并行，状态空间模型路线（如Mamba）则直接放弃注意力，在线性复杂度下建模长程依赖。

**超越注意力架构：状态空间模型。** 除上述基于注意力机制的高效化路线外，另一条革命性的思路是直接放弃注意力机制，采用状态空间模型（State Space Model, SSM）作为序列建模的核心。Mamba（Gu & Dao, 2023）基于选择性状态空间模型，在保持线性序列长度的同时，实现了与Transformer相媲美甚至超越的建模能力。

选择性SSM的核心方程将输入x映射到输出y：

$$
h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C^\top h_t
$$

其中$\bar{A}, \bar{B}$为依赖于输入x的时间-varying参数（这是"选择性"的关键）。在离散化形式下，使用零阶保持（ZOH）方法：

$$
h_t = \exp(\Lambda (t)) h_{t-1} + \exp(\Lambda (t)) B(t) x_t, \quad y_t = C(t)^\top h_t + D(t) x_t
$$

当$\bar{A}$为负定矩阵时，历史信息呈指数衰减，模型天然具备选择性遗忘机制，无需像注意力机制那样显式地存储所有历史token。这使Mamba在长序列建模上具有O(n)的理论复杂度，同时在语言建模任务上取得了与同规模Transformer相当的性能。

Mamba-2（Dao & Gu, 2024）将选择性SSM与结构化状态空间对偶（SSD）框架统一，证明了Mamba-2的矩阵乘法和Transformer的注意力矩阵乘法在数学上的深层联系。Mamba-2使用半可分（semiseparable）矩阵结构，可以利用硬件优化的矩阵乘法内核（如FlashAttention风格的块状计算）进行高效实现。这一发现表明，未来的序列建模架构可能在注意力机制和状态空间模型之间找到统一的数学描述，从而兼得两者的优势。

---

## 第2节 大语言模型的推理能力：思维链与超越

人类解决复杂问题时，并非一步到位给出答案，而是通过一系列中间推理步骤逐步逼近结论。这种"慢思考"（System 2 thinking，Kahneman, 2011）模式正是大语言模型推理能力研究的核心灵感来源。从Chain-of-Thought到Tree-of-Thought再到Graph-of-Thought，推理方法经历了从线性序列到树状探索再到图状组合的演进。本节将从形式化角度深入分析每种方法的数学原理与算法设计。

### 第2.1节 Chain-of-Thought：分步推理的提示范式

**动机与形式化。** Wei等人（2022）提出的Chain-of-Thought（CoT） prompting 的核心洞察是：通过在提示中提供包含中间推理步骤的示例，可以显著提升大语言模型在算术推理、常识推理和符号推理任务上的表现。从计算复杂性理论的角度看，CoT将模型从本质上改变了解题范式——不再是直接从输入映射到输出（$f: \mathcal{X} \to \mathcal{Y}$），而是生成一个推理轨迹（reasoning trace）$z = (z_1, z_2, \ldots, z_T)$，最终答案为$\hat{y} = g(z_T)$。

更形式化地，设$p_\theta(y|x)$为标准语言模型在输入x下生成答案y的概率。CoT通过引入中间变量z，将后验分解为：

$$
p_\theta(y|x) = \sum_{z \in \mathcal{Z}} p_\theta(z|x) p_\theta(y|z, x)
$$

在测试时，CoT通过自一致性（self-consistency，Wang et al., 2022）方法选择最优推理路径。给定m条采样得到的推理链$\{z^{(i)}\}_{i=1}^m$，自一致性答案为：

$$
\hat{y} = \arg\max_{y \in \mathcal{Y}} \sum_{i=1}^m \mathbb{I}[g(z^{(i)}_T) = y]
$$

这等价于在推理链的终结状态上进行多数投票。从理论角度，自一致性可被理解为一种蒙特卡洛近似，用于估计后验分布$p_\theta(y|x)$的众数。

**Zero-shot CoT与指令微调。** 无需在提示中提供示例，简单的"Let's think step by step"提示即可触发模型的链式推理能力（Kojima et al., 2022）。这一现象可以从信息论角度解释：CoT提示实际上增加了模型的先验概率质量，使得模型更倾向于选择那些具有清晰中间步骤的解码路径。指令微调（instruction tuning）进一步将CoT能力内化到模型参数中——通过监督学习让模型直接生成高质量的推理链。

**CoT的局限性。** 标准CoT生成单一的线性推理链，存在两个根本性问题：（1）一旦某一步推理出现错误，后续步骤将持续建立在错误前提之上，产生级联错误；（2）对于需要探索多条可能路径的复杂问题（如多步数学证明题），单条链无法回溯和比较。Tree-of-Thought和Graph-of-Thought正是针对这些局限性提出的。

### 第2.2节 Tree-of-Thought：树状推理与系统化探索

**核心思想。** Yao等人（2023）提出的Tree-of-Thought（ToT）框架将语言模型的推理过程建模为一棵树$T = (V, E)$，其中每个节点$v \in V$代表一个"思维"（thought，即一段中间推理文本），每条边$e \in E$代表从父思维到子思维的推导关系。

**算法框架。** ToT框架包含四个关键组件：

1. **思维分解（Thought Decomposition）。** 将解决问题所需的推理过程分解为k个步骤（或"深度"），每个步骤的思维候选集合记为$\mathcal{C}_i$。对于24点游戏，第一步选择两个数字进行运算，第二步选择运算结果与剩余数字继续运算，等等。

2. **思维生成（Thought Generation）。** 在每个节点$v$处，使用语言模型生成其子节点的候选思维集合。形式化地，给定当前推理状态$s_v$（包含历史思维序列），生成函数为：

$$
c \sim p_\theta(\cdot | s_v, \text{prompt}_{\text{generate}})
$$

其中$\text{prompt}_{\text{generate}}$引导模型生成在当前状态下合理的下一步推理。

3. **状态评估（State Evaluation）。** 这是ToT与CoT最关键的区别——ToT在每个中间思维节点上部署一个启发式评估函数$f: \mathcal{S} \to \mathbb{R}$，评估当前部分推理状态$s_v$的"promising程度"。评估可以通过以下方式实现：
   - **独立评估**：$f_{\text{ind}}(s_v) = p_\theta(\text{``promising''} | s_v)$，即让模型独立判断当前状态是否有望导向正确答案
   - **搜索终态评估**：$f_{\text{final}}(s_v) = p_\theta(\text{answer} | s_v)$，即让模型从当前状态出发生成完整答案的概率
   - **过程奖励模型（PRM）**：$f_{\text{PRM}}(s_v) = r_\phi(s_v)$，其中$r_\phi$是一个经过训练的奖励模型

4. **搜索算法（Search Algorithm）。** ToT支持多种搜索策略：
   - **广度优先搜索（BFS）**：在每个深度层级，生成所有候选思维，选择评分最高的前b个进入下一层。时间复杂度为$O(b^d)$，其中d为搜索深度。
   - **深度优先搜索（DFS）**：沿最高分分支深入，若达到死胡同则回溯。在最好的情况下复杂度为$O(bd)$，最坏情况下仍为$O(b^d)$。
   - **束搜索（Beam Search）**：在每个深度保留前k个思维状态。复杂度为$O(k \cdot b \cdot d)$。

**数学分析。** ToT的一个关键性质是，它通过树结构显式地对推理过程进行价值函数的动态规划近似。定义$V^*(s)$为从状态s出发能获得的期望最优最终得分，则有Bellman方程：

$$
V^*(s) = \max_{c \in \mathcal{C}(s)} \left[ f(s \circ c) + \gamma \cdot \mathbb{E}_{s' \sim p_\theta(\cdot|s \circ c)}[V^*(s')] \right]
$$

其中$\mathcal{C}(s)$为状态s的候选思维集合，$s \circ c$表示将思维c附加到状态s，$\gamma \in [0, 1]$为折扣因子。ToT的搜索过程是对这个最优Bellman方程的有限深度近似。

**实验验证。** Yao等人（2023）在Game of 24、创意写作和跨词逻辑推理任务上验证了ToT的有效性。在24点游戏中，CoT的通过率为4.8%，ToT+BFS的通过率提升至74%（b=5），DFS的通过率为74%，束搜索（beam width=1）的通过率为7%。这证明了系统化探索相比贪婪采样在复杂推理任务上的显著优势。

### 第2.3节 Graph-of-Thought：图状推理与思维组合

**超越树结构的动机。** Tree-of-Thought虽然引入了搜索机制，但其树结构限制了思维之间的连接方式——每个思维只能有一个父节点。Besta et al.（2023）提出的Graph-of-Thought（GoT）进一步将推理结构推广为有向无环图（DAG）$G = (V, E)$，允许一个思维被多个父思维引用，也允许同一父思维产生多个独立的分支思维后再进行融合。需要区分的是，Long（2023）的工作主要讨论语言模型引导的树状推理框架，并非 Graph-of-Thought 本身。

**图状推理的形式化。** GoT定义了以下操作，使语言模型能够对思维图进行变换：

1. **Thought Decomposition（思维分解）**：$op_{\text{decomp}}: v \to \{v_1, v_2, \ldots, v_k\}$——将思维v分解为k个子思维。在数学证明中，这相当于将一个复杂断言分解为更基本的子断言。

2. **Thought Generation（思维生成）**：$op_{\text{generate}}: \{v_1, \ldots, v_k\} \to v'$——从多个思维生成一个新的综合思维。这实现了不同推理分支的信息融合。

3. **Thought Scoring（思维评分）**：$op_{\text{score}}: v \to \mathbb{R}$——对单个思维的质量进行评分。

4. **Thought Pruning（思维剪枝）**：$op_{\text{prune}}: G \to G'$——移除图中评分低于阈值的思维节点及其关联边。

5. **Thought Aggregation（思维聚合）**：$op_{\text{agg}}: \{v_1, \ldots, v_k\} \to v^*$——从多个候选思维中选择最优思维。

GoT的推理过程可以表示为一系列图变换操作的序列：

$$
G_0 \xrightarrow{op_1} G_1 \xrightarrow{op_2} \cdots \xrightarrow{op_T} G_T
$$

其中$G_0$的根节点包含问题描述，$G_T$的终态节点包含最终答案。

**思维聚合的数学建模。** 最关键的创新是Thought Aggregation操作。设当前有m个候选思维$v_1, \ldots, v_m$，每个思维可以关联到一个向量表示$h_i = \text{LLM}_{\text{encoder}}(v_i) \in \mathbb{R}^d$。聚合操作可以形式化为：

$$
v^* = \text{LLM}_{\text{decoder}}\left(\text{AttentionPooling}(\{h_1, \ldots, h_m\})\right)
$$

其中$\text{AttentionPooling}$使用注意力机制对候选思维进行加权融合：

$$
\alpha_i = \frac{\exp(w^\top h_i)}{\sum_{j=1}^m \exp(w^\top h_j)}, \quad c = \sum_{i=1}^m \alpha_i h_i
$$

这允许模型根据任务相关性动态地决定每个候选思维的权重，而非简单地取多数投票（如ToT）或平均。

**与MCTS的关系。** GoT可以看作是对Monte Carlo Tree Search（MCTS）在语言推理领域的推广。在MCTS中，每个节点代表一个游戏状态，而GoT中每个节点代表一个思维状态。GoT的聚合操作类似于MCTS中的backpropagation，将叶节点的价值信息沿边反向传播到根节点。不同的是，GoT中的"状态"是结构化的文本（思维），而非离散的游戏棋盘。

### 第2.4节 推理能力的Scaling与涌现

除了prompting范式，推理能力的另一个关键维度是模型规模。Wei等人（2022）发现，当模型参数量超过一定阈值（约100B）时，模型会"涌现"出在较小模型上不具备的CoT推理能力。这种涌现行为可以从以下角度理解：

**神经路径假说。** 大规模Transformer中，不同注意力头和不同层对应着不同的"神经功能模块"。在小型模型中，这些模块是纠缠的（entangled），无法独立执行特定功能；在大规模模型中，模块化结构逐渐清晰，某些注意力头专门负责"回溯"、"验证"或"规划"等推理子任务。

**计算图的可表达性。** 从理论上，深度$L$、宽度$d$的Transformer可以模拟任意$L$层深、宽度为$O(d)$的布尔电路。当$L \cdot d$超过某个阈值时，电路恰好能够表示实现链式推理所需的最小深度。这一阈值与模型的隐式推理链长度正相关。

**推测解码（Speculative Decoding）与推理加速。** 在推理阶段，可以使用较小的草稿模型（draft model）生成候选token序列，然后由目标大模型并行验证。设草稿模型的接受率为$r$，目标模型的单步解码时间为$t_{\text{large}}$，草稿模型为$t_{\text{small}}$，则期望解码时间为$t_{\text{small}} + (1-r) \cdot t_{\text{large}}$。当$r$较高时，整体加速显著。

---

## 第3节 代码大模型：从文本生成到程序合成

代码大模型（Code LLMs）是NLP精细化在特定领域的重要体现。与通用语言模型不同，代码大模型需要理解形式化语法、类型系统、API文档以及代码的执行语义。Codex、GitHub Copilot和StarCoder代表了这一领域的三个里程碑式工作，分别从模型能力、产品化和开放生态三个维度推动了代码智能的发展。

### 第3.1节 OpenAI Codex：代码生成的GPT-3微调

**模型架构与训练。** Codex（Chen et al., 2021）是在 GPT 系列模型上做代码领域微调的产物；公开报告的最大变体约 **12B**（如 Codex-12B），**并非** 175B GPT-3 的直接微调版。其架构为 Transformer decoder，训练数据与目标针对代码生成优化。

训练数据来源于公开的GitHub代码仓库，经过严格的过滤和质量控制。过滤标准包括：
- 使用基于启发式的恶意代码检测与过滤
- 基于文件扩展名过滤（仅保留.py、.js、.java、.c、.cpp、.go、.rs等）
- 自动语法解析过滤：使用ANTLR或tree-sitter解析代码AST，丢弃解析失败的文件
- 长度过滤：移除过长文件（>1MB）和过短文件（<100字节）
- 自动PII（个人可识别信息）检测与移除

最终训练数据集包含约159GB的代码，涵盖GitHub上排名前250万的热门仓库。Codex的训练分为两个阶段：

**阶段一：领域自适应预训练（Domain-Adaptive Pretraining, DAPT）。** 在原始GPT-3 checkpoint的基础上，使用过滤后的代码数据继续预训练$L_2$个token。训练目标为标准语言建模目标：

$$
\mathcal{L}_{\text{LM}} = -\sum_{t=1}^{T} \log p_\theta(x_t | x_{<t})
$$

其中$x_{<t}$表示位置t之前的token序列。这一阶段使模型学习代码的统计规律——语法模式、API使用习惯、命名约定等。

**阶段二：带有解决方案的微调（Solution-Tuned）。** 在DAPT之后，进一步使用编程竞赛数据（如Codeforces、AtCoder）进行微调。每条样本包含问题描述和参考解决方案对$(p, s)$。使用标准语言建模损失，但输入中额外加入了问题描述的编码：

$$
\mathcal{L}_{\text{ST}} = -\sum_{t=1}^{|s|} \log p_\theta(s_t | p, s_{<t})
$$

**HumanEval基准与pass@k指标。** 为了评估代码生成质量，Chen等人（2021）构建了HumanEval基准——164个手写Python编程问题，每个问题包含函数签名、文档字符串和单元测试。评估使用pass@k指标：

$$
\text{pass}@k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
$$

其中n为从模型中采样的总候选数，c为通过所有单元测试的候选数。当$k=c$时，$\text{pass}@k = 1 - \prod_{i=1}^k \frac{n-c-i+1}{n-i+1}$。这衡量了在k次尝试中至少生成一个正确答案的概率。

Codex在HumanEval上达到28.8%的pass@1和71.2%的pass@100。相比之下，未经过代码微调的GPT-3仅达到11.4%的pass@1。值得注意的是，Codex在涉及库使用的任务（如使用numpy、pytorch）上表现显著优于纯算法任务，这是因为训练数据中包含了大量使用这些库的实际代码。

### 第3.2节 GitHub Copilot：AI结对编程助手

**系统架构。** GitHub Copilot是Codex的商业化产品，将代码生成能力集成到VS Code、JetBrains等主流IDE中。其核心架构可以分解为三个层次：

**层次一：上下文检索层。** Copilot在IDE中监听用户操作，实时构建一个包含以下信息的丰富上下文（context）：
- 当前文件内容（full file content）
- 光标前后各$N_c$行（通常$N_c = 6000$个字符）
- 同一项目中其他相关文件的内容（通过文件路径相似度、导入关系等启发式规则选取，最多~20个文件）
- 文件路径信息

上下文的总长度通常为8K-32K个字符（取决于使用的模型版本）。

**层次二：填充物化（Fill-in-the-Middle）。** 传统代码补全模型仅从左到右生成，即$p(x_{\text{cursor}+1}, \ldots, x_{\text{end}} | x_{\text{start}}, \ldots, x_{\text{cursor}})$。然而，实际编程中用户经常需要修改已有代码的中间部分。Copilot采用FIM（Fill-In-the-Middle）训练范式（Bavarian et al., 2022），将输入构造成：

$$
x_{\text{FIM}} = [\text{pre}]\langle \text{FIM\_MID} \rangle[\text{suf}]\langle \text{FIM\_SUF} \rangle
$$

模型学习在$\langle \text{FIM\_MID} \rangle$位置生成被删除的代码片段。具体地，训练目标为：

$$
\mathcal{L}_{\text{FIM}} = -\sum_{t=\text{start}}^{\text{end}} \log p_\theta(x_t | x_{\text{FIM}, <t})
$$

在推理时，将用户代码拆分为前缀（pre）和后缀（suf），中间留空，模型填充$\langle \text{FIM\_MID} \rangle$部分。FIM使Copilot在代码修改和重构场景下的准确率提升了约15-20个百分点。

**层次三：延迟优化与建议队列。** IDE环境对响应延迟有严格要求（<200ms感知阈值）。Copilot使用推测解码（speculative decoding）来加速推理：一个轻量级的草稿模型（~1B参数）在本地运行生成候选token，同时大模型在服务器端并行验证。若草稿模型的输出被大模型接受，则节省了解码时间；若被拒绝，则使用大模型的输出。

Copilot还维护一个建议队列（suggestion queue），将多个候选补全按置信度排序，用户可以按Tab键在不同候选之间切换。建议队列的置信度通过模型的输出概率和长度归一化Perplexity来排序：

$$
\text{score}(y_{1:T}) = \frac{1}{T} \sum_{t=1}^T \log p_\theta(y_t | y_{<t})
$$

即平均对数概率（与困惑度呈负相关：$\mathrm{PPL}=\exp(-\mathrm{score})$）。分数越高（平均对数概率越高、困惑度越低），建议越可靠。

**性能数据。** GitHub官方报告（2023年），Copilot被超过150万开发者和37000家组织使用，用户接受率约为30%。在Python开发中，Copilot辅助编写的代码中约46%的内容由Copilot生成。接受Copilot建议的开发者完成任务速度平均提升55%。

### 第3.3节 StarCoder与BigCode：开放源代码大模型

**训练数据：The Stack。** StarCoder（Li et al., 2023）由BigCode项目（Hugging Face与ServiceNow合作）开发，最大的变体StarCoderBase拥有15.5B参数，在The Stack数据集上进行训练。The Stack是一个精心构建的大规模代码数据集，包含约1.2万亿token、超过300种编程语言。

The Stack的数据处理流程体现了开源社区对数据治理的关注：

1. **源数据获取：** 从GitHub Archive的2021年2月快照中提取公开可用的Git仓库
2. **去重：** 使用MinHash + LSH（局部敏感哈希）进行文件级和仓库级去重，移除完全重复和高度相似的代码
3. **个人可识别信息（PII）移除：** 使用正则表达式和Microsoft的Presidio工具检测电子邮件、密钥、姓名等PII，并进行替换或移除
4. ** opted-out 机制：** 允许开发者将自己的仓库从数据集中移除。最终有约10%的原始数据因开发者请求而被移除
5. **低质量过滤：** 使用基于困惑度的分类器过滤低质量代码，基于行数过滤极端长/短文件

**模型架构与训练细节。** StarCoderBase基于 transformers 库的GPT-2架构（decoder-only），关键配置如下：

- 参数量：15.5B
- 层数：40层
- 隐藏维度：6144
- 注意力头数：48
- 序列长度：8192 token（比标准代码模型更长）
- 位置编码：可学习的绝对位置嵌入（learned absolute positional embeddings），最大位置 8192
- 注意力结构：Multi-Query Attention（MQA）——所有注意力头共享同一组键值投影，显著降低了推理时的显存占用
- 词汇表：49,152 token的BPE词汇表，针对多种编程语言联合训练

训练使用900GB数据（包括The Stack的86%和约400GB的精选自然语言数据），在256张A100 GPU（80GB）上训练约24天。

**StarCoder：指令微调变体。** 在StarCoderBase的基础上，Li et al. (2023)进一步使用自instruction数据集进行指令微调，得到StarCoder。指令数据集包含约22,000条指令-响应对，涵盖代码生成、代码补全、代码翻译、代码解释、调试等多种任务。指令数据通过以下方式构建：
- 人工编写的种子指令（约200条）
- 基于模板的程序化生成（约5000条）
- 使用Instructor模型自动生成的指令-代码对（约17,000条）

**Fill-In-the-Middle能力。** 与Copilot类似，StarCoder原生支持FIM训练。训练时以50%的概率使用前缀-后缀-中间（PSM）格式，50%的概率使用标准的从左到右格式。FIM格式的输入构造为：

$$
\langle \text{fim\_prefix} \rangle \text{prefix} \langle \text{fim\_suffix} \rangle \text{suffix} \langle \text{fim\_middle} \rangle \text{middle} \langle \text{eos} \rangle
$$

模型需要预测$\langle \text{fim\_middle} \rangle$之后的内容。

**评测结果。** StarCoder在HumanEval上达到33.6%的pass@1，在HumanEval-Infilling（填充评测）上达到18.0%的pass@1。在多语言HumanEval（包括Python、JavaScript、Java、C++等语言）上，StarCoder优于同规模的所有开源模型。在MultiPL-E（Cassano et al., 2023）多语言评测上，StarCoder在18种语言中的15种达到最优。

**Open RAIL-M许可证与数据治理。** StarCoder采用Open Responsible AI Model（Open RAIL-M）许可证，允许商业使用但设置了使用限制（如禁止生成恶意代码）。这一许可框架平衡了开放性与安全性的需求。GitHub Copilot的数据使用因Copilot的"黑箱"特性而备受争议，而StarCoder的透明数据治理为社区提供了可供审计的基准。

### 第3.4节 代码大模型的技术挑战与未来方向

**代码理解 vs. 代码生成。** 当前代码大模型主要聚焦于代码生成任务（给定上下文预测下一个token），但代码理解任务（如代码搜索、漏洞检测、代码摘要）同样重要。代码理解要求模型建立代码的语义表示，而不仅仅是学习表面的token统计规律。

**执行反馈与形式化验证。** 代码与自然语言的根本区别在于代码是可执行的。AlphaCode（Li et al., 2022）利用单元测试作为执行反馈：生成大量候选解，通过测试用例筛选有效解。进一步地，将形式化验证工具（如SMT solver）集成到生成-验证循环中，可以显著提高生成代码的正确性保证。

**多模态代码理解。** 代码不仅仅是文本——它包含结构信息（AST）、控制流信息（CFG）、数据流信息（DFG）以及UI渲染信息（前端代码的屏幕截图）。将代码的多种模态信息统一到预训练框架中，是未来的重要方向。TransCoder-ST（Rozière et al., 2023）等多语言代码翻译模型展示了结构信息在跨语言代码迁移中的价值。

**小模型与端侧代码生成。** 目前最强大的代码模型（如GPT-4级别）需要大型GPU集群运行，无法部署在本地开发机上。CodeLlama（Rozière et al., 2023）和Phi-2（Microsoft, 2023）等小模型在特定代码任务上接近大模型水平，展示了小模型在代码领域的潜力。模型蒸馏、量化（INT4/INT8）和投机解码是缩小模型的关键技术。

---

## 第4节 多语言大模型：跨越语言巴别塔

人类语言多样性是NLP面临的核心挑战之一——全球现存约7000种语言，但其中不到2%拥有足够的数字资源用于训练大规模神经网络模型。多语言大语言模型（Multilingual LLMs, mLLMs）的目标是构建单一模型，使其能够理解、生成和翻译多种语言，同时实现零样本跨语言迁移（zero-shot cross-lingual transfer）。本节将从表示学习、架构设计和训练策略三个层面深入分析mBERT、XLM-R、mT5和PaLM的技术细节。

### 第4.1节 多语言预训练的表示空间理论

**表示对齐的几何视角。** 在多语言预训练中，一个核心理论问题是：不同语言的表示在共享空间中是否被对齐？Luo et al. (2020)的形式化分析表明，当使用共享Transformer权重处理多种语言时，模型倾向于学习一种"语言无关"的语义表示空间。更精确地，对于语义等价的句子对$(x^{(i)}, x^{(j)})$（分别来自语言$\ell_i$和$\ell_j$），模型产生的表示满足：

$$
\|h_{\ell_i}(x^{(i)}) - h_{\ell_j}(x^{(j)})\| < \epsilon
$$

其中$\epsilon$随训练步数单调递减。这种对齐可以通过以下机制实现：

**机制一：共享词表与联合BPE。** 所有语言共享一个子词词汇表$\mathcal{V}$，通过多语言SentencePiece或BPE联合训练获得。词汇表的融合迫使不同语言的相似概念共享子词单元。例如，英语的"information"和法语的"information"共享相同的token序列，自然映射到相似的表示向量。

**机制二：反向翻译（Back-Translation）。** 在训练数据中，将翻译对$(x^{(i)}, x^{(j)})$同时以$(x^{(i)}, x^{(j)})$和$(x^{(j)}, x^{(i)})$的形式出现（即双向翻译对），模型被迫学习对称的语义映射。

**机制三：语言间负采样。** 在对比学习框架下，将同一语义的不同语言表述作为正样本，不同语义的跨语言表述作为负样本。对比损失函数为：

$$
\mathcal{L}_{\text{contra}} = -\log \frac{\exp(\text{sim}(h_{\ell_i}, h_{\ell_j}) / \tau)}{\sum_{k \in \mathcal{N}} \exp(\text{sim}(h_{\ell_i}, h_{\ell_k}) / \tau)}
$$

其中$\tau$为温度参数，$\mathcal{N}$为负样本集合。

**Hubness问题。** 多语言表示空间中存在一个被称为"Hubness"的现象（Dinu et al., 2014）：某些语言（通常是资源丰富的语言，如英语）的表示向量成为k近邻搜索中的"Hub"——它们作为大量其他语言向量的近邻出现，但自身不一定语义相近。这导致跨语言检索的准确率下降。缓解方法包括：表示归一化（length normalization）、 Hubness纠正（使用 Mutual Proximity 方法）以及重要性加权。

### 第4.2节 mBERT：多语言BERT的突破与局限

**模型架构。** mBERT（Devlin et al., 2018）是BERT-base架构的多语言扩展，使用104种语言的维基百科数据进行预训练。具体配置：

- 参数总量：110M（与单语言BERT-base相同）
- 层数：12层Transformer encoder
- 隐藏维度：768
- 注意力头数：12
- 最大序列长度：512 token
- 词汇表：119,547个WordPiece token（跨104种语言共享）

**训练数据与课程学习。** mBERT使用各语言维基百科的dumps进行训练，总token数约为12.7B（127亿）。训练采用采样策略$p(\ell) \propto n_\ell^{0.7}$，其中$n_\ell$为语言$\ell$的token数。指数0.7 < 1的设计实现了对低资源语言的过采样（oversampling），避免高资源语言完全主导训练梯度。这一采样权重可被视为一种简单的课程学习（curriculum learning）：模型先接触多样化的低资源语言样本，再逐渐过渡到高资源语言的高频模式。

**跨语言迁移机制。** mBERT的跨语言能力并非显式设计（没有使用翻译对或平行语料），而是从共享词汇表和联合训练中"emergent"的。实验表明，mBERT在零样本跨语言设置下的表现令人惊讶地好——在XNLI（Conneau et al., 2018）基准上，mBERT在15种语言上的平均准确率达到66.4%，仅比单语言BERT平均低约5个百分点。

**结构对齐与语言类型学距离。** 属性级跨语言迁移的性能高度依赖于源语言和目标语言之间的类型学距离（typological distance）。例如，从英语迁移到西班牙语（同为SVO语序、屈折变化丰富）的效果远好于从英语迁移到日语（SOV语序、话题优先）。Pires等人（2019）发现，共享词汇字符集（如拉丁字母 vs. 西里尔字母）是比词汇重叠率更强的迁移预测因子。这表明mBERT的跨语言能力在字符层面就已经部分对齐。

**mBERT的局限性：**
1. **词汇表瓶颈：** mBERT的WordPiece词汇表主要是为拉丁字母语言设计的，对中文、日文、阿拉伯文等非拉丁文字的支持有限。没有专门针对中文的tokenization，中文句子被逐字切分，丢失了词级语义。
2. **资源不平等：** 训练数据中英语占约42%，中文占约11%，而许多低资源语言（如斯瓦希里语、约鲁巴语）的数据量不到百万token，导致模型在这些语言上的表示质量显著下降。
3. **容量限制：** 110M参数对于104种语言来说，每语言的参数容量极其有限。模型需要在所有语言之间共享参数，导致单语言能力被稀释。

### 第4.3节 XLM-R：基于RoBERTa的大规模多语言预训练

**改进方向。** XLM-R（Conneau et al., 2019/2020）针对mBERT的局限性进行了系统改进：
1. 使用更大的模型（RoBERTa-large，355M参数）
2. 使用CC-100语料——覆盖约100种语言、约2.5TB文本
3. 使用SentencePiece而非WordPiece进行tokenization
4. 移除下一句预测（NSP）任务，仅使用MLM目标
5. 更大的批次（8192）和更长的训练步数

**掩码语言建模的多语言变体。** XLM-R（XLM-RoBERTa，Conneau et al., 2020）基于RoBERTa架构，仅使用Masked Language Modeling（MLM）目标。MLM目标为：

$$
\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{\mathcal{D}} \left[ \sum_{x_i \in \mathcal{M}(x)} \log p_\theta(x_i | x_{\setminus \mathcal{M}}) \right]
$$

其中$\mathcal{M}(x)$为随机掩码的token位置集合（掩码比例15%），$x_{\setminus \mathcal{M}}$为其余token。掩码策略使用：
- 80%概率替换为`[MASK]`
- 10%概率替换为随机token
- 10%概率保持不变

这种混合策略在微调阶段不存在预训练-微调分布偏移问题。

**结果。** XLM-R在XNLI上达到80.9%的零样本准确率（相比mBERT提升14.5个百分点），在NER、POS tagging等序列标注任务上也大幅超越mBERT。特别地，XLM-R在低资源语言上的提升最为显著——这表明模型容量和数据规模是跨语言迁移的关键瓶颈。

### 第4.4节 mT5：多语言文本到文本转移模型

**文本到文本范式。** mT5（Xue et al., 2020）是T5（Text-to-Text Transfer Transformer）的多语言扩展。与BERT系列的"理解-生成"双阶段范式不同，T5将所有NLP任务统一为文本到文本的生成问题——给定输入文本，生成输出文本。这一范式天然支持多语言场景，无需区分编码器和解码器架构。

**模型规模系列。** mT5 提供从 small 到 xxl 的完整规模系列（参数量约为）：
- mT5-small：约 300M 参数
- mT5-base：约 580M 参数
- mT5-large：约 1.2B 参数
- mT5-xl：约 3.7B 参数
- mT5-xxl：约 13B 参数

mT5 使用与 T5 一致的标准 encoder–decoder **稠密自注意力**（并非 Sparse Attention）；复杂度仍为序列长度的二次方，规模优势来自参数与多语言语料，而非稀疏注意力降复杂度。

**多语言词汇表。** mT5使用SentencePiece的Unigram模型，词汇表大小为250,100 token。相比XLM-R的250K词汇表，mT5的词汇表更加面向生成任务优化——包含更多完整词的token，减少了生成时的碎片化（fragmentation）。

**mC4语料。** mT5使用基于Common Crawl的多语言语料mC4进行预训练。mC4包含101种语言，总token数约6.3万亿。获取过程为：
1. 从Common Crawl（2020年4月快照）中提取文本
2. 使用cld3（Google Compact Language Detector）检测语言
3. 过滤低质量文本（如超短句子、非自然语言文本）
4. 按语言采样权重$p(\ell) \propto n_\ell^{0.3}$进行训练采样

**泰勒序列展开与深度扩展。** mT5-xxl的约13B参数使得模型在低资源语言上的表现显著提升。在XGLUE（Liang et al., 2020）基准上，mT5-xxl在9种语言的文本分类任务上平均达到80.5 F1，比XLM-R-large提升约5 F1点。mT5还展示了在零样本跨语言摘要和问答任务上的能力——这些生成任务在BERT架构下难以直接处理。

### 第4.5节 PaLM：Pathways架构与大规模多语言统一模型

**Pathways 与 PaLM。** PaLM（Chowdhery et al., 2022）是 Google 在 Pathways 系统上训练的**稠密解码器**语言模型。Pathways 主要是大规模训练/调度与系统软件栈，**不等于**模型内部的稀疏专家路由（MoE）。PaLM 本身是标准稠密 Transformer，而非稀疏路由混合专家。PaLM 的核心架构选择包括：

- **模型规模：** PaLM提供了540B、62B和8B三种规模。540B版本使用 Pathways系统在6144个TPU v4芯片上训练，总计算量为约$2.5 \times 10^{24}$ FLOPs。
- **前馈网络变体：** 使用SwiGLU激活函数而非标准ReLU或GeLU。SwiGLU定义为：
  
  $$
  \text{SwiGLU}(x, W, V, b, c) = \text{SiLU}(xW + b) \otimes (xV + c)
  $$
  
  其中$\text{SiLU}(x) = x \cdot \sigma(x)$，$\sigma$为sigmoid函数。SwiGLU结合了门控机制和非线性激活，在保持计算效率的同时提升了模型的表达力。
  
- **并行层（Parallel Layers）：** 标准Transformer将注意力层和前馈网络串行堆叠（先注意力后FFN），PaLM采用并行排列（attention和FFN的输入均为同一层归一化后的隐藏状态），然后通过残差连接相加：
  
  $$
  x' = x + \text{Attention}(\text{LN}(x)) + \text{FFN}(\text{LN}(x))
  $$
  
  这种排列方式使训练速度提升了约15%，可能是因为FFN和Attention的并行计算更好地利用了GPU的Tensor Core。

**多语言训练数据。** PaLM 约在 780B tokens 的混合语料上训练，涵盖英语（约 78%）、代码与多语言自然文本（法语、西班牙语、德语、日语、中文等）。具体组成：
- 社交媒体对话：55%
- 过滤后的网页：27%
- 书籍：13%
- 维基百科：4%
- 新闻：1%

对于非英语数据，PaLM没有使用显式的语言采样权重，而是让自然数据分布主导。由于英语在互联网内容中占据主导地位，模型自然以英语为主，但仍保持了相当的多语言能力。

**少样本多语言能力。** PaLM在少样本设置下展示了令人印象深刻的多语言推理能力。在MGSM（Shi et al., 2022）多语言数学推理基准上，PaLM 540B在8种语言（包括英语、日语、孟加拉语、马拉地语等）上使用8-shot prompting的准确率分别达到61.2%（英语）、40.6%（日语）、36.3%（孟加拉语）等。这些结果证明了超大参数模型在低资源语言上的知识迁移能力。

**思维链与多语言推理。** PaLM 540B在少样本CoT设置下的表现尤为突出。按PaLM论文报告，在GSM8K数学推理任务上，8-shot CoT达到约58%准确率，8-shot CoT+Self-Consistency可进一步提升至约74%。有趣的是，当使用非英语语言（如日语）进行CoT推理时，模型的性能下降但并未崩溃——这表明推理能力在一定程度上是语言无关的，模型可以利用不同语言中的CoT模式。

**PaLM的后续演进：PaLM 2与Gemini。** PaLM 2（Google, 2023）在PaLM的基础上改进了数据混合（data mixture）和模型架构，在HumanEval、MATH等代码和推理基准上大幅超越PaLM。Gemini系列则进一步将PaLM 2的多语言能力扩展到了多模态（文本+图像+视频），代表了Google在通用人工智能方向上的最新进展。

---

## 第5节 RAG：检索增强生成的原理与实践

大语言模型的核心瓶颈之一是知识截止——模型的知识冻结在训练数据的时间戳，无法获取训练之后的实时信息，且容易产生事实性幻觉。检索增强生成（Retrieval-Augmented Generation, RAG，Lewis et al., 2020）通过从外部知识库检索相关信息并注入生成过程，有效缓解了这一瓶颈。RAG系统的核心挑战在于检索的精确度和生成的一致性之间的平衡。

### 第5.1节 RAG的形式化框架

**概率图模型视角。** RAG可以被理解为一种模块化生成模型。给定查询q，RAG的生成过程为：

$$
p_{\text{RAG}}(y|q) = \sum_{z \in \text{Top-}k(D)} p_\theta(y|z, q) \cdot p(z|q)
$$

其中：
- $z$为从知识库$\mathcal{D}$中检索到的文档
- $p(z|q)$为检索模型给出的相关性概率
- $p_\theta(y|z, q)$为生成模型基于检索文档和查询生成回答的条件概率

在推断时，Top-k检索将求和转化为：

$$
p_{\text{RAG}}(y|q) \approx \sum_{i=1}^k p_\theta(y|z^{(i)}, q) \cdot \frac{\exp(\text{sim}(q, z^{(i)}) / \tau)}{\sum_{j=1}^k \exp(\text{sim}(q, z^{(j)}) / \tau)}
$$

其中$\text{sim}(\cdot, \cdot)$为检索相似度函数，$\tau$为温度参数。

**RAG的训练目标。** Lewis等人（2020）提出了两种训练变体：
- **RAG-Sequence模型：** 生成概率对所有检索文档取平均，即$p(y|q) = \sum_z p(y|z, q)p(z|q)$。所有检索文档的生成结果共享相同的概率质量。
- **RAG-Token模型：** 生成过程的每个token可以使用不同的检索文档，即$p(y|q) = \prod_{t=1}^T \sum_z p(y_t|z, y_{<t}, q)p(z|q)$。这允许模型在不同生成步骤关注不同文档的不同方面。

### 第5.2节 密集检索（Dense Retrieval）

**双塔架构。** 现代RAG系统普遍采用双塔（bi-encoder）架构进行高效检索。查询编码器$f_q: \mathcal{Q} \to \mathbb{R}^d$和文档编码器$f_d: \mathcal{D} \to \mathbb{R}^d$分别将查询和文档映射到d维向量空间。相似度使用内积或余弦相似度：

$$
\text{sim}(q, d) = f_q(q)^\top f_d(d)
$$

知识库中的文档在离线阶段预先编码为向量，构建向量索引（如使用FAISS或ScaNN）。在线检索时，仅需对查询进行一次编码，然后执行近似最近邻（ANN）搜索，检索时间复杂度为$O(\log |\mathcal{D}|)$而非$O(|\mathcal{D}|)$。

**DPR的对比学习训练。** DPR（Dense Passage Retrieval，Karpukhin et al., 2020）使用二部图对比损失训练双塔模型。每条训练样本包含一个查询$q$、一个正样本文档$d^+$和$n-1$个负样本文档$\{d^-\}_i$：

$$
\mathcal{L}_{\text{DPR}} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\exp(\text{sim}(q, d^+) / \tau) + \sum_{i=1}^{n-1} \exp(\text{sim}(q, d_i^-) / \tau)}
$$

负样本的来源包括：
- **In-batch negatives：** 同一训练batch中的其他正样本文档作为负样本
- **BM25 negatives：** 使用BM25检索得到的假正例（false positives）
- **Hard negatives：** 使用当前模型检索得到的高排名但非正确答案的文档

**ColBERT的细粒度交互。** 尽管双塔架构高效，但单向量表示丢失了查询与文档之间的细粒度词级匹配信息。ColBERT（Khattab & Zaharia, 2020）通过Late Interaction保留了更丰富的交互：

$$
\text{Sim}_{\text{ColBERT}}(Q, D) = \sum_{i \in Q} \max_{j \in D} (E_Q[i] \cdot E_D[j])
$$

其中$E_Q \in \mathbb{R}^{m \times d}$和$E_D \in \mathbb{R}^{n \times d}$分别为查询和文档的token级嵌入矩阵。MaxSim操作对查询的每个token在文档中找到最相似的token，然后求和。这等价于一个soft注意力机制，但仅在编码后的表示上计算，保持了检索效率。

ColBERTv2（Santhanam et al., 2022）进一步引入知识蒸馏和压缩，将ColBERT的存储需求从约100GB压缩到约2GB，同时保持检索质量。

**向量索引的量化与压缩。** 对于大规模知识库（亿级文档），精确ANN搜索的存储和计算成本仍然很高。常用的压缩技术包括：
- **Product Quantization (PQ)：** 将d维向量分割为m段，每段独立使用k-means量化。存储从d个浮点数降至m个码字索引（每个log2(k)比特）。
- **Scalar Quantization (SQ / INT8)：** 将浮点向量量化为INT8，压缩4倍，几乎无损。
- **OPQ（Optimized PQ）：** 在量化前对向量进行正交变换，减少段间相关性，提高量化精度。

### 第5.3节 稀疏检索（Sparse Retrieval）

**BM25的概率解释。** BM25（Robertson & Zaragoza, 2009）是传统信息检索的黄金标准。其评分函数为：

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

其中：
- $f(t, d)$为词项t在文档d中的词频
- $|d|$为文档d的长度
- $\text{avgdl}$为语料库的平均文档长度
- $k_1$和$b$为可调参数（通常$k_1 \in [1.2, 2.0], b = 0.75$）
- $\text{IDF}(t) = \log\frac{N - n(t) + 0.5}{n(t) + 0.5}$为逆文档频率，N为文档总数，$n(t)$为包含词项t的文档数

BM25可以看作是一个非参数化的词袋（bag-of-words）模型，其优势在于无需训练、可解释性强、推理速度快（使用倒排索引实现毫秒级检索）。缺点是无法捕捉语义相似度——同义词或释义不会被匹配。

**SPLADE：稀疏语义检索。** SPLADE（SParse Lexical AnD Expansion model，Formal et al., 2021）结合了稀疏检索的可解释性和密集检索的语义理解能力。SPLADE使用BERT架构，在[CLS] token上增加一个$|\mathcal{V}|$维的输出层（词汇表大小），通过Gumbel-Softmax或 top-k 阈值选择重要的词项扩展。

给定查询q，SPLADE输出一个稀疏向量$v_q \in \mathbb{R}^{|\mathcal{V}|}$，其中大部分元素为零。文档d同样编码为$v_d \in \mathbb{R}^{|\mathcal{V}|}$。相似度计算为：

$$
\text{sim}_{\text{SPLADE}}(q, d) = \sum_{t \in \mathcal{V}} \min(v_q[t], v_d[t])
$$

即两个稀疏向量的交集。这等价于扩展后的查询词项与文档词项的重叠度。

SPLADE的训练目标结合了MLM和排序损失：
- MLM损失确保模型学习词汇表级别的词项预测
- 排序损失（ListNet或Margin-MSE）确保检索质量

训练完成后，SPLADE在保持稀疏表示的同时实现了与ColBERT相当的检索质量，且索引大小与BM25相当（使用倒排索引存储）。

### 第5.4节 密集与稀疏检索的融合

**互补性分析。** 密集检索和稀疏检索各有优劣。在BEIR基准（Thakur et al., 2021）上：
- **BM25**在实体/关键词丰富的任务（如TREC-COVID、NFCorpus）上优于密集检索
- **DPR**在需要语义理解的问答任务（如HotpotQA、Natural Questions）上优于BM25
- 平均来看，单一方法在18个评测数据集上的nDCG@10为41.2（BM25）vs 40.9（DPR）

这种互补性催生了融合检索（Hybrid Retrieval）方法。

**线性融合。** 最简单的融合方式是加权求和：

$$
\text{score}_{\text{hybrid}}(q, d) = \alpha \cdot \text{score}_{\text{dense}}(q, d) + (1-\alpha) \cdot \text{score}_{\text{sparse}}(q, d)
$$

其中$\alpha \in [0, 1]$为可调超参数。在多个数据集上的最优$\alpha$值不同：对于需要精确关键词匹配的任务（如法律检索），$\alpha = 0.3-0.4$（偏向BM25）；对于需要语义理解的任务（如开放域问答），$\alpha = 0.6-0.7$（偏向DPR）。

**Reciprocal Rank Fusion (RRF)。** Cormack等人（2009）提出的RRF不依赖于分数归一化，直接使用排序位置：

$$
\text{RRF-Score}(q, d) = \sum_{r \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + \text{rank}_r(q, d)}
$$

其中k为常数（通常k=60），$\text{rank}_r(q, d)$为文档d在检索方法r下的排名位置。RRF的优势在于对不同的评分分布具有鲁棒性，无需手动调整分数范围。

**基于学习的融合。** 更先进的方法使用Learning-to-Rank (LTR)模型学习融合权重。给定查询-文档对$(q, d)$的特征向量$\phi(q, d) = [\text{score}_{\text{dense}}(q, d), \text{score}_{\text{sparse}}(q, d), \text{overlap}(q, d), |d|, \ldots]$，使用LambdaMART或RankNet学习排序函数：

$$
f_{\text{LTR}}(q, d) = w_1 \cdot \phi_1 + w_2 \cdot \phi_2 + \ldots
$$

**RAG与生成式检索的对比。** 近年来的生成式检索（Generative Retrieval，Tay et al., 2023）提出直接使用语言模型生成文档标识符（如文档标题或特殊token序列），而非检索文档向量。DIALOCI (BehnamGhader et al., 2023)等研究表明，生成式检索在封闭域任务上具有竞争力，但在开放域大规模检索中，检索-排序管线（retrieve-then-rerank）仍是最可靠的方法。

---

## 第6节 Agent与工具使用：从语言模型到行动者

大语言模型的本质是"文本补全器"——它们生成文本但不采取行动。Agent框架将LLM提升为能够感知环境、使用工具、执行多步计划和自我修正的自主系统。ReAct、Toolformer和AutoGPT代表了这一演进过程中的三个关键节点：推理-行动交替范式、工具使用能力的内化训练，以及全自主Agent框架。

### 第6.1节 ReAct：推理与行动的交替

**核心思想。** Yao等人（2022）提出的ReAct（Reason + Act）框架打破了"先推理后行动"的顺序范式，将推理过程和行动执行交织进行。在每一步，Agent交替执行：
1. **Thought（推理）：** 分析当前状态，推理下一步该做什么
2. **Action（行动）：** 选择一个具体工具/API并执行
3. **Observation（观察）：** 接收工具执行的结果，更新环境状态

这种交替循环可以用马尔可夫决策过程（MDP）的形式化描述：

$$
\pi(a_t | s_t) = p_\theta(a_t | \text{prompt}_{\text{ReAct}}(h_t))
$$

其中$h_t = (s_0, a_0, o_0, \text{thought}_1, a_1, o_1, \ldots, \text{thought}_{t-1}, a_{t-1}, o_{t-1})$为历史轨迹，$s_t$为当前状态，$a_t$为行动，$o_t$为观察结果。

**ReAct提示模板。** 在少样本设置下，ReAct通过手工编写的推理-行动轨迹作为in-context示例引导模型。标准模板如下：

```
Question: 一个存储了2L、浓度为3mol/L的溶液，需要稀释到1L，求最终浓度是多少？

Thought 1: 我需要使用稀释公式 C1 × V1 = C2 × V2。
Action 1: Calculator[C1=3, V1=2, V2=1]
Observation 1: C2 = 6 mol/L

Thought 2: 计算结果已经得出。
Action 2: Finish[6 mol/L]
```

相比于仅使用CoT的prompting：
```
Question: 一个存储了2L、浓度为3mol/L的溶液...
Answer: 使用稀释公式，C2 = C1 × V1 / V2 = 3 × 2 / 1 = 6 mol/L。
```

ReAct通过显式地使用计算器工具，避免了CoT中可能出现的计算错误。在HotpotQA和FEVER基准上，ReAct相比CoT分别提升了约3-6个百分点的准确率，同时在推理链的可解释性上也具有优势——每步推理都有对应的工具执行结果作为证据。

**ReAct的搜索视角。** 从搜索算法的角度看，ReAct执行的是在线深度优先搜索。在每个状态，模型选择最有希望的下一步行动，如果遇到死胡同（工具返回错误或无意义结果），模型可以在下一步推理中调整策略。这与ToT的离线搜索不同——ReAct不需要预先生成完整的搜索树，而是在线探索。

**形式化分析。** 设$s$为环境状态空间，$\mathcal{A}$为可用行动集合，$T(s'|s, a)$为状态转移概率（由外部工具决定），$R(s)$为奖励函数（由任务是否完成决定）。ReAct的目标是找到策略$\pi$最大化累积奖励：

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^T \gamma^t R(s_t)\right]
$$

其中$\tau = (s_0, a_0, o_0, \ldots, s_T)$为轨迹，$\gamma$为折扣因子。ReAct使用语言模型作为策略网络$\pi(a|s) \propto p_\theta(a|h)$，其中$h$为对话历史。由于环境的确定性（工具执行是确定性的），$T(s'|s, a)$为确定性函数，可以直接跟踪状态。

### 第6.2节 Toolformer：让语言模型学会使用工具

**动机。** ReAct依赖手工设计的prompt模板和示例，模型本身并未真正"学会"使用工具——它只是在in-context学习中模仿示例模式。Schick等人（2023）提出的Toolformer通过微调让模型自主决定何时使用工具、使用哪个工具以及如何使用工具。

**工具API设计。** Toolformer集成了四类工具：
1. **计算器（Calculator）：** 执行基本算术运算。输入格式：`[[ + 235 × 738 ]]`，输出：173430
2. **问答系统（Question Answering）：** 回答事实性问题。输入格式：`[[Q: What is the capital of France?]]`，输出：Paris
3. **搜索引擎（Search Engine）：** 返回Web搜索结果摘要
4. **翻译系统（Translation System）：** 执行机器翻译

每个工具被定义为API调用，输入输出均为纯文本。

**自监督训练信号。** 核心挑战是标注数据——需要为每个可能使用工具的位置标注工具调用。Toolformer的关键创新是利用LM本身来判断工具调用是否有益。具体流程为：

**步骤1：候选工具调用的生成。** 对于训练语料中的每个样本，在文本中随机选择位置插入工具调用。对于位置i，使用以下prompt生成候选API调用：

```
What API call is needed for the next word? Context: "... [previous text] ..."
Possible APIs: Calculator, QA, Search, Translate
API call format: [[API_NAME(input)]]
```

**步骤2：自监督过滤。** 对于每个候选调用$c$，评估其信息增益：
- 在插入工具调用后，模型生成的后续文本的困惑度（perplexity）降低
- 工具输出提供的信息是模型原本无法生成的

形式化地，对于候选调用$c$在位置i处，定义：

$$
\Delta \text{PPL}(c) = \text{PPL}(x_{>i} | x_{\leq i}) - \text{PPL}(x_{>i} | x_{\leq i}, c, \text{tool\_output}(c))
$$

若$\Delta \text{PPL}(c) > \tau$（$\tau$为阈值），则保留该工具调用；否则丢弃。

**步骤3：微调。** 使用筛选后的工具调用标注数据对基础语言模型进行监督微调。训练目标为标准语言建模目标：

$$
\mathcal{L}_{\text{TF}} = -\sum_{t=1}^{T} \log p_\theta(x_t | x_{<t})
$$

其中$x$包含原始文本和工具调用-输出的交替序列。

**实验结果。** 在零样本设置下，Toolformer-GPT2-large（774M参数）在数学推理（GSM8K）上的准确率从8.6%（基线）提升至42.5%，在问答（WebQuestions）上从10.5%提升至33.5%。Toolformer展示了通过自监督微调让模型自主学会工具使用的可行性，无需人工标注工具调用数据。

### 第6.3节 AutoGPT与全自主Agent框架

**系统架构。** AutoGPT（Toran Bruce, 2023）是首个广泛流行的自主Agent框架，它将GPT-4（或其他LLM）作为"大脑"，围绕其构建了完整的Agent循环：

1. **规划（Planning）：** 使用任务分解prompt将复杂目标分解为子任务列表
2. **优先级排序（Prioritization）：** 使用LLM对子任务按紧迫性和重要性排序
3. **执行（Execution）：** 依次执行子任务，使用工具获取信息或采取行动
4. **评估（Evaluation）：** 评估当前进展是否接近目标，决定是否调整计划
5. **记忆（Memory）：** 维护短期记忆（对话历史）和长期记忆（向量数据库）

**Memory系统。** AutoGPT的Memory系统使用向量数据库（如Pinecone、Weaviate）存储Agent的经验。每个经验条目包含：
- 文本描述（如"搜索了Python列表去重的最佳实践，发现使用set()是最快方法"）
- 嵌入向量（使用text-embedding-ada-002编码）
- 元数据（时间戳、相关性标签）

在规划阶段，Agent首先从Memory中检索与当前任务最相关的历史经验，作为规划的参考依据。这实现了跨任务的迁移学习——在先前任务中学到的经验可以指导新任务的解决。

**Agent Prompting的核心技巧。** AutoGPT的有效性高度依赖精心设计的系统提示。关键 prompting 技巧包括：

- **角色设定：** "You are an expert research analyst capable of breaking down complex problems..."
- **反例提示：** 在prompt中包含常见错误模式及其纠正方法
- **自我反思（Self-Reflection）：** 在prompt中明确要求模型"Evaluate your previous action. Was it successful? What could be improved?"
- **上下文窗口管理：** 当对话历史超出上下文窗口时，使用摘要压缩历史

**BabyAGI与任务驱动Agent。** Nakajima（2023）提出的BabyAGI是另一个有影响力的自主Agent框架。其核心循环为：

```
while 任务列表非空:
    任务 = 任务列表.popleft()
    结果 = 执行(任务, 上下文)
    存储(结果)
    新任务 = LLM(任务, 结果, 上下文)
    任务列表.extend(新任务)
```

BabyAGI使用两个独立的LLM调用：一个用于执行任务（execution agent），另一个用于根据执行结果生成新任务（task creation agent）。这种职责分离使得Agent可以在执行层面专注任务完成，在规划层面关注整体目标推进。

**局限性。** AutoGPT和BabyAGI虽然展示了自主Agent的潜力，但仍面临根本性挑战：
1. **无限循环问题：** Agent可能陷入重复执行相同操作的循环中
2. **累积误差：** 多步推理中的错误会随步骤累积，最终导致任务失败
3. **成本与延迟：** 自主Agent可能需要数十次甚至数百次LLM API调用，成本高昂且耗时
4. **安全性：** 自主Agent在不受监督的情况下执行行动存在安全风险

**Voyager：开放式探索与技能库。** Wang等人（2023）提出的Voyager解决了上述局限性中的循环和技能复用问题。Voyager在Minecraft游戏环境中展示了 lifelong learning 能力。其核心机制包括：
- **自动课程（Automatic Curriculum）：** 使用LLM根据Agent当前能力生成逐步递增难度的子目标
- **技能库（Skill Library）：** 将成功执行的行动序列编码为可复用的"技能代码"，存储在向量数据库中。未来遇到类似情境时检索并复用技能。
- **迭代式世界模型更新：** 环境状态的表示随探索经验不断更新

Voyager展示了从任务特定Agent向通用持续学习Agent演进的可能性。

---

## 第7节 NLP的评估体系：从单一任务到通用理解

随着大语言模型的能力从单一任务扩展到通用语言理解，评估体系经历了从任务特定基准到综合性基准的演进。GLUE/SuperGLUE代表了有监督微调时代的评估标准，而MMLU则反映了对大规模多任务语言理解能力的系统评测需求。本节从信息论和统计学习理论的角度深入分析这些评估体系的设计原理。

### 第7.1节 GLUE与SuperGLUE：通用语言理解的基准化

**GLUE的设计哲学。** GLUE（General Language Understanding Evaluation，Wang et al., 2018）的出发点是：单一任务的SOTA（最先进的）提升不一定意味着通用语言理解能力的提升——模型可能只是在特定任务上进行了过拟合或数据污染。GLUE通过聚合9个多样化的NLP任务，要求单一模型在所有任务上同时表现良好，从而更真实地衡量模型的通用理解能力。

GLUE包含的任务及其统计特性：

| 任务 | 类型 | 训练样本数 | 验证样本数 | 测试样本数 | 指标 |
|------|------|-----------|-----------|-----------|------|
| CoLA | 语法可接受性 | 8.5K | 1K | 1K | MCC |
| SST-2 | 情感分析 | 67K | 1.8K | 1.8K | Acc |
| MRPC |  paraphrase 检测 | 3.7K | 408 | 1.7K | F1/Acc |
| QQP | 问题对相似度 | 364K | 40K | 391K | F1/Acc |
| STS-B | 语义文本相似度 | 5.7K | 1.5K | 1.4K | Pearson-Spearman |
| MNLI | 自然语言推理 | 393K | 20K | 20K | Acc |
| QNLI | 问答NLI | 105K | 5.2K | 5.7K | Acc |
| RTE | 识别文本蕴含 | 2.5K | 277 | 3K | Acc |
| WNLI | Winograd NLI | 634 | 146 | 146 | Acc |

任务之间的多样性体现在多个维度：
- **语言学层次：** 从词汇级（STS-B的相似度判断）到句法级（CoLA的可接受性判断）到篇章级（MNLI的推理）
- **输入结构：** 单句任务（SST-2、CoLA）vs. 句对任务（MRPC、QQP、MNLI等）
- **数据规模：** 从634（WNLI）到364K（QQP）跨越约600倍
- **难度分布：** 人类基线在RTE上约93%准确率，而早期模型仅约60%

**GLUE评分机制。** GLUE的总体得分为各任务得分的简单平均。对于F1/Acc双指标任务（MRPC、QQP），取两者均值。所有任务使用0-100的评分尺度。设置了一个"公共排行榜"，研究者提交模型预测文件后获得匿名测试集评分。

**SuperGLUE的升级。** GLUE发布后不久，顶级模型的性能迅速饱和——到2019年，人类基线（87.1分）已被T5（89.3分）和ALUM（89.4分）超越。这表明GLUE的区分度不足。SuperGLUE（Wang et al., 2019）通过以下方式提升了难度：
1. 移除已饱和的SST-2和QQP，新增更具挑战性的任务：
   - **COPA（Choice of Plausible Alternatives）：** 因果推理，需要理解事件之间的因果关系
   - **MultiRC（Multi-Sentence Reading Comprehension）：** 多句阅读理解，需要跨句整合信息
   - **ReCoRD（Reading Comprehension with commonsense Reasoning requiring temporal ordering）：** 需要常识推理的填空式阅读理解
2. 增大任务规模（如MNLI-m的验证集从392K扩展到393K）
3. 移除容易的数据泄漏问题（如QQP中的重复问题对）

SuperGLUE的人类基线为89.8分，而初始最好的模型（BERT-large）仅约76分，提供了约14分的提升空间。

**GLUE/SuperGLUE的统计理论基础。** 从统计学习理论的角度，GLUE分数$\mathcal{S}_{\text{GLUE}} = \frac{1}{N}\sum_{i=1}^N s_i$（$s_i$为第i个任务的得分）是一个复合统计量。其方差满足：

$$
\text{Var}(\mathcal{S}_{\text{GLUE}}) = \frac{1}{N^2} \sum_{i=1}^N \text{Var}(s_i) + \frac{2}{N^2} \sum_{i<j} \text{Cov}(s_i, s_j)
$$

如果模型在不同任务上的表现高度相关（如大语言模型可能在所有任务上都表现良好），则方差主要由方差项决定。GLUE的局限性在于各任务的样本量差异巨大（WNLI仅634条训练样本 vs. QQP的364K），导致小任务上的分数方差较大，对总体得分产生噪声。

### 第7.2节 MMLU：大规模多任务语言理解

**设计动机与任务构建。** MMLU（Massive Multitask Language Understanding，Hendrycks et al., 2020）的目标是评测模型在 STEM、人文社科、商业、医学等57个学科上的专业知识和问题解决能力。MMLU的问题来自精心筛选的标准化考试和 textbooks：
- **STEM：** 数学、物理、化学、计算机科学、工程等
- **人文：** 历史、哲学、法律、艺术等
- **社会科学：** 经济学、心理学、社会学等
- **其他：** 医学、商业、教育学等

每个学科包含约200-400道4选1单选题，总计15,908道题。

**MMLU的难度层次。** MMLU根据考试级别将问题分为三个难度层次：
- **初级（Elementary）：** 约20%——高中水平知识
- **中级（High School / Early College）：** 约50%——大学一二年级水平
- **高级（Advanced Professional）：** 约30%——研究生/专业水平

人类专家的平均准确率约为89.8%（通过众包平台汇集多位专家的标注），而GPT-3（175B）仅为43.9%，Chinchilla（70B）为51.0%。这表明MMLU对模型的专业深度提出了很高要求。

**few-shot vs. zero-shot评估。** MMLU同时提供few-shot（5-shot）和zero-shot两种评估设置。Few-shot使用任务特定的示例（从训练集中采样），模型看到示例后回答问题。Zero-shot直接输入问题，模型不参考任何示例。

5-shot准确率与0-shot准确率的差值$\Delta = \text{Acc}_{5\text{-shot}} - \text{Acc}_{0\text{-shot}}$反映了模型从示例中"学习任务格式"的能力。对于简单分类任务，$\Delta$较小；对于需要理解复杂推理链的任务（如形式逻辑），$\Delta$较大。

**MMLU的分析性发现：**
1. **规模定律（Scaling Law）：** 模型性能与参数量、训练token数之间存在幂律关系：$\text{Acc} \approx a \cdot N^{-\alpha} + b \cdot D^{-\beta} + c$，其中N为参数量，D为训练数据量。
2. **知识 vs. 推理：** MMLU中的问题可以划分为"纯记忆"（如"DNA的全称是什么"）和"推理需要"（如"给定以下电路图，求等效电阻"）。纯记忆问题随模型规模提升最快，而推理问题存在平台期——模型可能在某个规模"卡住"，需要进一步增大规模或改进训练方法才能突破。
3. **学科间迁移：** 模型在相关学科之间表现正相关（如数学和物理的准确率相关系数r ≈ 0.85），但不相关学科之间几乎独立（如艺术史和有机化学，r ≈ 0.05）。这为课程学习（curriculum learning）提供了依据——先训练相关学科任务，再训练不相关学科任务。

**MMLU-Pro与动态评估。** MMLU的问题相对固定，模型可能通过数据污染取得虚高的分数。MMLU-Pro（Wang et al., 2024）通过增加选项数量（从4选1增加到10选1）、替换过时问题和引入更难的推理子集，将基准的难度提升了约20%。此外，动态评估基准（如HELM、BIG-bench）使用持续更新的测试集，以对抗数据污染。

### 第7.3节 大模型评估的新挑战

**数据污染（Data Contamination）。** 当模型的训练数据中包含评估基准的题目时，评估结果不再可信。Grosse等人（2023）估计，在MMLU的约16K道题中，约10-15%可能以某种形式出现在模型训练数据中。检测数据污染的方法包括：
- **n-gram重叠检测：** 计算模型训练数据与测试题目的n-gram重叠率
- **成员推断攻击（Membership Inference Attack）：** 通过模型对特定题目的困惑度差异推断该题目是否在训练集中
- **Canary Strings：** 在数据集发布时嵌入特殊标记，后续扫描训练数据中是否包含该标记

**指令遵循与对齐评测。** 传统评估关注"模型能否做对"，而大模型时代需要关注"模型是否按照指令做对"。IFEval（Zhou et al., 2023）评测模型对格式约束（如"回答不多于3句话"、"使用项目符号"）的遵循能力。评测指标为严格遵循率（strict accuracy）和宽松遵循率（loose accuracy）。

**LLM-as-Judge。** 随着任务复杂度的提升（如开放域对话、创意写作），传统的基于规则或分类器的自动评估方法不再适用。Zheng等人（2023）提出使用LLM本身作为评估器（judge），通过prompt让模型对两个生成结果进行比较评分。研究发现，GPT-4作为judge与人类判断的一致率超过80%，超过了基于ROUGE或BERTScore的自动评估方法。

---

## 第8节 大语言模型的幻觉问题：事实性与一致性的数学刻画

大语言模型的幻觉（Hallucination）是指模型生成的内容在事实层面不正确，或在逻辑层面不一致。Ji等人（2023）将幻觉分为三大类：输入引导型幻觉（输入中的错误信息诱导模型生成错误输出）、知识缺失型幻觉（训练数据中不存在相关知识）和推理错误型幻觉（模型推理过程中出现逻辑错误）。本节从数学角度刻画幻觉的成因和度量方法。

### 第8.1节 事实性幻觉的形式化

**定义。** 形式化地，设$p_\theta(y|x)$为语言模型在输入x下生成y的概率。对于事实性断言$a$，定义其真值函数$V(a) \in \{0, 1\}$（1表示真，0表示假）。模型生成包含断言a的文本时的事实性幻觉率为：

$$
P_{\text{halluc}}(a|x) = 1 - \sum_{y \ni a: V(a(y))=1} p_\theta(y|x)
$$

其中$y \ni a$表示生成文本y包含断言a。直觉上，如果模型生成的真断言集合的累积概率远低于1，则存在事实性幻觉。

**知识截止与分布偏移。** 事实性幻觉最直接的来源是训练数据的知识截止。设训练数据的时间分布为$p_{\text{train}}(t)$，测试时查询的时间为$t_q$。若$t_q$在训练数据的时间范围之外，则模型没有关于$t_q$事件的信息，只能基于训练数据中的先验分布进行推测，这天然导致事实性错误。

进一步地，设训练数据中关于实体e的事实为$F_{\text{train}}(e)$，真实世界的事实为$F_{\text{world}}(e)$。当$F_{\text{train}}(e) \neq F_{\text{world}}(e)$时（如人物的死亡日期、公司的合并信息），模型将输出过时的事实。

**幻觉概率的上界。** 设模型的词汇表为$\mathcal{V}$，生成文本的最大长度为T。对于任何事实性断言a，有：

$$
P_{\text{halluc}}(a|x) \leq 1 - \max_{y \in \mathcal{V}^T: V(a(y))=1} p_\theta(y|x)
$$

当不存在任何包含真断言的生成路径时（即模型从未在训练数据中见过正确的a），上界等于1。当存在至少一条正确路径时，上界取决于最高概率正确路径的生成概率。

### 第8.2节 一致性与Faithfulness幻觉

**定义一致性幻觉。** 事实性幻觉关注"生成内容是否与客观事实一致"，而一致性幻觉（consistency hallucination）关注"生成内容是否与自身先前陈述一致"。设模型在对话历史$h = (u_1, a_1, \ldots, u_t)$的基础上生成回答$a_t$。一致性幻觉发生在：

$$
\exists i < t, \text{ s.t. } \text{SemanticContradiction}(a_i, a_t) = \text{True}
$$

其中$\text{SemanticContradiction}$为语义矛盾检测函数。

**自洽性作为幻觉检测器。** 一个自然的问题是：能否通过多次生成来检测幻觉？如果模型对同一查询多次生成不同答案，则至少有一个答案是幻觉。形式化地，设从模型独立采样m次，得到回答集合$\{y_1, \ldots, y_m\}$。定义一致性分数：

$$
C(x) = \frac{1}{m(m-1)/2} \sum_{i<j} \text{sim}(y_i, y_j)
$$

其中$\text{sim}$为语义相似度（使用嵌入模型的余弦相似度）。低一致性分数$C(x)$暗示模型对该问题缺乏稳定的知识，高幻觉概率。Wang等人（2022）的自一致性（self-consistency）方法利用了这一原理：在推理链层面采样多条路径，通过投票选择最高频的答案，不一致的路径被视为幻觉路径。

**Faithfulness to the Context。** 在RAG场景中，还有一种特殊的幻觉类型——上下文不忠实（unfaithful to context）。模型可能生成与检索文档在事实上一致但与用户问题无关的内容，或者在整合多个文档时引入文档之间的虚假关联。设检索文档集合为$\mathcal{Z} = \{z_1, \ldots, z_k\}$，生成回答为$y$。 faithfulness（忠实性）定义为：

$$
\text{Faithful}(y, q, \mathcal{Z}) = \mathbb{I}\left[\forall a \in y, \exists z_j \in \mathcal{Z} \text{ s.t. } a \text{ is entailed by } z_j\right]
$$

即生成回答中的每个断言都可以被检索文档所蕴涵。当前大多数RAG系统不做 faithfulness 保证——模型可能在检索文档的"知识空白"处自由发挥，产生文档无法支持的事实。

### 第8.3节 幻觉的归因分析

**注意力头的幻觉角色。** 通过分析Transformer的注意力模式，研究发现某些注意力头专门负责"幻觉生成"。在事实性问答任务中，"幻觉头"的注意力模式呈现以下特征：
1. **全局关注：** 注意力分数分散到大量无关token，而非聚焦于与查询相关的关键token
2. **早期层活跃：** 幻觉通常在较低层（1-6层）开始形成，然后在较高层被放大
3. **复制机制：** 某些头专门负责复制训练数据中的高频但错误的关联（如将"爱因斯坦"与错误的出生年份关联）

**知识在参数中的存储方式。** 从表示学习的角度，事实性知识存储在Transformer的FFN层权重中。Dai等人（2021）证明了FFN层实际上实现了一个连续的键值存储（key-value memory）：

$$
\text{FFN}(x) = f(xW_k)W_v \approx \sum_{i} \sum_{j} v_{i,j} \cdot \sigma(xW_{k,i} + b_{k,i})_j
$$

其中$v_{i,j}$为第i层第j个神经元的值向量，$\sigma$为激活函数。事实性知识被编码为$(W_k, W_v)$中的特定模式：$W_k$的某些行对应"查询模式"（如"X的出生年份是什么"），$W_v$的对应行存储答案（如"1879"）。当查询匹配到错误的键时，就会检索到错误的值，产生幻觉。

**训练数据中事实的统计偏差。** 训练数据中的事实分布并不均匀。例如，关于"苹果公司CEO"的信息在训练数据中出现频率远高于"2024年诺贝尔物理学奖得主"（因为后者的获奖时间可能在训练数据截止之后）。模型通过学习数据中的统计规律来"回答"事实性问题，当训练数据中某事实的频率低于某个阈值时，模型无法可靠地检索该事实，倾向于生成高频但错误的替代事实。

### 第8.4节 缓解幻觉的技术路线

**检索增强作为事实性保障。** 最直接的缓解幻觉方法是将生成锚定在检索文档上。RAG框架（见6.5节）通过提供检索到的相关文本作为生成上下文，使模型有据可查。然而，即使提供了检索文档，模型仍可能生成与文档不一致的内容。提高RAG忠实性（faithfulness）的方法包括：
- **事实性引导微调：** 在微调数据中显式标注"仅根据提供的文档回答，不要使用训练数据中的知识"
- **引用标注：** 训练模型在生成事实性断言时附上文档来源引用，事后验证引用是否支持断言
- **约束解码：** 在解码时限制模型只能生成检索文档中出现过的n-gram，或使用T5-style的span extraction直接提取答案片段

**不确定性估计。** 对于模型自身不确定的事实性问题，理想的行为是模型表达不确定性（如"我不确定"）而非编造答案。不确定性估计方法包括：
- **语义熵（Semantic Entropy，Kuhn et al., 2023）：** 采样多个生成结果，计算语义聚类后的熵。高语义熵意味着模型对答案缺乏确定性。
- **数学上，语义熵定义为：**
  
  $$
  H_{\text{sem}} = -\sum_{C \in \text{Clusters}} p(C) \log p(C)
  $$
  
  其中聚类基于语义等价性（如"巴黎是法国的首都"和"法国的首都是巴黎"属于同一聚类），$p(C)$为聚类C的概率（通过对聚类内所有生成的概率求和得到）。

- **SAR（Self-Awareness with Retrieval）：** 在生成答案前，先检索知识库。如果检索结果与模型生成的知识一致，则置信度高；如果不一致或检索为空，则置信度低。

**后验事实核验。** 对于已生成的文本，可以使用外部知识源进行事实核验。FActScore（Min et al., 2023）将长文本分解为原子断言（atomic claims），然后对每个断言查询知识库进行验证。事实性分数定义为被验证为真的断言比例：

$$
\text{FActScore}(y) = \frac{1}{|y|} \sum_{a \in \text{Claims}(y)} \mathbb{I}[\text{Verify}(a, \mathcal{K}) = \text{True}]
$$

其中$\text{Claims}(y)$为从y中提取的原子断言集合，$\mathcal{K}$为知识库，$\text{Verify}$为事实验证函数。

**对比解码（Contrastive Decoding）。** Li等人（2023）提出的对比解码利用 expert 模型和 amateur 模型的概率差异来减少幻觉。Expert模型（大模型，如PaLM）具有丰富的知识但可能产生幻觉；Amateur模型（小模型，如GPT-2）知识有限但幻觉模式不同。对比解码的token级概率为：

$$
p_{\text{CD}}(y_t | y_{<t}) \propto \frac{p_{\text{expert}}(y_t | y_{<t})^\lambda}{p_{\text{amateur}}(y_t | y_{<t})}
$$

其中$\lambda > 1$为温度参数。当expert模型对某个token赋予高概率但amateur模型也赋予高概率时（即该token是"普通语言"而非"专业知识"），比值接近1，不进行修改。当expert模型对某个token赋予高概率但amateur模型赋予低概率时（该token是expert模型独有的"知识"），比值大于1，放大该token的概率。当expert模型本身不确定时（概率较低），amateur模型的概率可能更高，比值小于1，抑制该token的生成。

这种方法的一个关键假设是：amateur模型不知道expert模型的"幻觉知识"，因此幻觉token在两个模型上的概率差异较小（比值接近1），而非幻觉的知识token在expert上概率远高于amateur（比值显著大于1）。实验表明，在long-form生成任务上，对比解码将FActScore从约28提升至35（PaLM-2 540B），同时保持了生成文本的流畅度。

### 第8.5节 幻觉的度量与评测体系

**事实性评测基准。**
- **FActScore（Min et al., 2023）：** 将长生成分解为原子断言，使用Wikipedia或Google搜索验证
- **HaluEval（Li et al., 2023）：** 包含对话、摘要和问答三种任务格式的幻觉检测数据集
- **TruthfulQA（Lin et al., 2022）：** 评测模型是否会被常见误解所诱导产生错误回答
- **RARR（Ruhnau et al., 2022）：** 评测生成文本的事实准确性、 attributedness 和检索利用率

**幻觉检测模型。** 专门训练幻觉检测器来判断给定文本是否包含幻觉：
- **HaluEval evaluator：** 在合成幻觉数据上训练的分类器
- **LLM-based detector：** 使用GPT-4等大模型作为幻觉检测器，prompt引导模型判断每个断言的真实性

**信息论度量。** 从信息论角度，幻觉可以视为模型输出中的"意外信息"——训练数据分布中概率极低但在生成中被输出的内容。使用困惑度（perplexity）作为代理指标：

$$
\text{PPL}(y) = \exp\left(-\frac{1}{T}\sum_{t=1}^T \log p_\theta(y_t | y_{<t})\right)
$$

高困惑度片段更可能是幻觉，因为模型在生成这些片段时"犹豫不决"。然而，困惑度并非完美的幻觉指示器——模型可能对错误的信息非常"自信"（训练数据中的错误被反复强化）。

**人类反馈作为黄金标准。** 在最前沿的系统中，人类反馈强化学习（RLHF）被用于直接减少幻觉。通过收集人类对生成质量的标注（包括事实性评分），训练奖励模型来惩罚幻觉生成。Anthropic的Constitutional AI和OpenAI的InstructGPT都使用了这一策略，显著降低了生成文本的幻觉率。

### 第8.6节 本章小结

大语言模型的幻觉问题是其知识获取方式和概率生成本质的必然产物。从数学上看，幻觉源于训练数据分布$p_{\text{train}}$与真实世界分布$p_{\text{world}}$之间的分布偏移$D_{\text{KL}}(p_{\text{world}} \| p_{\text{train}})$。完全消除幻觉需要模型能够访问$p_{\text{world}}$——即实时更新的世界知识——而这在当前的技术框架下只能通过检索增强、工具调用和多模态感知来近似实现。

幻觉问题的解决不是单点技术的突破，而是需要系统层面的综合方案：更高质量的训练数据、更好的知识编辑方法、检索增强的深度融合、不确定性估计的准确化，以及人类反馈对齐的精细化。这些方向的进展将决定下一代语言系统的可信度和实用性。







