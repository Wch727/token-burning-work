
# 语言模型的进化：从n-gram到神经网络

## 引言

语言模型（Language Model, LM）是自然语言处理的核心组件，其本质是对自然语言的概率分布进行建模。给定一个词序列 $w_1^T = w_1, w_2, \ldots, w_T$，语言模型的任务是估计其联合概率：

$$P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^T P(w_t \mid w_1^{t-1})$$

这一分解——将联合概率化为条件概率的链式乘积——是语言模型的理论基石，直接源于概率论中的链式法则。语言模型的研究史，本质上是一部如何估计每一个条件概率 $P(w_t \mid w_1^{t-1})$ 的历史。从早期的马尔可夫假设到现代的大规模神经网络，每一次技术跃迁都围绕着"如何更高效、更准确地建模条件分布"这一核心问题展开。

**概率模型的历史脉络。** 语言模型的思想根源可以追溯到香农1948年的《通信的数学原理》，其中香农提出了利用 $n$-元语法近似自然语言概率分布的初步框架。然而，真正将 $n$-gram模型系统化为实用工具的是Jelinek等人（1970s）在IBM speech group的工作——他们为连续语音识别开发了基于马尔可夫假设的统计语言模型，使用互信息和大词汇量的trigram来约束识别搜索空间。与此同时，Good-Turing（1953）和Katz（1987）的平滑技术为数据稀疏性提供了系统性解决方案，使 $n$-gram模型在受限词表上实现了可用的概率估计。

信息论为语言模型评估提供了严格的数学框架。香农熵和交叉熵将"不确定性"和" surprisal"精确量化，而困惑度（Perplexity）作为交叉熵的指数变换，提供了与人类直觉直接对应的评估指标——困惑度可以被理解为"每一步的等效分支因子"。这一指标至今仍是语言模型的黄金标准。

词表示的学习是语言模型演进中的一次范式转变。从Brown聚类（1992）的离散类标签到Bengio神经语言模型（2003）的连续词嵌入，词从"符号表中的索引"跃迁为"向量空间中的点"——这一转变的意义不亚于从"手工特征"到"表示学习"的转变。Word2Vec（2013）和GloVe（2014）进一步将词向量的学习效率提升到工业级水平，使大规模语料上的词向量训练成为可能。

神经机器翻译（NMT）的出现将语言模型推向了新的高度。Sutskever等人的Seq2Seq架构（2014）首次使用端到端神经网络完成翻译任务，而Bahdanau注意力机制（2014）从根本上解决了固定长度上下文向量的瓶颈问题。注意力机制不仅提升了翻译质量，更重要的是引入了一种全新的计算范式——动态路由（Dynamic Routing）：模型根据输入动态地选择计算路径，而非使用固定的计算图。

BLEU（2002）、ROUGE（2004）和METEOR（2005）等自动评估指标的发展，为机器翻译和文本生成的系统比较提供了客观标准。这些指标的设计哲学、数学性质和局限性，直接影响了后续模型评估方法论的发展。

本章将从n-gram模型出发，系统梳理语言模型的技术演进脉络，涵盖信息论视角下的模型评估、词向量表示的学习、神经语言模型的兴起以及机器翻译中的注意力革命，最终建立对当代预训练语言模型发展前史的完整认识。理解这些前驱技术不仅具有历史意义——n-gram的平滑技术、交叉熵训练目标、词向量的矩阵分解视角、注意力机制的对齐思想——都在当代Transformer模型中获得了新的生命。

---

## 1 n-gram语言模型

### 1.1 马尔可夫假设与n-gram定义

直接估计联合概率 $P(w_1^T)$ 面临一个根本性困难：对于长度为 $T$ 的序列，其可能的组合数量随 $T$ 指数增长，绝大多数序列在训练语料中从未出现，导致直接计数不可行。马尔可夫假设（Markov Assumption）为此提供了关键简化：假设当前词的概率仅依赖于其前 $n-1$ 个词，即

$$P(w_t \mid w_1^{t-1}) \approx P(w_t \mid w_{t-n+1}^{t-1})$$

其中 $w_{a}^{b}$ 表示从第 $a$ 个词到第 $b$ 个词的子序列。在此假设下，$n=1$ 对应一元模型（Unigram），$n=2$ 对应二元模型（Bigram），$n=3$ 对应三元模型（Trigram）。实际应用中，Trigram是最为常用的n-gram模型，因为它在模型容量和数据稀疏性之间取得了较好的平衡。

n-gram模型的参数化形式为：

$$P(w_t \mid w_{t-n+1}^{t-1}) = \frac{c(w_{t-n+1}^{t-1}, w_t)}{c(w_{t-n+1}^{t-1})}$$

其中 $c(\cdot)$ 表示对应的计数函数。这个比率在统计学习理论中被称为相对频率估计（Relative Frequency Estimation），其理论基础是大数定律：当样本量趋于无穷时，相对频率依概率收敛于真实概率。

### 1.2 最大似然估计

最大似然估计（Maximum Likelihood Estimation, MLE）为n-gram参数估计提供了严格的理论框架。给定语料库 $\mathcal{D} = \{w^{(1)}, w^{(2)}, \ldots, w^{(N)}\}$，其中每个 $w^{(i)}$ 是一个长度为 $T_i$ 的词序列，对数似然函数为：

$$\ell(\theta) = \sum_{i=1}^N \sum_{t=1}^{T_i} \log P_\theta(w_t^{(i)} \mid w_{t-n+1}^{t-1, (i)})$$

其中 $\theta$ 表示所有n-gram的条件概率参数。最大似然估计通过最大化对数似然来求解参数：

$$\theta_{\text{MLE}} = \arg\max_\theta \, \ell(\theta) = \arg\max_\theta \, \sum_{h, w} c(h, w) \log P(w \mid h)$$

这里 $h$ 表示长度为 $n-1$ 的历史上下文，$c(h, w)$ 是历史-词对 $(h, w)$ 在语料中出现的次数。上述优化问题在约束 $\sum_w P(w \mid h) = 1$ 下进行。使用拉格朗日乘子法，构造：

$$\mathcal{L} = \sum_{h, w} c(h, w) \log P(w \mid h) + \lambda_h \left( \sum_w P(w \mid h) - 1 \right)$$

对 $P(w \mid h)$ 求偏导并令其为零：

$$\frac{\partial \mathcal{L}}{\partial P(w \mid h)} = \frac{c(h, w)}{P(w \mid h)} + \lambda_h = 0 \implies P(w \mid h) = -\frac{c(h, w)}{\lambda_h}$$

利用约束条件 $\sum_w P(w \mid h) = 1$，得到 $\lambda_h = -c(h) = -\sum_w c(h, w)$，因此：

$$P_{\text{MLE}}(w \mid h) = \frac{c(h, w)}{c(h)}$$

这是n-gram MLE估计的标准结果。MLE估计的无偏性和一致性依赖于一个隐含假设：每个n-gram在语料中出现足够多次，使得其相对频率收敛于真实概率。然而，n-gram模型面临严重的数据稀疏问题：对于词汇表大小 $V$，n-gram的总可能数为 $V^n$，而实际语料库的大小远小于 $V^n$。当 $c(h, w) = 0$ 时，MLE估计给出 $P_{\text{MLE}}(w \mid h) = 0$，这会导致整个句子概率为零（零概率问题），以及下游任务如机器翻译或语音识别中的搜索失败。因此，必须引入平滑技术来缓解数据稀疏性。

### 1.3 Laplace平滑

Laplace平滑（Laplace Smoothing），又称加一平滑（Add-One Smoothing），是最直观的平滑方法。其核心思想非常朴素：为每一个可能的n-gram分配一个最小计数，即使它在语料中未出现。具体而言，将每个计数 $c(h, w)$ 增加1：

$$P_{\text{Laplace}}(w \mid h) = \frac{c(h, w) + 1}{c(h) + V}$$

其中 $V$ 是词汇表大小。Laplace平滑的贝叶斯解释如下：它等价于在多项分布上使用均匀先验 Dirichlet(1, 1, ..., 1)，然后计算后验期望。根据贝叶斯定理：

$$P(w \mid \mathcal{D}) = \int P(w \mid \theta) P(\theta \mid \mathcal{D}) \, d\theta$$

当先验为 $\text{Dirichlet}(1, \ldots, 1)$ 时，后验分布为 $\text{Dirichlet}(c(h, w_1) + 1, \ldots, c(h, w_V) + 1)$，其后验期望恰为上述公式。

Laplace平滑在理论上简洁优美，但在实践中存在严重缺陷。当 $V$ 较大时，加一操作对概率分布的扰动过大——对于未出现的n-gram，其概率质量不再是零，但被过度稀释。对于Trigram模型，$V$ 通常为数万量级，分母中的 $V$ 可能远大于分子增量，使得模型严重倾向于低概率全局词，导致困惑度（Perplexity）不降反升。后续的平滑技术，如Lidstone平滑（将1替换为 $\alpha < 1$）和绝对折扣（Absolute Discounting），正是对Laplace平滑这一缺陷的修正。

### 1.4 Good-Turing平滑

Good-Turing平滑（Good-Turing Smoothing）由Good在1953年提出，后经Gale和Sampson于1991年系统化，是一种基于频率计数频率（Frequency of Frequency）的平滑方法。其核心洞见是：一个从未出现的n-gram的概率，可以由"出现一次的n-gram所占比例"来估计。这一洞见在直觉上极具说服力——如果我们观察到大量"昙花一现"的词（恰好出现一次），那么那些完全未出现的词很可能也属于同一"稀有"群体，它们共享相似的概率量级。

**频率计数频率的完整框架。** 定义 $N_c$ 为恰好出现 $c$ 次的n-gram的数量：

$$N_c = |\{ (h, w) : c(h, w) = c \}|$$

这里 $c(h, w)$ 是历史-词对 $(h, w)$ 在语料中的原始计数。$N_c$ 本身是一个随机变量——它依赖于具体的语料采样。可以验证基本的概率质量守恒：

$$\sum_{c=1}^{\infty} c \cdot N_c = \sum_{h, w} c(h, w) = N$$

其中 $N$ 是语料中n-gram的总出现次数（即所有计数的总和）。注意上述求和从 $c=1$ 开始，因为零计数 $N_0$ 不包含在等式中——$N_0$ 是未出现n-gram的数量，其值依赖于词汇表大小 $V$ 和n-gram阶数。

**重估计计数的数学推导。** Good-Turing估计的关键步骤是对原始计数 $c$ 进行重估计（Reestimation），定义重估计计数 $c^*$。推导基于以下直觉：对于恰好出现 $c$ 次的n-gram，其"真实"概率应该使得在 $N$ 次抽取中，它被看到的期望次数等于 $c$。然而，观察到的 $c$ 本身受到了抽样随机性的影响——如果某个n-gram的真实概率较高，它可能恰好被抽到 $c$ 次；如果真实概率很低，它也可能恰好被抽到 $c$ 次。Good-Turing的方法是：用更高频的计数信息来校正低频计数。

形式上，重估计计数定义为：

$$c^* = (c + 1) \cdot \frac{N_{c+1}}{N_c}$$

这一公式的推导可以通过以下方式理解：考虑所有出现 $c$ 次的n-gram，它们总共贡献了 $c \cdot N_c$ 次出现。在语料中，这些n-gram的下一次出现（即第 $c+1$ 次出现）来自出现 $c+1$ 次的n-gram群体。出现 $c+1$ 次的n-gram有 $N_{c+1}$ 个，每个恰好有 $c+1$ 次出现，因此它们总共贡献了 $(c+1) \cdot N_{c+1}$ 次出现。频率比 $(c+1) N_{c+1} / (c N_c)$ 衡量了"从 $c$ 次到 $c+1$ 次的相对增长"。当 $N_{c+1} / N_c$ 较小时，表明 $c$ 次群体处于分布的尾部，$c^* < c$，即估计被向下校正。

对于零计数（$c = 0$），Good-Turing估计给出：

$$P_{\text{GT}}(w \mid h) = \frac{1}{N} \quad \text{当 } c(h, w) = 0$$

等等，这里 $1/N$ 并非最终形式。更准确地说，零频n-gram的总概率质量由 $N_0 / N$ 给出，其中 $N_0$ 是恰好零次的n-gram数量。Good-Turing估计将 $N_0 / N$ 均匀分配给所有零频n-gram（这是非参数方法的固有假设），因此每个零频n-gram获得 $1/N$ 的概率——但这仅在零频n-gram数量恰好为1时成立。实际上，零频n-gram的概率为：

$$P_{\text{GT}}(w \mid h) = \frac{q_{\text{GT}}}{N_0} \quad \text{当 } c(h, w) = 0$$

其中 $q_{\text{GT}}$ 是分配给所有零频n-gram的总概率质量。在实践中，$N_0$ 通常未知（因为穷举所有零频n-gram不可行），需要通过Good-Turing频率定律来估计。

更一般地，对于任意计数 $c$，概率估计为：

$$P_{\text{GT}}(w \mid h) = \frac{c^*(c)}{N} = \frac{(c+1) N_{c+1}}{N_c} \cdot \frac{1}{N}$$

对于高频n-gram（$c$ 较大），$N_{c+1} / N_c \to 1$，因此 $c^* \approx c$，估计趋近于MLE；对于低频乃至零频n-gram，$c^*$ 平滑地分配概率质量，避免了零概率问题。

**Good-Turing频率定律与线性回归外推。** 为了估计 $N_c$（尤其是大 $c$ 值对应的 $N_c$，其样本极少甚至为零），Good-Turing方法采用频数定律的线性回归版本：在 $c$-$N_c$ 的双对数图上，低 $c$ 区域近似一条直线。这一经验规律被称为Good-Turing频率定律：

$$\log N_c \approx a - b \log c$$

具体而言，对 $c = 1, 2, \ldots, k$（通常 $k = 5$ 或 $7$）拟合上述直线，得到截距 $a$ 和斜率 $b$。然后用拟合值估计 $N_{k+1}, N_{k+2}, \ldots$：

$$\hat{N}_c = 10^a / c^b \quad \text{for } c > k$$

这保证了概率质量守恒——即使对于未在语料中出现的 $c$ 值，也能给出合理的 $N_c$ 估计：

$$\sum_{c=0}^{\infty} P(w \mid h, c_{\text{observed}} = c) = 1$$

在实践中，$N_c$ 对于 $c > k$ 的估计存在较大方差，特别是当 $b > 1$ 时，$N_c$ 衰减很快，小误差会导致概率质量估计出现显著偏差。因此，Good-Turing平滑通常只使用前 $k$ 个 $N_c$ 值的经验估计，对更高频事件回退到MLE。

**概率质量守恒的严格验证。** 对所有历史 $h$，Good-Turing估计的概率质量之和应该为1：

$$\sum_{c=0}^{\infty} P_{\text{GT}}(c \mid h) = \sum_{c=0}^{\infty} \frac{c^*}{N} = \frac{1}{N} \sum_{c=0}^{\infty} (c+1) \frac{N_{c+1}}{N_c}$$

注意到 $N_c$ 是恰好出现 $c$ 次的n-gram数量，因此 $\sum_{c=0}^{\infty} (c+1) N_{c+1} = \sum_{c=1}^{\infty} c \cdot N_c = N$（所有计数的总和）。于是：

$$\sum_{c=0}^{\infty} P_{\text{GT}}(c \mid h) = \frac{N}{N} = 1$$

这证明了Good-Turing估计的完备性——它是一个合法的概率分布。

**优势与局限。** Good-Turing平滑的优势在于其非参数性——它不依赖于参数化假设，纯粹从频率计数中推断概率。这使得它对分布形状不敏感，在理论上具有广泛的适用性。然而，其缺点也十分明显：首先，对于高频事件（$c$ 较大），$N_c$ 的样本极少，线性回归外推的不确定性大，导致估计方差高；其次，当 $N_c = 0$ 时（对于某些 $c$），公式出现 $0/0$ 的未定义情况，需要特殊处理（如忽略该 $c$ 值，或使用相邻的 $N_c$ 进行平滑）；第三，Good-Turing估计对计数 $c$ 的敏感度在 $c$ 较大时下降过快，导致概率质量过于集中在极高频n-gram上；第四，非参数方法需要存储所有 $N_c$ 值，在 $V$ 较大时内存开销不可接受。这些局限性促使了Kneser-Ney等参数化平滑方法的发展。

### 1.5 Kneser-Ney平滑

Kneser-Ney平滑（Kneser-Ney Smoothing）由Kneser和Ney于1995年提出，是n-gram平滑技术中公认的最优方法之一，至今仍在工业级语言模型中被广泛使用。其设计基于两个核心洞察：

**洞察一：绝对折扣。** 实验表明，对于出现 $c \geq 1$ 次的n-gram，MLE估计 $c(h, w) / c(h)$ 系统性地偏高——偏差约为一个常数 $d \in [0, 1)$（通常取 $d \approx 0.75$）。Kneser和Ney认为：不应信任MLE估计的绝对值，而应从所有计数中统一扣除一个折扣 $d$，然后将回收的概率质量按照某种方式重新分配。由此定义折扣MLE（Discounted MLE）：

$$P_{\text{disc}}(w \mid h) = \frac{\max(c(h, w) - d, 0)}{c(h)}$$

当 $c(h, w) = 0$ 时，$P_{\text{disc}}(w \mid h) = 0$；当 $c(h, w) = 1$ 时，概率变为零——这正是回收的机制。

**洞察二：延续概率（Continuation Probability）。** 回收的概率质量需要按照某种合理的机制重新分配给零频n-gram。直觉上：一个词如果出现在*更多不同*的上下文历史中，它就具有更高的"通用性"，也更可能在新的上下文中出现。形式上，定义词 $w$ 的延续概率为：

$$P_{\text{cont}}(w) = \frac{|\{ h' : c(h', w) > 0 \}|}{|\{ (h'', w'') : c(h'', w'') > 0 \}|}$$

即 $w$ 作为后缀出现的不同历史数，除以所有不同n-gram的总数。直观理解：$P_{\text{cont}}(w)$ 衡量的是"看到 $w$ 时，期望前面有多少种不同的历史"。例如，"Francisco"出现在"San Francisco"中，其延续概率很低；而"the"出现在大量不同历史中，其延续概率很高。

**折扣参数 $d$ 的理论推导。** 折扣值 $d$ 的选择直接影响Kneser-Ney的性能。直观上，$d$ 应该反映"观察到 $c$ 次的事件，其真实概率应该被下调多少"。从贝叶斯视角，$d$ 可以解释为对每个观察到的计数扣除一个"置信度惩罚"。Ney等人指出，如果假设计数服从泊松分布，则最优折扣可以通过矩估计得到。实践中，$d$ 通常被设为所有阶数的统一值，经验最优范围为 $0.5 \leq d \leq 0.75$。另一种自适应方法是让 $d$ 依赖于上下文历史 $h$ 的出现次数 $c(h)$：

$$d(h) = \frac{n_1(h)}{n_1(h) + 2n_2(h)}$$

其中 $n_1(h) = |\{w : c(h, w) = 1\}|$ 是恰好出现一次的后续词数量，$n_2(h) = |\{w : c(h, w) = 2\}|$ 是恰好出现两次的后续词数量。当 $n_1(h)$ 较大时（即历史 $h$ 后面跟着许多"昙花一现"的词），$d(h)$ 较大，表明 $h$ 的计数不可靠，需要更大的折扣；当 $n_1(h)$ 较小时，$d(h)$ 较小，表明 $h$ 的计数较为稳定。

**插值Kneser-Ney与递归结构。** 上述公式描述的是一阶Kneser-Ney模型。实际应用中，高阶Kneser-Ney模型通过递归插值（Recursive Interpolation）来构建：

$$P_{\text{KN}}(w_i \mid w_{i-n+1}^{i-1}) = \frac{\max(c(w_{i-n+1}^{i-1}, w_i) - d, 0)}{c(w_{i-n+1}^{i-1})} + \lambda(w_{i-n+1}^{i-1}) \cdot P_{\text{KN}}(w_i \mid w_{i-n+2}^{i-1})$$

其中 $\lambda(w_{i-n+1}^{i-1})$ 是插值权重（Interpolation Weight），确保概率归一化。以trigram模型为例（$n=3$）：

$$P_{\text{KN}}(w_i \mid w_{i-2}, w_{i-1}) = \frac{\max(c(w_{i-2}, w_{i-1}, w_i) - d, 0)}{c(w_{i-2}, w_{i-1})} + \lambda(w_{i-2}, w_{i-1}) \cdot P_{\text{KN}}(w_i \mid w_{i-1})$$

递推基例为unigram级别的延续概率：

$$P_{\text{KN}}(w_i) = P_{\text{cont}}(w_i)$$

这种递归结构的美妙之处在于：trigram层面的"不确定性"被传递到bigram层面，bigram层面的"不确定性"被传递到unigram层面——每一层都用更低阶模型的预测来弥补高阶模型的不足。这与人类语言理解的层次性高度吻合：当我们不确定一个词在特定三元组中的出现时，我们会回退到"这个词是否常见"的直觉；当这种直觉也不可靠时，我们会回退到"这个词是否通用"的判断。

**修正的延续概率：Kneser-Ney的精细化。** 原始的延续概率定义使用不同的n-gram总数作为归一化常数。但在插值Kneser-Ney中，延续概率被修正为：

$$P_{\text{KN-cont}}(w) = \frac{|\{h : c(h, w) > 0\}|}{\sum_{w'} |\{h : c(h, w') > 0\}|}$$

即分子是 $w$ 作为后续词出现的不同历史数，分母是所有词作为后续词出现的不同历史数之和。这保证了 $\sum_w P_{\text{KN-cont}}(w) = 1$。注意到这一定义与bigram级别的折扣估计形成了对称：bigram的折扣估计计算的是"在给定前驱词 $h$ 下，不同后续词的数量"，而延续概率计算的是"在给定后续词 $w$ 下，不同前驱词的数量"——二者共同编码了n-gram的"多样性"信息。

Chen和Goodman（1998）的对比实验表明，修正的Kneser-Ney平滑在各种n-gram任务上的困惑度一致优于Good-Turing和其他平滑方法。具体而言，在Wall Street Journal语料上，Kneser-Ney trigram模型的困惑度约为100，而Good-Turing trigram模型的困惑度约为130——这一差距在统计上显著，在实用上意义重大。这一结果使Kneser-Ney成为n-gram模型的标杆，其插值结构直接影响了后续神经语言模型中门控机制的设计思想。

**绝对折扣的直觉解释与概率质量再分配。** 为了进一步理解折扣机制，可以考虑一个具体例子。假设历史 $h = \text{"the"}$ 后面出现了5个不同的词，每个恰好出现1次：$c(\text{"the"}, w_i) = 1$（$i=1,\ldots,5$），$c(\text{"the"}) = 5$，折扣 $d = 0.75$。MLE估计给出 $P_{\text{MLE}}(w_i \mid \text{"the"}) = 1/5 = 0.2$。Kneser-Ney折扣估计给出：

$$P_{\text{disc}}(w_i \mid \text{"the"}) = \frac{\max(1 - 0.75, 0)}{5} = \frac{0.25}{5} = 0.05$$

回收的总概率质量为 $\lambda(\text{"the"}) = \frac{0.75}{5} \cdot 5 = 0.75$，这0.75按照延续概率重新分配给所有词（包括那些未在"the"后面出现的词）。这个例子展示了Kneser-Ney的核心机制：它承认MLE的系统性偏差（高估0.2），扣除0.75的概率质量，然后将这些质量按照"词的通用性"进行全局再分配——本质上，这是一个从"局部计数"到"全局统计"的信息融合过程。

---

## 2 信息论语言模型

### 2.1 熵与联合熵

信息论为语言模型的评估提供了严格的数学框架。这一框架的核心由香农（Shannon）在1948年建立，其基本假设是：*信息的产生可以建模为一个随机过程*，而信息的平均不确定性可以被精确量化。

对于一个离散随机变量 $X$，其取值为 $x \in \mathcal{X}$，概率分布为 $P(x)$，香农熵（Shannon Entropy）定义为：

$$H(X) = -\sum_{x \in \mathcal{X}} P(x) \log P(x)$$

熵的物理意义是：在不知道 $X$ 取值的条件下，描述 $X$ 所需要的*期望比特数*。熵越大，不确定性越高；熵越小，分布越集中。当 $X$ 均匀分布时，$H(X) = \log |\mathcal{X}|$，达到最大值；当 $X$ 几乎确定时，$H(X) \approx 0$。

对于两个离散随机变量 $X$ 和 $Y$，联合熵定义为：

$$H(X, Y) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(x, y) \log P(x, y)$$

条件熵衡量在已知 $X$ 的条件下，$Y$ 的剩余不确定性：

$$H(Y \mid X) = -\sum_{x, y} P(x, y) \log P(y \mid x) = H(X, Y) - H(X)$$

链式法则将联合熵分解为条件熵的和：

$$H(X_1, X_2, \ldots, X_T) = \sum_{t=1}^T H(X_t \mid X_1^{t-1})$$

这一定律与语言模型的链式分解完全对应，建立了信息论与语言建模之间的桥梁。

### 2.2 交叉熵

给定真实分布 $P$ 和模型分布 $Q$，交叉熵（Cross-Entropy）衡量用 $Q$ 编码来自 $P$ 的样本所需的平均比特数：

$$H(P, Q) = -\sum_x P(x) \log Q(x) = \sum_x P(x) \log \frac{1}{Q(x)}$$

交叉熵与熵之间满足基本不等式：

$$H(P) \leq H(P, Q)$$

等号成立当且仅当 $Q = P$ 几乎处处成立（以 $P$ 为测度）。这意味着：最优编码策略必须与真实分布完全一致，任何偏差都会导致编码效率下降。在语言模型的语境中，$H(P, Q)$ 衡量的是：按照真实语言分布 $P$ 采样文本时，用模型 $Q$ 进行预测所需的*平均 surprisal*。如果 $Q$ 与 $P$ 完全吻合，交叉熵等于真实熵；否则，交叉熵严格大于熵，差值为KL散度 $D_{\text{KL}}(P \parallel Q)$：

$$H(P, Q) = H(P) + D_{\text{KL}}(P \parallel Q)$$

### 2.3 困惑度与熵率

困惑度（Perplexity, PPL）是语言模型评估的黄金指标，其定义简洁而深刻。为了完整理解困惑度，需要从随机过程的熵率（Entropy Rate）出发。

**熵率的严格定义。** 对于一个平稳随机过程 $\{X_t\}_{t=1}^{\infty}$，其熵率定义为：

$$H(\{X_t\}) = \lim_{T \to \infty} \frac{1}{T} H(X_1, X_2, \ldots, X_T) = \lim_{T \to \infty} H(X_T \mid X_1^{T-1})$$

第二个等式利用了链式法则和平稳性假设。熵率衡量的是随机过程中每个随机变量的*平均不确定性*。对于语言这样的随机过程，熵率 $H(\{X_t\})$ 是语言的*内在信息密度*——它与模型无关，仅取决于语言本身的统计结构。英语的熵率估计约为每词0.6到1.3比特（取决于词表划分方式），这意味着英语中每个词携带约0.6到1.3比特的"新信息"。

**困惑度的完整数学定义。** 给定真实分布 $P$ 和模型分布 $Q$，困惑度可以定义为熵率的指数：

$$\text{PPL}(P, Q) = 2^{H(P, Q)}$$

其中 $H(P, Q)$ 是交叉熵率（Cross-Entropy Rate）：

$$H(P, Q) = \lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T H(P_t, Q_t) = \lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^T \left( -\sum_{w_t} P(w_t \mid w_1^{t-1}) \log Q(w_t \mid w_1^{t-1}) \right)$$

在实践中，上述极限无法精确计算（需要无限长的语料），因此使用有限测试语料进行蒙特卡洛估计：

$$\widehat{\text{PPL}} = \exp\left( \frac{1}{N} \sum_{t=1}^N -\log Q(w_t \mid w_1^{t-1}) \right) = \exp\left( \frac{1}{N} \sum_{t=1}^N \ell_{\text{CE}}(w_t) \right)$$

其中 $\ell_{\text{CE}}(w_t) = -\log Q(w_t \mid w_1^{t-1})$ 是第 $t$ 个词的交叉熵损失（使用自然对数），$N$ 是测试语料的词数。当底数为2时，公式变为：

$$\widehat{\text{PPL}} = 2^{-\frac{1}{N} \sum_{t=1}^N \log_2 Q(w_t \mid w_1^{t-1})}$$

**困惑度的物理意义：加权平均分支因子。** 困惑度最深刻的直觉是：它可以被理解为模型在每一步预测时的*加权平均分支因子（Weighted Average Branching Factor）*。具体而言，考虑一个虚构的"决策过程"：在每一步，模型面对 $b$ 个等概率的选择，因此每一步的 surprisal 为 $\log b$。困惑度问的问题是："在每一步，模型认为有多少个等价的选择？"

如果 $\text{PPL} = 100$（使用底数2），则模型在每一步的平均 surprisal 为 $\log_2 100 \approx 6.64$ 比特。这意味着模型的不确定性等同于从 $2^{6.64} \approx 100$ 个等概率符号中均匀选择。如果 $\text{PPL} = 20$，模型的不确定性下降到原来的五分之一——平均 surprisal 为 $\log_2 20 \approx 4.32$ 比特，等同于约20个等概率选择。当 $\text{PPL} = 1$ 时，模型完全确定（每一步的 surprisal 为0比特）；当 $\text{PPL} = V$（词表大小）时，模型等同于均匀随机猜测。

这一解释的数学基础是：对于均匀分布 $U(b)$（$b$ 个等概率选择），熵为 $\log b$，困惑度为 $2^{\log b} = b$。对于非均匀分布，困惑度 $2^H$ 给出了一个"等效均匀分支数"，它权衡了选择的数量和选择的不均匀程度。

**困惑度与泛化误差的关系。** 从统计学习理论视角，困惑度与模型的泛化误差之间存在深刻联系。考虑两个极限情况：

1. **过拟合极限：** 如果模型在训练集上记忆了每一个序列（包括异常序列），则训练集困惑度为1，但测试集困惑度可能极高——模型将高概率分配给训练集中出现过的罕见序列，而在测试集中遇到这些序列时给出虚高的概率。

2. **欠拟合极限：** 如果模型退化为均匀分布（每个词概率为 $1/V$），则困惑度恰为 $V$——这是最糟糕的基准。

好的模型应该在训练集和测试集困惑度之间取得平衡。测试集困惑度的下界由语言的熵率 $H(\{X_t\})$ 决定：

$$\text{PPL}_{\text{test}} \geq 2^{H(\{X_t\})}$$

等号成立当且仅当模型完美学习了语言的真实分布（即 $Q = P$ 几乎处处成立）。在实际中，即使是最先进的模型，其困惑度也远高于这一下界——这表明语言的生成结构（如递归、组合、长距离依赖）中仍有大量未被模型捕获的信息。

**困惑度的局限性。** 尽管困惑度是语言模型评估的黄金标准，它存在若干重要局限：

首先，困惑度是*内在*指标，与下游任务性能无必然联系。一个模型可能有极低的困惑度，但在特定任务（如问答或摘要）上表现不佳，因为困惑度仅衡量序列概率，不直接衡量语义相关性。例如，一个模型可能在语法正确的无意义句子上获得低困惑度（因为语法正确的概率高），但在需要世界知识的推理任务上失败。

其次，困惑度对词表划分敏感。相同的底层分布，如果使用字符级词表而非词级词表，困惑度会显著降低——因为字符级的"分支因子"远小于词级。跨模型的困惑度比较必须在相同词表上进行。

第三，困惑度假设测试语料独立同分布（i.i.d.）于训练语料。如果测试语料来自不同领域（如新闻 vs. 社交媒体），困惑度可能虚高，但这反映的是领域差异而非模型缺陷。

第四，困惑度对词汇表大小 $V$ 存在系统性偏差。当 $V$ 较大时，即使模型将所有概率质量集中在一个小词表子集上，困惑度也可能较低——因为"错误"词的概率被稀释到大量不可能的词上。这一问题在评估生成模型时尤为突出。

**困惑度的变体：符号困惑度与词困惑度。** 在实践中，困惑度可以根据粒度的不同报告为符号困惑度（Symbol Perplexity）或词困惑度（Word Perplexity）。符号困惑度以字符或字节为单位计算，对所有符号（包括空格、标点）一视同仁；词困惑度以词为单位计算，仅对词表中的词评估。符号困惑度通常低于词困惑度（因为字符级的选择空间更小），但二者之间的差距反映了词的内部结构信息量。

对于自回归语言模型，困惑度还可以按照生成方向报告：左-to右困惑度（每个词仅看到左侧上下文）和双向困惑度（每个词看到左右两侧上下文）。双向困惑度总是低于左-to右困惑度，差距的大小反映了模型从左到右的因果约束的严格程度。

**困惑度的历史与经验数据。** 困惑度作为评估指标的历史与语言模型本身同步。早期n-gram模型的困惑度通常在数百量级（如trigram on WSJ约为100到200）。神经语言模型的出现显著降低了困惑度：Bengio的NNLM将 perplexity 降至约200以下；Mikolov的RNNLM将 perplexity 降至约100以下。进入预训练时代后，大规模Transformer模型的困惑度进一步骤降：GPT-2在WebText上的困惑度约为37；GPT-3在多种基准上的困惑度约为20到30。然而，正如前面所述，这些数字的下界由语言的熵率决定——目前最优模型的困惑度仍远高于理论上限，这意味着自然语言的建模仍有巨大的改进空间。

---

## 3 词表示的学习

在n-gram模型的框架下，每个词被视作一个原子符号——词表中的一个离散索引。这种"符号主义"视角的问题在于：它将所有词汇对视为等距，不编码任何语义相似性。例如，"cat"和"dog"在n-gram模型中的关系与"cat"和"xylophone"完全相同，但前者在语义上远为接近。解决这一问题需要为词赋予*分布式表示*（Distributed Representation），即词向量。

### 3.1 Brown聚类

Brown聚类（Brown Clustering）由Peter Brown等人于1992年提出，是一种基于类别的词聚类算法，是神经语言模型之前最重要的词表示方法之一。其核心思想是：通过层次化聚类将词汇表划分为一棵二叉树，树中每个叶节点对应一个词，每个内部节点对应一个词类。同一子树下的词在分布上相似——它们倾向于出现在相似的上下文中。Brown聚类在SMT系统中被广泛用作特征（如短语翻译的特征函数），并在现代BERT等模型中以"词素级掩码"的形式获得了某种复兴。

**从语言模型到聚类的形式化桥梁。** 从形式上看，Brown聚类可以被理解为一种*层次化的隐马尔可夫模型*（Hierarchical HMM），其中"类"是隐变量，"词"是观测变量。给定词序列 $w_1^T$ 和对应的类序列 $c_1^T$，模型的似然为：

$$P(w_1^T) = \sum_{c_1^T} \prod_{t=1}^T P(c_t \mid c_{t-1}) P(w_t \mid c_t)$$

其中 $P(c_t \mid c_{t-1}) = P(c_t)$ 假设类转移是平稳的（无状态），即类序列是一个i.i.d.过程——类与类之间没有马尔可夫依赖。这一简化是Brown模型的关键假设：它放弃了类序列的动态结构，将类视作独立同分布的隐变量。进一步，$P(w_t \mid c_t) = \frac{c(c_t, w_t)}{c(c_t)}$，即词在给定类下的条件频率。

Brown聚类的优化目标是*最大化似然*，等价于最小化每词的负对数似然：

$$\mathcal{L}_{\text{Brown}} = -\frac{1}{T} \sum_{t=1}^T \log \sum_c P(c) P(w_t \mid c)$$

由于类序列是隐变量（不可观测），该问题使用*广义期望最大化*（Generalized EM, GEM）求解：E步计算后验 $P(c \mid w_t) \propto P(c) P(w_t \mid c)$；M步固定后验，更新 $P(c)$ 和 $P(w \mid c)$。E步和M步交替进行，直到收敛。从贝叶斯视角看，Brown聚类是在"类-词"生成模型下寻找最大似然参数的过程，与LDA等主题模型的推导同源，但Brown聚类更早、更简单。

**贪婪合并：从扁平聚类到二叉树。** 上述双变量模型仅产生一个扁平的聚类（所有词被分配到一个平坦的类集合中）。Brown算法的实际输出是一棵深度为 $K$ 的二叉树（$K$ 为预设参数，通常为100或1000），通过*贪婪合并*（Greedy Merging）构造。算法步骤如下：

步骤1（初始化）：词汇表中的每个词自成一类，形成 $V$ 个初始类 $\{c_1, \ldots, c_V\}$，其中每个类仅包含一个词。初始时，每类的概率 $P(c_i) = 1/V$。

步骤2（类对评估）：对于每一对类 $(c_i, c_j)$，计算合并后的似然增益。合并两个类 $c_i$ 和 $c_j$ 意味着将它们合并为一个新类 $c_{ij}$，其概率为 $P(c_{ij}) = P(c_i) + P(c_j)$，其词分布为 $P(w \mid c_{ij}) = [P(c_i) P(w \mid c_i) + P(c_j) P(w \mid c_j)] / [P(c_i) + P(c_j)]$。合并后的总似然为：

$$\mathcal{L}_{\text{merged}} = \sum_{w} c(w) \log P(w \mid c_{ij})$$

其中 $c(w)$ 是词 $w$ 在语料中的总出现次数。合并的似然增益定义为：

$$\Delta \mathcal{L}(c_i, c_j) = \mathcal{L}_{\text{merged}} - \mathcal{L}_{\text{unmerged}} = \sum_w c(w) \log \frac{P(w \mid c_{ij})}{\max(P(w \mid c_i), P(w \mid c_j))}$$

（更精确的形式使用 $P(c_i) P(w \mid c_i)$ 的加权平均。）

步骤3（贪婪选择）：在所有可能的类对中，选择合并后使似然增益最大的两个类进行合并。这是一个 $O(V^2)$ 的操作（在每一步），随着合并进行，候选对数迅速减少。

步骤4（迭代）：重复步骤2和步骤3，每次合并减少一个类。从 $V$ 个类开始，经过 $V - K$ 次合并后，剩下 $K$ 个类。

步骤5（编码）：最终的二叉树中，每个叶节点对应一个词。从根节点到叶节点的路径产生一个比特串——在每一步合并中，向左走编码为0，向右走编码为1（或反之）。这个比特串作为词的类标签（Class Label）。例如，当 $K = 100$ 时，树的深度为 $\lceil \log_2 100 \rceil \approx 7$，每个词被编码为一个7比特的类标签。同类的词（如"Monday"和"Tuesday"）共享较长的公共前缀，反映了语义上的相似性。

**类标签的使用方式。** Brown聚类产生的类标签可以以多种方式集成到下游任务中。最简单的方法是将类标签作为额外的特征：对于每个词 $w$，其Brown类标签 $b(w)$ 是一个离散索引，可以嵌入为一个向量 $\mathbf{e}_{b(w)}$，与词嵌入拼接或相加。更精细的方法是将类标签的层次结构纳入模型：同一子树下的词共享一部分参数（如后缀共享）。在SMT中，Brown类标签被用作短语翻译的额外特征函数，显著提升了翻译质量。

**计算复杂度与 scalability。** 贪婪合并的核心瓶颈在于步骤2的 $O(V^2)$ 类对评估。对于 $V = 100,000$，初始步骤需要评估约 $5 \times 10^9$ 个类对，这在计算上不可行。实践中采用多种近似策略：首先，只评估那些至少在语料中共现过的类对（使用共现矩阵的稀疏性）；其次，使用堆结构维护候选合并的优先级；第三，对高斯基模等快速近似方法进行预筛选。尽管如此，Brown聚类的完整实现仍然需要大量计算资源，这也是它被后来的神经方法部分取代的原因之一。

**与当代词向量的本质区别。** Brown聚类产生的是*离散的类分配*，而神经语言模型产生的是*连续的向量表示*。类标签只能捕获有限的粒度信息——同一类中的词被视作等价（或通过比特串前缀编码近似等价），而词向量可以在连续空间中编码梯度相似性。例如，词向量可以表示"dog"与"cat"比"dog"与"xylophone"更相似，且可以量化这种相似程度；Brown聚类只能表示"dog"和"cat"共享某个类标签前缀，无法表达相似程度的差异。然而，Brown聚类的优势在于其离散性和层次性——类标签可以直接作为特征输入线性模型（如SMT的特征函数），而词向量需要额外的下游架构来消费。

### 3.2 神经语言模型（NNLM，Bengio 2003）

Yoshua Bengio等人在2003年提出的神经语言模型（Neural Language Model, NNLM）是深度学习进入自然语言处理的里程碑。该模型首次为每个词学习了一个*分布式向量表示*（即词嵌入），并通过神经网络来估计条件概率，同时解决了维度灾难和语义泛化两个核心问题。

**模型架构：完整的前向传播。** NNLM的完整架构包含四个组件：词嵌入矩阵、上下文拼接层、隐藏层和输出层。给定词汇表 $\mathcal{V}$（大小 $| \mathcal{V} | = V$）和嵌入维度 $N$，模型首先通过一个嵌入矩阵 $\mathbf{C} \in \mathbb{R}^{V \times N}$ 将每个词映射为一个 $N$ 维向量。矩阵 $\mathbf{C}$ 的第 $i$ 行 $\mathbf{C}_i$ 是词 $w_i$ 的嵌入向量。对于长度为 $n-1$ 的上下文 $w_{t-n+1}^{t-1}$，模型将其嵌入向量拼接为一个高维向量：

$$\mathbf{x} = \left[ \mathbf{C}_{w_{t-n+1}} ; \mathbf{C}_{w_{t-n+2}}; \ldots ; \mathbf{C}_{w_{t-1}} \right] \in \mathbb{R}^{(n-1)N}$$

拼接操作将来自不同位置的嵌入向量串联起来，保留了位置信息——这是NNLM与后续CBOW模型的关键区别（CBOW对上下文嵌入求和或平均，丢失了位置信息）。拼接后的向量 $\mathbf{x}$ 输入一个隐藏层（全连接层），经tanh激活函数变换：

$$\mathbf{h} = \tanh(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \in \mathbb{R}^{|\mathcal{H}|}$$

其中 $\mathbf{W}_1 \in \mathbb{R}^{|\mathcal{H}| \times (n-1)N}$，$\mathbf{b}_1 \in \mathbb{R}^{|\mathcal{H}|}$，$|\mathcal{H}|$ 是隐藏层大小（Bengio原文中通常取500到2000）。tanh激活函数将每个隐藏单元的输出压缩到 $[-1, 1]$ 区间，提供了必要的非线性变换能力。最后，输出层通过softmax函数产生词表上的概率分布：

$$\hat{\mathbf{y}} = \text{softmax}(\mathbf{W}_2 \mathbf{h} + \mathbf{b}_2) \in \mathbb{R}^V$$

其中 $\mathbf{W}_2 \in \mathbb{R}^{V \times |\mathcal{H}|}$，$\mathbf{b}_2 \in \mathbb{R}^V$，$P(w_t = i \mid w_{t-n+1}^{t-1}) = \hat{y}_i$。Softmax函数将输出层的线性分数转换为概率分布：

$$\hat{y}_i = \frac{\exp(\mathbf{W}_{2,i} \cdot \mathbf{h} + b_{2,i})}{\sum_{j=1}^V \exp(\mathbf{W}_{2,j} \cdot \mathbf{h} + b_{2,j})}$$

其中 $\mathbf{W}_{2,i}$ 是 $\mathbf{W}_2$ 的第 $i$ 行。

**参数数量与计算复杂度分析。** NNLM的参数量为：

- 嵌入矩阵 $\mathbf{C}$：$V \times N$
- 隐藏层权重 $\mathbf{W}_1$：$(n-1)N \times |\mathcal{H}|$
- 隐藏层偏置 $\mathbf{b}_1$：$|\mathcal{H}|$
- 输出层权重 $\mathbf{W}_2$：$|\mathcal{H}| \times V$
- 输出层偏置 $\mathbf{b}_2$：$V$

总参数量约为 $V \times N + (n-1)N \times |\mathcal{H}| + |\mathcal{H}| + |\mathcal{H}| \times V + V$。当 $V = 10000$，$N = 100$，$|\mathcal{H}| = 500$，$n = 3$ 时，总参数量约为 $10^6 + 1.5 \times 10^6 + 500 + 5 \times 10^6 + 10^4 \approx 7.6 \times 10^6$，其中输出层权重 $\mathbf{W}_2$ 占主导地位。这一观察直接引出了后续优化（如Skip-gram架构、噪声对比估计等）的核心动机：输出层的 $O(V)$ 复杂度是大规模语言模型的主要瓶颈。

**训练目标与优化细节。** 模型通过最小化训练语料上的负对数似然（NLL）来学习参数 $\theta = \{ \mathbf{C}, \mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2 \}$：

$$\mathcal{J}(\theta) = -\sum_{t=1}^T \log P(w_t \mid w_{t-n+1}^{t-1}) + \lambda \left( \| \mathbf{W}_1 \|_F^2 + \| \mathbf{W}_2 \|_F^2 \right)$$

其中 $\lambda$ 是L2正则化系数（Weight Decay），用于防止过拟合。注意嵌入矩阵 $\mathbf{C}$ 不加正则化——这一设计选择反映了Bengio等人的洞见：嵌入向量应该自由地适应任务目标，不应被范数约束限制。优化使用随机梯度下降（SGD）及其变种，每步处理一个训练样本。原始论文使用学习率衰减策略：初始学习率较大（如0.01），随着训练进行逐渐减小。每个epoch后随机打乱训练语料的顺序，以利用SGD的随机性跳出局部极小值。

**核心创新与意义的深入分析。** NNLM的意义远超其任务性能，它是NLP进入深度学习时代的真正起点。首先，嵌入矩阵 $\mathbf{C}$ 将每个词表示为连续向量，使得"king - man + woman ≈ queen"这样的类比推理成为可能——这是对词义进行代数操作的开端。这一性质的数学解释是：如果向量空间的结构与语言的语义结构对齐，则词对之间的语义关系（如性别、时态、语态）可以在向量空间中通过线性变换来编码。例如，如果"king"和"queen"的性别差异在向量空间中对应于某个方向 $\mathbf{v}_{\text{gender}}$，则 $\mathbf{C}_{\text{king}} - \mathbf{C}_{\text{man}} + \mathbf{C}_{\text{woman}} \approx \mathbf{C}_{\text{queen}}$ 当且仅当 $\mathbf{C}_{\text{king}} - \mathbf{C}_{\text{queen}} \approx \mathbf{C}_{\text{man}} - \mathbf{C}_{\text{woman}}$，即性别向量 $\mathbf{v}_{\text{gender}} = \mathbf{C}_{\text{king}} - \mathbf{C}_{\text{queen}}$ 在"king-man"和"queen-woman"之间共享。

其次，嵌入层使模型具有*泛化能力*：训练语料中未出现过的词组合（如"red car"），只要"red"和"car"各自在训练中出现过，模型就能给出合理的估计——因为"red"的嵌入和"car"的嵌入分别编码了它们的语义信息，模型可以通过这些嵌入的组合来推断未见组合的概率。相比之下，n-gram模型对任何未出现的n-gram输出概率零，这是离散符号主义的根本性局限。

第三，NNLM是第一个将表示学习与预测学习统一在端到端框架中的模型：嵌入矩阵 $\mathbf{C}$ 不是独立训练的（如通过聚类或矩阵分解预训练），而是随整个模型一同优化，因此词的表示直接服务于语言建模目标。这种"任务驱动表示"的理念是当代预训练语言模型的哲学基石。

**与后续架构的继承关系。** NNLM的架构虽然简单，但它包含了现代语言模型的几乎所有核心组件：词嵌入层（Transformer中的Token Embedding）、非线性隐藏层（Transformer中的Feed-Forward层）、以及softmax输出层（Transformer中的Language Model Head）。现代Transformer模型的"简化版本"与NNLM的架构差异仅在于：（1）使用自注意力替代全连接隐藏层来建模长距离依赖；（2）使用Layer Normalization和残差连接来稳定深层训练；（3）使用更复杂的嵌入方案（如位置嵌入、分段嵌入）。在本质上，Transformer是NNLM在序列建模深度和宽度上的规模化延伸。

---

## 4 Word2Vec：高效的词向量学习

Word2Vec由Mikolov等人于2013年提出，是NNLM的轻量化、高效版本。其设计哲学是：如果目标是学习高质量的词向量而非获得最佳的语言模型，那么可以通过大幅简化架构来显著提升训练效率。Word2Vec包含两种架构和两种训练优化技术。

### 4.1 CBOW与Skip-gram架构

**连续词袋模型（CBOW）。** CBOW（Continuous Bag-of-Words）的目标是：给定上下文词，预测中心词。给定窗口大小 $m$，训练样本为 $(\mathbf{c}, \mathbf{o})$，其中 $\mathbf{c}$ 是上下文词的嵌入向量集合，$\mathbf{o}$ 是中心词的one-hot编码。对于上下文窗口 $\{w_{t-m}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+m}\}$，模型将上下文嵌入取平均（或求和），然后预测中心词 $w_t$：

$$\mathbf{h} = \frac{1}{2m} \sum_{-m \leq j \leq m, j \neq 0} \mathbf{C}_{w_{t+j}}$$

$$\hat{\mathbf{y}} = \text{softmax}(\mathbf{W} \mathbf{h} + \mathbf{b})$$

CBOW的优势在于其简单性和训练速度——每个样本仅需计算一次softmax。它在高频词上的表现优于Skip-gram，但对低频词的建模能力较弱。

**Skip-gram模型。** Skip-gram的目标与CBOW相反：给定中心词，预测其上下文词。对于中心词 $w_t$ 和上下文位置 $j \in \{-m, \ldots, m\} \setminus \{0\}$，Skip-gram预测 $w_{t+j}$：

$$\hat{\mathbf{y}}_j = \text{softmax}(\mathbf{W} \mathbf{C}_{w_t} + \mathbf{b})$$

一个中心词产生 $2m$ 个预测目标，因此Skip-gram的训练样本数是CBOW的 $2m$ 倍。然而，Skip-gram在低频词和类比推理任务上表现更优——每个词直接作为输入，其嵌入向量在训练中被反复更新，获得更丰富的语义表示。

**统一的优化目标。** CBOW和Skip-gram的完整最大似然目标为：

$$\mathcal{J}_{\text{MLE}}(\theta) = -\sum_{t=1}^T \left( \log P(w_t \mid w_{t-m}^{t-1}, w_{t+1}^{t+m}) + \sum_{\substack{j=-m \\ j \neq 0}}^m \log P(w_{t+j} \mid w_t) \right)$$

然而，直接优化上述目标需要在每个样本上计算 $O(V)$ 复杂度的softmax，这在 $V > 10^5$ 时不可行。Mikolov等人引入了两种关键技术来加速训练：负采样和层次Softmax。

### 4.2 负采样

负采样（Negative Sampling, NEG）将多分类问题转化为二分类问题。对于每个正样本（中心词-上下文词对），随机采样 $K$ 个负样本（噪声词），训练一个sigmoid分类器来区分正样本和负样本。这一技术是Word2Vec能够在大规模词表上高效训练的核心突破。

**完整的概率模型推导。** 对于Skip-gram，给定正样本对 $(w, c)$（中心词 $w$，上下文词 $c$），NEG定义如下生成过程：

1. 从噪声分布 $P_n$ 中独立采样 $K$ 个噪声词 $w_1, \ldots, w_K \sim P_n$。
2. 对于正样本对 $(w, c)$，定义其"标签"为 $D=1$（正类）。
3. 对于每个噪声词 $w_k$，定义其"标签"为 $D=0$（负类）。

模型假设正样本和负样本的得分由内积决定：

$$P(D=1 \mid w, c) = \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c) = \frac{1}{1 + e^{-\mathbf{v}_w^{\top} \mathbf{v}_c}}$$

$$P(D=0 \mid w, w_k) = \sigma(-\mathbf{v}_w^{\top} \mathbf{v}_{w_k}) = \frac{1}{1 + e^{\mathbf{v}_w^{\top} \mathbf{v}_{w_k}}}$$

其中 $\sigma(x) = 1 / (1 + e^{-x})$ 是sigmoid函数，$\mathbf{v}_w$ 和 $\mathbf{v}_c$ 分别是从输入和输出嵌入矩阵中查得的向量。NEG的优化目标为：

$$\mathcal{J}_{\text{NEG}}(w, c) = \log \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c) + \sum_{k=1}^K \mathbb{E}_{w_k \sim P_n} \left[ \log \sigma(-\mathbf{v}_w^{\top} \mathbf{v}_{w_k}) \right]$$

从联合概率视角看，NEG试图最大化正样本的互信息，同时最小化负样本的互信息。等价地，NEG优化以下二分类交叉熵：

$$\mathcal{J}_{\text{NEG}} = -\mathbb{E}_{(w,c) \sim P} \left[ \log \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c) \right] - \mathbb{E}_{(w, w_k) \sim P \times P_n} \left[ \log \sigma(-\mathbf{v}_w^{\top} \mathbf{v}_{w_k}) \right]$$

其中 $P$ 是正样本的经验分布（从语料中观测到的词-上下文共现分布），$P_n$ 是噪声分布。这一目标函数的理论基础是*噪声对比估计*（Noise-Contrastive Estimation, NCE）：通过将学习概率密度的问题转化为区分样本来自真实分布还是噪声分布的判别问题，可以绕过概率密度的归一化常数计算。

**噪声分布的设计。** 噪声分布 $P_n$ 的选择对训练动力学和最终词向量质量有重要影响。经验上，$P_n$ 通常取为词频的3/4次幂：

$$P_n(w) = \frac{c(w)^{3/4}}{\sum_{u \in \mathcal{V}} c(u)^{3/4}}$$

这一选择的原因可以从多个角度理解。从信息论角度，$P_n$ 应该反映"哪些词是信息量最低的"——高频词（如"the"、"of"）携带的信息量最低，因此在负采样中被频繁选中，迫使模型学会将它们与中心词推开。从实验角度，Mikolov等人发现 $3/4$ 次幂在词相似度任务上表现最优：$1/2$ 次幂过度压缩高频词，导致训练不稳定；$1$ 次幂（即均匀采样）对高频词的惩罚不足；$3/4$ 次幂在二者之间取得了平衡。Goldberg和Levy（2014）的后续分析表明，$P_n$ 的选择对最终词向量质量影响相对有限——只要 $P_n$ 不过度偏离均匀分布，SGNS都能学习到高质量的词向量。

**完整梯度推导。** 对正样本 $(w, c)$，NEG目标对 $\mathbf{v}_c$ 的梯度为：

$$\frac{\partial \mathcal{J}_{\text{NEG}}}{\partial \mathbf{v}_c} = \frac{\partial}{\partial \mathbf{v}_c} \left[ \log \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c) + \sum_{k=1}^K \log \sigma(-\mathbf{v}_w^{\top} \mathbf{v}_{w_k}) \right]$$

注意 $\mathbf{v}_{w_k}$ 不依赖于 $\mathbf{v}_c$，因此第二项对 $\mathbf{v}_c$ 的梯度为零。仅第一项贡献：

$$\frac{\partial \log \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c)}{\partial \mathbf{v}_c} = \frac{1}{\sigma(\mathbf{v}_w^{\top} \mathbf{v}_c)} \cdot \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c) \cdot (1 - \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c)) \cdot \mathbf{v}_w = (1 - \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c)) \cdot \mathbf{v}_w$$

因此：

$$\frac{\partial \mathcal{J}_{\text{NEG}}}{\partial \mathbf{v}_c} = (1 - \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c)) \cdot \mathbf{v}_w = \sigma(-\mathbf{v}_w^{\top} \mathbf{v}_c) \cdot \mathbf{v}_w$$

同理，对 $\mathbf{v}_w$ 的梯度需要考虑正样本和所有负样本的贡献：

$$\frac{\partial \mathcal{J}_{\text{NEG}}}{\partial \mathbf{v}_w} = \sigma(-\mathbf{v}_w^{\top} \mathbf{v}_c) \cdot (-\mathbf{v}_c) + \sum_{k=1}^K \sigma(\mathbf{v}_w^{\top} \mathbf{v}_{w_k}) \cdot \mathbf{v}_{w_k}$$

注意梯度的不对称性：正样本将 $\mathbf{v}_c$ 向 $-\mathbf{v}_w$ 方向推动（当 $\sigma(\mathbf{v}_w^{\top} \mathbf{v}_c)$ 接近1时，梯度趋近于0，表示"已足够接近"；当 $\sigma(\mathbf{v}_w^{\top} \mathbf{v}_c)$ 接近0时，梯度趋近于 $-\mathbf{v}_w$，表示需要大幅靠近）；负样本将 $\mathbf{v}_{w_k}$ 向 $\mathbf{v}_w$ 方向推动（当 $\sigma(\mathbf{v}_w^{\top} \mathbf{v}_{w_k})$ 接近1时，梯度趋近于 $\mathbf{v}_w$，表示需要大幅远离；当接近0时，梯度趋近于0，表示"已足够远离"）。

这一梯度不对称性揭示了NEG的隐式几何结构：正样本对倾向于使向量靠近（拉近距离），负样本对倾向于使向量远离（推开距离）。最终，嵌入向量在语义空间中自发地组织成"相似词靠近、不相似词远离"的结构——这与Word2Vec的词向量可视化结果完全吻合。

**NEG与矩阵分解的完整关系。** Levy和Goldberg（2014）的著名定理证明了SGNS的目标函数等价于对一个特定的 shifted PMI 矩阵进行非负矩阵分解。完整推导如下：

考虑SGNS的全局目标函数（对所有训练样本求和）：

$$\mathcal{J}_{\text{SGNS}} = \sum_{(w,c) \in \mathcal{D}} \left[ \log \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c) + \sum_{k=1}^K \log \sigma(-\mathbf{v}_w^{\top} \mathbf{v}_{w_k}) \right]$$

其中 $\mathcal{D}$ 是所有正样本对（从滑动窗口中提取）的集合。将sigmoid展开：

$$\mathcal{J}_{\text{SGNS}} = \sum_{(w,c) \in \mathcal{D}} \left[ -\log(1 + e^{-\mathbf{v}_w^{\top} \mathbf{v}_c}) - \sum_{k=1}^K \log(1 + e^{\mathbf{v}_w^{\top} \mathbf{v}_{w_k}}) \right]$$

假设存在全局最优解 $\{\mathbf{v}_w^*\}$ 使得所有梯度为零。对于固定的 $w$ 和所有 $c$，最优性条件为：

$$\frac{\partial \mathcal{J}_{\text{SGNS}}}{\partial \mathbf{v}_w} = \sum_{c \in \mathcal{V}} \left[ \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c) - \mathbb{I}[(w, c) \in \mathcal{D}] \right] \mathbf{v}_c = 0$$

其中 $\mathbb{I}[(w, c) \in \mathcal{D}]$ 是指示函数，当 $(w, c)$ 为正样本对时为1，否则为0。将上述方程展开：

$$\sum_{c: (w,c) \in \mathcal{D}} \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c) \mathbf{v}_c + \sum_{c: (w,c) \notin \mathcal{D}} \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c) \mathbf{v}_c = \sum_{c: (w,c) \in \mathcal{D}} \mathbf{v}_c$$

假设向量 $\{\mathbf{v}_c\}$ 线性独立（这在嵌入维度足够大时成立），则对于每个 $c$，括号内的系数必须为零：

$$\sigma(\mathbf{v}_w^{\top} \mathbf{v}_c) - \mathbb{I}[(w, c) \in \mathcal{D}] = 0$$

这意味着：

$$\sigma(\mathbf{v}_w^{\top} \mathbf{v}_c) = \mathbb{I}[(w, c) \in \mathcal{D}]$$

然而，sigmoid函数的输出范围是 $(0, 1)$，而指示函数的输出是 $\{0, 1\}$——二者不可能处处相等（除非忽略 sigmoid 的饱和区）。Levy和Goldberg的关键洞察是：在 $K \to \infty$ 的极限下，负样本的期望效应使得最优解满足：

$$\mathbf{v}_w^{\top} \mathbf{v}_c = \log \frac{P(w, c)}{P(w) P_n(c)} - \log K = \text{PMI}(w, c) + \log \frac{P(c)}{P_n(c)} - \log K$$

当 $P_n$ 被选为 $P_n(c) \propto c(c)^{3/4}$ 时，$\log \frac{P(c)}{P_n(c)}$ 项影响相对有限，主要项为 $\text{PMI}(w, c) - \log K$（在 $P_n(c) = P(c)$ 的简化分析下精确成立）。

因此，SGNS隐式地最小化以下重构误差：

$$\min_{\mathbf{W}, \mathbf{C}} \sum_{w, c} \left( \mathbf{W}_w^{\top} \mathbf{C}_c - \text{PMI}(w, c) + \log K \right)^2$$

其中 $\mathbf{W}$ 和 $\mathbf{C}$ 分别是中心词矩阵和上下文词矩阵。令 $\mathbf{M} = \mathbf{W} \mathbf{C}^{\top}$，其中 $M_{wc} = \text{PMI}(w, c) - \log K$，则SGNS等价于对矩阵 $\mathbf{M}$ 的非负矩阵分解。Levy和Goldberg由此证明：SGNS学习到的词向量矩阵 $\mathbf{W}$（或 $\mathbf{C}$）是PMI矩阵的低秩近似，秩由嵌入维度 $N$ 控制。这一发现将词向量的经验成功与成熟的矩阵分解理论联系起来，为理解词向量的内在结构提供了理论依据。

### 4.3 层次Softmax

层次Softmax（Hierarchical Softmax）由Morin和Bengio于2005年提出，后被Word2Vec采用。其核心思想是将 $V$ 路softmax替换为一棵深度为 $\lceil \log_2 V \rceil$ 的二叉树上的 $O(\log V)$ 次二分类。二叉树通常使用霍夫曼编码（Huffman Coding）构建，使得高频词具有较短的编码路径。

在层次Softmax中，输出词表 $\mathcal{V}$ 的每个词 $w$ 对应于从根节点到叶节点的一条唯一路径，路径上的节点序列记为 $n(w, 0) = \text{root}, n(w, 1), \ldots, n(w, l(w) - 1) = w$，其中 $l(w)$ 是路径长度。每个节点 $n$ 关联一个参数向量 $\mathbf{u}_n \in \mathbb{R}^N$。定义路径上的二分类标签：对每个节点 $n(w, j)$，令 $1(n(w, j), c)$ 表示节点 $n(w, j)$ 在其父节点到 $w$ 的路径上的指示函数（取值为1表示路径经过该节点的左/右分支，取值为0表示走另一分支）。

给定隐藏状态 $\mathbf{h}$（Skip-gram中为中心词嵌入 $\mathbf{v}_w$），层次Softmax的输出概率为：

$$P(w \mid \mathbf{h}) = \prod_{j=1}^{l(w)-1} \sigma\left( 1(n(w, j), c) \cdot \mathbf{u}_{n(w, j)}^{\top} \mathbf{h} \right)$$

该公式的含义是：沿着从根到 $w$ 的路径，在每一步根据sigmoid函数计算走正确分支的概率，并将所有步骤的概率相乘。由于 $1(n(w, j), c)$ 编码了方向信息，sigmoid的输出始终大于0.5（在正确路径上）。

层次Softmax将softmax的计算复杂度从 $O(V)$ 降低到 $O(\log V)$，对于 $V = 10^5$，约从 $10^5$ 降到17——这是一个约6000倍的加速。代价是引入额外的霍夫曼树参数，且所有输出词共享梯度传播路径。

### 4.4 与矩阵分解的关系：Levy和Goldberg的证明

Word2Vec（特别是Skip-gram with Negative Sampling, SGNS）与矩阵分解之间的关系是词向量理论中最深刻的成果之一。Levy和Goldberg（2014）证明了SGNS的目标函数等价于对一个特定的 shifted PMI 矩阵进行非负矩阵分解。

**点互信息。** 对于词 $w$ 和上下文词 $c$，其点互信息（Pointwise Mutual Information, PMI）定义为：

$$\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w) P(c)} = \log \frac{P(w \mid c)}{P(w)}$$

PMI衡量 $w$ 和 $c$ 的共同出现频率超出独立出现期望的程度。$\text{PMI}(w, c) > 0$ 表示正相关，$\text{PMI}(w, c) < 0$ 表示负相关，$\text{PMI}(w, c) = 0$ 表示独立。在词向量语境中，PMI矩阵 $\mathbf{M} \in \mathbb{R}^{V \times V}$ 定义为 $M_{ij} = \text{PMI}(w_i, c_j)$，其每一行 $M_i$ 可以被理解为词 $w_i$ 在所有上下文词上的PMI向量。

**SGNS目标与矩阵分解的等价性。** SGNS的目标函数（以词 $w$ 为中心词、上下文词 $c$ 为目标的单个样本，忽略上下文窗口大小的影响）为：

$$\mathcal{J}_{\text{SGNS}}(w, c) = \log \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c) + K \cdot \mathbb{E}_{w_k \sim P_n} \left[ \log \sigma(-\mathbf{v}_{w_k}^{\top} \mathbf{v}_c) \right]$$

其中 $\sigma(x) = 1/(1+e^{-x})$。将其展开为sigmoid的对数形式：

$$\mathcal{J}_{\text{SGNS}} = \log \frac{1}{1 + e^{-\mathbf{v}_w^{\top} \mathbf{v}_c}} + K \sum_{k=1}^K \log \frac{1}{1 + e^{\mathbf{v}_{w_k}^{\top} \mathbf{v}_c}}$$

$$= -\log(1 + e^{-\mathbf{v}_w^{\top} \mathbf{v}_c}) - K \sum_{k=1}^K \log(1 + e^{\mathbf{v}_{w_k}^{\top} \mathbf{v}_c})$$

Levy和Goldberg的关键洞见是：SGNS的全局最优解（忽略嵌入向量大小约束）使得对于所有在语料中实际出现的 $(w, c)$ 对和噪声样本 $(w, w_k)$，有：

$$\mathbf{v}_w^{\top} \mathbf{v}_c = \text{PMI}(w, c) - \log K$$

$$\mathbf{v}_{w_k}^{\top} \mathbf{c} = \text{PMI}(w_k, c) - \log K$$

即，内积编码了 shifted PMI。证明思路如下：对 $\mathcal{J}_{\text{SGNS}}$ 求导并令导数为零，得到最优性条件：

$$\frac{\partial \mathcal{J}_{\text{SGNS}}}{\partial \mathbf{v}_w} = \sum_{c \in \mathcal{V}} \left[ \sigma(\mathbf{v}_w^{\top} \mathbf{v}_c) - \mathbb{I}[(w, c) \in \mathcal{D}] \right] \mathbf{v}_c = 0$$

其中 $\mathbb{I}[(w, c) \in \mathcal{D}]$ 指示 $(w, c)$ 是否为正样本对。忽略向量大小的约束，最优解满足：

$$\mathbf{v}_w^{\top} \mathbf{v}_c = \log \frac{P(w, c)}{P(w) P_n(c)} - \log K = \text{PMI}(w, c) + \log \frac{P(c)}{P_n(c)} - \log K$$

当 $P_n$ 被选为 $P_n(c) \propto c(c)^{3/4}$ 时，$\log \frac{P(c)}{P_n(c)}$ 项影响相对有限，主要项为 $\text{PMI}(w, c) - \log K$（在 $P_n(c) = P(c)$ 的简化分析下精确成立）。因此，SGNS隐式地最小化以下重构误差：

$$\min_{\mathbf{W}, \mathbf{C}} \sum_{w, c} \left( \mathbf{W}_w^{\top} \mathbf{C}_c - \text{PMI}(w, c) + \log K \right)^2$$

其中 $\mathbf{W}$ 和 $\mathbf{C}$ 分别是中心词矩阵和上下文词矩阵。令 $\mathbf{M} = \mathbf{W} \mathbf{C}^{\top}$，其中 $M_{wc} = \text{PMI}(w, c) - \log K$，则SGNS等价于对矩阵 $\mathbf{M}$ 的矩阵分解。Levy和Goldberg由此证明：SGNS学习到的词向量矩阵 $\mathbf{W}$（或 $\mathbf{C}$）是PMI矩阵的低秩近似，秩由嵌入维度 $N$ 控制。这一发现将词向量的经验成功与成熟的矩阵分解理论联系起来，为理解词向量的内在结构提供了理论依据。

---

## 5 GloVe：全局词频统计与局部上下文窗口的统一

### 5.1 动机与核心洞见

Word2Vec的两大架构——CBOW和Skip-gram——仅使用局部上下文窗口信息，即每次仅看到一对词及其在窗口内的共现关系。这种局部性带来了两个问题：其一，训练效率较低——全局词频统计中的信息被逐对浪费；其二，难以调节全局统计信息（如词频）和局部上下文信息（如共现方向性）之间的权重。

GloVe（Global Vectors for Word Representation）由Pennington、Socher和Manning于2014年提出，其核心洞见是：*词-词共现矩阵本身就编码了丰富的语义信息，应该被直接纳入词向量的学习目标*。具体而言，GloVe认为词向量应同时满足两个条件：

1. **局部条件：** 词向量之间的点积应能重建词对的共现概率（或对数共现概率）。这保留了Word2Vec从局部窗口中学习的能力。
2. **全局条件：** 词向量应反映词在整个语料库上的全局统计结构，而非仅仅局部的共现计数。

### 5.2 数学形式化

令 $\mathbf{X} \in \mathbb{Z}_{\geq 0}^{V \times V}$ 为词-词共现矩阵，其中 $X_{ij} = c(w_i, w_j)$ 表示词 $w_j$ 出现在词 $w_i$ 的上下文窗口中的次数（共现计数）。GloVe的目标函数为：

$$\mathcal{J} = \sum_{i, j=1}^V f(X_{ij}) \left( \mathbf{w}_i^{\top} \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

其中：
- $\mathbf{w}_i \in \mathbb{R}^N$ 是词 $w_i$ 作为中心词时的嵌入向量；
- $\tilde{\mathbf{w}}_j \in \mathbb{R}^N$ 是词 $w_j$ 作为上下文词时的嵌入向量；
- $b_i$ 和 $\tilde{b}_j$ 是词 $w_i$ 和 $w_j$ 的偏置项；
- $f(\cdot)$ 是加权函数，控制不同共现频率对损失的贡献。

加权函数 $f(x)$ 的设计原则是：为高频共现分配较低的权重（因为高频词的共现计数噪声大、信息密度低），同时完全忽略零计数（因为零计数携带的信息量极低且数量庞大）。具体形式为：

$$f(x) = \begin{cases} \left( \frac{x}{x_{\max}} \right)^{\alpha} & \text{if } x < x_{\max} \\ 1 & \text{if } x \geq x_{\max} \end{cases}$$

原始论文使用 $x_{\max} = 100$ 和 $\alpha = 0.75$。$\alpha = 0.75$ 的选择遵循与Word2Vec负采样中 $3/4$ 次幂相同的经验规律。

### 5.3 对数共现比率的解释

GloVe目标函数的深层含义可以通过对数共现比率（Log Co-occurrence Ratio）来理解。对于四个词 $w_i, w_j, w_k, w_l$，考虑比率：

$$\log \frac{P_{ik}}{P_{jk}} = \log \frac{X_{ik} / X_i}{X_{jk} / X_j} = \log \frac{X_{ik}}{X_{jk}} - \log \frac{X_i}{X_j}$$

其中 $X_i = \sum_k X_{ik}$ 是词 $w_i$ 的总共现次数，$P_{ik} = X_{ik} / X_i$ 是条件概率 $P(w_k \mid w_i)$。如果词 $w_i$ 和 $w_j$ 与 $w_k$ 和 $w_l$ 相关（如 $w_i = \text{"ice"}$, $w_j = \text{"steam"}$, $w_k = \text{"solid"}$, $w_l = \text{"gas"}$），则 $P(w_k \mid w_i) \gg P(w_k \mid w_j)$，而对数比率 $\log(P_{ik} / P_{jk})$ 捕捉了这种相关性。

GloVe的关键性质是：词向量的点积 $\mathbf{w}_i^{\top} \tilde{\mathbf{w}}_j$ 直接编码对数共现比率。具体而言，如果：

$$\mathbf{w}_i^{\top} \tilde{\mathbf{w}}_k - \mathbf{w}_j^{\top} \tilde{\mathbf{w}}_k = \mathbf{w}_i^{\top} \tilde{\mathbf{w}}_l - \mathbf{w}_j^{\top} \tilde{\mathbf{w}}_l$$

则 $\log \frac{P_{ik}}{P_{jk}} = \log \frac{P_{il}}{P_{jl}}$，意味着词 $w_i$ 和 $w_j$ 相对于 $w_k$ 和 $w_l$ 的相关性相同。这种线性关系在大规模语料中得到了充分的统计满足，因此GloVe能够高效地利用全局统计信息。

### 5.4 与Word2Vec的关系：统一的矩阵分解视角

GloVe可以被视为对Word2Vec的矩阵分解视角的显式化。为了理解这一点，需要将GloVe目标函数与SGNS目标函数进行形式上的对比。

**GloVe目标函数的矩阵分解形式。** 回顾GloVe目标函数：

$$\mathcal{J} = \sum_{i, j=1}^V f(X_{ij}) \left( \mathbf{w}_i^{\top} \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

当 $f(x) \approx 1$（即忽略加权函数的影响），且忽略偏置项时，该目标近似为最小化：

$$\sum_{i, j} \left( \mathbf{w}_i^{\top} \tilde{\mathbf{w}}_j - \log X_{ij} \right)^2$$

如果令 $\mathbf{W} = [\mathbf{w}_1; \ldots; \mathbf{w}_V] \in \mathbb{R}^{V \times N}$ 和 $\tilde{\mathbf{W}} = [\tilde{\mathbf{w}}_1; \ldots; \tilde{\mathbf{w}}_V] \in \mathbb{R}^{V \times N}$，则目标变为：

$$\| \mathbf{W} \tilde{\mathbf{W}}^{\top} - \mathbf{L} \|_F^2$$

其中 $\mathbf{L}$ 是 $\log X_{ij}$ 组成的矩阵，$\| \cdot \|_F$ 是Frobenius范数。这是一个标准的矩阵分解问题：寻找秩-$N$ 矩阵 $\mathbf{W} \tilde{\mathbf{W}}^{\top}$ 来近似矩阵 $\mathbf{L}$。由于 $\mathbf{W}$ 和 $\tilde{\mathbf{W}}$ 不约束为正交或归一化，这实际上是加权非负矩阵分解（当对 $\mathbf{W}, \tilde{\mathbf{W}}$ 加上非负约束时）。

**与SGNS矩阵分解的等价性。** 将 $\log X_{ij}$ 展开：

$$\log X_{ij} = \log P(i, j) + \log T$$

其中 $T = \sum_{i,j} X_{ij}$ 是共现矩阵的总计数。因此：

$$\mathbf{W} \tilde{\mathbf{W}}^{\top} \approx \log P(i, j) + \log T = \log P(i) + \log P(j \mid i) + \log T$$

而SGNS的最优条件给出：

$$\mathbf{W}_w^{\top} \mathbf{C}_c = \text{PMI}(w, c) - \log K = \log \frac{P(w, c)}{P(w) P(c)} - \log K$$

对比两个公式可以发现：GloVe的内积近似 $\log P(j \mid i) + \text{const}$（条件概率的对数），而SGNS的内积近似 $\text{PMI}(w, c)$（联合概率与独立概率之比的对数）。两者都隐式地对一个与共现概率相关的矩阵进行矩阵分解，但目标矩阵的具体形式不同：GloVe使用 $\log X_{ij}$（即对数共现计数），SGNS使用 $\text{PMI}(w, c) - \log K$（即移位点互信息）。

**加权函数 $f(x)$ 的深层作用。** GloVe目标函数中的加权函数 $f(x)$ 不仅是工程上的平滑手段，它从根本上决定了矩阵分解关注的是"哪些共现对更重要"。形式上，GloVe等价于加权最小二乘矩阵分解：

$$\min_{\mathbf{W}, \tilde{\mathbf{W}}} \sum_{i, j} w_{ij} ( \mathbf{w}_i^{\top} \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij} )^2$$

其中 $w_{ij} = f(X_{ij})$ 是样本权重。权重函数的设计反映了以下直觉：

1. 低频共现（$X_{ij}$ 很小）：可能源于偶然共现，噪声大，应赋予低权重。$f(x)$ 在 $x \to 0$ 时趋于0，自动忽略零计数。
2. 中频共现（$X_{ij}$ 中等）：携带最有价值的语义信息（如"ice"和"solid"共现约50次），应赋予高权重。$f(x)$ 在 $x_{\max}$ 之前单调递增。
3. 高频共现（$X_{ij}$ 很大）：可能源于功能词的普遍共现（如"the"和所有词的高频共现），语义区分度低，应适度降权。$f(x)$ 在 $x > x_{\max}$ 时饱和为1。

原始论文选择 $x_{\max} = 100$ 和 $\alpha = 0.75$，这组超参数在多个词相似度基准上取得了最优性能。$x_{\max} = 100$ 意味着：当共现次数超过100时，权重不再增加——这防止了极高频共现（如"the"和"of"的共现可能达数千次）主导优化目标。$\alpha = 0.75$ 控制了权重增长的速度，遵循与Word2Vec负采样中 $3/4$ 次幂相同的经验规律。

**对称性与不对称性的统一。** GloVe的中心词嵌入 $\mathbf{w}_i$ 和上下文词嵌入 $\tilde{\mathbf{w}}_j$ 在目标函数中具有对称性：$\mathbf{w}_i^{\top} \tilde{\mathbf{w}}_j$ 的形式对 $i$ 和 $j$ 对称。这意味着最优解满足 $\mathbf{w}_i \approx \tilde{\mathbf{w}}_i$（当 $i=j$ 时，自共现被对称处理）。实践中，GloVe最终词向量通常取 $\mathbf{w}_i + \tilde{\mathbf{w}}_i$ 或 $\mathbf{w}_i$（或 $\tilde{\mathbf{w}}_i$）作为词的最终表示——这两种选择在性能上差异很小，因为对称性保证了二者高度相似。

相比之下，SGNS中 $\mathbf{W}$（输入矩阵）和 $\mathbf{C}_{\text{out}}$（输出矩阵）的角色不同：$\mathbf{W}$ 作为中心词时被正样本拉向上下文词，$\mathbf{C}_{\text{out}}$ 作为负样本时被推离中心词。这种不对称性导致 $\mathbf{W}$ 和 $\mathbf{C}_{\text{out}}$ 在语义空间中位于不同的区域——Levy和Goldberg发现，$\mathbf{W}$ 更擅长捕捉词的功能相似性（如词性），而 $\mathbf{C}_{\text{out}}$ 更擅长捕捉词的语义相似性（如同义词）。GloVe的对称性避免了这一问题，但也牺牲了一定程度的语义区分能力。

**收敛性与训练动力学。** GloVe使用交替最小二乘（ALS）或随机梯度下降进行优化。由于目标函数涉及完整的共现矩阵（非稀疏采样），每次迭代的计算成本为 $O(V^2 N)$（遍历所有共现对），但可以通过共现矩阵的稀疏性（绝大多数 $X_{ij} = 0$）和截断（$f(0) = 0$）来加速。在实践中，GloVe通常使用AdaGrad自适应学习率，在4到50个epoch内收敛——远快于SGNS需要数十个epoch的遍历。收敛速度的优势来源于GloVe对全局统计信息的直接访问：每个训练步骤都利用了完整的共现矩阵，而SGNS每次仅采样一个局部窗口。

**实验对比的深层结论。** 在词相似度任务（如WordSim-353、SimLex-999）上，GloVe和SGNS的性能相当，各有优劣：GloVe在大型语料上通常略微领先（因为全局统计信息更充分），SGNS在小语料上更鲁棒（因为局部窗口采样降低了噪声）。在类比推理任务（如"man:king :: woman:?"）上，SGNS通常优于GloVe——这可能是因为SGNS的移位PMI矩阵更直接地编码了词对之间的语义关系。在句法任务上，GloVe与SGNS差异不大。综合来看，两种方法代表了词向量学习的两个互补视角：GloVe强调"全局共现统计的结构"，SGNS强调"局部预测任务的判别能力"。当代的预训练语言模型（如BERT）可以理解为将这两种视角统一到统一的掩码预测框架中。

---

## 6 早期神经机器翻译：Encoder-Decoder与注意力机制

### 6.1 Encoder-Decoder架构的雏形

机器翻译（Machine Translation, MT）是自然语言处理中最具挑战性的任务之一。统计机器翻译（Statistical Machine Translation, SMT）长期占据主导地位，但其 pipelines 复杂、特征工程繁琐、错误传播不可避免。神经机器翻译（Neural Machine Translation, NMT）的出现标志着一个新时代的到来。

NMT的核心架构是Encoder-Decoder框架，其雏形可追溯到Sutskever等人（2014）和Cho等人（2014）的独立工作。该框架将翻译任务建模为一个序列到序列（Sequence-to-Sequence, Seq2Seq）的映射过程。

**编码器：LSTM与GRU的完整机制。** 在Sutskever等人的原始实现中，编码器是一个循环神经网络，最常用的是LSTM（Long Short-Term Memory）或GRU（Gated Recurrent Unit）。LSTM通过门控机制解决了传统RNN的梯度消失/爆炸问题，其核心是三个门控单元：

- **遗忘门（Forget Gate）：** $f_t = \sigma(\mathbf{W}_f \mathbf{h}_{t-1} + \mathbf{U}_f x_t + b_f)$，决定从细胞状态中丢弃多少信息。
- **输入门（Input Gate）：** $i_t = \sigma(\mathbf{W}_i \mathbf{h}_{t-1} + \mathbf{U}_i x_t + b_i)$，决定将多少新信息写入细胞状态。
- **输出门（Output Gate）：** $o_t = \sigma(\mathbf{W}_o \mathbf{h}_{t-1} + \mathbf{U}_o x_t + b_o)$，决定从细胞状态中输出多少信息到隐藏状态。

细胞状态的更新为：$\mathbf{c}_t = f_t \odot \mathbf{c}_{t-1} + i_t \odot \tanh(\mathbf{W}_c \mathbf{h}_{t-1} + \mathbf{U}_c x_t + b_c)$，隐藏状态为：$\mathbf{h}_t = o_t \odot \tanh(\mathbf{c}_t)$。LSTM的关键创新在于细胞状态 $\mathbf{c}_t$ 的"高速公路"结构——它通过逐元素乘法直接从前一步传递信息，梯度可以几乎无损地传播数百步，这对于长句子的编码至关重要。

GRU是LSTM的简化版本，将遗忘门和输入门合并为更新门（Update Gate），并取消了独立的细胞状态。GRU的更新为：$z_t = \sigma(\mathbf{W}_z \mathbf{h}_{t-1} + \mathbf{U}_z x_t + b_z)$（更新门），$r_t = \sigma(\mathbf{W}_r \mathbf{h}_{t-1} + \mathbf{U}_r x_t + b_r)$（重置门），$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h (r_t \odot \mathbf{h}_{t-1}) + \mathbf{U}_h x_t + b_h)$，$\mathbf{h}_t = (1 - z_t) \odot \mathbf{h}_{t-1} + z_t \odot \tilde{\mathbf{h}}_t$。GRU的参数更少（约LSTM的3/4），训练更快，在许多任务上与LSTM性能相当。

在NMT的Encoder-Decoder中，编码器通常是一个双向LSTM或双向GRU——正向LSTM从左到右读取源句，反向LSTM从右到左读取源句，最终隐藏状态 $\mathbf{h}_j = [\overrightarrow{\mathbf{h}}_j; \overleftarrow{\mathbf{h}}_j]$ 拼接了双向信息，提供了更丰富的源句表示。

**解码器（Decoder）。** 解码器是一个另一个RNN，以 $\mathbf{c}$ 为初始隐藏状态，自回归地生成目标语言句子 $\mathbf{y} = (y_1, y_2, \ldots, y_{T_y})$：

$$p(y_1, \ldots, y_{T_y} \mid x_1, \ldots, x_{T_x}) = \prod_{t=1}^{T_y} p(y_t \mid y_1^{t-1}, \mathbf{c})$$

每个时间步的概率由softmax计算：

$$p(y_t \mid y_1^{t-1}, \mathbf{c}) = \text{softmax}(\mathbf{W}_o \mathbf{s}_t + \mathbf{b}_o)$$

其中 $\mathbf{s}_t$ 是解码器在时间步 $t$ 的隐藏状态：

$$\mathbf{s}_t = g(\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c})$$

函数 $g$ 通常是一个LSTM或GRU单元，将前一步的隐藏状态 $\mathbf{s}_{t-1}$、前一个生成的词 $y_{t-1}$ 和上下文向量 $\mathbf{c}$ 作为输入。

**训练与推理。** 训练使用*教师强制*（Teacher Forcing）：在每个解码步输入真实的先前目标词，而非模型自身生成的词。推理时使用*束搜索*（Beam Search）或贪婪搜索来生成目标句子——由于解码是自回归的，一旦生成错误，后续步骤无法纠正。

### 6.2 Bahdanau注意力机制

Sutskever等人的Encoder-Decoder架构虽然开创了NMT的先河，但其使用固定长度上下文向量 $\mathbf{c}$ 的设计存在根本性缺陷。对于长句，一个固定维度的向量无法充分编码所有信息——这被称为"瓶颈问题"（Bottleneck Problem）。Bahdanau等人（2014）提出了*注意力机制*（Attention Mechanism），从根本上解决了这一问题。

**注意力机制的核心思想与信息论解释。** 与其将整个源句压缩为一个固定向量，不如让解码器在每一步*动态地关注源句的不同部分*。从信息论角度看，注意力机制解决的核心问题是"信息瓶颈"：固定长度上下文向量 $\mathbf{c}$ 必须在一个固定带宽的通道中传递源句的全部信息，而动态上下文向量 $\mathbf{c}_t$ 允许在不同时间步使用不同的带宽分配——某些源词在特定目标词生成时获得高带宽（高注意力权重），其他源词获得低带宽。

具体而言，解码器在生成目标词 $y_t$ 时，计算源句各隐藏状态 $\{\mathbf{h}_1, \ldots, \mathbf{h}_{T_x}\}$ 的加权和作为动态上下文向量 $\mathbf{c}_t$：

$$\mathbf{c}_t = \sum_{j=1}^{T_x} \alpha_{tj} \mathbf{h}_j$$

其中 $\alpha_{tj}$ 是注意力权重（Attention Weight），衡量在生成第 $t$ 个目标词时，源句第 $j$ 个位置的重要性。注意力权重通过softmax归一化：

$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{T_x} \exp(e_{tk})}$$

Softmax归一化保证了 $\sum_{j=1}^{T_x} \alpha_{tj} = 1$，因此 $\mathbf{c}_t$ 是 $\{\mathbf{h}_j\}$ 的凸组合（Convex Combination）——这一性质在几何上意味着 $\mathbf{c}_t$ 位于 $\{\mathbf{h}_j\}$ 构成的凸包内，不会超出源句隐藏状态的线性范围。

**能量函数与对齐模型的完整数学族。** $e_{tj}$ 是注意力能量（Attention Energy），由注意力模型（Alignment Model）计算。Bahdanau等人提出的原始注意力模型是一个单隐藏层前馈神经网络：

$$e_{tj} = \mathbf{v}_a^{\top} \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_j)$$

其中 $\mathbf{W}_a \in \mathbb{R}^{d_a \times d_s}$，$\mathbf{U}_a \in \mathbb{R}^{d_a \times d_h}$，$\mathbf{v}_a \in \mathbb{R}^{d_a}$ 是可学习的参数，$d_s$ 和 $d_h$ 分别是解码器和编码器隐藏状态的维度。$\tanh$ 激活函数引入非线性，使注意力模型能够捕获复杂的对齐关系。这一设计被称为*加性注意力*（Additive Attention）或*Bahdanau注意力*。

加性注意力的核心计算步骤可以展开为：

1. 将解码器隐藏状态 $\mathbf{s}_{t-1}$ 投影到 $d_a$ 维空间：$\mathbf{W}_a \mathbf{s}_{t-1}$
2. 将编码器隐藏状态 $\mathbf{h}_j$ 投影到 $d_a$ 维空间：$\mathbf{U}_a \mathbf{h}_j$
3. 逐元素求和并应用 $\tanh$：$\tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_j)$
4. 再投影到标量：$\mathbf{v}_a^{\top} (\cdot)$

**Luong注意力的对比：乘性注意力。** 后续Luong等人（2015）提出了*乘性注意力*（Multiplicative Attention），其能量函数为：

$$e_{tj} = \mathbf{s}_{t-1}^{\top} \mathbf{W}_a \mathbf{h}_j$$

其中 $\mathbf{W}_a \in \mathbb{R}^{d_s \times d_h}$ 是一个可学习的矩阵。乘性注意力的计算更高效（无需额外的 $\mathbf{v}_a$ 和 $\tanh$），在隐藏维度较大时（$d_s, d_h > 256$）通常优于加性注意力——这可能是因为大维度下，点积操作比加性操作保留了更多的几何信息。Luong还提出了"全局注意力"（考虑所有源词）和"局部注意力"（仅考虑源词的一个窗口）的变体。

两种能量函数的选择本质上反映了对"对齐"这一概念的参数化假设。加性注意力使用了一个通用的非线性函数（MLP），理论上可以拟合任意复杂的对齐关系；乘性注意力使用了一个双线性形式，假设对齐关系可以通过隐藏状态的几何内积来捕获。实验表明，当 $d_s = d_h$ 时，加性注意力和乘性注意力性能相当；当 $d_s \neq d_h$ 时，乘性注意力需要额外的投影矩阵，计算效率下降。

**解码器更新与门控机制。** 在Bahdanau注意力机制下，解码器在时间步 $t$ 的隐藏状态更新为：

$$\mathbf{s}_t = g(\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c}_t)$$

其中 $\mathbf{c}_t$ 是动态计算的上下文向量。函数 $g$ 通常是一个GRU或LSTM单元，以拼接（Concatenation）或加法（Addition）方式融合上下文向量：

- **拼接融合：** $[\mathbf{s}_{t-1}; y_{t-1}; \mathbf{c}_t]$ 作为LSTM/GRU的输入。
- **加法融合：** $\mathbf{s}_{t-1}' = \mathbf{s}_{t-1} + \mathbf{c}_t$，然后 $g(\mathbf{s}_{t-1}', y_{t-1})$。

Bahdanau原文使用拼接融合，将 $\mathbf{s}_{t-1}$、$y_{t-1}$ 和 $\mathbf{c}_t$ 拼接后输入GRU。拼接的优势在于保留了三者的独立信息，但增加了参数量；加法的优势在于简洁，且对维度对齐有严格要求。

**联合训练与端到端可微性。** 注意力模型的参数和解码器、编码器的参数一同通过最小化负对数似然来训练：

$$\mathcal{J} = -\sum_{t=1}^{T_y} \log p(y_t \mid y_1^{t-1}, \mathbf{x})$$

注意力权重 $\alpha_{tj}$ 不是显式参数化的，而是由 $\mathbf{s}_{t-1}$ 和 $\mathbf{h}_j$ 通过注意力模型隐式决定的。由于整个模型（编码器LSTM、解码器LSTM、注意力MLP）都是可微的，梯度可以通过标准的反向传播通过时间（BPTT）传播。具体而言，损失 $\mathcal{J}$ 对注意力能量 $e_{tj}$ 的梯度为：

$$\frac{\partial \mathcal{J}}{\partial e_{tj}} = \sum_{t'} \frac{\partial \mathcal{J}}{\partial \alpha_{t'j}} \frac{\partial \alpha_{t'j}}{\partial e_{tj}} = \sum_{t'} \frac{\partial \mathcal{J}}{\partial \alpha_{t'j}} \alpha_{t'j} (\mathbb{I}[t'=j] - \alpha_{tj})$$

这一梯度的形式展示了softmax函数的Jacobian结构——$\partial \alpha_{t'j} / \partial e_{tj} = \alpha_{t'j} (\mathbb{I}[t'=t] - \alpha_{tj})$。

**Bahdanau注意力的理论意义与历史地位。** Bahdanau注意力机制在概念上完成了一次根本性的转变：它将"对齐"（Alignment）从独立的后处理步骤（SMT中的Giza++字对齐）内化到了神经网络的端到端训练中。这一转变的意义超越了NMT——注意力机制是"动态路由"思想在深度学习中的首次成功应用：模型不再使用固定的计算图（如编码器最后一个隐藏状态），而是根据输入动态地选择计算路径（如源句的不同位置）。

更重要的是，注意力权重 $\alpha_{tj}$ 提供了翻译的*内在可解释性*。在可视化中，$\alpha_{tj}$ 矩阵通常呈现出清晰的"对角线"结构——源词 $w_j$ 和目标词 $y_t$ 在时间步上大致对齐。这种"软对齐"（Soft Alignment）与SMT中的"硬对齐"（Hard Alignment，每个源词恰好对应一个目标词）形成对比：注意力允许一个源词同时影响多个目标词（如英语"it"对应法语中的性别代词），也允许多个源词共同影响一个目标词（如法语冠词+名词对应英语名词）。

Bahdanau注意力的局限性在于其计算复杂度：对于每个目标词 $y_t$，需要计算 $T_x$ 次能量函数（每次处理一个源词隐藏状态 $\mathbf{h}_j$），总复杂度为 $O(T_x \cdot T_y \cdot d_a)$。当源句很长（$T_x > 200$）时，这种"逐步计算"成为瓶颈。这一问题直接导致了后续Transformer架构中"自注意力"的设计——自注意力通过矩阵乘法一次性计算所有词对之间的注意力权重，将复杂度降低到 $O(T^2)$（对序列长度 $T$），且可以充分并行化。

---

## 7 评估方法：BLEU、ROUGE与METEOR

### 7.1 BLEU

BLEU（Bilingual Evaluation Understudy）由Papineni等人于2002年提出，是机器翻译领域最具影响力的自动评估指标。其设计哲学是：*好的翻译应该与参考翻译在n-gram层面具有较高的重叠度*。BLEU的提出标志着MT评估从人工评估向自动评估的范式转变，至今仍是工业界和学术界的基准指标。

**修改后的n-gram精度：完整推导。** BLEU的核心是修改后的n-gram精度（Modified n-gram Precision）。对于候选翻译 $\hat{\mathbf{y}}$ 和一组参考翻译 $\mathbf{y}^{(1)}, \ldots, \mathbf{y}^{(K)}$（通常 $K = 1$ 到 $4$），定义：

$$p_n = \frac{\sum_{\text{n-gram} \in \hat{\mathbf{y}}} \min(\text{count}(\text{n-gram}, \hat{\mathbf{y}}), \max_{k} \text{count}(\text{n-gram}, \mathbf{y}^{(k)}))}{\sum_{\text{n-gram} \in \hat{\mathbf{y}}} \text{count}(\text{n-gram}, \hat{\mathbf{y}})}$$

分子的Clip操作（取候选计数与参考最大计数的较小值）防止了候选通过重复高频n-gram来"刷分"。例如，如果候选中"the"出现7次，但所有参考中"the"最多出现4次，则分子中"the"的贡献不超过4。分母使用候选中的n-gram总数（未Clip），因此Clip操作直接降低了精度——对于冗余的候选，精度被惩罚。

修改后精度的几何平均形式为：

$$\exp\left( \sum_{n=1}^N w_n \log p_n \right) = \prod_{n=1}^N p_n^{w_n}$$

通常使用 $N = 4$（即计算1-gram到4-gram的精度）和均匀权重 $w_n = 1/N$。$N=4$ 的选择反映了以下直觉：1-gram精度衡量"信息覆盖度"（候选词是否出现在参考中），4-gram精度衡量"词序准确性"（四个词的序列是否一致），而2-gram和3-gram精度衡量"局部流畅性"。使用几何平均而非算术平均的原因在于：算术平均对单项零值过于敏感（任何一个 $p_n = 0$ 会使BLEU归零），而几何平均在 $p_n \to 0$ 时渐进地趋于零，提供了更平滑的衰减。

**简洁性惩罚的完整推导。** 仅使用精度会导致短翻译获得虚高的分数——短翻译容易通过选择高频参考词获得高精度，但信息量不足。BLEU引入简洁性惩罚（Brevity Penalty, BP）：

$$\text{BP} = \begin{cases} 1 & \text{if } c > r \\ \exp(1 - r/c) & \text{if } c \leq r \end{cases}$$

其中 $c$ 是候选翻译的长度，$r$ 是*最佳参考长度*——即与候选长度最接近的参考翻译的长度。当有多个参考长度与 $c$ 同样接近时，$r$ 取这些长度的平均值。BP在 $c \leq r$ 时指数衰减：当 $c = r$ 时 $\text{BP} = 1$；当 $c = r/2$ 时 $\text{BP} = \exp(-1) \approx 0.37$；当 $c \to 0$ 时 $\text{BP} \to 0$。这一指数形式的选择确保了惩罚的平滑性和对超短翻译的强烈抑制。

**完整公式与取值范围。** BLEU Score定义为：

$$\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^N w_n \log p_n \right)$$

BLEU值的范围在理论上是 $[0, 1]$（乘以100后习惯上报告为0到100之间的整数）。BLEU=1（或100）意味着候选与参考在每个n-gram级别完全匹配，且长度恰好等于最佳参考长度；BLEU=0意味着至少一个n-gram级别的修改精度为零。由于修改精度和BP都依赖于具体语料和参考数量，BLEU分数在不同测试集之间不可直接比较——只有在相同测试集和相同参考数量下的BLEU才具有可比性。

**BLEU的统计性质与显著性。** BLEU分数是一个样本统计量（基于测试集的有限样本），因此存在估计方差。对于 $N=4$ 的BLEU，每个样本提供的信息非常稀疏——4-gram在测试集中出现的次数可能很少，导致 $p_4$ 的估计方差极大。Clark等人（2011）的研究表明，BLEU差异为5到6个点时，在95%置信水平下可能不具有统计显著性——这意味着在比较两个系统时，BLEU差异需要达到一定阈值才能排除随机波动。后续工作（如Yang和Agarwal 2018）提出了BLEU的置信区间估计方法，基于自助法（Bootstrap Resampling）来量化BLEU的不确定性。

**局限性：语义盲区与评估偏差。** BLEU的已知局限可以从数学角度精确刻画：

1. **缺乏语义等价性：** BLEU仅衡量表面n-gram重叠，不考虑语义等价性。"car"和"automobile"不会获得任何分数，即使它们在语义上等价。这一局限的根源在于BLEU基于精确字符串匹配，缺乏任何语义词典或推理机制。

2. **缺乏对流畅性的直接衡量：** BLEU不直接衡量生成文本的语法流畅性。候选"the the the the"（如果参考恰好包含"the"）可能获得非零BLEU分数——这显然不是一个流畅的句子。修改精度惩罚了高频词重复，但仅通过Clip操作间接实现，而非通过语法模型直接评估。

3. **对参考数量和质量的敏感性：** BLEU随参考数量 $K$ 增加而单调递增（因为更多的参考提供了更多的匹配机会）。单参考BLEU通常低于多参考BLEU，且差距可能很大。这使得跨论文的BLEU比较变得困难——一篇论文使用1个参考，另一篇使用4个参考，即使系统性能相同，BLEU分数也可能不同。

4. **对长度的不敏感性（过长翻译）：** 当 $c \gg r$ 时，$\exp(1 - r/c) \to e$（约2.718），BP仅惩罚到这一下限。过长的翻译（如重复参考内容的翻译）不会被充分惩罚——它们可能获得与长度适中的翻译相近甚至更高的BLEU分数。

5. **对n-gram顺序的刚性要求：** 即使词的集合完全相同，不同的词序也会导致BLEU分数显著下降。例如，"the cat sat on the mat"和"the mat sat on the cat"有相同的unigram精度（1.0），但4-gram精度为零，导致BLEU=0。这种刚性在某些语言对（如SOV语言到SVO语言）中尤为成问题。

### 7.2 ROUGE

ROUGE（Recall-Oriented Understudy for Gisting Evaluation）由Lin（2004）提出，最初面向自动摘要评估，后被广泛应用于机器翻译和文本生成的评估。与BLEU不同，ROUGE以召回率（Recall）为核心——它衡量参考翻译中出现在候选中的n-gram比例。

**ROUGE-N的完整形式化。** ROUGE-N定义为：

$$\text{ROUGE-N} = \frac{\sum_{\text{n-gram} \in \mathcal{R}} \min(\text{count}(\text{n-gram}, \mathcal{R}), \max_{k} \text{count}(\text{n-gram}, \hat{\mathbf{y}}))}{\sum_{\text{n-gram} \in \mathcal{R}} \text{count}(\text{n-gram}, \mathcal{R})}$$

其中 $\mathcal{R}$ 表示所有参考翻译中n-gram的集合（多重集合并），$\hat{\mathbf{y}}$ 是候选翻译。分子统计参考n-gram在候选中被覆盖的次数（Clipping与BLEU相同），分母是参考n-gram的总数。ROUGE-N以召回率为导向，这与BLEU的精确率导向形成互补。当候选翻译包含参考中的大多数n-gram时，ROUGE-N接近1；当候选与参考几乎无重叠时，ROUGE-N接近0。

ROUGE-1（unigram ROUGE）衡量词级别的召回率，对词汇选择敏感但不要求词序；ROUGE-2（bigram ROUGE）衡量相邻词对的召回率，对局部词序敏感；ROUGE-3和ROUGE-4衡量更长词序的召回率。在摘要任务中，ROUGE-1和ROUGE-2是最常用的变体——ROUGE-1与信息覆盖度高度相关，ROUGE-2与流畅性相关。

**ROUGE-L：最长公共子序列的数学性质。** ROUGE-L基于最长公共子序列（Longest Common Subsequence, LCS）。LCS是两个序列中都出现的、不必连续的子序列（与子字符串不同）。例如，"the cat sat on the mat"和"the cat was sitting on the mat"的LCS为"the cat on the mat"（长度5）。LCS的长度可以通过动态规划在 $O(T_x \cdot T_y)$ 时间内计算，其中 $T_x$ 和 $T_y$ 分别是候选和参考的长度。

ROUGE-L定义为F1分数：

$$F_{\text{LCS}} = \frac{(1 + \beta^2) R_{\text{LCS}} P_{\text{LCS}}}{\beta^2 R_{\text{LCS}} + P_{\text{LCS}}}$$

其中 $R_{\text{LCS}} = \text{LCS}(\hat{\mathbf{y}}, \mathbf{y}) / |\mathbf{y}|$，$P_{\text{LCS}} = \text{LCS}(\hat{\mathbf{y}}, \mathbf{y}) / |\hat{\mathbf{y}}|$，$\beta$ 通常取1（即F1分数）。ROUGE-L不要求n-gram连续，因此对词序的局部变化更加鲁棒——插入一个词不会破坏整个n-gram，只会缩短LCS。

ROUGE-L的一个重要性质是：当候选是参考的子序列时，$R_{\text{LCS}} = 1$（完全覆盖），但 $P_{\text{LCS}}$ 可能小于1（如果候选比参考短）。这确保了ROUGE-L同时惩罚信息遗漏（低召回率）和冗余信息（低精确率）。

**ROUGE-S/U：Skip-bigram与并集。** ROUGE-S允许n-gram中有"跳跃"（Skip，即词可以不连续出现），其形式化定义为skip-bigram的召回率。Skip-bigram是两个词之间可以间隔任意其他词的对——这比LCS更宽松，因为skip-bigram不要求词序连续（只要求词序相对），而LCS要求词序连续。

ROUGE-SU结合了skip-bigram的召回率和unigram的召回率，在DUC（Document Understanding Conference）评测中被广泛使用。ROUGE-SU的动机是：纯unigram召回率（ROUGE-1）忽略词序，纯bigram召回率（ROUGE-2）对插入/删除过于敏感——skip-bigram提供了一个中间选择，允许有限的不连续性。

**与BLEU的系统性对比。** BLEU强调精确率，ROUGE强调召回率。在翻译任务中，BLEU更适合评估流畅性和忠实度（避免冗余词），ROUGE更适合评估信息覆盖度（不遗漏重要信息）。在摘要任务中，ROUGE由于关注信息完整性，通常比BLEU更合适——摘要的核心要求是"覆盖参考的关键信息"，而非"用词精确匹配参考"。

从数学角度看，BLEU和ROUGE可以统一到* F-分数框架*中：

$$F_\alpha = \frac{(1 + \alpha^2) P R}{\alpha^2 P + R}$$

BLEU使用几何平均（隐含地）和精确率导向（通过BP惩罚短翻译），ROUGE使用算术F1和召回率导向（分母为参考长度）。选择精确率还是召回率取决于任务的核心评价标准：翻译中"多说"比"少说"更容易纠正（精确率导向），摘要中"少说漏掉关键信息"比"多说冗余信息"更不可接受（召回率导向）。

### 7.3 METEOR

METEOR（Metric for Evaluation of Translation with Explicit ORdering）由Banerjee和Lavie于2005年提出，旨在克服BLEU的两个核心局限：缺乏对语义等价性的考量，以及缺乏对词序结构的显式建模。

**对齐与匹配：贪心最大匹配的算法细节。** METEOR的第一步是在候选和参考之间建立词级对齐。与BLEU的精确n-gram匹配不同，METEOR使用以下匹配类型：

1. **精确匹配（Exact Match）：** 词形完全一致（区分大小写）。
2. **词干匹配（Stem Match）：** 词干相同但词形不同（如"running"和"run"），使用Porter词干提取器。Porter词干提取器通过一系列规则（如将"-ing"、"-ed"、"-s"后缀去除）将词还原为词干，例如"running"→"run"，"happily"→"happili"（Porter提取器的输出有时不是标准词形，但足以用于匹配）。
3. **语义匹配（Synonym Match）：** 词义等价但词形不同（如"car"和"automobile"），使用WordNet等同义词词典。WordNet将英语词组织为synsets（同义词集），每个synset编码了一个概念；如果两个词属于同一个synset，则它们被视为语义等价。

对齐过程采用*贪心 maximal matching*：在所有可能的匹配中，选择跨距（Span）最小的匹配序列。具体算法为：按匹配类型的优先级（精确匹配 > 词干匹配 > 语义匹配）排序所有可能的匹配对；依次选择优先级最高、跨距最小的匹配；将已匹配的词从候选中移除，继续选择下一对匹配。这一步保证了每个词至多被匹配一次，避免了重复匹配导致的分数虚高。

** Precision、Recall和F-mean的加权调和平均。** 令 $m$ 为匹配的词数，$| \hat{\mathbf{y}} |$ 和 $| \mathbf{y} |$ 分别为候选和参考的长度。则：

$$P = \frac{m}{|\hat{\mathbf{y}}|}, \quad R = \frac{m}{|\mathbf{y}|}$$

$$F_{\text{mean}} = \frac{10 P R}{R + 9P}$$

METEOR使用加权调和平均，精确率权重为9，召回率权重为1——这意味着精确率比召回率重要9倍。这一权重选择的动机是：在翻译评估中，"不遗漏信息"（召回率）比"避免多余信息"（精确率）更重要——多余的词可以通过后处理删除，但遗漏的词无法恢复。某些实现使用标准调和平均 $F_1 = \frac{2PR}{P+R}$，但METEOR原始论文明确使用 $F_{9:1}$。

**碎片惩罚：词序结构的量化。** METEOR的独到之处在于显式度量词序质量。定义 $\text{chunks}$ 为候选中与参考保持相同词序的*连续匹配块*（Chunks）的数量。例如：

候选: "the cat sat on the mat"
参考: "the cat was sitting on the mat"

对齐后，匹配的词为 "the", "cat", "sat", "on", "the", "mat"（假设通过词干匹配将"sat"和"sitting"匹配）。这些匹配词在候选中形成两个chunks：$\{ \text{the cat} \}$ 和 $\{ \text{on the mat} \}$。中间插入了"sitting"（未匹配），将匹配分割为两个块。chunks越多，表示匹配越碎片化，词序质量越差。

碎片惩罚为：

$$\text{Penalty} = 0.5 \cdot \left( \frac{\text{chunks}}{m} \right)^3$$

Chunks的最小值为1（所有匹配词形成一个连续块），此时Penalty=0.5；chunks的最大值为 $m$（每个匹配词独立成块），此时Penalty=0.5。因此，Penalty的范围是 $[0, 0.5]$。指数 $3$ 的选择使得惩罚对碎片化程度的增长呈加速趋势——当chunks从1增加到2时，惩罚从0.5增加到0.5（无变化）；当chunks接近 $m$ 时，惩罚急剧上升。

**完整公式与计算流程。** METEOR分数为：

$$\text{METEOR} = (1 - \text{Penalty}) \cdot F_{\text{mean}}$$

完整计算流程为：（1）对候选和参考进行分词和词干提取；（2）建立词级对齐（精确匹配 > 词干匹配 > 语义匹配，贪心最大匹配）；（3）计算匹配词数 $m$，精确率 $P$ 和召回率 $R$；（4）计算F-mean；（5）识别chunks，计算碎片惩罚；（6）输出 $(1 - \text{Penalty}) \cdot F_{\text{mean}}$。

**性质与优势的深入分析。** METEOR相比BLEU具有若干理论优势：

首先，它显式引入了语义等价性（通过WordNet），使评价更贴近人工判断。在翻译中，"car"和"automobile"的等价性被捕获，这在BLEU中完全被忽略。

其次，碎片惩罚使词序质量被直接量化——BLEU的4-gram精度间接反映了词序，但仅通过n-gram匹配的几何平均来体现；METEOR通过chunks概念显式地将词序结构纳入评分。

第三，METEOR在句子级别上计算召回率和精确率的调和平均，对每个样本单独评估，而BLEU通常在文档级别聚合统计量（将所有候选的n-gram计数合并后计算精度）。句子级别的评估使METEOR对个体差异更敏感，适合短翻译和对话系统的评估。

第四，METEOR的评分范围更广——BLEU的有效范围通常在0到1之间（但大多数实际BLEU值集中在0.05到0.35之间），METEOR的有效范围约为0到1，且在不同语言对上更具可比性。

**局限性与计算成本。** METEOR的局限性同样明显：首先，它依赖外部语言学资源（如WordNet和Porter词干提取器），这使得它在低资源语言上的应用受限——许多语言没有成熟的WordNet等价物。其次，其运行速度慢于BLEU——METEOR需要在每个评估样本上执行词干提取、语义匹配和LCS计算，时间复杂度约为 $O(T_x \cdot T_y)$，而BLEU仅为 $O(T)$。第三，碎片惩罚中的chunks定义在匹配词不连续时产生多个chunks，这可能导致对某些合法但非常规的词序过度惩罚。第四，METEOR的 $F_{9:1}$ 权重（精确率权重9倍于召回率）在理论上偏向精确率，这与设计初衷（召回率导向）存在张力——不同的权重选择会显著改变分数排序。

---

## 8 总结与过渡：从统计语言模型到大语言模型

从n-gram到神经语言模型，语言模型的发展历程呈现出一条清晰的技术主线：**从离散符号到连续表示，从局部统计到全局优化，从孤立预测到序列建模**。n-gram模型以马尔可夫假设为基石，通过最大似然估计和精巧的平滑技术，在受限词表上实现了可用的概率估计；信息论提供了交叉熵和困惑度等严格的评估工具，使模型的优劣可以被量化比较；词表示的学习将词从离散索引提升为连续向量，开启了语义计算的先河；Word2Vec和GloVe通过不同的优化视角——预测视角与计数视角——共同确立了词向量作为NLP基础设施的地位；Bahdanau注意力机制将"对齐"这一核心语言概念内化到端到端的神经网络中，为Transformer的登场铺平了道路。

**技术传承的深层逻辑。** 这些技术和思想并非彼此孤立，而是层层叠加、相互支撑。理解n-gram的平滑技术有助于理解正则化和先验的作用：Good-Turing的非参数频率重估计可视作贝叶斯先验的一种朴素形式；Kneser-Ney对MLE的系统性偏差修正可视作偏差-方差权衡的早期实践。理解交叉熵有助于理解神经语言模型的训练目标：最小化负对数似然等价于最小化交叉熵，而困惑度作为交叉熵的指数，将信息论中的"比特数"与语言建模中的"不确定性"统一在同一框架下。理解SGNS与矩阵分解的关系有助于理解BERT等预训练模型中"预测 masked token"的深层结构——BERT的掩码语言模型目标本质上是对上下文-目标词的PMI矩阵进行矩阵分解，与SGNS共享相同的数学骨架，只是将"词-上下文"的局部关系扩展到了"句子-句子"或"token-token"的全局关系。

**从注意力机制到Transformer。** Bahdanau注意力机制的思想在Vaswani等人（2017）的Transformer架构中得到了极致的发展和规模化。Transformer完全抛弃了循环结构，使用自注意力（Self-Attention）替代RNN来建模序列依赖。自注意力可以被理解为注意力机制的"全连接"版本：对于序列中的每个位置，计算序列中所有位置（包括自身）的注意力权重，形成一个 $T \times T$ 的注意力矩阵。这一矩阵化的计算可以充分利用GPU/TPU的并行能力，将训练速度提升到RNN难以企及的高度。

Transformer的多头注意力（Multi-Head Attention）进一步将注意力机制推广为多个独立的注意力子空间，每个头学习不同的对齐关系——这与Bahdanau注意力中单一注意力模型形成对比。位置编码（Positional Encoding）为自注意力注入了序列顺序信息（自注意力本身不感知词序），残差连接和层归一化（Layer Normalization）稳定了深层Transformer的训练。编码器-解码器架构在Transformer中被堆叠为6层（原始配置），每层包含一个多头自注意力子层和一个前馈网络子层。

**从预训练到微调：当代大语言模型的哲学谱系。** 当代的大规模预训练语言模型——GPT系列、BERT系列、T5系列、LLaMA系列——在表面上与本章讨论的技术相去甚远，但其核心框架——自回归/去噪序列建模、softmax输出、词嵌入、注意力机制——都可以在n-gram到神经机器翻译的演进脉络中找到源头。具体而言：

- **自回归语言模型（GPT系列）** 是n-gram链式分解的神经网络扩展：$P(w_t \mid w_1^{t-1})$ 通过Transformer解码器估计，词嵌入提供分布式表示，注意力机制建模长距离依赖。
- **掩码语言模型（BERT系列）** 是n-gram的"双向"扩展：通过掩码部分输入词，模型从双向上下文预测掩码词，这与n-gram中使用左右上下文估计词概率的思想同源，但使用Transformer编码器的双向注意力替代了单向马尔可夫链。
- **T5系列（Text-to-Text Transfer Transformer）** 将所有NLP任务统一为"文本到文本"的Seq2Seq问题，这与Sutskever等人的原始Encoder-Decoder愿景一脉相承，但使用Transformer替代LSTM，使用大规模预训练替代小规模监督训练。
- **词嵌入的当代形式** 从Word2Vec的静态嵌入演进为BERT的上下文嵌入（Contextualized Embedding）：同一个词"bank"在"river bank"和"bank account"中具有不同的向量表示。这一演进解决了Word2Vec无法处理一词多义的根本性缺陷，其代价是失去了静态嵌入的简洁性和可解释性。

**未来的挑战与方向。** 尽管取得了巨大进展，语言模型仍面临若干根本性挑战。首先，困惑度的理论上限（语言熵率）仍未达到——最优模型的困惑度仍比理论上限高出一个数量级以上，这意味着自然语言的生成结构中仍有大量未被捕获的信息。其次，大规模语言模型的计算成本和碳排放引发了可持续性担忧——训练一个千亿参数模型所需的能量相当于数百辆汽车终身排放量。第三，语言模型的"幻觉"（Hallucination）问题——生成流畅但不真实的文本——揭示了概率建模与事实 grounding 之间的根本性张力。第四，语言模型的伦理对齐（Alignment）问题——如何使模型的输出与人类价值观一致——已成为研究的前沿。

理解n-gram到神经机器翻译的演进脉络，不仅有助于我们把握NLP的历史，更有助于我们理解当代模型的设计选择——每一个看似"革命性"的创新，几乎都可以在历史脉络中找到其思想的萌芽。技术演进不是断裂的革命，而是连续的积累；每一个新的架构都在前一个架构的肩膀上，解决前一个架构的瓶颈，同时继承其核心洞见。

字数统计：约19800字