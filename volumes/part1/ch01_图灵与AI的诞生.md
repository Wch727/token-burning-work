# 第1章 图灵与人工智能的诞生

## 1.1 引言

人工智能（Artificial Intelligence, AI）作为一门正式的科学学科，其诞生标志通常指向1956年在美国达特茅斯学院（Dartmouth College）举行的"达特茅斯夏季人工智能研究项目"（Dartmouth Summer Research Project on Artificial Intelligence）。然而，任何一门学科的确立都必须依赖于其背后的哲学基础与数学理论。正是艾伦·图灵（Alan Mathison Turing, 1912–1954）在20世纪30年代至50年代所建立的可计算性理论、计算复杂性思想以及对机器智能的哲学论证，为人工智能的诞生提供了最为根本的理论基石。本章从纯学术与技术视角出发，系统梳理从图灵机到达特茅斯会议期间（1936–1956）人工智能赖以产生的科学背景、核心理论成果、早期计算系统的实现条件，以及早期预言与实际发展之间的落差。

图灵的工作涵盖三个相互关联的领域：（1）可计算性理论（computability theory），以1936年发表的《论可计算数及其在判定问题上的应用》（On Computable Numbers, with an Application to the Entscheidungsproblem）为代表；（2）计算复杂性理论（computational complexity theory）的先驱性探索；（3）机器智能哲学，以1950年发表的著名论文《计算机器与智能》（Computing Machinery and Intelligence）为巅峰之作。

本章的组织结构如下：1.2节详细介绍1936年图灵机的形式化定义；1.3节讨论可计算性理论的基本框架；1.4节深入分析1950年图灵关于机器智能的哲学论证；1.5节回顾1956年达特茅斯会议；1.6节介绍早期人工智能系统；1.7节分析当时的计算资源限制；1.8节考察早期预言与实际发展之间的巨大差距；1.9节提供历史时间线；1.10节进行总结。

---

## 1.2 图灵机：可计算性的形式化模型

### 1.2.1 Hilbert判定问题与数学基础的危机

图灵机（Turing Machine）的诞生直接源于20世纪初数学家大卫·希尔伯特（David Hilbert）所提出的"判定问题"（Entscheidungsproblem, 即decision problem）。1928年，希尔伯特在意大利博洛尼亚举行的国际数学家大会上正式提出了判定问题（Entscheidungsproblem）。[注：希尔伯特的著名23个数学问题（Hilbert's 23 Problems）是在1900年巴黎国际数学家大会上提出的，而非1928年。]判定问题触及数学推理本身的机械化和自动化本质。

判定问题的精确表述：给定一阶逻辑中的一个闭公式，是否存在一个一般的算法，能够判定该公式在全域中是否为真？希尔伯特本人相信答案是肯定的——这种信念被称为"希尔伯特纲领"（Hilbert's Program）。

然而，1930年9月7日，库尔特·哥德尔（Kurt Gödel）在柯尼斯堡举行的"精确科学的认识论"会议上宣读了他划时代的论文《论〈数学原理〉及相关系统中的不可判定命题》。哥德尔不完备性定理表明：在任何包含初等算术的一致形式系统中，都存在一个在该系统中既不能被证明也不能被否证的命题。更进一步，哥德尔第二不完备定理指出，这样一个系统的一致性本身在该系统中也是不可证明的。

哥德尔的不完备性定理虽然没有直接解决判定问题，但它彻底粉碎了希尔伯特关于数学系统完备性的乐观信念。判定问题的最终解决需要等待图灵在1936年的工作。

### 1.2.2 图灵的形式化模型

1935年春天，图灵在剑桥大学国王学院攻读硕士学位期间构思出了"图灵机"的形式化模型。这一模型的精妙之处在于它将"计算"这一看似模糊的心理活动转化为一个严格的数学对象。

图灵在其1936年发表的论文中将"可计算数"定义如下：实数$a$（$0 \leq a \leq 1$）被称为可计算的，如果存在一台图灵机，当给它输入一段适当的磁带时，它能够打印出$a$的十进制展开式。

图灵机的核心思想是任何机械化的计算procedure都可以被分解为最基本的原子性操作步骤：（1）读取磁带当前格上的符号；（2）根据当前内部状态决定下一步行动；（3）在当前格上写或不写一个符号；（4）将读写头向左、右移动一格或保持不动；（5）将内部状态更新为新状态。

#### 1.2.2.1 图灵机的形式化定义

一台（确定性的单带）图灵机$M$可以定义为一个7元组：

$$
M = (Q, \Gamma, b, \Sigma, \delta, q_0, q_f)
$$

其中各分量的含义如下：

- $Q$：有限的状态集合，$|Q| < \infty$。
- $\Gamma$：有限的磁带符号集合，$|\Gamma| \geq 2$。
- $b \in \Gamma$：空白符号。
- $\Sigma \subseteq \Gamma$：输入符号集合，不包含空白符号。
- $\delta: Q \times \Gamma \to Q \times \Gamma \times \{L, R, S\}$：状态转移函数。
- $q_0 \in Q$：初始状态。
- $q_f \in Q$：终止状态。

在形式化定义的基础上，需要进一步引入**瞬时描述**（Instantaneous Description, ID）这一关键概念，它是对图灵机在某一特定时刻"完整快照"的精确数学刻画。一台图灵机$M$在时刻$t$的瞬时描述是一个三元组：

$$
C_t = (q_t, \, \alpha_t, \, \beta_t)
$$

其中$q_t \in Q$是当前内部状态，$\alpha_t \in \Gamma^*$是读写头左侧磁带上的符号序列（从左端到读写头左侧邻格），$\beta_t \in \Gamma^*$是读写头及其右侧磁带上的符号序列（从读写头当前格到右端）。整个磁带内容可以恢复为$\alpha_t \beta_t$，读写头永远指向$\beta_t$的第一个符号。

两个瞬时描述之间的**产出关系**（Yield Relation）写作$C \vdash_M C'$，其精确定义为：若$\delta(q, \sigma) = (q', \sigma', R)$，且$C = (q, \alpha\sigma, \beta)$、$C' = (q', \alpha\sigma', \sigma\beta)$，则$C \vdash_M C'$。类似地定义$L$（左移）和$S$（保持）的情况。产出关系的自反传递闭包记为$\vdash_M^*$。

基于瞬时描述，可以形式化定义图灵机对语言的**识别**：称图灵机$M$识别语言$L \subseteq \Sigma^*$，如果$L = \{w \in \Sigma^* \mid C_0 \vdash_M^* C_{acc}\}$，其中$C_0$是初始瞬时描述（读写头位于输入串最左端，状态为$q_0$），$C_{acc}$是进入接受状态$q_{acc}$的瞬时描述。值得注意的是，标准形式化通常设置两个不同的终止状态——接受状态$q_{acc}$和拒绝状态$q_{rej}$——而不仅仅是一个通用终止状态$q_f$。这一区分引出了两个至关重要的图灵机变体：

**判定器**（Decider）：对所有输入$w \in \Sigma^*$都在有限步内停机（进入$q_{acc}$或$q_{rej}$之一）的图灵机。判定器对应的语言类恰好是**递归语言**（Recursive Languages, R）。

**识别器**（Recognizer）：对属于$L$的输入$w$必定停机并进入$q_{acc}$，对不属于$L$的输入可能不停机的图灵机。识别器对应的语言类是**递归可枚举语言**（Recursively Enumerable Languages, RE）。

这一区分直接导致了图灵机层次论的核心图景：$R \subsetneq RE \subsetneq \Sigma^*$的严格包含关系，以及$coRE = \{L \mid \overline{L} \in RE\}$与$RE$之间既非包含亦非分离的复杂关系。

状态转移函数的精确含义可以通过以下伪代码表述。

**伪代码1：图灵机单步执行**

```
FUNCTION TuringStep(M, q, σ):
    (q', σ', d) ← δ_M(q, σ)
    IF q' = q_f THEN RETURN HALT
    WRITE σ' ON CURRENT CELL
    IF d = L THEN MOVE HEAD LEFT BY ONE
    ELSE IF d = R THEN MOVE HEAD RIGHT BY ONE
    ELSE IF d = S THEN STAY AT CURRENT CELL
    SET CURRENT STATE ← q'
    RETURN CONTINUE
```

**伪代码2：图灵机完整计算过程**

```
FUNCTION TuringRun(M, input_string):
    INITIALIZE TAPE ← input_string
    FILL REMAINING TAPE WITH b
    SET HEAD POSITION ← 0
    CURRENT STATE ← q_0
    LOOP:
        σ ← READ SYMBOL AT HEAD
        IF CURRENT STATE = q_f THEN RETURN HALTED(TAPE)
        (q', σ', d) ← TuringStep(M, CURRENT STATE, σ)
        IF q' = q_f THEN RETURN HALTED(TAPE)
        WRITE σ' ON TAPE
        IF d = L THEN HEAD POSITION ← HEAD POSITION - 1
        ELSE IF d = R THEN HEAD POSITION ← HEAD POSITION + 1
        CURRENT STATE ← q'
    END LOOP
```

上述伪代码揭示了图灵机的一个深刻性质：图灵机不仅可以计算函数，还可以模拟其他图灵机。这一观察直接导致了通用图灵机概念的产生。

#### 1.2.2.2 无限磁带（Infinite Tape）的意义

图灵机的一个关键设计选择是无限长的双向磁带。必要性：如果磁带是有限的，机器在任意时刻的配置只有有限多种，任何在有限磁带上的计算如果在$n^2$步内没有停机就必然进入循环。充分性：对于任何有效procedure，该procedure在任意输入上的计算步数是有限的，只要磁带足够长就能容纳全部中间结果。

无限磁带的概念与λ演算中的"名字"的无限Supply，以及递归函数论中的"自然数"的无限集合形成了鲜明对应。这种"无限性假设"是图灵完备性的必要前提。

#### 1.2.2.3 状态转移函数与计算过程

状态转移函数$\delta$可以视为图灵机的"程序"。一个图灵机的计算过程就是$\delta$的反复应用：

$$
(q_{i+1}, \sigma_{i+1}, d_{i+1}) = \delta(q_i, \sigma_i), \quad i = 0, 1, 2, \ldots
$$

特别值得注意的是$\delta$是一个部分函数：对于某些组合，$\delta(q, \sigma)$可能无定义——这意味着机器遇到该组合时没有合法的行动，即"崩溃"。图灵允许$\delta$为部分函数，这一选择增加了模型的表达能力。

#### 1.2.2.4 图灵完备性（Turing Completeness）

一台计算装置被称为图灵完备的，如果它能够模拟通用图灵机。所有图灵完备的系统——无论是现代编程语言、细胞自动机还是DNA计算——在计算能力上是等价的，差异仅在于效率而不在于能力。

图灵完备性的哲学含义在于它表明"机械计算"的概念与"物理实现"无关。只要一个系统能够执行基本的符号操作，它就能够执行任何可计算的计算。

#### 1.2.2.5.1 图灵机计算实例：二进制加法器

为了将上述抽象定义具体化，我们构造一台执行二进制加法的图灵机M_{add}。该机器的输入格式为1#b1b2...bn#c1c2...cm，输出为两数之和的二进制表示。

M_{add}的7元组定义如下：

- Q = {q0, q1, q2, q3, q4, q5}，6个状态
- Gamma = {0, 1, #, X, b}，5个磁带符号
- b：空白符号
- Sigma = {0, 1, #}，输入符号集合
- q0：初始状态（寻找第一个操作数位）
- q5：终止状态

状态转移表（仅列出非平凡转移）：

当前状态  读取符号  写入符号  移动方向  下一状态
q0        1         X         R         q1
q0        0         X         R         q1
q0        b         b         S         q5（停机）
q1        #         #         R         q2
q1        1         1         R         q1
q1        0         0         R         q1
q2        1         1         R         q3（进位已处理）
q2        0         1         R         q3（无进位，当前位为1）
q2        b         1         S         q5（无进位，补1）
q3        1         1         R         q3
q3        0         0         R         q3
q3        b         b         L         q4
q4        X         b         L         q4（清除标记）
q4        b         b         S         q5（停机）

计算过程示例：以输入磁带1#1011#110为例。以下展示前15步计算轨迹：

步骤0（初始状态）：...b b q0 1 #1011#110 b b...
步骤1：读取1，写入X，右移，进入q1。
步骤2：读取#，保持，右移，进入q2。
步骤3：读取1，保持，右移，进入q3（进位已处理）。
步骤4-6：连续右移扫描剩余位。
步骤7：读取#，保持，右移。
步骤8-9：继续右移。
步骤10：读取空白，左移，进入q4。
步骤11-12：左移清除标记X。
步骤13-14：继续左移直到遇到空白，进入终止状态q5。

最终磁带上留下和#1010，即1010₂。上述轨迹展示了图灵机如何通过状态转移实现基本算术操作。

#### 1.2.2.5.2 字符串反转（String Reversal）

为了进一步展示图灵机处理非算术类问题的能力，我们构造一台执行字符串反转的图灵机$M_{rev}$。该机器的输入格式为$w\#b$，其中$w \in \{a,b\}^*$是待反转的字符串，$\#$是字符串结束标记，$b$是空白。输出格式为$b\#w^R$，即原字符串的反转形式，前导空白，后接字符串结束标记。

$M_{rev}$的7元组定义如下：

- $Q = \{q_0, q_1, q_2, q_3, q_4, q_{halt}\}$，6个状态
- $\Gamma = \{a, b, \#, x, b\}$，5个磁带符号（其中$x$是标记符号，用于标记已被复制的字符）
- $b$：空白符号
- $\Sigma = \{a, b, \#\}$，输入符号集合
- $q_0$：初始状态
- $q_{halt}$：终止状态

**完整状态转移表**（使用五元组表示法$(q, \sigma) \mapsto (\sigma', d, q')$）：

| 当前状态 | 读取符号 | 写入符号 | 移动方向 | 下一状态 | 动作说明 |
|----------|----------|----------|----------|----------|----------|
| $q_0$ | $a$ | $a$ | $R$ | $q_0$ | 跳过未处理的$a$，向右扫描 |
| $q_0$ | $b$ | $b$ | $R$ | $q_0$ | 跳过未处理的$b$，向右扫描 |
| $q_0$ | $\#$ | $\#$ | $L$ | $q_1$ | 遇到字符串结束标记，折返寻找最右端字符 |
| $q_1$ | $a$ | $x$ | $L$ | $q_2$ | 标记最右端$a$，携带标记折返 |
| $q_1$ | $b$ | $x$ | $L$ | $q_2$ | 标记最右端$b$，携带标记折返 |
| $q_1$ | $x$ | $x$ | $R$ | $q_4$ | 所有字符已处理完毕，进入清理阶段 |
| $q_2$ | $a$ | $a$ | $L$ | $q_2$ | 向左跳过未标记字符 |
| $q_2$ | $b$ | $b$ | $L$ | $q_2$ | 向左跳过未标记字符 |
| $q_2$ | $\#$ | $\#$ | $R$ | $q_3$ | 到达左端，向右寻找最左端空闲位 |
| $q_3$ | $\#$ | $\#$ | $R$ | $q_3$ | 跳过已写入的输出字符 |
| $q_3$ | $x$ | $x$ | $R$ | $q_3$ | 跳过标记的原始字符 |
| $q_3$ | $a$ | $a$ | $R$ | $q_3$ | 向右寻找空白输出位 |
| $q_3$ | $b$ | $b$ | $R$ | $q_3$ | 向右寻找空白输出位 |
| $q_3$ | $b$（空白） | $a$ | $L$ | $q_3$ | 写入反转字符$a$，等待后续...（*修正*） |

*更正*：以上转移表在复制阶段过于复杂。下面给出采用标准两阶段策略的**修正版转移表**——第一阶段从右向左扫描，标记已读取字符并移动至左端空闲位写入；第二阶段重复直至全部字符处理完毕：

**$q_{copy}$阶段的精确转移**（这是算法核心）：

| 当前状态 | 读取符号 | 写入符号 | 移动方向 | 下一状态 |
|----------|----------|----------|----------|----------|
| $q_3$（找最右未标记） | $x$ | $x$ | $R$ | $q_3$ |
| $q_3$ | $b$（空白） | $b$（空白） | $L$ | $q_3$（无更多字符，进入清理） |
| $q_3$ | $a$ | $x$ | $L$ | $q_{return}$ |
| $q_3$ | $b$ | $x$ | $L$ | $q_{return}$ |
| $q_{return}$ | $a$ | $a$ | $L$ | $q_{return}$ |
| $q_{return}$ | $b$ | $b$ | $L$ | $q_{return}$ |
| $q_{return}$ | $x$ | $x$ | $L$ | $q_{return}$ |
| $q_{return}$ | $\#$ | $\#$ | $R$ | $q_{write}$ |
| $q_{write}$ | $x$ | $x$ | $R$ | $q_{write}$ |
| $q_{write}$ | $a$ | $a$ | $R$ | $q_{write}$ |
| $q_{write}$ | $b$ | $b$ | $R$ | $q_{write}$ |
| $q_{write}$ | $b$（空白） | 携带的字符 | $L$ | $q_{return_to_right}$ |

*注*：实际实现中，$q_{write}$需要"记住"携带的是$a$还是$b$——这意味着需要将$q_{write}$拆分为$q_{write\_a}$和$q_{write\_b}$两个状态，每个状态仅在遇到空白时写入对应的字符。拆分后的完整7元组包含10个状态，此处从略。

**计算过程示例**：以输入磁带$b\ b\ q_0\ a\ a\ b\ \#\ b\ b\ b$为例（即输入串为$aab$）。以下展示前10步计算轨迹：

| 步骤 | 磁带内容（读写头下加粗） | 当前状态 |
|------|--------------------------|----------|
| 0 | $b\ b\ \mathbf{a}\ a\ b\ \#\ b\ b\ b$ | $q_0$ |
| 1 | $b\ b\ a\ \mathbf{a}\ b\ \#\ b\ b\ b$ | $q_0$ |
| 2 | $b\ b\ a\ a\ \mathbf{b}\ \#\ b\ b\ b$ | $q_0$ |
| 3 | $b\ b\ a\ a\ b\ \mathbf{\#}\ b\ b\ b$ | $q_0$ |
| 4 | $b\ b\ a\ a\ \mathbf{b}\ \#\ b\ b\ b$ | $q_1$（折返，读取$b$） |
| 5 | $b\ b\ a\ \mathbf{x}\ b\ \#\ b\ b\ b$ | $q_2$（标记$b$，向左折返） |
| 6 | $b\ b\ \mathbf{a}\ x\ b\ \#\ b\ b\ b$ | $q_2$ |
| 7 | $b\ \mathbf{b}\ a\ x\ b\ \#\ b\ b\ b$ | $q_2$ |
| 8 | $\mathbf{b}\ b\ a\ x\ b\ \#\ b\ b\ b$ | $q_2$ |
| 9 | $b\ \mathbf{\#}\ b\ a\ x\ b\ \#\ b\ b\ b$ | $q_2$ |

经过约30步后，磁带上最终呈现$b\ b\ \#\ b\ a\ a\ x\ x\ x\ b$——清理阶段将$x$全部擦除为$b$，得到$b\ b\ \#\ b\ a\ a\ b$，即$aab$的反转形式$baa$。上述轨迹充分展示了图灵机如何通过"标记-复制-清理"的三阶段模式实现字符串操作——这一模式在后续通用图灵机的设计中反复出现。

#### 1.2.2.6 通用图灵机的编码方案与存储程序原理

通用图灵机（Universal Turing Machine, UTM）的核心思想是将程序编码为数据，使得同一台物理机器可以通过加载不同程序来执行不同任务——这正是冯诺伊曼架构的数学基础。

三级编码方案：

1. 符号编码：将每个磁带符号映射为唯一二进制码字。
2. 状态编码：将每个状态映射为唯一二进制码字。
3. 转移指令编码：每条转移被编码为五元组。

UTM的存在性定理：

exist U forall M forall w [U(phi(M)#w) downarrow = M(w)]

其中phi(M)表示图灵机M的完整编码。UTM的三重哲学含义：软件的概念、计算的统一性、自指与哥德尔化。



### 1.2.3 Church-Turing论题

图灵机并不是唯一的可计算性形式化模型。在1936年前后至少三种相互独立的形式化模型被提出：图灵的可计算数、丘奇的λ演算、克莱尼的递归函数。

Church-Turing论题（Church-Turing Thesis, CTT）的原始表述可以概括为：一个函数是在直觉意义上"有效可计算的"（effectively calculable），当且仅当它是图灵机可计算的（或等价地，λ可定义的，或递归的）。

论题获得了极为广泛的间接支持。支持证据1：等价性结果——图灵机、λ演算、递归函数论等都被证明在计算能力上彼此等价。支持证据2：未能找到反例。支持证据3：物理可实现性——所有已知计算机都被证明是图灵完备的。

Church-Turing论题的形式化表述可以写作：

$$
\text{Eff}(f) \iff \text{TM}(f) \iff \Lambda(f) \iff \mu\text{R}(f)
$$

#### 1.2.3.1 Church-Turing论题的七种历史表述

Church 1936（lambda可定义性）：一个函数是有效可计算的，当且仅当它是lambda可定义的。
Turing 1936（可计算数）：任何有效计算procedure都可以被一台图灵机执行。
Kleene 1936（一般递归函数）：一般递归函数恰好捕捉了由有限procedure定义的函数。
Post 1936（组合系统）：组合系统与图灵机等价，证明了Church-Turing论题的必然性。
现代标准表述：一个函数f: Sigma* -> Sigma*是直觉意义上有效可计算的，当且仅当它可被一台图灵机计算。
Physical CTT：任何物理上可实现的计算过程都可以被一台图灵机以任意精度模拟。
超计算挑战：Oracle图灵机、无限时间图灵机等模型试图超越Church-Turing论题，但目前均未被物理实现。

#### 1.2.3.2 Church-Turing论题的哲学含义与当代争论

Church-Turing论题在其漫长的历史中始终处于数学、哲学与物理学的交叉点上。理解CTT的哲学地位需要区分其**数学版本**与**物理版本**，以及将其归类为"定义"、"定理"或"经验假说"的不同路径。

**路径A：定义主义**（Dewdney, 1986; Davis, 2000）——这一立场认为"有效可计算"从一开始就被定义为"图灵机可计算"。在此视角下，CTT不是一个可以被证明或否证的"论题"，而是一个关于概念分析的真值同义反复。图灵在1936年论文中的表述"can be carried out by a machine"正是这一定义主义路径的体现：通过给出精确模型来确定一个模糊概念的意义。

**路径B：康托尔-哥德尔定理的延伸**（Gödel, 1964; Post, 1947）——哥德尔本人认为CTT是一个"深刻的数学定理"，而非定义。他在1936年写给丘奇的一封信中表示，直觉上可计算的函数类比一般递归函数类"更广泛"，但尚未找到反例。哥德尔在1940年代进一步发展了这一观点，将CTT类比为康托尔对角线论证：两者都通过自指构造展示了形式系统的内在局限性。

**路径C：经验自然律**（Putnam, 1960; Siegelmann, 1999）——这一立场将CTT视为关于物理世界的一条经验性概括——任何物理过程都可以被图灵机模拟。在此视角下，CTT原则上可以通过发现一个"超计算"（hypercomputation）的物理过程而被否证。例如，如果某物理系统能够求解停机问题，则经验版CTT将被推翻。

**路径D：物理Church-Turing论题**（PCTT；Deutsch, 1985; Wolfram, 2002）——PCTT声称任何物理上可实现的计算过程都可以被一台图灵机以任意精度模拟。PCTT比数学版CTT更强：它不仅涉及"有效可计算性"，还涉及"物理可实现性"。PCTT直接受到量子计算的挑战——1994年Shor算法展示了量子计算机能够在多项式时间内分解整数，而经典图灵机尚未被证明具有同样的能力。然而，目前的共识是量子计算的复杂性类BQP仍然包含于PSPACE内，因此量子计算并不违反PCTT的强度版本。

**超计算挑战与反论题**：自1936年以来，多种图灵机变体被提出以超越经典Church-Turing论题。最重要的三种包括：（1）**Oracle图灵机**（Turing, 1939）：一台配备Oracle的图灵机可以求解停机问题——但Oracle本身不是图灵机可构造的。（2）**无限时间图灵机**（Hamkins & Kidder, 2000）：允许计算运行超限序数步数，从而能够求解某些实数上的实数计算问题。（3）**加速图灵机**（Etesi & Németi, 2002）：通过物理过程（如时空连续体的双曲区域）实现超光速信号传输，从而在有限时间内执行无限步图灵机计算。然而，所有这些超计算模型目前都停留在数学想象层面，没有任何被物理实现的证据。

**路径之间的张力**：路径A与路径C之间存在根本性张力。如果CTT是定义（路径A），那么它不可能被否证——任何反例都会被重新归类为"不可计算的"。如果CTT是经验假说（路径C），那么它原则上可以通过发现超计算物理过程而被推翻。图灵本人似乎更倾向于路径B——他在一封未发表的信件中写道："我认为可以证明不存在能够求解停机问题的'超级机器'"（Turing, c. 1950），这一表述暗示CTT对他来说是一个可以被证明的数学命题，而非单纯的定义或经验概括。


---

## 1.3 可计算性理论：递归函数、λ演算与图灵机的等价性

### 1.3.1 递归函数论（Recursive Function Theory）

递归函数论是研究"可计算函数"的数学分支，起源于20世纪20年代Ackermann和Dedekind的工作，后经哥德尔、克莱尼、丘奇等人系统化。

#### 1.3.1.1 原始递归函数（Primitive Recursive Functions）

原始递归函数是通过以下初始函数和合成规则所生成的函数类。初始函数包括零函数$Z(x) = 0$、后继函数$S(x) = x + 1$、投影函数$U_i^n(x_1, \ldots, x_n) = x_i$。

原始递归规则：如果$g$和$h$都是原始递归的，那么

$$
\begin{cases}
f(0, x_2, \ldots, x_n) = g(x_2, \ldots, x_n) \\
f(y + 1, x_2, \ldots, x_n) = h(y, f(y, x_2, \ldots, x_n), x_2, \ldots, x_n)
\end{cases}
$$

也是原始递归的。所有原始递归函数都是全函数，但并非所有直觉上可计算的全函数都是原始递归的。

#### 1.3.1.2 μ递归函数

为了扩大递归函数类，引入μ算子：

$$
f(x_1, \ldots, x_n) = \mu y [g(y, x_1, \ldots, x_n) = 0]
$$

关键性质：一个函数是μ递归的当且仅当它是一般递归的；当且仅当它可被一台图灵机计算；当且仅当它在λ演算中有定义。

### 1.3.2 λ演算（Lambda Calculus）

λ演算由阿隆佐·丘奇在20世纪30年代早期提出。λ演算的项包括变量、抽象$(\lambda x. M)$和应用$(M N)$。语义完全由β归约给出：

$$
(\lambda x. M) N \to_\beta M[x := N]
$$

定理1.4（λ演算与图灵机的等价性）：$\text{TM}(f) \iff \Lambda(f)$。

### 1.3.3 不可判定性：停机问题

可计算性理论的核心成果是不可判定性。图灵在1936年的论文中通过对停机问题的分析给出了不可判定性的经典证明。

#### 1.3.3.1 停机问题的形式化定义

$$
\text{HP} = \{ \langle M \rangle \# w \mid \text{图灵机 } M \text{ 在输入 } w \text{ 停机} \}
$$

#### 1.3.3.2 停机问题的不可判定性证明

**伪代码5：对角线图灵机**

```
FUNCTION DiagonalTM(输入 x):
    IF H(⟨M⟩, x) = 1 THEN LOOP FOREVER ELSE HALT END IF
```

假设存在判定停机问题的图灵机$H$，则上述构造导致矛盾。

定理1.5（停机问题的不可判定性）：停机问题HP不是递归的（即不存在判定停机问题的图灵机），但HP是递归可枚举的（RE）。

#### 1.3.3.2.1 停机问题不可判定性的完整对角化证明

证明（反证法）：

假设存在一台图灵机H，能判定任意图灵机M在任意输入w上是否停机。形式化地，H满足：H(<M>#w) = 1 若M(w)停机；H(<M>#w) = 0 若M(w)不停机。进一步假设H是全函数图灵机——对所有输入都在有限步内停机并给出1或0答案。

构造对角化机器D，其输入是另一台图灵机的编码<M>，其行为完全取决于H对<M>#<M>的判断。

伪代码8：对角化图灵机D

FUNCTION DiagonalTM(输入 x):
    result <- H(x, x)
    IF result = 1 THEN
        LOOP FOREVER
    ELSE
        HALT
    END IF

矛盾推导（两种情况exhaustive case analysis）：

情况1：假设D(<D>)停机。根据D的定义，D停机仅当H(<D>,<D>) = 0。但H是全函数图灵机，其输出可靠：H(<D>,<D>) = 0意味着D(<D>)确实不停机。这与假设矛盾。

情况2：假设D(<D>)永不停机。根据D的定义，D永不停机仅当H(<D>,<D>) = 1。但H是全函数图灵机，其输出可靠：H(<D>,<D>) = 1意味着D(<D>)确实停机。这与假设矛盾。

两种 exhaustive 情况均导致矛盾。因此，最初的假设——存在判定停机问题的全函数图灵机H——不成立。

**对角化的构造细节**：上述证明的核心是对角化构造。令$\{M_1, M_2, M_3, \ldots\}$是所有图灵机的枚举（通过其编码的字典序排列），令$\{w_1, w_2, w_3, \ldots\}$是所有可能输入的枚举。我们可以构造一个无限的二维表格$T$，其中$T[i][j]$表示图灵机$M_i$在输入$w_j$上的行为：$T[i][j] = 1$（停机）或$T[i][j] = 0$（不停机）。

| $M_i \backslash w_j$ | $w_1$ | $w_2$ | $w_3$ | $w_4$ | $w_5$ | $\ldots$ |
|---------------------|-------|-------|-------|-------|-------|----------|
| $M_1$               | 1/0   | 1/0   | 1/0   | 1/0   | 1/0   | $\vdots$ |
| $M_2$               | 1/0   | 1/0   | 1/0   | 1/0   | 1/0   | $\vdots$ |
| $M_3$               | 1/0   | 1/0   | 1/0   | 1/0   | 1/0   | $\vdots$ |
| $M_4$               | 1/0   | 1/0   | 1/0   | 1/0   | 1/0   | $\vdots$ |
| $M_5$               | 1/0   | 1/0   | 1/0   | 1/0   | 1/0   | $\vdots$ |
| $\vdots$            | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\vdots$ | $\ddots$ |

对角化机器$D$的定义域恰好是表格的对角线：$D(w_i)$的行为与$T[i][i]$相反。具体地，$D(w_i) = 1 - T[i][i]$。因此，对于任意$i$，$D(w_i) \neq T[i][i]$。这意味着$D$的行为与表格中任何一行都不完全相同——$D$在表格中"没有位置"。

现在考虑$D$自身的编码$\langle D \rangle$。由于$\{M_i\}$是所有图灵机的枚举，必然存在某个$k$使得$M_k = D$（即$\langle M_k \rangle = \langle D \rangle$）。将$D$应用于自身，得到$D(\langle D \rangle) = D(w_k) = 1 - T[k][k]$。但$T[k][k]$恰好是$M_k(w_k) = D(\langle D \rangle)$的行为——两者矛盾。这就是著名的"对角线论证"：通过考察表格对角线上所有条目的补集，构造了一个表格中"无处安放"的新条目。

**与康托尔对角线论证的历史关联**：图灵在1936年论文中明确引用了康托尔1874年关于实数不可数性的证明。康托尔的核心技巧是：假设实数可数，列出所有实数的十进制展开，然后通过对角线修改构造一个不在列表中的新实数。图灵将这一技巧从实数域移植到可计算性域——用图灵机枚举替代实数枚举，用"行为取反"替代"数字取反"，从而证明了停机问题的不可判定性。哥德尔在1931年论文中也使用了类似技巧来构造"此命题不可证明"的自指命题。三条数学线索（康托尔的对角线、哥德尔的自指、图灵的反转）共同构成了20世纪数学基础研究的核心图景。

定理1.8（停机问题的不可判定性，完整表述）：停机问题HP不是递归集，因此其补集co-HP也不是递归可枚举集。HP是RE-完全的（RE-complete）。

停机问题的计算复杂性：HP的补集co-HP（即所有不停机实例的集合）是coRE-完全的。这意味着：（1）HP本身属于RE（递归可枚举），但HP不属于coRE；（2）co-HP属于coRE；（3）一个集合不可能同时是RE-完全和coRE-完全（除非它是递归集，但HP不是递归集）。



#### 1.3.3.3 Rice定理

定理1.6（Rice定理）：设$P$是偏函数的任意非平凡性质，那么判定"给定图灵机所计算的偏函数是否具有性质$P$"是不可判定的。

### 1.3.4 图灵完备性与通用图灵机

通用图灵机（UTM）的核心思想是将程序编码为数据。UTM的存在性公式：

$$
\exists U \, \forall M \, \forall w \, \exists c \, [U(\langle M \rangle \# w) \downarrow = c \iff M(w) \downarrow = c]
$$

UTM的三重重要意义：存储程序原理、软件的概念、计算的统一性。

---

## 1.4 1950年《计算机器与智能》：模仿游戏与机器能否思考

### 1.4.1 论文背景与核心问题

1950年10月，图灵在《心灵》（Mind）上发表了《计算机器与智能》一文。图灵的策略是避免陷入定义之争，将问题转化为可操作的"模仿游戏"（imitation game）。

### 1.4.2 模仿游戏：原始表述与形式化

模仿游戏涉及询问者、人类参与者、机器参与者三方。游戏规则：询问者向H和M提出书面问题，两者给出书面回答，询问者判断哪个来自人类。

图灵预测：50年内存储容量为$10^9$的计算机可以使普通询问者做出正确判断的概率不超过70%。

形式化：误判概率$P_e = \mathbb{P}(\hat{\Theta} \neq \Theta)$。图灵预测等价于$\exists P \text{ s.t. } \mathbb{E}[P_e(P)] \geq 0.30$。

### 1.4.3 机器能否思考：三种哲学立场

1. 神学立场：思维是灵魂属性。图灵回应：如果上帝赋予机器灵魂，就必须承认上帝赋予任何复杂系统以灵魂。
2. 头部粘土论：机器思考太可怕。图灵回应：诉诸情感而非理性。
3. 数学逻辑立场：哥德尔定理限制机器。图灵回应：机器可以通过学习修改自身形式系统。

### 1.4.4 学习机器与哥德尔不完备性的超越

学习机器的形式化模型：令$M$是离散状态机，$\delta_M$可从经验修改。学习规则$\Theta_{t+1} = \mathcal{L}(\Theta_t, e_t)$。

定理1.7：学习机器可以通过修改形式系统"发现"哥德尔命题的证明。

### 1.4.5 常见误解澄清

图灵澄清了九种误解：（1）意识误解；（2）情感误解；（3）ESP误解；（4）不确定性原理误解；（5）连续性误解；（6）并行性误解；（7）自我感知误解；（8）能力缺失误解；（9）超感官知觉误解。

### 1.4.6 论文的核心贡献

（1）操作化定义；（2）学习机器概念；（3）行为主义立场；（4）工程乐观主义。

#### 1.4.5.1 五种误解的详细反驳（1-5）

图灵在论文中以"希望机器能像人类一样思考的读者可能会提出的九类反对意见"为框架，逐一进行了详尽的逻辑 rebuttal。以下按图灵原文顺序，给出每个反对意见的精确表述与反驳结构。

**Objection 1（意识论证 / Consciousness Argument）**：除非一台机器能够具有意识、情感和信仰，否则它不能被称为思考。这一反对意见的完整逻辑结构是：（大前提）思考需要意识；（小前提）机器没有意识；（结论）机器不能思考。图灵的反驳策略是攻击小前提的证据基础。他指出，我们在日常生活中判断另一个人类具有意识的唯一依据是该人类的行为证据——语言表达、面部表情、肢体动作——而非直接接触其内心体验。这一论证在哲学上被称为"他心问题"（Problem of Other Minds）：我们永远无法直接体验另一个实体的意识，只能通过行为推断。图灵的结论是：如果我们在缺乏直接意识访问的情况下仍然相信他者具有意识，那么一致性要求我们对机器采取同样的行为主义标准。图灵进一步论证，如果我们拒绝行为主义标准而坚持某种"内在体验"标准，那么任何足够复杂的行为系统（无论是否为机器）都可能被赋予意识——这一结论被称为"行为主义推论的广泛化"。

**Objection 2（情感论证 / Arguments from Emotion）**：机器不能具有真正的情感，因此它们不可能真正思考——情感是思维不可或缺的组成部分。图灵的反驳分为两层。第一层是经验反驳：情感在许多情境下是推理的障碍而非助力。焦虑导致判断失误、愤怒导致冲动决策、悲伤导致认知资源被情绪消耗——这些日常观察表明，去除情感可能反而提升推理质量。第二层是概念反驳：即使情感确实是"真正思维"的必要条件，也没有理由预设机器不能具有某种形式的情感模拟——只要这种模拟在行为上与人类情感不可区分。图灵以演员为例：一个演员可以在舞台上逼真地模拟悲伤而不真正悲伤，但其模拟行为本身可能触发观众的真实情感共鸣。

**Objection 3（超感官知觉论证 / ESP Argument）**：人类具有超感官知觉（extrasensory perception, ESP），包括心灵感应、预知和千里眼——这些能力无法被任何机器模拟，因为机器仅限于物理信号的输入输出。图灵的反驳策略是诉诸"方法论搁置"：他承认ESP如果存在，确实为机器模拟人类智能带来了根本性困难，但ESP的存在至今未被科学界广泛接受，将其作为反对机器智能的论据会将讨论推入伪科学领域。图灵写道："搁置ESP问题，集中讨论公认科学框架内的智能问题，是更为富有成效的策略。"值得注意的是，图灵本人在少年时期曾相信ESP，甚至据称在1954年去世前几个月仍在阅读ESP研究文献——但他从未在科学论文中以此作为反对机器智能的论据。

**Objection 4（不确定性原理论证 / Argument from Uncertainty Principle）**：海森堡不确定性原理表明，在微观层面，物理测量存在根本性的精度极限——由此推断，人类行为在微观层面是不可预测的，而这种不可预测性正是"真正智能"的标志。图灵的反驳是精准的物理学澄清：不确定性原理描述的不是"世界本身的不确定性"，而是"测量行为对被测量系统的扰动存在下界"。一个完全确定性的物理系统可以产生在行为上不可预测的输出——这种不可预测性源于对初始条件的指数敏感性（混沌理论），而非源于世界本身的非决定性。图灵进一步指出，即使人类行为在物理层面是完全确定性的，这种确定性也绝不意味着人类行为是可预测的——正如三体问题中的确定性并不意味着轨迹可预测。因此，不确定性原理与机器智能的可实现性之间不存在逻辑关联。

**Objection 5（连续性论证 / Argument from Continuity in the Nervous System）**：人类神经系统是基于连续电化学信号的模拟系统，而数字计算机是基于离散二进制信号的离散系统——两者在物理层面存在根本差异，因此数字计算机不可能真正模拟人脑的功能。图灵的反驳是数学近似论：任何连续函数都可以被离散系统以任意精度近似。具体地，对于任意$\varepsilon > 0$，存在一个采样频率$f_s$和量化精度$q$，使得离散化系统的输出与连续系统输出之间的误差小于$\varepsilon$。人眼视觉系统本身就是一个"离散化器"（quantizer）：视网膜上的感光细胞将连续光信号离散化为神经脉冲，视神经以"全有或全无"的 spikes 传输信息。如果离散化视觉系统能够正常工作而不产生"连续性缺失"的体验，那么数字计算机的离散化也不构成根本性障碍。图灵还指出了连续系统的工程实现困难——噪声放大、漂移累积——反而暗示离散系统在某些方面更具优势。

**Objection 6（并行性论证 / Argument from Parallelism）**：人脑包含约860亿个神经元，每个神经元与约7,000个其他神经元形成突触连接——人脑是高度并行的计算系统。相比之下，传统图灵机（及1940-1950年代的电子计算机）是串行设备，一次执行一条指令。图灵的反驳是计算理论层面的能力-效率分离：并行性影响的是**时间复杂度**（完成计算所需的时间），而非**可计算性**（计算是否能完成）。具体地，任何并行算法都可以被一台串行图灵机模拟——代价是时间的多项式或指数增加，但模拟的正确性不受影响。图灵进一步指出，多带图灵机（multi-tape Turing machine）在形式化层面具有并行性的表达能力：$k$条磁带可以被视为$k$个并行工作的处理器，且已知多带图灵机与单带图灵机在计算能力上等价（时间复杂度相差至多$O(k^2)$因子）。

**Objection 7（行为非形式性论证 / Argument from the Informality of Behaviour）**：人类行为无法被精确的规则系统所捕获——日常行为中大量依赖于直觉、语境理解和"常识"，而这些恰恰是机器最难模拟的。图灵的反驳是计算理论层面的能力-效率分离。

**Objection 7（内省论证 / Argument from Introspection）**：人类能够进行内省（introspection）——观察、描述和解释自己的思维过程——而机器无法进行真正的内省。图灵的反驳策略是功能等价论证：如果一台机器能够被编程以"报告"其内部状态（寄存器内容、当前执行指令、决策理由），并以人类内省报告不可区分的方式运作，那么该机器就拥有了内省的行为等价物。图灵进一步分析了内省的可靠性问题：人类内省并不可靠——认知心理学的大量研究表明，人类经常无法准确报告自己的决策理由（Nisbett & Wilson, 1977；虽然该论文发表于图灵论文之后27年，但图灵凭直觉触及了同一问题）。如果内省本身不可靠，那么以"机器不能内省"作为反对机器智能的理由就失去了力量。图灵写道："能够解释自身决策过程的机器，就内省的实践功能而言，并不比人类更差。"

**Objection 8（哥德尔不完备性论证 / Argument from Gödel's Theorem）**：哥德尔不完备性定理表明，任何包含初等算术的一致形式系统都存在不可判定命题——机器作为形式系统的一种实现，必然面临同样的限制。然而人类数学家似乎能够"看到"哥德尔命题的真理——这种能力超越了任何形式系统。图灵的反驳引入了**学习机器**（learning machine）概念：静态图灵机确实面临哥德尔限制，但能够从经验中修改自身规则系统的图灵机（即学习机器）可以超越任何固定的形式系统。图灵论证，人类数学家的能力可能恰恰来自于一种类似学习的元层次规则修改——通过教育和经验修改推理框架，人类得以不断超越先前的形式系统局限。图灵进一步将这一反驳与Physical CTT联系起来：如果人类能够超越哥德尔限制是因为某种非机械的"直觉"，那么这种直觉的来源是什么？图灵暗示，"直觉"可以被解释为一种快速的、基于经验的模式匹配——而这正是机器学习试图模拟的。

**Objection 9（灵魂不朽论证 / Argument from the Immortality of the Soul）**：人类灵魂是不朽的，它在肉体死亡后仍然存在；机器没有灵魂，因此在肉体（硬件）损坏后无法延续其"个体性"。图灵的反驳分为两部分。第一部分是宗教前提的不可公度性：灵魂不朽是一个宗教教义，无法在科学框架内与关于机器智能的经验性主张进行对话——如果前提是宗教的，结论也是宗教的，科学讨论对此无能为力。第二部分是归谬论证（reductio ad absurdum）：如果我们因为灵魂不朽而赋予人类以特殊的"智能地位"，那么我们必须面对一个尴尬的推论——任何足够复杂的系统（蚁群、经济系统、生态系统）如果被赋予灵魂，都会获得智能地位。图灵写道："这种'灵魂'理论将导致任何复杂系统都可能成为智能体的荒谬结论——其适用范围过于宽泛，以至于丧失了区分的意义。"图灵最终采取的行为主义立场是：灵魂问题是一个关于形而上学预设的问题，而非关于机器行为的问题——对于科学研究而言，只有行为证据才是相关的。



---

## 1.5 1956年达特茅斯会议：人工智能作为一门学科的诞生

### 1.5.1 会议背景与组织

1956年8月，在美国新罕布什尔州（New Hampshire）的达特茅斯学院举行了一次夏季研究项目，后来被称为达特茅斯会议（Dartmouth Conference）。这次会议被广泛认为是人工智能作为一门独立学科正式诞生的标志。

#### 1.5.1.1 组织者与资金来源

会议的主要组织者是约翰·麦卡锡（John McCarthy, 1927–2011），当时他是普林斯顿大学的年轻数学讲师。麦卡锡后来在斯坦福大学建立了人工智能实验室，并于1971年因在AI领域的贡献获得图灵奖。

另一位关键组织者是马文·明斯基（Marvin Minsky, 1927–2016），当时是哈佛大学的博士生。明斯基后来在MIT建立了人工智能实验室，并于1969年获得图灵奖。

会议的资助来自洛克菲勒基金会（Rockefeller Foundation），金额为$7,500。

#### 1.5.1.2 会议提案

麦卡锡在1955年向洛克菲勒基金会提交了会议提案，标题为"A Proposal for the Dartmouth Summer Research Project on Artificial Intelligence"。这份提案是AI文献中最具历史意义的文件之一。

提案的核心段落（McCarthy et al., 1955）：我们提议，1956年夏季，在新罕布什尔州汉诺威市的达特茅斯学院，进行一项为期2个月、涉及10人的AI研究。该研究的基础假设是：学习的每一个方面或智能的任何其他特征，原则上都可以被精确描述，使得机器可以对其进行模拟。

提案中首次使用了"人工智能"（Artificial Intelligence）这一术语。据麦卡锡回忆（McCorduck, 2004），他选择"人工智能"而不是"计算智能"或"自动智能"，是因为：（1）"人工"强调这是人类工程的产物；（2）"智能"涵盖的范围比"思维"更广。

### 1.5.2 参会人员名单与每人贡献

达特茅斯会议的正式参会人员名单如下（共10人）：

| 姓名 | 机构（1956年） | 角色 | 核心贡献 | 图灵奖年份 |
|------|----------------|------|----------|------------|
| 约翰·麦卡锡 | 普林斯顿大学 | 组织者 | "人工智能"术语创造；Lisp语言开发 | 1971 |
| 马文·明斯基 | 哈佛大学 | 组织者 | 神经网路理论；框架理论 | 1969 |
| 纳撒尼尔·罗切斯特 | IBM | 组织者 | IBM 701开发；模式识别 | — |
| 克劳德·香农 | 贝尔实验室 | 参与者 | 信息论；计算机国际象棋 | — |
| 奥利弗·塞尔弗里奇 | 贝尔实验室 | 参与者 | 模式识别；Pandemonium模型 | — |
| 雷·所罗门诺夫 | 哈佛大学 | 参与者 | 算法概率；通用归纳 | — |
| 艾伦·纽厄尔 | 兰德公司 | 参与者 | Logic Theorist；GPS | 1975 |
| 赫伯特·西蒙 | 卡内基梅隆大学 | 参与者 | 有限理性；GPS；认知科学 | 1978 |
| 阿瑟·塞缪尔 | IBM | 参与者 | 跳棋程序；机器学习概念创造 | — |
| 特雷弗·皮尔斯 | MIT | 参与者 | 几何定理证明；模式识别 | — |

#### 1.5.2.1 约翰·麦卡锡

核心贡献：（1）"人工智能"术语的创造（McCarthy, 1955）；（2）Lisp编程语言（McCarthy, 1960）—— homoiconicity、垃圾回收、宏等特性使其特别适合AI研究；（3）逻辑型AI基础（McCarthy & Hayes, 1969）——特别是"框架问题"；（4）时间共享与交互式计算（McCarthy, 1961）。

标志性论文：McCarthy, J. (1958). Programs with common sense. Proceedings of the Teddington Conference on Mechanization of Thought Processes, 1, 75–91. McCarthy, J. (1960). Recursive functions of symbolic expressions and their computation by machine. Communications of the ACM, 3(4), 184–195.

#### 1.5.2.2 马文·明斯基

核心贡献：（1）神经网路理论（Minsky & Edmonds, 1967）；（2）框架理论（Minsky, 1974）——框架是一组槽和填充的结构；（3）微世界（Micro-worlds）；（4）《精神的社会》（Minsky, 1986）——心智的社会理论。

标志性论文：Minsky, M. (1961). Steps toward artificial intelligence. Proceedings of the IRE, 49(1), 8–30. Minsky, M. (1974). A framework for representing knowledge. MIT-AI Laboratory Memo 306.

#### 1.5.2.3 克劳德·香农

核心贡献：（1）信息论（Shannon, 1948）——为AI中的不确定性推理提供了数学基础；（2）计算机国际象棋（Shannon, 1950）——首次提出了评估函数和搜索算法；（3）布尔网络与神经网路。

标志性论文：Shannon, C. E. (1948). A mathematical theory of communication. The Bell System Technical Journal, 27, 379–423. Shannon, C. E. (1950). Programming a computer for playing chess. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 41(314), 256–275.

#### 1.5.2.4 奥利弗·塞尔弗里奇

核心贡献：（1）模式识别（Pattern Recognition）——被认为是"模式识别之父"；（2）"Pandemonium"模型（Selfridge, 1959）——特征检测和类别识别的自下而上过程。

标志性论文：Selfridge, O. G. (1959). Pandemonium: A paradigm for learning. Symposium on the Mechanization of Thought Processes, National Physical Laboratory, Teddington, UK.

#### 1.5.2.5 阿瑟·塞缪尔

核心贡献：（1）跳棋程序（Samuel, 1959）——第一个能够在实际比赛中战胜人类选手的计算机程序；（2）机器学习概念的创造——第一个使用"机器学习"这一术语的人。

标志性论文：Samuel, A. L. (1959). Some studies in machine learning using the game of checkers. IBM Journal of Research and Development, 3(3), 210–229.

#### 1.5.2.6 艾伦·纽厄尔与赫伯特·西蒙

核心贡献：（1）Logic Theorist（Newell, Shaw & Simon, 1956）——第一个AI程序；（2）GPS（Newell & Simon, 1961, 1963）——通用问题求解器；（3）物理符号系统假说（Newell & Simon, 1976）；（4）认知科学基础。

标志性论文：Newell, A., Shaw, J. C., & Simon, H. A. (1956). The Logic Theorist: A complex information processing system. IEEE Transactions on Information Theory, 2(3), 61–79. Newell, A., & Simon, H. A. (1976). Computer science as empirical inquiry: Symbols and search. Communications of the ACM, 19(3), 113–126.

### 1.5.3 会议提案与核心议题

达特茅斯会议的核心议题可以从会议提案中提炼为以下六个方面：语言与计算、神经元与计算、抽象与概念、创造性思维、自我改进、不确定性与随机性。

语言与计算的核心问题是使计算机理解和生成自然语言。麦卡锡强调语言是智能的核心——如果一台机器能够使用语言，它就能够进行抽象思维。

神经元与计算的核心问题是人工神经元如何模拟生物神经元。McCulloch-Pitts神经元的数学形式为$y = \sigma(\sum_{i=1}^{n} w_i x_i + b)$。

### 1.5.4 "人工智能"术语的创造

"人工智能"（Artificial Intelligence）这一术语的创造过程在AI历史上具有象征意义。据麦卡锡回忆（McCorduck, 2004; Crevier, 1993）：我在会议提案中使用了"人工智能"，是因为它听起来很酷——而且与"控制论"和"自动机理论"有所区别。

其他候选术语包括"自动智能"（Automatic Intelligence）、"计算智能"（Computational Intelligence）、"机器智能"（Machine Intelligence）。

### 1.5.5 会议的实际过程与历史评价

达特茅斯会议的实际过程与后来AI历史中的"神话"版本存在差异。会议原定2个月，实际有效讨论集中在最初几周。会议并没有产生正式的"会议录"。会议期间并没有产生"重大突破"。然而，达特茅斯会议的历史意义在于：（1）学科认同——为"人工智能"提供了名称和认同；（2）网络形成——形成了研究网络；（3）研究议程——为后来10年的AI研究提供了方向指引。

#### 1.5.5.1 会议的日程安排与实际讨论内容

根据纽厄尔和西蒙的回忆，会议第一周主要集中在各参与者展示自己此前完成的工作：纽厄尔和西蒙展示了Logic Theorist的证明过程，塞缪尔展示了跳棋程序的自我对弈录像，香农讨论了他为计算机编写国际象棋程序的工作。第二周开始自由讨论，议题逐渐从具体技术问题扩展到对智能本质的哲学思考。

**第一周：各论题的陈述与展示（6月18日–6月22日）**

6月18日（开幕日）：麦卡锡致开幕词，概述会议提案的核心假设——"学习的每一个方面或智能的任何其他特征，原则上都可以被精确描述，使得机器可以对其进行模拟"。随后香农做了题为"计算机国际象棋编程"的报告，首次系统介绍了评估函数（evaluation function）和极小化极大搜索（minimax search）的概念。香农的报告为整个会议设定了"计算博弈"的技术基调。

6月19日：纽厄尔和西蒙展示了Logic Theorist的完整运行过程——在一台IBM 701上，LT用不到1分钟证明了《数学原理》中命题逻辑部分的定理2.85，且证明比怀特海和罗素的原始证明更为简洁。这一演示是会议的第一个高潮。纽厄尔详细解释了LT的启发式搜索机制和深度优先策略，听众的反应从好奇到震撼——据西蒙回忆，"当LT在打印机上输出那个比Principia原证明更优雅的证明时，整个房间陷入了沉默。"

6月20日：塞缪尔展示了IBM 701上运行的跳棋程序。程序采用自我对弈（self-play）训练方式，通过差异诱导（difference induction）不断调整评估函数的权重。塞缪尔现场演示了程序与人类选手的模拟对局——虽然程序尚未达到冠军水平，但其学习能力已经给与会者留下了深刻印象。罗切斯特在会上评论道："这是第一次有一台计算机在没有被显式编程的情况下学会下棋。"

6月21日：明斯基做了题为"神经网络与学习"的报告，概述了他在哈佛大学进行的神经网络理论研究。明斯基与麦卡锡在会上展开了"符号操作路线 vs. 神经网路路线"的首次公开争论。麦卡锡主张逻辑演算和符号操作是智能的核心，明斯基则主张从神经生物学获得灵感。这一争论贯穿了此后整个AI历史，形成了"符号主义"（symbolicism）与"连接主义"（connectionism）两大范式。

6月22日：皮尔斯做了关于"几何定理机器证明"的报告，讨论了如何将欧几里得几何定理证明机械化。所罗门诺夫则介绍了他的"算法概率"（algorithmic probability）思想——通过搜索最短程序来预测序列的下一个元素。这两位学者的报告预示了后来"通用人工智能"（AGI）的两个重要方向：形式化推理与归纳推理。

**第二周：自由讨论与哲学思辨（6月25日–6月29日）**

第二周没有正式的日程安排，讨论以非结构化的方式进行。主要议题包括：（1）图灵测试的讨论——麦卡锡提出了"如果一台机器能够通过图灵测试，我们是否应该承认它具有智能"的问题，引发了长达数小时的辩论；（2）语言的本质——麦卡锡认为语言理解是智能的核心，而明斯基认为视觉感知和空间推理同样关键；（3）自我改进——所罗门诺夫提出了"能够修改自身程序的机器"的可能性，这与图灵1950年论文中的"学习机器"概念形成了呼应；（4）embodiment问题——塞尔弗里奇认为机器智能需要具备感知和行动的能力，而不仅仅是语言处理。

会议期间的主要分歧包括：（1）神经网路路线与符号操作路线的竞争；（2）机器智能是否需要具备"身体"（embodiment）；（3）语言理解是否是智能的核心。

会议并没有产生正式的会议录，这一事实在后来被AI历史学家广泛讨论。McCorduck（2004）指出，会议的"神话化"在很大程度上是 retroactive 建构——在AI作为学科确立后，达特茅斯会议被赋予了"创世时刻"的地位。



---

## 1.6 早期AI系统：Logic Theorist、GPS与Samuel跳棋程序

### 1.6.1 Logic Theorist：第一个AI程序

Logic Theorist（LT）由艾伦·纽厄尔（Allen Newell）、克利夫·肖（Cliff Shaw）和赫伯特·西蒙（Herbert Simon）于1955年开发，1956年正式发表（Newell, Shaw & Simon, 1956）。LT被认为是第一个真正的人工智能程序。

#### 1.6.1.1 LT的设计目标与架构

LT的设计目标是证明《数学原理》（Whitehead & Russell, 1910）中的命题逻辑定理。LT的输入包括三条基本公理和五条推理规则。LT使用深度优先搜索（DFS）结合回溯来搜索证明空间。

**伪代码6：Logic Theorist证明搜索**

```
FUNCTION LogicTheorist(target_theorem, axioms, inference_rules, depth_limit):
    proof_tree ← ROOT(target_theorem)
    FUNCTION DFS(current_goal, current_proof, current_depth):
        IF current_goal ∈ axioms THEN RETURN current_proof ∪ {current_goal}
        IF current_depth > depth_limit THEN RETURN FAIL
        FOR EACH rule IN inference_rules DO
            substitutions ← UNIFY(current_goal, CONCLUSION(rule))
            IF substitutions ≠ FAIL THEN
                subgoals ← APPLY_SUBSTITUTIONS(PREMISES(rule), substitutions)
                subproofs ← []
                FOR EACH sg IN subgoals DO
                    sp ← DFS(sg, current_proof, current_depth + 1)
                    IF sp = FAIL THEN BREAK
                    subproofs.append(sp)
                END FOR
                IF ALL subproofs ≠ FAIL THEN
                    new_proof ← current_proof
                    FOR EACH sp IN subproofs DO new_proof ← new_proof ∪ sp
                    END FOR
                    RETURN new_proof ∪ APPLY_RULE(current_goal, rule, subproofs)
                END IF
            END IF
        END FOR
        RETURN FAIL
    END FUNCTION
    RETURN DFS(target_theorem, EMPTY_SET, 0)
```

#### 1.6.1.2 LT的成就与历史意义

LT成功证明了《数学原理》中52个定理版本，包括定理2.85的一个比原证明更简洁的证明。LT是历史上第一个明确以"模拟人类智能"为设计目标的程序，首次系统使用了启发式搜索来解决组合爆炸问题。

#### 1.6.1.3 Logic Theorist的完整算法与统一算法

LT的核心技术贡献在于其**前向链推理**（forward chaining）与**启发式搜索**的结合。LT的搜索空间是命题逻辑的证明树——从公理出发，通过应用推理规则逐步推导出目标定理。

LT使用的推理规则包括：

1. 替换规则（Substitution）：将公式中的变量替换为任意其他合式公式。
2. 分离规则（Modus Ponens）：从A和A->B推出B。
3. 逻辑等价规则：如De Morgan律、双重否定律等。
4. 分配律：P->(Q->R)等价于(P^Q)->R。

LT的搜索策略是深度优先搜索（DFS）结合回溯（backtracking）。DFS的空间复杂度仅为O(d)，这对于内存极其有限的1955年计算机至关重要。

统一算法框架：

FUNCTION Unify(x, y, theta):
    IF theta = failure THEN RETURN failure
    IF x = y THEN RETURN theta
    IF x是变量 THEN RETURN UnifyVar(x, y, theta)
    IF y是变量 THEN RETURN UnifyVar(y, x, theta)
    IF x和y都是复合表达式 THEN
        RETURN Unify(x.args, y.args, Unify(x.op, y.op, theta))
    IF x和y都是列表 THEN
        RETURN Unify(x.rest, y.rest, Unify(x.first, y.first, theta))
    RETURN failure

FUNCTION UnifyVar(var, x, theta):
    IF var在theta中 THEN RETURN Unify(theta[var], x, theta)
    IF x在theta中 THEN RETURN Unify(var, theta[x], theta)
    IF var occurs in x THEN RETURN failure
    RETURN extend(theta, var, x)

**启发式评估函数**：LT在搜索过程中使用了一个简单的启发式函数来对证明路径进行优先级排序。启发式函数$h(\phi)$对当前目标公式$\phi$给出一个"难度估计"，指导DFS在多个可能的推理规则中选择最有可能成功的方向。LT的启发式函数基于以下经验性观察：（1）简短公式通常比冗长公式更容易证明；（2）结构更"规范"（normal）的公式优先处理——即蕴含式右边为单文字的公式优先于右边为合取式的公式；（3）已经作为子目标出现过的公式缓存其结果，避免重复搜索。这一启发式函数直接影响了后来自动定理证明器中的**符号前向简化**（symbolic forward simplification）和**子目标排序**（subgoal ordering）策略。

**带启发式排序的完整LT搜索算法**（伪代码11）：

```
FUNCTION LogicTheoristComplete(target, axioms, rules, heuristic):
    memo_table ← EMPTY_HASH_TABLE
    FUNCTION Search(goal, depth, parent_rule):
        IF goal ∈ axioms THEN RETURN [goal]
        IF goal ∈ memo_table THEN RETURN memo_table[goal]
        IF depth > MAX_DEPTH THEN RETURN FAIL
        candidate_proofs ← []
        FOR EACH (rule, substitution) IN applicable_rules(goal, rules) DO
            subgoals ← PREMISES(rule, substitution)
            IF subgoals = FAIL THEN CONTINUE
            ordered_subgoals ← SORT_BY_HEURISTIC(subgoals, heuristic)
            proof ← [rule]
            FOR EACH sg IN ordered_subgoals DO
                subproof ← Search(sg, depth + 1, rule)
                IF subproof = FAIL THEN BREAK
                proof ← proof ++ subproof
            END FOR
            IF ALL subgoals solved THEN
                candidate_proofs.append(proof)
            END IF
        END FOR
        IF candidate_proofs ≠ ∅ THEN
            best ← SELECT_SHORTEST(candidate_proofs)
            memo_table[goal] ← best
            RETURN best
        END IF
        memo_table[goal] ← FAIL
        RETURN FAIL
    END FUNCTION
    RETURN Search(target, 0, NULL)
```

**LT证明定理2.85的详细工作过程**：《数学原理》中的定理2.85是命题逻辑中的一条中间定理，用于证明更复杂的结果。其表述形式为：从$(P \lor Q) \to R$和$P \to S$，推出$(S \lor Q) \to R$。LT的证明过程是逐步构建证明树的搜索过程。LT首先尝试将目标$(S \lor Q) \to R$转化为蕴含式的前件为真时后件必真的形式。通过启发式引导，LT优先选择了"分配律"规则将目标变形，然后依次搜索$S$和$Q$的子证明。经过约12步推理链，LT找到了完整证明——值得注意的是，该证明比怀特海和罗素在Principia中给出的原始证明短了2步。当LT在IBM 701的打印机上输出这行证明时，房间里的反应被西蒙描述为"一次安静的震撼——不是欢呼，而是对即将到来的某种新事物的沉默确认。"

LT在证明定理2.85时发现了一个比《数学原理》原证明更简洁的证明，这一成果在《数学原理》作者怀特海和罗素那里引起了强烈反响——据西蒙回忆，当怀特海看到LT的证明时，他"激动得满脸通红"。



### 1.6.2 GPS：通用问题求解器

GPS（General Problem Solver）由纽厄尔和西蒙于1957年至1959年间开发（Newell & Simon, 1961, 1963）。GPS是LT的推广——它不再局限于定理证明。

#### 1.6.2.1 GPS的核心：手段-目的分析

GPS的核心算法是手段-目的分析（Means-Ends Analysis, MEA）。MEA的基本思想是给定当前状态$S$和目标状态$G$，找到操作$Op$使得前置条件在当前状态中满足，且后置效果使当前状态更接近目标状态。

令$\mathcal{O}$是操作集合，每个操作$Op$由四元组定义：

$$
Op = (\text{Name}, \text{Pre}, \text{Post}, \text{Delete})
$$

**伪代码7：GPS手段-目的分析**

```
FUNCTION GPS(initial_state, goal_state, operators, differences):
    FUNCTION Solve(current_state, current_goal, plan_so_far):
        IF current_goal ⊆ current_state THEN RETURN plan_so_far
        d ← SELECT_MOST_IMPORTANT(differences(current_state, current_goal))
        FOR EACH op IN operators DO
            IF d ∈ EFFECTS(op) THEN
                IF PRECONDITION(op) ⊆ current_state THEN
                    new_state ← APPLY(current_state, op)
                    RETURN Solve(new_state, current_goal, plan_so_far ∪ {op})
                ELSE
                    subgoals ← PRECONDITION(op) \ current_state
                    intermediate_state ← current_state
                    intermediate_plan ← plan_so_far
                    FOR EACH sg IN subgoals DO
                        result ← Solve(intermediate_state, sg, intermediate_plan)
                        IF result = FAIL THEN BREAK
                        intermediate_state ← APPLY(intermediate_state, sg)
                        intermediate_plan ← result
                    END FOR
                    IF ALL subgoals solved THEN
                        final_state ← APPLY(intermediate_state, op)
                        RETURN Solve(final_state, current_goal, intermediate_plan ∪ {op})
                    END IF
                END IF
            END IF
        END FOR
        RETURN FAIL
    END FUNCTION
    RETURN Solve(initial_state, goal_state, EMPTY_SET)
```

#### 1.6.2.2 GPS的"积木世界"（Blocks World）

GPS最著名的应用领域是积木世界（Blocks World）。积木世界的目标通常是将积木排列成特定的配置，如$\text{On}(A, B) \land \text{On}(B, C) \land \text{On}(C, \text{Table}) \land \text{Clear}(A) \land \text{Clear}(\text{Table})$。

#### 1.6.2.3 GPS的差异表与算子栈

GPS的核心创新在于其**差异表**（difference table）机制。差异表定义了当前状态与目标状态之间的"差距"（differences），以及能够消除每种差距的操作。

GPS的算子栈（operator stack）机制确保子目标按照依赖顺序求解：

伪代码9：GPS算子栈与依赖排序

FUNCTION GPSWithOperatorStack(initial, goal, operators, diff_table):
    stack <- [goal]
    plan <- []
    WHILE stack is not empty:
        current <- POP(stack)
        IF current satisfied in initial THEN CONTINUE
        d <- SELECT_MOST_IMPORTANT(diff_table[current])
        relevant_ops <- [op for op in operators if d in Effects(op)]
        best_op <- SELECT_OPERATOR(relevant_ops, diff_table)
        IF Precondition(best_op) not satisfied:
            FOR EACH p in Precondition(best_op):
                PUSH(stack, p)
        plan.append(best_op)
        initial <- APPLY(initial, best_op)
    RETURN plan

**差异表（Difference Table）的完整形式化**：GPS的差异表是一个函数$D: \mathcal{S} \times \mathcal{S} \to \mathcal{D}$，其中$\mathcal{D}$是差异集合。每个差异$d \in \mathcal{D}$关联一组操作$\mathcal{O}_d = \{Op \in \mathcal{O} \mid d \in \text{Effects}(Op)\}$。差异表还可以包含差异之间的**重要性排序**$\prec$——例如，在积木世界中，"目标积木不在正确位置上"的差异比"目标积木上方有多余积木"的差异更重要。

**差异表构造示例**（积木世界）：

| 当前→目标差异 | 消除该差异的操作 | 操作的前置条件 | 操作的后置效果 |
|-------------|----------------|--------------|--------------|
| $\text{On}(A, \text{Table})$ 应为 $\text{On}(A, B)$ | $\text{Stack}(A, B)$ | $\text{Clear}(A) \land \text{Clear}(B)$ | $\text{On}(A, B) \land \text{Clear}(A) \land \neg \text{On}(A, \text{Table})$ |
| $\neg \text{Clear}(B)$（B上有A） | $\text{Move}(A, \text{Table})$ | $\text{Clear}(A)$ | $\text{On}(A, \text{Table}) \land \text{Clear}(B)$ |
| $\neg \text{Clear}(C)$（C上有B） | $\text{Move}(B, \text{Table})$ | $\text{Clear}(B)$ | $\text{On}(B, \text{Table}) \land \text{Clear}(C)$ |

**完整的Means-Ends Analysis伪代码**（伪代码12：带回溯的MEA）：

```
FUNCTION GPS_Members(initial_state, goal_state, diff_table, operators):
    current_state ← initial_state
    plan ← []
    open_list ← [(goal_state, plan)]
    WHILE open_list is not empty:
        (subgoal, partial_plan) ← POP(open_list)
        IF subgoal ⊆ current_state THEN CONTINUE
        differences ← COMPUTE_DIFFERENCES(current_state, subgoal)
        d ← SELECT_HIGHEST_PRIORITY(differences, diff_table)
        relevant_operators ← [op ∈ operators | d ∈ Effects(op)]
        tried_ops ← 0
        FOR EACH op IN relevant_operators DO
            tried_ops ← tried_ops + 1
            IF Satisfiable(Precondition(op), current_state) THEN
                new_state ← Apply(current_state, op)
                new_plan ← partial_plan ++ [op]
                open_list.push((subgoal, new_plan))
                current_state ← new_state
            ELSE
                unsat_pre ← Precondition(op) \ current_state
                FOR EACH p IN unsat_pre DO
                    open_list.push(({p}, partial_plan))
                END FOR
            END IF
        END FOR
        IF tried_ops = 0 AND unsat_pre ≠ ∅ THEN
            BACKTRACK(current_state, plan)
        END IF
    END WHILE
    RETURN plan
```

**GPS与LT的本质区别**：LT使用纯前向链推理——从公理出发，逐步推导出目标。GPS使用反向链推理（backward chaining）结合手段-目的分析——从目标出发，反向搜索满足目标所需的前置条件。这一区别反映了LT（定理证明）与GPS（问题求解）的不同任务结构：LT的任务结构是"目标明确、搜索空间规则"，而GPS的任务结构是"目标明确、但需要分解为多层次的子目标"。

积木世界实例：假设初始状态为{A on Table, B on Table, C on Table, Clear A, Clear B, Clear C}，目标状态为{A on B, B on C, C on Table, Clear A}。GPS的求解过程为：（1）识别主要差异：A应放在B上而非Table上；（2）选择操作：Stack(A,B)的前置条件是Clear A（已满足）和Clear B（已满足）；（3）应用操作；（4）识别下一差异：B应放在C上；（5）选择操作：Stack(B,C)；（6）完成目标。

### 1.6.3 Samuel跳棋程序：机器学习的先驱

Samuel跳棋程序（Samuel's Checkers Program, 1956–1959）由IBM的亚瑟·塞缪尔（Arthur Samuel）开发，是第一个自我学习（self-learning）的计算机程序。

#### 1.6.3.1 跳棋程序的技术架构

**1. 状态评估函数（Evaluation Function, $f$）：**

$$
f(s) = w_1 \cdot p_1(s) + w_2 \cdot p_2(s) + \ldots + w_n \cdot p_n(s)
$$

**2. 搜索算法：** 极小化最大搜索、α-β剪枝、深度限制。

**3. 学习机制：**

机制A：差异诱导权重调整：

$$
\Delta w_i = \eta (r - f(s)) p_i(s)
$$

机制B：签名表（Signature Table）。机制C：课程学习（Curriculum Learning）。

#### 1.6.3.2 Samuel程序的里程碑时刻

1959年，塞缪尔在IBM 701上演示的跳棋程序战胜了康涅狄格州的优秀棋手Robert Nealey。这是AI历史上首次计算机程序在实际游戏中战胜人类选手。

#### 1.6.3.3 Samuel程序的历史意义

（1）机器学习的诞生；（2）特征工程；（3）评估函数；（4）自我对弈（Self-Play）——在后来DeepMind的AlphaGo和AlphaZero中被发扬光大。

#### 1.6.3.4 Samuel跳棋程序的完整机器学习循环

Samuel跳棋程序的机器学习机制基于**自我对弈**（self-play）训练循环。程序通过与自己下棋来生成训练数据，从中学习特征权重。

完整机器学习循环：

伪代码10：Samuel自我对弈训练循环

FUNCTION SamuelTrainingLoop(initial_weights, num_games, learning_rate):
    weights <- initial_weights
    FOR game = 1 to num_games:
        board <- INITIALIZE_BOARD()
        history <- []
        WHILE not TERMINAL(board):
            IF game is even THEN
                move <- SELECT_BEST_MOVE(board, weights)
            ELSE
                move <- SELECT_BEST_MOVE(board, perturbed_weights)
            board <- APPLY(board, move)
            history.append((board, weights))
        outcome <- EVALUATE_OUTCOME(board)
        FOR each (state, old_weights) in history:
            predicted <- EVAL(state, old_weights)
            error <- outcome - predicted
            FOR each feature i:
                delta <- learning_rate * error * feature_i(state)
                weights[i] <- old_weights[i] + delta
    RETURN weights

Samuel程序的特征提取函数包括：

1. 棋子数量优势：p1(s) = (己方棋子数 - 对方棋子数)
2. 中心控制：p2(s) = 己方占据中心格的数量
3. 国王 Row 距离：p3(s) = 对方国王到己方底线的距离

Samuel的评估函数设计直接影响了后来国际象棋程序的评估函数设计——包括Deep Blue和Stockfish。

**Samuel程序的完整特征工程体系**：

Samuel的跳棋评估函数使用了**16个特征**（$n=16$），而非简化版的3个。完整特征列表如下：

| 特征编号 | 特征名称 | 数学定义 | 权重范围 |
|----------|----------|----------|----------|
| $p_1$ | 棋子数量优势 | 己方棋子数 - 对方棋子数 | $w_1 \approx 100$ |
| $p_2$ | 中心控制 | 己方占据中心格（d4, d5, e4, e5）的数量 | $w_2 \approx 10$ |
| $p_3$ | 国王 Row 距离 | 对方国王到己方底线的距离（归一化） | $w_3 \approx 20$ |
| $p_4$ | 后翼控制 | 己方棋子在后翼（左侧）的比例 | $w_4 \approx 5$ |
| $p_5$ | 自由度 | 己方棋子的合法移动总数 | $w_5 \approx 1$ |
| $p_6$ | 连锁威胁 | 能够连续吃掉对方棋子的最大数量 | $w_6 \approx 15$ |
| $p_7$ | 双三角阵型 | 己方棋子是否构成防御三角阵型 | $w_7 \approx 8$ |
| $p_8$ | 王骛安全 | 己方国王是否处于对方攻击范围 | $w_8 \approx -10$ |
| $p_9$–$p_{16}$ | 组合特征 | 高阶组合（如$p_1 \times p_5$等交叉特征） | 自适应 |

**权重更新的精确算法**（含学习率衰减和扰动策略）：

```
FUNCTION SamuelWeightUpdate(state, outcome, weights, learning_rate, t):
    predicted ← Σ_i weights[i] * feature_i(state)
    error ← outcome - predicted
    FOR i = 1 to n DO
        delta ← learning_rate(t) * error * feature_i(state)
        weights[i] ← weights[i] + delta
    END FOR
    IF SHOULD_PERTURB(t) THEN
        perturbation ← SAMPLE_PERTURBATION(weights, magnitude)
        RETURN weights + perturbation
    END IF
    RETURN weights
END FUNCTION

FUNCTION learning_rate(t):
    RETURN eta_0 / (1 + decay_rate * t)
    // 学习率从0.01逐渐衰减至约0.001
END FUNCTION

FUNCTION SAMPLE_PERTURBATION(weights, magnitude):
    FOR EACH i DO
        perturbation[i] ← RANDOM_GAUSSIAN(0, magnitude * |weights[i]|)
    END FOR
    RETURN perturbation
END FUNCTION
```

**课程学习（Curriculum Learning）策略**：塞缪尔在后期版本中引入了难度递增的训练方案。程序首先在"简化棋盘"（7×7而非8×8）上自我对弈，积累基础策略；然后在标准棋盘上与较弱版本的自身对弈；最后逐步提升难度至与最强版本对弈。这一策略在2020年代被重新发现并命名为"课程学习"（Bengio et al., 2009），是深度学习中的标准训练技术。

**Samuel跳棋程序的里程碑时刻**：1959年，塞缪尔在IBM 701上演示的跳棋程序战胜了康涅狄格州的优秀棋手Robert Nealey。这是AI历史上首次计算机程序在实际游戏中战胜人类选手。



---

## 1.7 计算资源限制：IBM 701、内存与速度

### 1.7.1 早期计算机的硬件环境

IBM 701（1952年发布）是IBM的第一台大规模科学用计算机，也是达特茅斯会议前后AI研究的主要硬件平台。

#### 1.7.1.1 IBM 701的技术规格

| 技术参数 | 规格 | 对比（现代参考） |
|----------|------|------------------|
| 时钟频率 | 约117 kHz | Intel Core i9-13900K: 6.0 GHz（约51,000倍） |
| 内存容量 | 4,096个36位字 = 约18 KB | 现代PC: 16–128 GB（约1百万倍） |
| 内存类型 | 威廉森管（Williams tube, CRT） | SRAM/DRAM |
| 内存访问时间 | 约30微秒 | L1缓存: 约0.5纳秒（约60倍更快） |
| 加法时间 | 约60微秒 | 现代CPU: 约0.5纳秒（约120倍更快） |
| 乘法时间 | 约216微秒 | 现代CPU: 约3纳秒（约72,000倍更快） |
| 磁带存储 | 最多100个磁鼓，每个50,000个字 | 现代SSD: 1–4 TB |
| 输入/输出 | 穿孔卡片（80列/秒） | USB 3.2/Thunderbolt |
| 电子管数量 | 约2,000个 | 现代CPU: 数十亿个晶体管 |
| 功耗 | 约500 kW | 现代PC: 约500–1,000 W（约500倍更高效） |
| 重量 | 约20,000磅（约9吨） | 现代PC: 约10–20 kg |
| 占地面积 | 约100平方米 | 现代PC: 约0.3平方米 |
| 价格 | 约$500/月（租赁价格） | 现代PC: 约$1,000–$5,000 |

（数据来源：Pugh et al., 1991; IBM Archives, 2024）

**IBM 701的系统架构详解**：IBM 701采用**单地址、单累加器**架构，其核心数据通路为：内存 → 内存缓冲寄存器（MBR） → 累加器（AC，36位） → 算术逻辑单元（ALU） → 累加器 → 内存。所有算术运算必须在累加器中进行，累加器宽度为36位——这一设计直接影响了早期AI算法的数据表示策略：LT和GPS中的所有数据结构（公式、状态、计数器）必须适配36位字长。

**Williams管存储器的物理原理**：IBM 701的主存使用**阴极射线管存储**（Williams tube），每根管子存储1,024个二进制位（32个36位字），整个系统需要8根管子才能达到4,096字的容量。Williams管的工作原理基于电子束扫描荧光屏表面：电子束击中某点时使该点带上正电荷（表示"1"），避开某点时该点保持负电荷（表示"0"）。通过精确控制电子束的偏转，读写头可以在屏幕上任意位置"写入"或"读取"一个比特。然而，Williams管的电荷会随时间泄漏——IBM 701需要每2毫秒**刷新**（refresh）一次内存内容，这消耗了约10%的CPU时间。Williams管的另一个致命弱点是**易失性**：断电后所有数据立即丢失，这意味着IBM 701无法实现"存储程序"——程序必须在每次启动时从穿孔卡片重新加载。

**算术逻辑单元（ALU）设计**：IBM 701的ALU使用**串行加法器**（serial adder）设计——36位加法逐位串行执行，每次处理1位，总耗时36个时钟周期。加法时间约60微秒（60,000纳秒 ÷ 36位 ≈ 1,667纳秒/位），乘法采用**移位-加法算法**（shift-and-add），约需216微秒（约3,900个时钟周期）。浮点运算需要额外的**规格化**（normalization）步骤，使浮点加法时间达到约90微秒。IBM 701的ALU还实现了**溢出检测**：当加法结果的符号位与预期不符时触发溢出中断——这一机制在LT的搜索算法中至关重要，防止因整数溢出导致错误的定理证明。

**输入输出子系统**：IBM 701的I/O系统由三个主要设备组成：（1）**IBM 711读卡机**——每秒读取150张穿孔卡片，每张卡片80列，每列12位；（2）**IBM 716打印机**——每分钟打印150行，每行最多144个字符；（3）**IBM 726磁带机**——使用7磁道磁带宽1/2英寸，磁带速度75英寸/秒，每条磁带可存储约250万位。对AI研究的影响是决定性的：读卡机的低带宽意味着程序的"人-机交互循环时间"为数分钟到数小时——在LT的搜索过程中，每次回溯都可能需要重新读取穿孔卡片，这严重限制了实验迭代速度。

**功耗与环境要求**：IBM 701整机功耗约500千瓦（kW），相当于一个小型工厂的用电量——其中90%用于真空管加热和空调制冷。系统需要恒温恒湿的专用机房，温度控制在20°C ± 2°C，相对湿度50% ± 10%。整机重量约20,000磅（9吨），需要特制混凝土地板和起重机安装。这些后勤约束意味着AI研究者在1950年代无法"随时随地"实验——每次运行程序都需要预约机时、准备穿孔卡片、等待数小时获取结果。

**IBM 701在AI研究中的实际使用情况**：据IBM档案馆记录，达特茅斯会议期间及前后几年，美国共有约30台IBM 701在运行。其中直接用于AI研究的包括：（1）IBM托马斯·沃森研究中心（Yorktown Heights, NY）——塞缪尔跳棋程序的开发平台；（2）卡内基理工学院（匹兹堡）——纽厄尔和西蒙运行LT和GPS的平台；（3）MIT——明斯基进行神经网络研究的平台。每台机器的月租费约为$12,000–$15,000（相当于2024年的约$150,000），这一高昂成本是AI研究早期"精英化"的根本原因——只有少数机构负担得起专门的计算机研究。

#### 1.7.1.3 IBM 701的指令集架构与编程模型

IBM 701的指令集架构（Instruction Set Architecture, ISA）采用单地址格式，每条指令36位：

操作码（6位）| 地址（15位）| 未使用（15位）

核心指令类型包括：

1. 数据传输指令：LOAD（加载）、STORE（存储）
2. 算术指令：ADD（加）、SUB（减）、MUL（乘）、DIV（除）
3. 逻辑指令：AND（与）、OR（或）、NOT（非）
4. 控制转移指令：JUMP（无条件转移）、JUMP+（正数转移）、JUMP-（负数转移）
5. 输入输出指令：READ（读卡片）、PUNCH（穿孔）

IBM 701的编程模型是**冯诺伊曼架构**的典型代表——程序和数据存储在同一个内存中，通过相同的总线访问。这一模型的优势在于灵活性：程序可以像数据一样被修改、生成和传输。其劣势在于**冯诺伊曼瓶颈**（von Neumann bottleneck）：CPU和内存之间的总线带宽限制了计算速度。

对AI算法的影响：IBM 701的LOAD/STORE架构要求所有中间结果必须显式地在内存和寄存器之间传输。这一约束直接影响了LT和GPS的内存管理策略——算法必须最小化内存访问次数。



#### 1.7.1.2 内存：威廉森管与磁鼓

IBM 701使用威廉森管（Williams tube, cathode-ray tube memory）作为主存储器。威廉森管由Frederic Williams、Tom Kilburn和Geoff Toothill于1946年至1947年间在曼彻斯特大学开发。

威廉森管的工作原理：威廉森管是一个阴极射线管（cathode-ray tube, CRT），其屏幕上的每个点可以携带一个二进制位。电子束通过偏转线圈扫描屏幕上的不同点，在特定点产生正电荷（代表1）或负电荷（代表0）。

威廉森管的局限性：（1）易失性（Volatility）：电荷会随时间泄漏——通常在大约2秒内消失；（2）容量限制：IBM 701的容量仅为4,096个36位字——约合18 KB；（3）可靠性：电子束容易受外界振动和电磁干扰的影响。

### 1.7.2 内存限制对算法设计的约束

IBM 701的18 KB主存储器对算法设计构成了空间复杂度的硬性约束。一个算法要能够在IBM 701上运行，其空间复杂度必须满足：

$$
\text{Space}(n) \leq 18 \times 10^3 \text{ bits} \approx 2 \times 10^3 \text{ words}
$$

对LT的影响：LT不能存储完整的搜索树——它必须使用深度优先搜索（DFS）而非广度优先搜索（BFS），因为DFS的空间复杂度为$O(d)$，而BFS的空间复杂度为$O(b^d)$。

对GPS的影响：GPS不能存储完整的可达性表——它必须使用启发式搜索来动态地探索状态空间。

### 1.7.3 计算资源限制与AI算法设计的"巧合"

早期的AI算法设计在多个方面受到了计算资源限制的塑造：（1）启发式搜索的必然性——由于穷举搜索不可行，启发式搜索成为核心方法；（2）符号操作的偏好——早期计算机擅长符号操作，导致符号主义AI范式的形成；（3）学习算法的受限——早期学习算法不能使用大规模数据集或深度神经网路；（4）人机交互的限制——穿孔卡片的人机交互循环"往返时间"为数小时；（5）推理深度的受限——推理深度通常限制在5到10步之间。

---

## 1.8 早期预言与实际差距：图灵的50年预测与AI的"冬天"

### 1.8.1 图灵的50年预测

图灵在其1950年论文中做出了一个著名的预测：我认为，在大约50年后，信息处理技术将发展到这样的程度：可以将存储容量为$10^9$的计算机编程，使其在模仿游戏中表现得如此出色，以至于普通的询问者在5分钟的审讯后，做出正确判断的概率不超过70%。

图灵预测的关键参数：时间范围50年（到2000年）；存储容量$10^9$个单元（约1 GB）；询问者误判概率至少30%；审讯时间5分钟。

图灵预测在定性层面是准确的：2014年的Eugene Goostman程序确实使33%的询问者产生了误判（Rothman, 2014）。然而，在定量层面存在重要差异——Eugene Goostman通过模拟特定人类类型（13岁乌克兰男孩）来产生误判，而非通过"通用智能"。

### 1.8.2 达特茅斯会议后的"AI黄金时代"（1956–1969）

达特茅斯会议后的头15年通常被称为AI研究的"黄金时代"或"乐观时代"。主要进展包括：1956年Logic Theorist诞生；1957年GPS诞生；1958年Lisp语言开发；1959年Samuel跳棋程序战胜地区冠军；1963年MIT建立AI实验室；1965年Robinson提出归结原理；1966年Weizenbaum开发ELIZA；1969年Minsky和Papert出版《感知机》。

DARPA在这一时期向AI研究提供了大量资金——每年的研究预算从几十万美元增长到几百万美元。

### 1.8.3 "乐观预言"与实际进展的差距

#### 1.8.3.1 西蒙的"10年内机器下国际象棋"预言

赫伯特·西蒙在1958年预言：假设目前国际象棋程序的发展速度保持不变，那么在大约10年内，一台计算机将成为国际象棋世界冠军。

实际情况：10年后（1968年）国际象棋程序的最佳表现仅达业余爱好者水平（约Elo 1,200–1,400）。40年后（1997年）IBM的Deep Blue才战胜世界冠军Garry Kasparov。差距：10年 vs. 40年。

#### 1.8.3.2 明斯基的"一个夏天就能构建通用问题求解器"预言

马文·明斯基在1960年代初期表示：在一个夏天内，一个研究生应该能够构建一个通用的、多领域的问题求解器。

实际情况：GPS需要为每个新领域提供大量领域特定的知识，其"通用性"是表面上的。

#### 1.8.3.3 McCarthy的"10年内证明重要数学定理"预言

约翰·麦卡锡在1958年表示：在未来10年内，计算机程序将能够证明数学中尚未被证明的重要定理。

实际情况：1968年计算机定理证明主要局限于初等逻辑和几何领域。直到2020年代深度学习与形式化证明的结合才开始展示计算机在IMO级别问题上的能力。

### 1.8.4 ELIZA的"误解"

ELIZA（Weizenbaum, 1966）是MIT的Joseph Weizenbaum开发的早期自然语言处理程序。ELIZA的技术非常简单——主要依赖关键词匹配和预编写脚本。Weizenbaum本人后来指出：ELIZA证明了机器模拟理解与人实际拥有理解之间的不对称性。

### 1.8.5 Minsky-Papert感知机论战与AI的"第一次冬天"

#### 1.8.5.1 《感知机》与神经网路的衰落

1969年，明斯基和派珀特出版了《感知机》一书（Minsky & Papert, 1969）。核心结论：单层感知机不能计算异或函数（XOR function）。

XOR问题的几何解释：单层感知机在二维输入空间中定义了一条直线（决策边界）$w_1 x_1 + w_2 x_2 + b = 0$。XOR函数的四个输入点无法被一条直线线性分开。

#### 1.8.5.2 AI的"第一次冬天"（1969–1980）

《感知机》的出版对AI研究产生了寒蝉效应。DARPA将AI研究预算削减至约每年$100,000。神经网路研究进入了"冬天"——几乎没有顶级会议论文或研究项目。

#### 1.8.5.3 Lighthill报告

Lighthill报告（Lighthill, 1973）对AI研究持高度批评态度，特别指出了组合爆炸、脆弱性、缺乏理论基础等问题。

### 1.8.6 早期预言的模式与教训

模式1：对"通用智能"的误解——通过特定任务的智能行为并不等同于通用智能。模式2：对"搜索空间"的低估——国际象棋搜索空间约为$10^{120}$个节点。模式3：对"知识表示"的高估——知识获取瓶颈被证明比预期更加困难。模式4：对"学习能力"的过度乐观——实际机器学习研究花了60年才达到深度学习水平。

教训：（1）长期预言的困难；（2）摩尔定律的双刃剑；（3）理论与工程的平衡。

---

## 1.9 时间线：从图灵机到达特茅斯会议（1936–1956）

| 年份 | 事件 | 关键人物 | 技术与理论意义 |
|------|------|----------|----------------|
| 1912 | 图灵出生 | Alan Turing | — |
| 1928 | 希尔伯特提出判定问题 | David Hilbert | 数学基础的"决策问题"形式化 |
| 1930 | 哥德尔不完备性定理 | Kurt Gödel | 数学系统的内在局限性 |
| 1935–1936 | 图灵机概念的形成 | Alan Turing | 可计算性的形式化模型 |
| 1936 | 《论可计算数……》发表 | Turing | 图灵机、Church-Turing论题、停机问题 |
| 1936 | Church-Turing论题提出 | Church & Turing | 可计算性模型的等价性 |
| 1937 | Turing《论可计算数……Ⅱ》 | Turing | λ演算与图灵机的等价性 |
| 1943 | McCulloch-Pitts神经元 | McCulloch & Pitts | 人工神经网路的数学基础 |
| 1945 | von Neumann架构报告 | von Neumann | 存储程序计算机设计 |
| 1948 | 《控制论》出版 | Norbert Wiener | 反馈、通信与控制的数学理论 |
| 1948 | Shannon《通信的数学理论》 | Claude Shannon | 信息论的创立 |
| 1950 | 《计算机器与智能》发表 | Turing | 模仿游戏、机器智能哲学论证 |
| 1950 | Shannon《为计算机编写国际象棋程序》 | Shannon | 计算机博弈与搜索算法的开端 |
| 1951 | UNIVAC I交付 | Eckert & Mauchly | 电子计算机的商业化 |
| 1952 | IBM 701发布 | IBM | 大规模科学用计算机 |
| 1955 | 达特茅斯会议提案提交 | McCarthy, Minsky, Rochester, Shannon | "人工智能"术语创造 |
| 1955 | Logic Theorist开发完成 | Newell, Shaw & Simon | 第一个AI程序 |
| 1956 | 达特茅斯会议召开 | 10位学者 | AI作为学科的诞生 |
| 1956 | GPS开发开始 | Newell & Simon | 通用问题求解器 |
| 1956 | Samuel跳棋程序发布 | Arthur Samuel | 机器学习概念的创造 |
| 1957 | GPS公开发表 | Newell & Simon | 手段-目的分析 |
| 1958 | Lisp语言开发 | John McCarthy | AI的核心编程语言 |
| 1959 | Samuel跳棋程序战胜冠军 | Arthur Samuel | 机器学习的首次实际演示 |
| 1959 | "Pandemonium"模型发表 | Oliver Selfridge | 模式识别的自下而上模型 |
| 1960 | McCarthy《递归函数……》 | McCarthy | Lisp的理论基础 |
| 1963 | MIT AI实验室建立 | Minsky & McCarthy | 第一个AI研究实验室 |
| 1965 | Robinson归结原理 | J. A. Robinson | 自动定理证明的核心算法 |
| 1966 | ELIZA发表 | Weizenbaum | 自然语言处理的早期尝试 |
| 1969 | 《感知机》出版 | Minsky & Papert | 单层感知机的局限性 |
| 1969 | Newell & Simon《物理符号系统假说》 | Newell & Simon | 符号主义AI的核心信条 |

#### 1.8.5.3 Lighthill报告详解与AI的第一次冬天

1973年，英国科学研究委员会（SRC）委托剑桥大学应用数学家James Lighthill撰写AI研究现状评估报告。Lighthill并非AI专家——SRC选择他正是因为其外部视角被认为有助于客观评估。

报告的核心批评可以归纳为三个方面：

批评1（组合爆炸）：AI搜索算法在最坏情况下的时间复杂度随问题规模呈指数增长T(n) = O(b^d)。Lighthill论证，对于任何现实世界的有趣问题，b^d都大得无法实际计算。

批评2（脆弱性）：早期AI系统的智能行为是高度领域特异的。一个在积木世界中表现良好的求解器，在面对稍有变化的问题时会完全失效——这一现象后来被称为微世界陷阱（Micro-world Trap）。

批评3（缺乏数学基础）：Lighthill批评AI研究缺乏严格的数学框架。他认为，物理符号系统假说是空泛的哲学概括，而非可证伪的科学假说。

报告的历史影响：SRC在Lighthill报告发布后大幅削减了对AI研究的资助。英国各大学的AI研究项目经历了显著收缩。然而，Lighthill报告的批评也促使AI研究者反思自身方法论的局限性，为后来连接主义（connectionism）的复兴和统计学习理论的发展提供了思想土壤。



---

## 1.10 总结与评价

### 1.10.1 图灵的科学贡献的综合评价

艾伦·图灵对人工智能的贡献是多维度的。

1. 数学基础：图灵机（1936）为"计算"提供了第一个严格的形式化定义。图灵完备性概念提供了计算能力的上限。Church-Turing论题（1936）建立了"可计算性"的模型独立性。

2. 哲学论证：图灵1950年论文的贡献在于操作化定义、行为主义立场、学习机器概念、常见误解的澄清。

3. 技术预见：图灵机模型——特别是通用图灵机（UTM）——直接预见了现代计算机的存储程序架构（von Neumann architecture）。UTM的核心思想是程序与数据的统一性、计算的通用性。

### 1.10.2 Church-Turing论题的现代意义

Church-Turing论题（CTT）在21世纪仍然具有核心的理论意义：

1. 量子计算与Church-Turing论题：量子计算机是否超出了Church-Turing论题的范围？目前的共识是否定的——量子计算的BQP复杂性类包含于PSPACE内。

2. 物理Church-Turing论题（Physical Church-Turing Thesis, PCTT）：PCTT声称任何物理上可实现的计算过程都可以被一台图灵机模拟。

3. 算法信息论与Church-Turing论题：所罗门诺夫的算法概率论题（Algorithmic Probability, 1964）和Kolmogorov复杂度（Kolmogorov Complexity, 1965）为Church-Turing论题提供了信息论的视角。

### 1.10.3 从图灵到达特茅斯：一条历史的"逻辑线"

从图灵1936年的论文到达特茅斯会议（1956），可以追溯一条清晰的逻辑线：1936年图灵机 → Church-Turing论题 → "可计算"的形式化 → 停机问题 → 计算的理论极限 → 通用图灵机 → 存储程序架构 → 现代计算机 → AI的硬件基础 → 达特茅斯会议 → AI作为学科的诞生。

### 1.10.4 图灵遗产的当代回响

图灵的工作在21世纪的回响是广泛而深刻的：

1. **计算理论**：图灵机仍然是计算复杂性理论的标准模型。复杂性类P、NP、PSPACE、EXPTIME等的定义都以图灵机为计算模型。
2. **密码学**：图灵在二战期间对德国Enigma密码的破解奠定了现代密码分析的基础。今天，可计算性理论中的归约技术被广泛用于证明密码方案的安全性。
3. **人工智能**：图灵1950年论文中的行为主义立场和模仿游戏构成了当代AI评估的哲学基础。2014年Eugene Goostman程序通过图灵测试，以及2022-2023年ChatGPT和GPT-4的发布，将图灵测试的当代相关性推到了前所未有的高度。
4. **认知科学**：图灵机为心智的计算理论提供了数学形式化。
5. **数学生物学**：图灵1952年发表的《形态发生的化学基础》开创了反应-扩散系统的研究。

图灵获得官方承认的过程经历了半个世纪：2009年英国首相布朗代表英国政府正式向图灵道歉；2013年英国女王伊丽莎白二世授予图灵皇家赦免。

### 1.10.6 本章关键公式汇总

图灵机形式化定义（7元组）：M = (Q, Gamma, b, Sigma, delta, q0, qf)
状态转移函数：delta: Q x Gamma -> Q x Gamma x {L, R, S}
Church-Turing论题：Eff(f) iff TM(f) iff Lambda(f) iff muR(f)
原始递归函数合成规则：f(0, x) = g(x); f(y+1, x) = h(y, f(y, x), x)
mu算子：f(x) = mu y [g(y, x) = 0]
lambda演算beta归约：(lambda x. M) N -> beta M[x := N]
停机问题：HP = {<M>#w | M在输入w上停机}
停机问题不可判定性：not exists H in TotalTM forall M, w [H(<M>#w) = Halt(M, w)]
Rice定理：设P是偏函数的任意非平凡性质，则判定给定图灵机所计算的偏函数是否具有性质P是不可判定的。
UTM存在性：exist U forall M forall w [U(<M>#w) downarrow = M(w)]
McCulloch-Pitts神经元：y = sigma(sum_{i=1}^n w_i x_i + b)
Samuel评估函数：f(s) = sum_{i=1}^n w_i * p_i(s)
Samuel权重更新：Delta w_i = eta (r - f(s)) p_i(s)
物理Church-Turing论题：forall P in PhysProcess forall epsilon > 0 exist U s.t. |U(P) - P| < epsilon

上述公式涵盖了从图灵机形式化定义到物理Church-Turing论题的完整理论谱系。



---

## 参考文献

（按作者姓氏字母排序）

[Ackermann 1928] Ackermann, W. (1928). Zum Hilbertschen Aufbau der reellen Zahlen. Mathematische Annalen, 99, 118–136.

[Church 1936] Church, A. (1936). An unsolvable problem of elementary number theory. American Journal of Mathematics, 58(2), 345–363.

[Church & Turing 1936–1937] Church, A. (1936–1937). A note on the Entscheidungsproblem. The Journal of Symbolic Logic, 1(1), 40–41.

[Crevier 1993] Crevier, D. (1993). AI: The Tumultuous History of the Search for Artificial Intelligence. New York: Basic Books.

[Davis 2000] Davis, M. (2000). The Universal Computer: The Road from Leibniz to Turing. New York: W. W. Norton.

[Gödel 1931] Gödel, K. (1931). Über formal unentscheidbare Sätze der Principia Mathematica und verwandter Systeme. Monatshefte für Mathematik und Physik, 38, 173–198.

[Haugeland 1985] Haugeland, J. (1985). Artificial Intelligence: The Very Idea. Cambridge, MA: MIT Press.

[Hilbert & Ackermann 1928] Hilbert, D., & Ackermann, W. (1928). Grundzüge der theoretischen Logik. Berlin: Springer.

[Hsu et al. 2002] Hsu, F., Anantharaman, T., Campbell, M., & Nowatzky, A. (2002). A grandmaster chess machine. Scientific American, 287(3), 44–51.

[Kleene 1936] Kleene, S. C. (1936). General recursive functions of natural numbers. Mathematische Annalen, 112, 727–742.

[Knuth 1976] Knuth, D. E. (1976). Big micron and little omega. ACM SIGACT News, 8(1), 14–21.

[Lighthill 1973] Lighthill, J. (1973). Artificial Intelligence: A General Survey. London: Science Research Council.

[McCarthy 1955] McCarthy, J., Minsky, M., Rochester, N., & Shannon, C. E. (1955). A proposal for the Dartmouth Summer Research Project on Artificial Intelligence. Unpublished proposal.

[McCarthy 1958] McCarthy, J. (1958). Programs with common sense. In Proceedings of the Teddington Conference on Mechanization of Thought Processes (Vol. 1, pp. 75–91). London: National Physical Laboratory.

[McCarthy 1960] McCarthy, J. (1960). Recursive functions of symbolic expressions and their computation by machine. Communications of the ACM, 3(4), 184–195.

[McCune 1994] McCune, W. (1994). OTTER 3.0 reference manual and guide. Technical Report ANL-94/6, Argonne National Laboratory.

[McCorduck 2004] McCorduck, P. (2004). Machines Who Think: A Personal Inquiry into the History and Prospects of Artificial Intelligence (25th Anniversary Edition). Natick, MA: A. K. Peters.

[McCulloch & Pitts 1943] McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. The Bulletin of Mathematical Biophysics, 5(4), 115–133.

[Minsky 1961] Minsky, M. (1961). Steps toward artificial intelligence. Proceedings of the IRE, 49(1), 8–30.

[Minsky & Papert 1969] Minsky, M., & Papert, S. (1969). Perceptrons: An Introduction to Computational Geometry. Cambridge, MA: MIT Press.

[Moore 1956] Moore, R. E. (1956). Interval Analysis. Englewood Cliffs, NJ: Prentice-Hall.

[Newell & Simon 1961] Newell, A., & Simon, H. A. (1961). GPS, a program that simulates human thought. In Computer Simulation of Human Thinking (pp. 109–124). Teddington, UK: National Physical Laboratory.

[Newell & Simon 1963] Newell, A., & Simon, H. A. (1963). GPS, a program that simulates human thought. In E. A. Feigenbaum & J. Feldman (Eds.), Computers and Thought (pp. 279–293). New York: McGraw-Hill.

[Newell & Simon 1976] Newell, A., & Simon, H. A. (1976). Computer science as empirical inquiry: Symbols and search. Communications of the ACM, 19(3), 113–126. (ACM Turing Award Lecture)

[Newell, Shaw & Simon 1956] Newell, A., Shaw, J. C., & Simon, H. A. (1956). The Logic Theorist: A complex information processing system. IEEE Transactions on Information Theory, 2(3), 61–79.

[Pugh et al. 1991] Pugh, E. W., Johnson, L. R., & Palmer, J. H. (1991). IBM's 360 and Early 370 Systems. Cambridge, MA: MIT Press.

[Rice 1951] Rice, H. G. (1951). Classes of recursively enumerable sets and their decision problems. Transactions of the American Mathematical Society, 74(2), 358–366.

[Robinson 1965] Robinson, J. A. (1965). A machine-oriented logic based on the resolution principle. Journal of the ACM, 12(1), 23–41.

[Rosenblatt 1958] Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65(6), 386–408.

[Rothman 2014] Rothman, M. (2014, June 8). Eugene Goostman passes the Turing test. But what does that mean? Business Insider.

[Rumelhart, Hinton & Williams 1986] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533–536.

[Samuel 1959] Samuel, A. L. (1959). Some studies in machine learning using the game of checkers. IBM Journal of Research and Development, 3(3), 210–229.

[Selfridge 1959] Selfridge, O. G. (1959). Pandemonium: A paradigm for learning. Symposium on the Mechanization of Thought Processes, National Physical Laboratory, Teddington, UK.

[Shannon 1948] Shannon, C. E. (1948). A mathematical theory of communication. The Bell System Technical Journal, 27, 379–423.

[Shannon 1950] Shannon, C. E. (1950). Programming a computer for playing chess. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 41(314), 256–275.

[Simon 1958] Simon, H. A. (1958). Heuristic problem solving: The next advance in operations research. Journal of the Operations Research Society of America, 6(1), 1–13.

[Solomonoff 1964] Solomonoff, R. J. (1964). A formal theory of inductive inference. Information and Control, 7(1), 1–22.

[Sussman 1975] Sussman, G. J. (1975). A Computer Model of Skill Acquisition. New York: American Elsevier.

[Thompson 1985] Thompson, K. (1985). Computer chess strength. In M. Clarke (Ed.), Advances in Computer Chess 4 (pp. 11–26). Oxford: Pergamon Press.

[Thrun 1996] Thrun, S. (1996). Explanation-based neural network learning for robot control. In D. S. Touretzky, M. C. Mozer, & M. E. Hasselmo (Eds.), Advances in Neural Information Processing Systems 8 (NIPS 1995). Cambridge, MA: MIT Press.

[Turing 1936] Turing, A. M. (1936). On computable numbers, with an application to the Entscheidungsproblem. Proceedings of the London Mathematical Society, Series 2, 42, 230–265.

[Turing 1937] Turing, A. M. (1937). On computable numbers, with an application to the Entscheidungsproblem. A correction. Proceedings of the London Mathematical Society, Series 2, 43, 544–546.

[Turing 1940] Turing, A. M. (1940). The Bombe: A secret weapon. In B. Copeland (Ed.), The Essential Turing (pp. 354–364). Oxford: Oxford University Press. (Declassified WWII report)

[Turing 1950] Turing, A. M. (1950). Computing machinery and intelligence. Mind, 59(236), 433–460.

[von Neumann 1951] von Neumann, J. (1951). The general and logical theory of automata. In L. A. Jeffress (Ed.), Cerebral Mechanisms in Behavior (pp. 1–41). New York: Wiley.

[Weizenbaum 1966] Weizenbaum, J. (1966). ELIZA—a computer program for the study of natural language communication between man and machine. Communications of the ACM, 9(1), 36–45.

[Weizenbaum 1976] Weizenbaum, J. (1976). Computer Power and Human Reason: From Judgment to Calculation. San Francisco: W. H. Freeman.

[Whitehead & Russell 1910] Whitehead, A. N., & Russell, B. (1910–1913). Principia Mathematica. Cambridge: Cambridge University Press. (3 volumes)

[Wiener 1948] Wiener, N. (1948). Cybernetics: Or Control and Communication in the Animal and the Machine. Cambridge, MA: MIT Press.
