# 第18章 计算机视觉的新纪元：ViT、扩散模型、视频理解

## 第1节 引言：从卷积到注意力——计算机视觉的范式转移

计算机视觉（Computer Vision, CV）自2012年AlexNet（Krizhevsky et al., 2012）引爆深度学习革命以来，卷积神经网络（Convolutional Neural Network, CNN）长期占据着该领域的主导地位。从VGG（Simonyan & Zisserman, 2014）到GoogLeNet（Szegedy et al., 2015），再到ResNet（He et al., 2016）及其后的各种变体——包括DenseNet（Huang et al., 2017）、EfficientNet（Tan & Le, 2019）和ConvNeXt（Liu et al., 2022）——CNN通过其固有的归纳偏置（inductive bias）——即局部连接性、权重共享和平移等变性——在图像分类、目标检测、语义分割等视觉任务上取得了令人瞩目的成就。ImageNet大规模视觉识别挑战赛（ILSVRC, Russakovsky et al., 2015）的top-1准确率从AlexNet的约62.5%提升到了EfficientNet-L2的90.2%以上（top-5准确率则从约84.7%提升至约98.7%），COCO目标检测的AP从约35%提升到了60%以上。这些进步部分归功于网络深度的增加（从AlexNet的8层到ResNet-152的152层再到 EfficientNet的数百层等效层），部分归功于更优的训练策略（如批量归一化、学习率调度、数据增强等）。

然而，CNN的核心局限也日益凸显：感受野受限于卷积核尺寸，深层网络的感受野虽然理论上可以覆盖全局，但实际训练中难以有效建模长距离依赖关系——Dilated卷积和空洞空间金字塔池（ASPP, Chen et al., 2017）虽然扩大了感受野，但本质上是手工设计的稀疏连接模式；池化操作虽然提供了空间不变性，却也丢失了精确的位置信息——这对于需要像素级精度的语义分割和关键点检测任务尤为不利；此外，CNN的层级结构是人工设计的（如ResNet中每个bottleneck块的通道数比例、每个stage的层数），模型本身无法自主决定如何分配计算资源到不同的图像区域——Attention U-Net（Oktay et al., 2018）等尝试将注意力引入CNN，但注意力机制仅在局部区域或低分辨率特征图上运行，无法实现全局级别的灵活计算资源分配。

与此同时，自然语言处理（NLP）领域正在经历一场由Transformer架构（Vaswani et al., 2017）驱动的革命。Transformer完全基于自注意力（Self-Attention）机制，摒弃了循环神经网络（RNN）和卷积结构，在序列建模任务中展现出了前所未有的能力。GPT系列（Radford et al., 2018; Brown et al., 2020）和BERT（Devlin et al., 2019）的成功表明，大规模预训练加Transformer架构的范式在语言领域具有极强的通用性和可扩展性——GPT-3（Brown et al., 2020）的1750亿参数展示了令人惊叹的少样本学习（few-shot learning）和推理能力，而BERT的预训练-微调范式在几乎所有NLP基准上都刷新了纪录。一个自然的问题随之浮现：Transformer架构能否被成功迁移到计算机视觉领域？如果可以，它将如何改变视觉任务的游戏规则？

2020年，Google Research团队发表的Vision Transformer（ViT）（Dosovitskiy et al., 2021）论文给出了一个初步但极具说服力的回答。ViT证明，当训练数据量足够大时，纯Transformer架构在图像分类任务上可以超越最优的CNN模型。这一发现彻底打破了"CNN是计算机视觉的必由之路"这一长达数十年的技术信条，开启了视觉领域的大规模预训练新时代。此后短短数年间，以Swin Transformer（Liu et al., 2021）、Masked Autoencoder（MAE）（He et al., 2022）为代表的视觉Transformer变体层出不穷，它们在效率、性能和通用性上不断刷新纪录。到2024年，视觉Transformer已经成为计算机视觉领域的主流架构，应用于几乎所有视觉子任务——从图像分类、目标检测、语义分割到图像生成、视频理解和三维视觉。

如果说Transformer的引入代表了视觉理解（Visual Understanding）领域的范式转移，那么扩散模型（Diffusion Model）的兴起则标志着视觉生成（Visual Generation）领域迎来了其最深刻的变革。自2015年Sohl-Dickstein等人提出扩散概率模型以来，这一生成范式在经历了数年的沉寂后，于2020年前后凭借DDPM（Ho et al., 2020）和DDIM（Song et al., 2021）等技术的突破性进展重新焕发生机。Stable Diffusion（Rombach et al., 2022）通过将扩散过程从像素空间转移到潜在空间，首次实现了高质量图像生成的普及化——在一个8GB显存的消费级GPU上即可生成512×512的高质量图像（后续SDXL等模型才将原生分辨率提升至1024×1024）。DALL-E 2（Ramesh et al., 2022）和Midjourney等系统则将文本到图像的生成质量推向了令人惊叹的高度——生成的图像在美学质量、语义一致性和细节丰富度上都达到了专业数字艺术品的水平。Sora（OpenAI, 2024）更进一步，展示了对动态视觉世界进行模拟的惊人潜力——能够生成长达60秒、多镜头切换、物理规律一致的高清视频。

本章将系统地梳理2020至2024年间计算机视觉领域最核心的技术突破。我们将从Vision Transformer的数学基础出发，逐步深入DeiT的数据效率优化、Swin Transformer的层次化架构设计，以及MAE的自监督预训练范式。随后，本章将对扩散模型进行完整而严格的数学推导，涵盖从正向扩散过程到反向去噪过程的每一个关键步骤——包括ELBO的完整推导、噪声预测网络的参数化、方差调度策略的设计原则——以及条件生成的技术细节，包括Classifier Guidance、Classifier-Free Guidance的理论推导及其相互关系。在此基础上，我们将剖析DALL-E、Stable Diffusion和Midjourney三大文本到图像系统的技术路线。接着，本章将探讨视频理解领域的最新进展，包括TimeSformer、VideoMAE以及对Sora技术原理的深入推测。最后，我们将详细介绍图像生成领域的核心评估指标——FID、IS、CLIP Score、Precision/Recall和sFID——的数学定义、理论意义和实际应用中的注意事项。

本章的目标是为读者提供一个关于计算机视觉新时代的全面而深入的技术导引，既涵盖关键的数学推导和算法细节，也提供对技术演变的宏观理解和未来方向的洞察。

---

## 第2节 Vision Transformer（ViT）：Transformer架构在视觉领域的首次突破

### 第2.1节 图像块（Patch）的数学表示

Vision Transformer的核心思想是将二维图像重新排列为一维的视觉词元（Visual Token）序列，然后将其输入标准的Transformer编码器进行处理。这一转化的关键在于图像块（Image Patch）的嵌入表示。

设输入图像为 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$，其中 $H$ 表示图像高度（像素行数），$W$ 表示图像宽度（像素列数），$C$ 表示通道数（对于标准RGB图像，$C=3$）。ViT将图像划分为大小为 $P \times P$ 的非重叠正方形块。由此，图像被分割为 $N = \frac{H}{P} \times \frac{W}{P}$ 个patch。当 $P=16$、$H=W=224$ 时，$N = 14 \times 14 = 196$ 个patch。每个patch可被重塑为一个长度为 $P^2 C$ 的向量：

$$
\mathbf{x}_p^i = \text{Reshape}(\mathbf{x}_{[i_h \cdot P : (i_h+1) \cdot P, i_w \cdot P : (i_w+1) \cdot P, :]}) \in \mathbb{R}^{P^2 C}, \quad i = 1, 2, \ldots, N
$$

其中 $i_h = \lfloor \frac{i-1}{W/P} \rfloor$ 和 $i_w = (i-1) \bmod (W/P)$ 分别表示第 $i$ 个patch在图像网格中的行和列索引。为严谨起见，patch的展平顺序采用行优先（row-major）排列。

为了将这些patch向量映射到Transformer所需的隐藏维度 $D$，ViT引入了一个可学习的线性投影矩阵 $\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$。从计算图的角度看，这等价于一个二维卷积层，其卷积核大小为 $P \times P$，步长（stride）为 $P$，输出通道数为 $D$，且不使用偏置项：

$$
\mathbf{z}_0 = \left[ \mathbf{x}_{\text{class}};\, \mathbf{x}_p^1 \mathbf{E};\, \mathbf{x}_p^2 \mathbf{E};\, \cdots;\, \mathbf{x}_p^N \mathbf{E} \right] + \mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}
$$

这里 $\mathbf{x}_{\text{class}} \in \mathbb{R}^{D}$ 为可学习的 [class] token（分类专用汇聚表示），$\mathbf{z}_0 \in \mathbb{R}^{(N+1) \times D}$ 是 Transformer 编码器的输入序列（$N$ 个 patch token 加上 1 个 [class] token），$\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ 为位置编码（详见第2节），分号表示序列维度上的拼接操作（concatenation along the sequence dimension）。

从线性代数的视角审视，patch嵌入操作 $\mathbf{x}_p^i \mapsto \mathbf{x}_p^i \mathbf{E}$ 是一个线性变换，它将原始的像素空间 $\mathbb{R}^{P^2 C}$ 映射到一个语义更为丰富的隐藏空间 $\mathbb{R}^D$。值得注意的是，当patch尺寸 $P=16$、隐藏维度 $D=768$ 时，对于典型的 $224 \times 224 \times 3$ 输入图像，每个patch展平后为 $16 \times 16 \times 3 = 768$ 维，恰好与隐藏维度 $D$ 相同，此时投影矩阵 $\mathbf{E}$ 退化为一个方阵，在实现上可以直接使用单层线性层完成嵌入，这种维度匹配在计算效率上具有一定的优势。

**多尺度Patch尺寸的理论考量：** 从信息论的角度，patch尺寸 $P$ 的选择反映了模型对局部细节和全局结构之间权衡的预设。较小的patch（如 $P=8$）保留了更多的空间细节，序列长度 $N$ 更大，自注意力计算量呈二次增长；较大的patch（如 $P=32$）丢失细节但序列更短、计算更快。ViT论文的消融实验表明，$P=16$ 在计算效率和性能之间取得了最佳平衡。值得指出的是，动态patch尺寸（Dynamic Patch Size）——即在不同层使用不同大小的patch——是后续研究（如Focal Transformer, Yang et al., 2021）探索的方向。

### 第2.2节 多头自注意力的数学机制

在深入讨论ViT的可行性之前，有必要对Transformer编码器中的核心计算单元——多头自注意力（Multi-Head Self-Attention, MHSA）——进行形式化描述。给定输入序列 $\mathbf{Z} \in \mathbb{R}^{N \times D}$，MHSA首先通过 $h$ 个独立的注意力头（attention head）并行计算注意力，然后将结果拼接后投影回 $D$ 维。

对于第 $k$ 个注意力头（$k = 1, \ldots, h$），定义查询矩阵 $\mathbf{Q}_k = \mathbf{Z} \mathbf{W}_Q^k$，键矩阵 $\mathbf{K}_k = \mathbf{Z} \mathbf{W}_K^k$，值矩阵 $\mathbf{V}_k = \mathbf{Z} \mathbf{W}_V^k$，其中 $\mathbf{W}_Q^k, \mathbf{W}_K^k, \mathbf{W}_V^k \in \mathbb{R}^{D \times d_k}$ 为可学习参数，$d_k = D/h$ 为每个头的维度。第 $k$ 个头的输出为：

$$
\text{Head}_k(\mathbf{Z}) = \text{Softmax}\left(\frac{\mathbf{Q}_k \mathbf{K}_k^\top}{\sqrt{d_k}}\right) \mathbf{V}_k
$$

缩放因子 $\sqrt{d_k}$ 的作用在于控制点积（dot-product）的幅度，避免在 $d_k$ 较大时Softmax函数进入梯度接近于零的饱和区。将所有头的输出在通道维度上拼接后，通过一个输出投影矩阵 $\mathbf{W}_O \in \mathbb{R}^{D \times D}$ 融合：

$$
\text{MHSA}(\mathbf{Z}) = \text{Concat}(\text{Head}_1, \ldots, \text{Head}_h) \mathbf{W}_O
$$

多头注意力的优势在于允许模型在不同的表示子空间中并行地关注不同位置的信息。从群论（Group Theory）的角度，每个注意力头可以看作是在一个 $d_k$ 维子空间中对关系模式的独立提取，多头的拼接则相当于在 $D$ 维空间中构建了一个更丰富的"关系基"（relational basis）。

完整的前馈网络（FFN）子层是两个线性变换之间的非线性映射，中间由GELU（Gaussian Error Linear Unit, Hendrycks & Gimpel, 2016）激活函数分隔：

$$
\text{FFN}(\mathbf{z}) = \text{GELU}(\mathbf{z} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2
$$

其中 $\mathbf{W}_1 \in \mathbb{R}^{D \times 4D}$，$\mathbf{W}_2 \in \mathbb{R}^{4D \times D}$ 为权重矩阵，$\mathbf{b}_1 \in \mathbb{R}^{4D}$，$\mathbf{b}_2 \in \mathbb{R}^{D}$ 为偏置向量。中间层维度扩展为 $4D$ 的设计在经验上被证明是最优的——过小的扩展比（如2）限制了模型的表达能力，过大的扩展比（如8以上）则引入了不必要的参数和计算开销。

### 第2.3节 位置编码的设计与理论意义

Transformer架构本身不具备处理序列顺序的能力——自注意力机制对输入token的位置是不变的（permutation-invariant）。在NLP中，这通过位置编码（Positional Encoding, PE）来解决。ViT沿用了这一设计，并采用了可学习的位置编码方案。

ViT的位置编码矩阵 $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ 是一个可学习的参数（与含 `[class]` token 的序列长度一致），与模型的其他参数一起通过反向传播进行优化。与NLP中广泛使用的正弦-余弦位置编码（Sinusoidal Positional Encoding）不同，ViT采用可学习位置编码的理由在于：视觉数据中"位置"的概念与语言中的"顺序"存在本质差异。图像中的空间关系是二维的，不存在天然的一维线性顺序；可学习的位置编码允许模型自主发现最优的空间编码方案。

形式化地，Transformer编码器由 $L$ 层相同的层堆叠而成，每一层包含一个多头自注意力（Multi-Head Self-Attention, MHSA）子层和一个前馈网络（Feed-Forward Network, MLP）子层，每个子层之后都有残差连接（Residual Connection）和层归一化（Layer Normalization, LN）：

$$
\begin{aligned}
\mathbf{z}'_\ell &= \text{MHSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1} \\
\mathbf{z}_\ell &= \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell, \quad \ell = 1, \ldots, L
\end{aligned}
$$

其中 $\mathbf{z}_0$ 为patch嵌入序列。最终，分类头（Classification Head）仅对[class] token对应的输出向量进行预测：

$$
\hat{y} = \text{Softmax}(\mathbf{W}_{\text{head}} \cdot \mathbf{z}_L^{[cls]})
$$

这里 $\mathbf{z}_L^{[cls]} \in \mathbb{R}^D$ 是第 $L$ 层输出序列中第一个token（即特殊的[class] token）对应的隐藏状态，$\mathbf{W}_{\text{head}} \in \mathbb{R}^{D \times K}$ 为分类头的权重矩阵，$K$ 为类别数量。

### 第2.4节 纯Transformer在CV上的可行性证明

ViT论文的核心贡献在于提供了一个系统性的大规模实验验证，证明纯Transformer架构在计算机视觉任务上可以与CNN相媲美甚至超越之。Dosovitskiy等人设计的实验架构清晰地揭示了数据规模与模型容量之间的相互作用。

**小规模数据集上的表现：** 当在ImageNet-1K（约120万张训练图像，1000个类别）上从头训练时，ViT的表现不及同等计算预算下的CNN基线（如ResNet）。具体而言，ViT-B/16（Base模型，12层Transformer编码器，隐藏维度768，12个注意力头，patch大小16×16，约86M参数）在ImageNet-1K上仅达到约78-79%的top-1准确率，而ResNet-50（约25M参数）可以达到约80%以上的top-1准确率，ResNet-152（约60M参数）更是达到约81.5%。这一差距的根本原因在于：CNN的归纳偏置（局部连接性和平移等变性）在小数据集上起到了正则化作用，防止了模型过拟合——这些归纳偏置大大缩减了假设空间（hypothesis space）的规模，使得模型在有限数据下就能学到有效的表征。相比之下，ViT缺乏这些归纳偏置，拥有更大的假设空间，需要更多的数据才能将搜索空间约束到正确的区域。

从统计学习理论的视角审视，CNN和ViT的差异可以形式化为假设空间复杂度（Hypothesis Space Complexity）的差异。设 $\mathcal{H}_{\text{CNN}}$ 和 $\mathcal{H}_{\text{ViT}}$ 分别为CNN和ViT的假设空间。由于CNN的卷积核共享和局部连接约束，$\mathcal{H}_{\text{CNN}}$ 的VC维（Vapnik-Chervonenkis Dimension）或Rademacher复杂度远低于 $\mathcal{H}_{\text{ViT}}$。根据统计学习理论，泛化误差界（Generalization Error Bound）与假设空间复杂度正相关：

$$
\mathbb{E}[L_{\text{gen}}] \leq \hat{L}_{\text{train}} + \mathcal{O}\left(\sqrt{\frac{\mathcal{C}(\mathcal{H})}{n}}\right)
$$

其中 $\mathcal{C}(\mathcal{H})$ 为假设空间 $\mathcal{H}$ 的复杂度度量，$n$ 为训练样本数。对于小样本 regime（$n$ 较小），$\mathcal{C}(\mathcal{H}_{\text{CNN}}) \ll \mathcal{C}(\mathcal{H}_{\text{ViT}})$，因此CNN的泛化误差界更紧；而对于大样本 regime（$n$ 足够大），两个界都趋于收敛到真实风险，此时表达能力更强的 $\mathcal{H}_{\text{ViT}}$ 能达到更低的训练误差和泛化误差。

**大规模数据集上的突破：** 当训练数据扩展到JFT-300M（包含约3亿张图像的大规模Google私有数据集，涵盖约18K类别）时，ViT的表现发生了质变。ViT-L/16（Large模型，24层Transformer编码器，隐藏维度1024，16个注意力头，约304M参数）在 JFT 预训练后于 ImageNet-1K 上 fine-tune 约达到 87.76% 的 top-1 准确率，优于此前最优的 CNN 模型（包括使用大量数据增强和自训练的 Noisy Student 等）。ViT-H/14（Huge 模型，32 层，隐藏维度 1280，16 个注意力头，约 632M 参数）更是将准确率推高至约 88.55%，将 ImageNet top-1 准确率的纪录向前推进了一大步。

Dosovitskiy等人在论文中进行了系统的消融实验（ablation study），揭示了几个关键的设计选择：

1. **Patch尺寸的影响：** 在相同计算预算下，较小的patch（$P=8$ 或 $P=16$）始终优于较大的patch（$P=32$ 或 $P=64$）。这是因为较小的patch保留了更多的空间细节，序列更长使得每个token的表征更局部化、更易于学习。

2. **网络深度的影响：** 对于给定的patch尺寸，更深的网络（更多层）始终表现更好，但收益递减。从12层到24层有明显的性能提升，但从24层到32层的提升幅度较小。

3. **架构变体的影响：** 论文比较了仅使用Transformer编码器（标准ViT）、加入卷积 Stem（将patch嵌入层替换为小型CNN）、以及使用更深层的混合架构（Hybrid Architecture，如ResNet-50的特征图作为patch输入）。结果表明，在相同数据量下，纯Transformer架构（标准ViT）优于混合架构——因为混合架构引入了CNN归纳偏置，限制了大模型的表达能力上限。

4. **预训练数据规模的影响：** ViT的性能优势高度依赖于预训练数据规模。在ImageNet-1K上从头训练时，ViT-B/16比ResNet-50差约1-2%；但在JFT-300M上预训练后，ViT-L/16比在相同数据上预训练的最优CNN（BiT-L, Big Transfer, Kolesnikov et al., 2020）高出约2个百分点。

这一实验结果具有深刻的理论含义。从统计学习理论的角度来看，CNN的归纳偏置可以被视为一种强先验（strong prior），它在数据量有限时能够引导模型快速收敛，但在数据量充足时会成为限制模型表达能力的"天花板"。ViT缺乏这些归纳偏置，因此具有更大的假设空间，当数据量足够大时，模型可以从数据中自主发现比人工设计的归纳偏置更为有效的模式。换言之，ViT的实验结果验证了以下结论：

> **视觉Transformer可行性定理：** 存在一个Transformer架构 $\mathcal{T}$ 和一个数据集规模阈值 $S^*$，使得对于足够大的视觉识别任务（类别数 $K$ 足够大，图像多样性足够丰富），当训练数据量 $S \geq S^*$ 时，$\mathcal{T}$ 在ImageNet等标准基准上的top-1准确率优于任何具有固有归纳偏置的CNN架构。

值得注意的是，ViT的性能优势在大规模数据预训练后迁移到下游任务时表现得更加明显。在VTAB（Visual Task Adaptation Benchmark, Zhu et al., 2019）包含的19个多样化视觉任务上的评估中，ViT-L/16在JFT-300M上预训练后在所有任务上的平均准确率超过了所有CNN基线，且平均领先幅度超过4个百分点。这表明ViT学习到的表征具有更好的迁移泛化能力（transfer generalization）——其"视觉词汇"更加通用，可以适应不同类型的视觉任务。

然而，ViT也面临着两个关键挑战。第一，全局自注意力的计算复杂度为 $O(N^2 D)$，其中 $N = HW/P^2$ 为序列长度。对于高分辨率图像（如 $1024 \times 1024$ 输入，$P=16$ 时 $N = 4096$），$N^2$ 可达约1670万，导致计算量和内存消耗急剧增加。第二，ViT产生的特征图分辨率随着网络加深而不断降低（最终为一个单一的[class]向量），缺乏多尺度特征表示，这严重限制了其在目标检测、语义分割等需要密集预测（dense prediction）任务上的应用。这两个问题正是后续Swin Transformer和层次化视觉Transformer所要解决的核心。

### 第2.5节 自注意力可视化与可解释性分析

ViT论文包含一项深刻而常被忽视的分析：通过对训练好的ViT模型进行自注意力模式的可视化，研究者揭示了Transformer如何在无需显式卷积结构的情况下，通过注意力机制自主学习到类卷积的空间归纳偏置。

具体而言，Dosovitskiy等人在不同层、不同注意力头上提取了注意力图（Attention Map），即每个token对其他所有token的注意力权重矩阵 $\mathbf{A} \in [0,1]^{N \times N}$（在Softmax之后）。他们发现了一个引人注目的现象：在网络的浅层（第1-4层），许多注意力头表现出强烈的空间局部性——它们主要关注查询token附近的patch，且注意力权重随空间距离的增加而单调递减。这种行为与CNN中卷积核的局部感受野高度相似，尽管Transformer从未被明确约束去学习这种局部模式。随着网络深度的增加，注意力模式逐渐变得更加全局化和语义化——高层注意力头往往关注来自图像中不同区域但属于同一物体的patch，表现出物体级别的整体性理解。

从优化理论的角度，这种从局部到全局的注意力演化模式可以解释为：浅层通过局部注意力捕获低阶视觉信息（边缘、纹理、颜色），这些信息的编码成本最低、梯度信号最丰富；随着训练深入，高层利用这些低级表征来建立长距离语义关联（物体部件之间的空间关系、场景布局等）。这种"由简入繁"的自组织学习过程，与人类视觉系统的处理层级（从V1简单细胞到IT物体的层级处理，Hubel & Wiesel, 1962）具有令人瞩目的功能相似性。

进一步的分析表明，ViT的注意力模式与CNN的卷积核特性存在本质差异。CNN的卷积核是静态的——一旦训练完成，每个卷积核的权重就固定不变，对所有输入图像使用相同的滤波模式。而ViT的注意力权重是动态的（content-based）——对于不同的输入图像，注意力权重会自适应地调整。这意味着ViT本质上是一种动态滤波器（Dynamic Filter）网络：网络根据当前输入的内容决定"看哪里"，而不是像CNN那样对所有输入采用固定的空间滤波模式。这种动态性是Transformer相比CNN的一个根本性优势，它使模型能够根据图像的语义内容灵活地分配计算资源——对于包含多个分散物体的复杂场景，不同的注意力头可以分别聚焦于不同的物体区域。

### 第2.6节 ViT与CNN的系统性对比分析

为了更清晰地理解ViT相对于传统CNN的优势与劣势，表18-1从多个维度对两者进行了系统性比较。这一比较涵盖了架构设计、计算特性、数据效率、泛化能力和下游任务适用性等核心维度。

在架构层面，ViT的核心是全局自注意力，任何两个patch之间都有直接的交互路径，路径长度为1（即一步注意力）；而CNN的感受野随深度指数增长，远距离patch之间的交互需要经过多层的卷积传播，路径长度与网络深度成正比。这意味着ViT在建模长距离依赖关系方面具有架构级的优势，但代价是自注意力的二次方复杂度。在数据效率方面，CNN的归纳偏置使其在小数据集上具有天然优势——卷积核的权重共享意味着参数数量远小于同等容量的全连接网络，局部连接性则天然编码了图像的统计结构假设。ViT在数据量超过约1400万张图像后开始超越CNN，这一阈值（cross-over point）与模型的容量正相关——更大的模型需要更多的数据才能发挥其优势。

在下游任务适用性方面，ViT的原始形态（固定分辨率的patch嵌入 + 全局注意力 + 单一[class]输出）天然适用于图像分类，但直接应用于目标检测和语义分割等密集预测任务需要大量的架构修改（如添加多尺度特征金字塔、可变形注意力等）。相比之下，CNN的层次化特征图结构（从低分辨率高语义到高分辨率低语义的金字塔）与密集预测任务的需求天然匹配，这也是为什么在Swin Transformer出现之前，CNN在密集预测任务上一直占据主导地位的根本原因。在计算效率方面，对于相同输入分辨率和模型容量，ViT的训练和推理速度通常慢于CNN——这主要是因为自注意力的二次方复杂度和缺乏高度优化的硬件感知算子（如FlashAttention, Dao et al., 2022）。然而，随着专用硬件（TPU）的发展和对注意力算子优化的持续投入（如xFormers, Meta, 2022），这一效率差距正在逐步缩小。

**表18-1 ViT与CNN的系统性对比**

| 维度 | Vision Transformer (ViT) | 卷积神经网络 (CNN) |
|------|------------------------|---------------------|
| **核心操作** | 全局/局部自注意力（动态、内容自适应） | 卷积（静态、空间共享滤波器） |
| **感受野** | 理论上全局（单层可达全图），实际受序列长度限制 | 随深度指数增长，需要深层才能覆盖全局 |
| **归纳偏置** | 极弱（仅通过位置编码引入空间信息） | 强（局部连接性、权重共享、平移等变性） |
| **参数效率** | 参数量较大，但单参数携带信息更丰富 | 参数效率高（权重共享），但单参数表达能力受限 |
| **数据需求** | 大规模数据（>10M图像）才能发挥优势 | 中小规模数据即可训练良好 |
| **计算复杂度** | $O(N^2 D)$（全局注意力），$O(N M^2 D)$（窗口注意力） | $O(N D^2)$（深度可分离卷积）或 $O(HWK^2 D)$（标准卷积） |
| **多尺度建模** | 原始ViT不支持（单尺度特征图） | 天然支持（通过下采样和不同通道数的stage） |
| **位置信息处理** | 显式位置编码（可学习或正弦-余弦） | 隐式通过卷积核的空间排列 |
| **动态性** | 注意力权重动态生成（输入自适应） | 滤波器权重固定（训练后不变） |
| **典型下游任务** | 图像分类（原生支持），检测/分割需修改架构 | 分类、检测、分割、关键点等（原生支持） |
| **训练收敛速度** | 较慢（大量预训练数据时收敛加速） | 较快（归纳偏置加速收敛） |
| **可解释性** | 注意力图提供空间注意力分布可视化 | 特征图可视化、滤波器可视化 |
| **硬件优化成熟度** | 中等（FlashAttention、xFormers持续优化） | 高度成熟（cuDNN、TensorRT深度优化） |

这一对比分析揭示了ViT和CNN各自的优势区间。ViT的优势在于其架构级的全局建模能力和动态适应性，使其在大规模数据 regime 下能够学习到比CNN更强大的表征；CNN的优势在于其强归纳偏置带来的数据效率和计算效率，使其在数据有限或计算资源受限的场景中更具实用性。后续的Swin Transformer和ConvNeXt等工作试图融合两者的优点，在不同的约束条件下取得最佳平衡。

---

## 第3节 DeiT：数据高效的Vision Transformer

### 第3.1节 训练策略与数据增强

Data-efficient Image Transformer（DeiT）（Touvron et al., 2020）旨在解决ViT的两个核心痛点：对大规模标注数据的极度依赖和较慢的收敛速度。DeiT的核心洞察是：通过精心设计的训练策略，可以在仅使用ImageNet数据（不使用JFT-300M等大规模数据集）上将ViT的性能提升到与CNN相当的水平。

DeiT采用的训练策略包括以下几个关键组件：

**1. 增强型数据增强（Augmentation Regime）：** DeiT采用了类似于CNN训练中使用的强数据增强策略，包括随机裁剪（Random Resized Crop）、水平翻转（Random Horizontal Flip）、随机擦除（Random Erasing, RE）、Cutout（DeVries & Taylor, 2017）、Mixup（Zhang et al., 2018）和CutMix（Yun et al., 2019）。这些增强策略对于防止ViT过拟合至关重要，因为它们在一定程度上弥补了模型缺乏CNN归纳偏置的不足。特别是Mixup和CutMix，它们通过对训练样本进行线性插值和局部替换，显著扩展了训练分布的覆盖范围。

**2. 正则化技术：** DeiT使用了标签平滑（Label Smoothing, LS）（Szegedy et al., 2016）和随机深度（Stochastic Depth, SD）（Huang et al., 2016）。标签平滑将硬标签 $y$ 替换为软标签 $\tilde{y} = (1-\epsilon) y + \epsilon/K$，其中 $\epsilon$ 为平滑参数（通常取0.1），$K$ 为类别数。这防止模型对预测过于自信，提高了泛化能力。随机深度则以概率 $p_\ell = \frac{\ell}{L}(1 - p_{\text{max}})$ 随机丢弃第 $\ell$ 层（即跳过该层的残差连接），其中 $p_{\text{max}}$ 为最大丢弃概率（通常取0.1或0.2）。

**3. 余弦退火学习率调度：** DeiT采用余弦退火（Cosine Annealing）学习率衰减策略（Loshchilov & Hutter, 2017），相对于多步衰减（Multi-step Decay）能够实现更平滑的优化轨迹和更好的最终性能。

**4. 指数移动平均（EMA）：** 在训练过程中对模型参数维护指数移动平均，记为 $\bar{\theta}_t = \beta \bar{\theta}_{t-1} + (1-\beta)\theta_t$，其中 $\theta_t$ 为第 $t$ 步的原始参数，$\beta$ 通常取0.9996。EMA模型通常比最终检查点具有更好的泛化性能。

### 第3.2节 知识蒸馏

DeiT最具创新性的贡献在于引入了基于Transformer的蒸馏机制（Distillation with a Transformer）。传统知识蒸馏（Knowledge Distillation, KD）（Hinton et al., 2015）使用一个大型教师模型（或模型集成）的软标签（soft labels）来指导学生模型的训练。然而，DeiT的关键发现是：使用CNN作为教师模型来蒸馏ViT学生模型时，仅仅使用硬标签（hard labels）或标准的KL散度损失不足以获得最佳效果。

DeiT提出了两种蒸馏损失的组合。设 $y$ 为真实标签的独热编码（one-hot encoding），$\mathbf{p}_S = \text{Softmax}(\mathbf{z}_S / \tau)$ 为学生模型的 softened 输出分布，$\mathbf{p}_T = \text{Softmax}(\mathbf{z}_T / \tau)$ 为教师模型的 softened 输出分布，其中 $\tau$ 为温度参数。则总损失函数为：

$$
\mathcal{L}_{\text{total}} = (1-\lambda) \cdot \mathcal{L}_{\text{CE}}(y, \mathbf{p}_S) + \lambda \cdot \tau^2 \cdot \mathcal{L}_{\text{KL}}(\mathbf{p}_T, \mathbf{p}_S)
$$

其中 $\mathcal{L}_{\text{CE}}$ 为交叉熵损失，$\mathcal{L}_{\text{KL}}$ 为KL散度损失，$\tau^2$ 因子用于补偿温度缩放带来的梯度缩小效应（Hinton et al., 2015），$\lambda$ 为平衡权重。

DeiT的一个巧妙设计是引入了一个额外的蒸馏token（distillation token）。这个token与[class] token并行存在，同样通过自注意力机制与所有patch token交互，但其唯一目的是预测教师模型的输出分布。具体地，在Transformer编码器的输入序列中，DeiT使用 $[\text{class}; \text{distill}; \mathbf{z}_p^1; \mathbf{z}_p^2; \ldots; \mathbf{z}_p^N]$ 作为输入序列。在训练时，分类头对[class] token的输出计算标准交叉熵损失，而对蒸馏token的输出计算蒸馏损失。在推理时，可以选择使用[class] token或蒸馏token的预测，也可以将两者取平均：

$$
\hat{y} = \frac{1}{2} \left( \text{Softmax}(\mathbf{z}_L^{[cls]}\mathbf{W}_{\text{head}}) + \text{Softmax}(\mathbf{z}_L^{[distill]}\mathbf{W}_{\text{head}}) \right)
$$

实验结果表明，这种设计使DeiT-B在ImageNet上达到了83.4%的top-1准确率，与使用大量增强和训练技巧的CNN模型相当；相对ResNet-50（约25M参数），DeiT-B（约86M参数）的参数量约为其3倍，而更轻量的DeiT-S则与ResNet-50参数量同量级。更重要的是，DeiT的收敛速度显著快于ViT——通常只需原来1/3到1/2的训练步数即可达到相同的性能。

---

## 第4节 Swin Transformer：层次化视觉表征

### 第4.1节 层次化特征图构建

Swin Transformer（Liu et al., 2021）的名字来源于"Shifted Window"（移位窗口），其最核心的创新在于构建了层次化的特征图（Hierarchical Feature Maps），同时通过移位窗口注意力机制实现了线性计算复杂度。这两项设计使Swin Transformer成为首个在密集预测任务（如目标检测和语义分割）上全面超越CNN的Transformer架构。

与ViT在整个网络中保持恒定分辨率不同，Swin Transformer模仿了CNN的层次化结构，通过在连续层之间逐步合并patch（patch merging）来产生层次化的特征表示。具体而言，Swin Transformer分为四个阶段（Stage），每个阶段的特征图分辨率不同：

- **Stage 1：** 输入图像被划分为 $4 \times 4$ 的patch，每个patch线性嵌入为维度 $C$ 的向量。经过若干层Transformer块后，输出特征图的空间分辨率为 $H/4 \times W/4$，通道数为 $C$。
- **Stage 2：** 通过Patch Merging层将相邻的 $2 \times 2$ patch区域合并。每个 $2 \times 2$ 区域的特征被拼接后通过一个线性层投影到 $2C$ 维。分辨率降为 $H/8 \times W/8$，通道数翻倍为 $2C$。
- **Stage 3：** 再次通过Patch Merging，分辨率降为 $H/16 \times W/16$，通道数为 $4C$。
- **Stage 4：** 最后一次Patch Merging，分辨率降为 $H/32 \times W/32$，通道数为 $8C$。

数学上，第 $i$ 个Patch Merging层的操作可表示为：

$$
\mathbf{z}^{(i)} = \text{Linear}(\text{Concat}(\mathbf{z}_{\text{patch}_1}^{(i-1)}, \mathbf{z}_{\text{patch}_2}^{(i-1)}, \mathbf{z}_{\text{patch}_3}^{(i-1)}, \mathbf{z}_{\text{patch}_4}^{(i-1)}))
$$

其中 $\mathbf{z}_{\text{patch}_j}^{(i-1)}$ 表示第 $i-1$ 阶段中一个 $2 \times 2$ 邻域内的第 $j$ 个 patch 的特征向量，Concat 操作沿通道维度进行拼接。以 Stage 1→2 为例，四个通道数为 $C$ 的 patch 拼接后为 $4C$ 维，Linear 层再将其投影到 $2C$ 维（一般地，若上一阶段通道数为 $C'$，则拼接后为 $4C'$，投影到 $2C'$）。

这种层次化结构使得Swin Transformer天然适用于目标检测和语义分割等下游任务——Stage 2至Stage 4的输出特征图分别对应不同尺度的语义信息，可以像FPN（Feature Pyramid Network, Lin et al., 2017）那样被多尺度特征金字塔结构所利用。

### 第4.2节 移位窗口注意力机制

Swin Transformer最核心的技术贡献是移位窗口注意力（Shifted Window Attention）。在标准的Vision Transformer中，全局自注意力的计算复杂度与序列长度的平方成正比，即 $O(N^2 D)$。对于高分辨率图像，当 $N$ 较大时，这一复杂度变得不可接受。Swin Transformer通过将自注意力限制在非重叠的局部窗口内，将复杂度降低到了线性级别。

设窗口大小为 $M \times M$，则每个窗口内的patch数量为 $M^2$。在连续的Transformer块中，Swin Transformer交替使用两种配置：

**规则窗口分区（Regular Partitioning）：** 将特征图均匀划分为大小为 $M \times M$ 的非重叠窗口。在第 $l$ 层（$l$ 为奇数），特征图 $\mathbf{z} \in \mathbb{R}^{H' \times W' \times C'}$ 被划分为 $\frac{H'}{M} \times \frac{W'}{M}$ 个窗口，每个窗口包含 $M^2$ 个patch。自注意力仅在每个窗口内部进行计算：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}} + \mathbf{B}\right)V
$$

其中 $Q, K, V \in \mathbb{R}^{M^2 \times d}$ 为对应窗口内所有patch的查询、键和值矩阵，$d$ 为每个注意力头的维度，$\mathbf{B} \in \mathbb{R}^{M^2 \times M^2}$ 为相对位置偏置（Relative Position Bias）。

相对位置偏置 $\mathbf{B}$ 的设计是Swin Transformer的一个重要创新。与绝对位置编码不同，$\mathbf{B}$ 直接编码窗口内任意两个patch之间的相对空间偏移。设 $\hat{\mathbf{B}}$ 为可学习的相对位置偏置表，其维度为 $(2M-1) \times (2M-1) \times N_h$，其中 $N_h$ 为注意力头数。对于窗口内位置为 $(i, j)$ 和 $(i', j')$ 的两个patch，其相对偏移为 $(\Delta i, \Delta j) = (i'-i, j'-j)$，对应的偏置为：

$$
\mathbf{B}_{i,j,i',j'} = \hat{\mathbf{B}}_{\Delta i + M - 1, \Delta j + M - 1}
$$

这种设计相比绝对位置编码具有更好的外推能力（extrapolation），且参数效率更高——$\hat{\mathbf{B}}$ 的参数数量仅为 $(2M-1)^2$，独立于输入分辨率。

**移位窗口分区（Shifted Partitioning）：** 在第 $l+1$ 层（$l$ 为偶数），特征图被循环移位（Cyclic Shift） $\lfloor M/2 \rfloor$ 个patch。这将导致窗口被"打乱"，原来的一个窗口中的patch可能分散到多个窗口中，同时也引入了跨窗口的交互。为了保持计算效率，移位后的特征图被重新划分为新的窗口集合（包含不同大小的窗口——部分窗口较小，因为它们位于特征图边界）。由于移位后的窗口可能包含来自原始不同窗口的patch，自注意力在重新划分的窗口内计算时自然实现了跨窗口的信息交互。

具体而言，设 $\mathbf{z}$ 为经过循环移位后的特征图，首先通过掩码（Mask）操作将其划分为新的窗口集合。对于每个新窗口，掩码矩阵 $\mathbf{M} \in \{-10^{15}\}^{N_w \times N_w}$ 被添加到注意力分数矩阵中（$N_w$ 为窗口内patch数），其中不属于同一原始窗口的patch对之间的注意力分数被设置为一个极大的负数，从而在Softmax操作后被抑制为零。这种掩码机制确保了移位窗口注意力在计算上等价于在扩展的局部区域内进行注意力计算，同时保持了窗口内的计算效率。

### 第4.3节 多尺度建模与语义融合

Swin Transformer的层次化架构使其天然支持多尺度建模。从Stage 1到Stage 4，特征图的空间分辨率依次降低为原始的1/4、1/8、1/16和1/32，而通道维度依次增加为 $C, 2C, 4C, 8C$。这种设计遵循了经典的"分辨率降低、通道增加"原则（如VGG和ResNet所采用的），与人类视觉系统的处理方式具有结构相似性——初级视觉皮层（V1区）处理高分辨率局部特征，而高级视觉皮层处理低分辨率全局语义。

在目标检测和实例分割等下游任务中，Swin Transformer 的 Stage 2 至 Stage 4 输出直接作为特征金字塔网络（FPN）或路径聚合网络（PANet, Liu et al., 2018）的输入。例如，在 Mask R-CNN（He et al., 2017）框架中，Swin Transformer 作为骨干网络（Backbone）替换原来的 ResNet-50/101：在标准 Mask R-CNN 设置下，Swin-T 约达到 box AP 46.0 / mask AP 41.6（相对 ResNet-50 基线约 42.0 / 37.8 有明显提升）。更高的约 51.1 box AP / 48.1 mask AP 等数字通常对应 Cascade Mask R-CNN、HTC 或更强检测头与更长训练日程等增强配置，不宜直接当作标准 Mask R-CNN 结果引用。更大的 Swin-S 和 Swin-B 在相应检测框架下进一步提升，全面超越了使用 ResNeXt-101-64x4d 骨干的强 CNN 基线。

从理论视角审视，Swin Transformer的成功揭示了视觉任务中一个深刻的设计权衡：归纳偏置的双刃剑效应。CNN的局部连接性和平移等变性在数据有限时是宝贵的先验知识，但在大规模数据时代，过于强烈的归纳偏置反而限制了模型的表征能力上限。Swin Transformer通过局部窗口注意力保留了部分归纳偏置（局部性），同时通过移位窗口实现了跨区域信息交互，在效率和性能之间取得了最佳平衡。事实上，Swin Transformer的计算复杂度为：

$$
\mathcal{C}_{\text{Swin}} = O(N \cdot M^2 \cdot C)
$$

其中 $N = HW/P^2$ 为patch序列长度，$M$ 为窗口大小。当 $M$ 固定时（如 $M=7$），复杂度与 $N$ 呈线性关系，这与ViT的 $O(N^2 \cdot C)$ 形成鲜明对比。

---

## 第5节 掩码自编码器（MAE）：自监督视觉预训练的新范式

### 第5.1节 编码器-解码器架构

Masked Autoencoder（MAE）（He et al., 2022）是Facebook AI Research（现Meta AI）在2021年底提出的自监督视觉预训练框架。MAE的核心思想源自自然语言处理领域的掩码语言模型（Masked Language Model, MLM）（Devlin et al., 2019），即BERT所采用的预训练目标——通过随机掩码输入序列中的一部分token，然后训练模型来重建被掩码的内容。然而，将这一策略直接迁移到视觉领域面临着一个根本性的挑战：图像数据是连续的、信息高度冗余的，而语言是离散的、信息高度密集的。在NLP中，掩码15%的token已经构成足够具有挑战性的预训练任务；但在图像中，即使掩码75%的patch，人类仍然能够根据剩余的信息推断出被掩码区域的内容。He等人的关键洞见是：恰恰是这种高比例的掩码，使得自监督视觉预训练成为可能且高效。

MAE采用了一个非对称的编码器-解码器架构：

**编码器（Encoder）：** 编码器仅处理未被掩码的可见patch。设原始图像被划分为 $N$ 个patch，随机掩码比例为 $r$（通常为75%），则保留的可见patch数量为 $N_{\text{vis}} = N \times (1-r)$。编码器（通常是一个ViT）接收这些可见patch的嵌入序列，输出对应的隐藏表征：

$$
\{\mathbf{z}_{\text{vis}, i}\}_{i=1}^{N_{\text{vis}}} = \text{Encoder}(\{\mathbf{x}_{\text{vis}, i}\}_{i=1}^{N_{\text{vis}}})
$$

其中 $\mathbf{x}_{\text{vis}, i} = \mathbf{x}_{p,i}\mathbf{E} + \mathbf{E}_{\text{pos},i}$ 为第 $i$ 个可见patch的嵌入表示。值得注意的是，编码器只看到 $N_{\text{vis}} \ll N$ 个patch，这使得计算量大幅降低。

**解码器（Decoder）：** 解码器接收所有patch（包括被掩码的patch和可见patch）的嵌入，重建原始像素值。对于被掩码的patch，使用一个可学习的掩码token（mask token） $\mathbf{m} \in \mathbb{R}^D$ 来表示；对于可见patch，使用编码器的输出 $\mathbf{z}_{\text{vis},i}$；所有patch都加上对应的位置编码。解码器通常是一个较浅的Transformer（如ViT-Decoder/8，即8层的小型Transformer），其输入序列长度为 $N$：

$$
\{\hat{\mathbf{x}}_i\}_{i=1}^N = \text{Decoder}([\mathbf{m}, \ldots, \mathbf{m}; \mathbf{z}_{\text{vis},1}, \ldots, \mathbf{z}_{\text{vis},N_{\text{vis}}}] + \mathbf{E}_{\text{pos}})
$$

其中 $[\mathbf{m}, \ldots, \mathbf{m}; \ldots]$ 表示 $N_{\text{mask}} = rN$ 个掩码token与 $N_{\text{vis}}$ 个编码器输出在序列维度上的拼接。解码器的输出经过一个投影层映射回像素空间，得到对每个patch的重建预测。

**重建目标：** MAE使用均方误差（MSE）作为重建损失，但仅在被掩码的patch上计算：

$$
\mathcal{L}_{\text{MAE}} = \frac{1}{N_{\text{mask}}} \sum_{i \in \mathcal{M}} \|\hat{\mathbf{x}}_i - \mathbf{x}_i\|^2
$$

其中 $\mathcal{M}$ 为被掩码patch的索引集合，$\mathbf{x}_i$ 为第 $i$ 个patch的原始像素值（经过patch嵌入之前的原始像素，或归一化后的像素值）。值得注意的是，解码器的输出维度需要匹配patch的像素维度（如 $16 \times 16 \times 3 = 768$），而不是编码器的隐藏维度。

### 第5.2节 高比例掩码策略

MAE最关键的创新在于其高比例掩码策略。He等人系统地比较了不同掩码比例对预训练效果的影响，发现当掩码比例为75%时，预训练性能最优；即使提高到90%，性能下降也很有限。这一发现可以从信息论的角度进行解释。

设图像的信息熵为 $H(\mathbf{x})$，可见patch携带的信息量为 $H(\mathbf{x}_{\text{vis}})$，被掩码patch的条件熵为 $H(\mathbf{x}_{\text{mask}} | \mathbf{x}_{\text{vis}})$。高比例掩码策略的有效性依赖于以下假设：

$$
I(\mathbf{x}_{\text{mask}}; \mathbf{x}_{\text{vis}}) \gg 0
$$

即可见patch与被掩码patch之间存在显著的信息冗余。在自然图像中，由于语义结构的高度结构化（边缘、纹理、形状等低级特征和物体部件、空间关系等高级特征均具有空间局部性和统计冗余性），这一假设在大多数情况下成立。

高比例掩码带来的好处是多方面的。从计算效率的角度，编码器仅需处理25%的patch，预训练速度提升了约4倍（相对于处理全部patch）。从表征学习的角度，高比例掩码迫使模型学习更高层次的语义信息——模型不能仅仅通过复制相邻patch的像素来完成重建任务，而必须理解图像中的物体结构、空间关系和语义内容。这类似于BERT中掩码语言模型迫使模型理解语法和语义，而不仅仅是进行局部n-gram匹配。

He等人的实验表明，在ImageNet-1K上，MAE 预训练后 **ViT-Huge** 微调可达约 **87.8%** top-1（更高分辨率微调设定）；**ViT-Large** 约为 **85.9%** 量级，不宜把 87.8% 归给 ViT-Large。该结果优于同期许多需要标签的监督预训练基线。更重要的是，MAE的预训练在更大的数据集（如ImageNet-21K）上展现出了更强的可扩展性（scalability）——模型越大、数据越多，MAE的收益越明显，这与BERT在NLP领域的观察一致。

MAE的另一个重要优势是其泛化能力。在COCO目标检测和ADE20K语义分割等下游任务上，MAE预训练的骨干网络在仅使用10%或1%标注数据微调时，性能显著优于在完整数据集上监督预训练的基线。这表明MAE学习到的表征更加强调通用语义理解，而非对特定分类标签的 memorization。

---

## 第6节 扩散模型：生成建模的数学之美

扩散模型（Diffusion Model）是近年来生成式人工智能领域最引人注目的技术突破之一。在2020年之前，生成对抗网络（GAN, Goodfellow et al., 2014）和自回归模型（如PixelRNN, Van den Oord et al., 2016）主导着图像生成领域。然而，GAN面临模式崩溃（mode collapse）和训练不稳定的固有问题，而自回归模型的采样速度极慢。扩散模型通过一个优雅的概率框架，在生成质量和训练稳定性之间取得了出色的平衡，并催生了Stable Diffusion、DALL-E 2、Midjourney等划时代应用。

### 第6.1节 前向扩散过程：高斯马尔可夫链

扩散模型的核心思想来源于非平衡热力学（Non-equilibrium Thermodynamics）（Sohl-Dickstein et al., 2015）。给定一个数据样本 $\mathbf{x}_0 \sim q(\mathbf{x}_0)$（从真实数据分布中采样），前向扩散过程（Forward Diffusion Process）通过逐步添加高斯噪声将数据分布逐渐转化为标准正态分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$。这一过程被定义为一个固定长度的马尔可夫链：

$$
q(\mathbf{x}_{1:T} | \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t | \mathbf{x}_{t-1})
$$

其中每一步的条件分布为一个高斯分布：

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})
$$

这里 $\beta_t$ 为第 $t$ 步的方差调度参数（variance schedule），通常满足 $\beta_1 < \beta_2 < \cdots < \beta_T \ll 1$。$\beta_t$ 的选取对扩散模型的性能有重要影响——如果 $\beta_t$ 过大，数据会被过快噪声淹没，导致信息丢失；如果过小，则需要过多的扩散步骤才能达到纯噪声分布。实践中常用的调度策略包括线性调度（$\beta_t$ 从 $10^{-4}$ 线性增长到 $10^{-2}$）和余弦调度（Nichol & Dhariwal, 2021）。

前向扩散过程的一个重要性质是，它可以被重新参数化为在任意时间步 $t$ 直接对 $\mathbf{x}_0$ 添加噪声的闭式表达式。利用马尔可夫链的传递性，有：

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

其中 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。这一闭式表达式具有极其重要的实践意义：它允许我们在任意时间步 $t \in \{1, \ldots, T\}$ 直接从原始数据 $\mathbf{x}_0$ 采样噪声版本 $\mathbf{x}_t$，而无需逐步迭代。同时，$\bar{\alpha}_t$ 随时间单调递减——当 $t=T$ 时，$\bar{\alpha}_T \approx 0$，因此 $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$，即数据分布已被完全转化为标准正态分布。

**后验分布：** 前向过程中任意两个时间步 $t-1$ 和 $t$ 之间的后验分布也是高斯分布：

$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})
$$

其中：

$$
\begin{aligned}
\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) &= \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{1 - \beta_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t \\
\tilde{\beta}_t &= \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t
\end{aligned}
$$

这一后验分布将在反向去噪过程中发挥关键作用。

### 第6.2节 反向去噪过程：Score Matching与DDPM

前向扩散过程 $q(\mathbf{x}_t | \mathbf{x}_{t-1})$ 是已知的（其参数 $\beta_t$ 在训练前已固定），但反向过程 $q(\mathbf{x}_{t-1} | \mathbf{x}_t)$ 无法直接计算，因为它依赖于难以处理的边缘分布 $q(\mathbf{x}_t)$。扩散模型的核心目标是学习一个反向过程（Reverse Denoising Process） $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ 来近似真实的反向后验分布 $q(\mathbf{x}_{t-1} | \mathbf{x}_t)$。

**参数化反向过程：** 类似于前向过程，反向过程也被参数化为一个高斯马尔可夫链：

$$
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)
$$

其中 $p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$ 为先验分布，每个条件分布参数化为：

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
$$

在原始的DDPM（Denoising Diffusion Probabilistic Model, Ho et al., 2020）中，方差 $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$ 被固定为 $\tilde{\beta}_t \mathbf{I}$（即使用前向过程后验方差），仅对均值 $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$ 进行参数化。后续研究表明，固定方差已经足够好，且简化了训练。

**从噪声预测角度重新参数化：** DDPM对均值函数进行了关键的重新参数化。注意到真实反向后验均值 $\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0)$ 可以改写为：

$$
\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{1}{\sqrt{1 - \beta_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_0 \right)
$$

其中 $\boldsymbol{\epsilon}_0$ 为生成 $\mathbf{x}_t$ 时添加的噪声（因为 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_0$）。这意味着，如果我们能够预测出原始噪声 $\boldsymbol{\epsilon}_0$，就可以精确地计算真实后验均值。

因此，DDPM将神经网络的预测目标从"预测均值"改为"预测噪声"。设噪声预测网络为 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$（通常是一个U-Net架构，以时间步 $t$ 为条件），则参数化的均值函数为：

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{1 - \beta_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)
$$

**训练目标：** DDPM的训练目标是最大化数据对数似度的变分下界（Variational Lower Bound, VLB），也称为证据下界（Evidence Lower Bound, ELBO）。对于扩散模型，ELBO可以简化为以下形式：

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(\mathbf{x}_0), t, \boldsymbol{\epsilon}} \left[ \frac{\beta_t^2}{2 \sigma_t^2 (1 - \beta_t)(1 - \bar{\alpha}_t)} \cdot \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, t) \|^2 \right] + C
$$

其中 $\sigma_t^2 = \tilde{\beta}_t$ 为第 $t$ 步的方差，$C$ 为与 $\theta$ 无关的常数项。Ho等人发现，进一步简化损失——忽略方差权重并仅保留简单的均方误差——在实践中同样有效且更容易训练：

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{q(\mathbf{x}_0), t, \boldsymbol{\epsilon}} \left[ \| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, t) \|^2 \right]
$$

这一简化损失是后续所有扩散模型工作的标准训练目标。

**改进的DDPM（Improved DDPM）：** Nichol & Dhariwal（2021）在原始DDPM的基础上提出了两项关键改进。第一，**学习方差（Learned Variance）**：原始DDPM将反向过程的方差固定为 $\sigma_t^2 = \tilde{\beta}_t$，而Improved DDPM让网络同时预测噪声 $\boldsymbol{\epsilon}_\theta$ 和方差插值参数。具体地，网络输出一个与图像同形状的插值系数 $v_\theta(\mathbf{x}_t, t) \in [0,1]$（通常经 sigmoid 约束），并在对数空间中对上界 $\beta_t$ 与后验方差 $\tilde{\beta}_t$ 做插值：

$$
\Sigma_\theta(\mathbf{x}_t, t) = \exp\!\big( v_\theta(\mathbf{x}_t, t)\,\log \beta_t + \big(1 - v_\theta(\mathbf{x}_t, t)\big)\,\log \tilde{\beta}_t \big)
$$

其中 $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t$。该参数化保证方差始终落在 $[\tilde{\beta}_t, \beta_t]$ 区间内，并在ELBO中让方差相关项完整参与梯度计算。实验表明，学习方差在CIFAR10上将FID从3.75进一步降低到2.92，在CelebA-HQ 256×256上从5.83降低到3.94，提升幅度虽然不及噪声预测架构改进（如U-Net设计、注意力机制），但在高质量图像生成中提供了额外的边际收益。

第二，**余弦方差调度（Cosine Variance Schedule）：** Improved DDPM将方差调度从线性调度改为余弦调度：

$$
\bar{\alpha}_t = f(t) = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)
$$

其中 $s$ 为一个小偏移量（通常取0.008），用于防止 $\beta_t$ 在初始时刻过小。余弦调度的优势在于其尾部行为——与线性调度在末尾急剧增大 $\beta_t$ 不同，余弦调度使噪声添加过程在整个扩散过程中更加均匀，这被证明在生成高细节图像时产生更好的视觉效果。

**采样过程（DDPM Sampling）：** 训练完成后，生成新样本的过程从标准正态噪声 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 出发，逐步应用学习到的反向过程去噪：

$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{1 - \beta_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}_t, \quad \mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad t = T, \ldots, 1
$$

其中最后一项 $\sigma_t \mathbf{z}_t$ 为添加的随机噪声（当 $t > 1$ 时），保证生成过程的随机性。标准的DDPM需要 $T=1000$ 步采样，速度较慢。

### 第6.3节 DDIM：确定性采样与加速

DDPM的1000步采样严重限制了扩散模型的实际部署效率。Denoising Diffusion Implicit Models（DDIM, Song et al., 2021）对这一局限进行了根本性的改进。DDIM的核心洞察是：前向扩散过程的马尔可夫性假设并非反向过程所必需的。通过移除反向过程的马尔可夫性约束，DDIM实现了大幅加速的确定性采样。

设 $\alpha_t = \bar{\alpha}_t$ 为累积乘积，DDIM将 $\mathbf{x}_t$ 到 $\mathbf{x}_{\tau_{i-1}}$ 的采样步骤重新定义为：

$$
\mathbf{x}_{\tau_{i-1}} = \sqrt{\alpha_{\tau_{i-1}}} \cdot \underbrace{\frac{\mathbf{x}_{\tau_i} - \sqrt{1 - \alpha_{\tau_i}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_{\tau_i}, \tau_i)}{\sqrt{\alpha_{\tau_i}}}}_{\text{预测的原始图像}} + \sqrt{1 - \alpha_{\tau_{i-1}} - \sigma_{\tau_i}^2} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_{\tau_i}, \tau_i) + \sigma_{\tau_i} \mathbf{z}
$$

其中 $\{\tau_i\}_{i=0}^K$ 是 $\{1, \ldots, T\}$ 的一个子序列（$\tau_0 = 0$ 对应原始数据，$\tau_K = T$ 对应纯噪声），$\sigma_{\tau_i}$ 控制采样过程中的随机性。DDIM的关键特性包括：

1. **确定性采样：** 当 $\sigma_{\tau_i} = 0$ 时，整个采样过程变为确定性的（给定初始噪声，输出唯一确定），且仍然可以使用在DDPM下训练的同一个噪声预测网络 $\boldsymbol{\epsilon}_\theta$。
2. **加速比可控：** 通过选择子序列 $\{\tau_i\}$，可以在保持生成质量的同时大幅减少采样步数。实验表明，使用仅10-20步的DDIM采样，生成质量与1000步的DDPM相当。
3. **与传统扩散过程的关系：** DDPM可以看作是DDIM的一个特例，当 $\sigma_{\tau_i}^2 = \frac{1 - \alpha_{\tau_{i-1}}}{1 - \alpha_{\tau_i}} \beta_{\tau_i}$ 时，DDIM退化为DDPM。

### 第6.4节 条件生成：Classifier Guidance与Classifier-Free Guidance

无条件扩散模型学习的是数据边缘分布 $p(\mathbf{x})$，而条件扩散模型学习的是条件分布 $p(\mathbf{x} | \mathbf{c})$，其中 $\mathbf{c}$ 为条件信息（如类别标签或文本描述）。条件生成技术使扩散模型从"无条件的随机生成器"进化为"可控的创意工具"，这是扩散模型能够支撑DALL-E、Stable Diffusion等应用的关键。

**Classifier Guidance（分类器引导）：** Dhariwal & Nichol（2021）提出的分类器引导方法利用一个独立的分类器 $p_\phi(\mathbf{c} | \mathbf{x}_t)$ 来引导采样过程。根据贝叶斯定理，条件分布可以分解为：

$$
\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t | \mathbf{c}) = \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log p(\mathbf{c} | \mathbf{x}_t)
$$

这意味着，在采样过程中，每一步的去噪方向可以被修改为：

$$
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - w \cdot \sqrt{1 - \bar{\alpha}_t} \cdot \nabla_{\mathbf{x}_t} \log p_\phi(\mathbf{c} | \mathbf{x}_t)
$$

其中 $w > 0$ 为引导强度（guidance scale）的超参数。当 $w=0$ 时退化为无条件采样，$w$ 越大则条件 $\mathbf{c}$ 对生成结果的影响越强。然而，分类器引导需要额外训练一个在噪声图像上表现良好的分类器，且在大规模开放词汇条件（如自由形式文本描述）下难以泛化。

**Classifier-Free Guidance（无分类器引导）：** Ho & Salimans（2021）提出的无分类器引导方法彻底消除了对独立分类器的依赖。其核心思想是在训练扩散模型时，以一定概率（如10%）丢弃条件信息 $\mathbf{c}$（即用无条件训练目标替代条件训练目标），使得同一个噪声预测网络 $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})$ 既能处理条件输入也能处理无条件输入。

在采样时，同时使用条件和无条件预测：

$$
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) + w \cdot \left( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) \right)
$$

其中 $\emptyset$ 表示空条件（dropout后的条件），$w \geq 1$ 为引导强度。当 $w=1$ 时，$\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c})$ 为标准条件采样。Classifer-Free Guidance的优势在于：（1）无需额外训练分类器；（2）支持任意类型的条件信号（类别标签、文本描述、深度图、分割图等）；（3）引导强度 $w$ 可以在推理时灵活调整，$w$ 越大生成图像与条件的关联性越强，但多样性会相应降低。

从概率论的角度，无分类器引导可以被理解为对条件分布进行再归一化（renormalization）：

$$
p_{\text{guided}}(\mathbf{x} | \mathbf{c}) \propto \frac{p(\mathbf{x} | \mathbf{c})^w}{p(\mathbf{x})^{w-1}} = p(\mathbf{x})\cdot p(\mathbf{c} | \mathbf{x})^w / Z(\mathbf{c})
$$

（等价地也可写为 $\propto p(\mathbf{c}|\mathbf{x})^w \cdot p(\mathbf{x})^{1-w}$ 的 Bayes 形式，但须与左侧 $p(\mathbf{x}|\mathbf{c})^w/p(\mathbf{x})^{w-1}$ 一致；勿写成 $p(\mathbf{x}|\mathbf{c})^w/p(\mathbf{x})$ 却再等于 $p(\mathbf{c}|\mathbf{x})^w p(\mathbf{x})^{1-w}$ 的错误等式。）
$$

当 $w > 1$ 时，分布被尖锐化（sharpened），更集中于条件 $\mathbf{c}$ 所指示的区域；当 $w < 1$ 时，分布被平坦化（flattened），增加了多样性。

### 第6.5节 Score Matching与随机微分方程（SDE）视角

上述从概率图模型角度对扩散模型的推导（前向马尔可夫链 + 反向参数化）是DDPM系列的经典框架。然而，扩散模型还可以从另一个深刻的数学视角来理解——**score matching**（得分匹配）和**随机微分方程**（Stochastic Differential Equation, SDE）。这一视角不仅提供了更优雅的理论解释，还催生了 consistency model、stochastic interpolant 等新一代生成模型。

**Score Matching的基本原理：** 设数据分布为 $p_{\text{data}}(\mathbf{x})$，其概率密度函数（PDF）为 $p(\mathbf{x})$。得分函数（score function）定义为概率密度的对数梯度：

$$
\nabla_{\mathbf{x}} \log p(\mathbf{x}) = \frac{\nabla_{\mathbf{x}} p(\mathbf{x})}{p(\mathbf{x})}
$$

得分函数指向概率密度增长最快的方向——在数据密集区域（高概率区域），得分函数的值较小；在数据稀疏区域（低概率区域），得分函数指向高概率区域。因此，如果能够学习得分函数 $\nabla_{\mathbf{x}} \log p(\mathbf{x})$，就可以通过以下 Langevin 动力学采样过程从任意先验分布生成数据：

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \frac{\epsilon}{2} \nabla_{\mathbf{x}} \log p(\mathbf{x}_k) + \sqrt{\epsilon} \mathbf{z}_k, \quad \mathbf{z}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

其中 $\epsilon$ 为步长。当 $\epsilon \to 0$ 且 $K \to \infty$ 时，$\mathbf{x}_K$ 的分布收敛到 $p(\mathbf{x})$（依据Langevin动力学与Fokker–Planck方程的经典结果）。

然而，直接估计 $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ 面临一个根本困难：我们无法直接计算 $p(\mathbf{x})$（它涉及对整个数据分布的归一化常数积分）。Denoising Score Matching（Vincent, 2011）通过引入一个受控的噪声过程绕过了这一问题：对数据样本 $\mathbf{x} \sim p(\mathbf{x})$ 添加噪声得到 $\mathbf{x}' = \mathbf{x} + \boldsymbol{\epsilon}$（$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$），然后估计加噪分布 $p(\mathbf{x}')$ 的得分函数。关键定理表明：

$$
\nabla_{\mathbf{x}} \log p(\mathbf{x}) = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} \left[ \nabla_{\mathbf{x}'} \log p(\mathbf{x}' | \mathbf{x}) \right]_{\mathbf{x}' = \mathbf{x} + \boldsymbol{\epsilon}}
$$

其中条件分布 $p(\mathbf{x}' | \mathbf{x}) = \mathcal{N}(\mathbf{x}'; \mathbf{x}, \sigma^2 \mathbf{I})$ 是已知的，其得分函数有闭式解 $\nabla_{\mathbf{x}'} \log p(\mathbf{x}' | \mathbf{x}) = -(\mathbf{x}' - \mathbf{x})/\sigma^2$。因此，学习得分函数转化为学习一个回归问题：给定含噪样本 $\mathbf{x}'$，预测其原始样本 $\mathbf{x}$。

**多尺度Score Matching与扩散过程的联系：** 在扩散模型中，前向过程 $q(\mathbf{x}_t | \mathbf{x}_0)$ 在多个噪声水平上对数据进行加噪（从 $\sigma_{\min}$ 到 $\sigma_{\max}$）。对于每个噪声水平 $t$，我们可以学习对应的得分函数 $s_\theta(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)$。Song等人在Score-Based Generative Modeling through Stochastic Differential Equations（Song et al., 2021）中证明：扩散过程的正向过程和反向过程都可以用SDE来描述。

**正向SDE：** 前向扩散过程可以表示为以下SDE的离散化：

$$
d\mathbf{x} = f(\mathbf{x}, t) \, dt + g(t) \, d\mathbf{w}
$$

其中 $f(\mathbf{x}, t)$ 为漂移系数（drift coefficient），$g(t)$ 为扩散系数（diffusion coefficient），$\mathbf{w}$ 为标准Wiener过程（Brownian Motion）。对于DDPM的正向过程，有 $f(\mathbf{x}, t) = -\frac{1}{2} \beta_t \mathbf{x}$ 和 $g(t) = \sqrt{\beta_t}$。当使用方差调度 $\beta_t$ 时，这一SDE的离散化恰好对应于DDPM的前向加噪步骤。

**反向SDE：** 根据Anderson（1982）的经典结果，正向SDE的时间反向过程也是一个SDE：

$$
d\mathbf{x} = \left[ f(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log q(\mathbf{x}) \right] dt + g(t) \, d\bar{\mathbf{w}}
$$

其中 $\bar{\mathbf{w}}$ 为反向时间的Wiener过程。注意反向SDE的漂移项中包含了得分函数 $\nabla_{\mathbf{x}} \log q(\mathbf{x})$——这正是我们需要学习的量。如果能够训练一个网络 $s_\theta(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)$，就可以通过数值方法（如Euler-Maruyama离散化）从纯噪声出发，沿反向SDE逐步去噪，最终采样得到数据样本。

**DDPM与SDE的统一视角：** DDPM的反向过程（见上文采样公式 $\mathbf{x}_{t-1} = \frac{1}{\sqrt{1 - \beta_t}}\big(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\big) + \sigma_t \mathbf{z}_t$）可以看作是上述反向SDE的概率ODE（Probability ODE）版本——当对反向SDE应用Tweedie公式并选择特定的离散化方案时，得到的结果与DDPM采样公式一致。反过来，DDIM（Song et al., 2021）可以被理解为对反向SDE的确定性离散化（忽略随机项 $g(t) \, d\bar{\mathbf{w}}$），这正是DDIM实现确定性采样的数学根源。

从SDE视角出发，研究者们提出了多种改进扩散模型的方法。**Variance Exploding（VE）**和**Variance Preserving（VP）**是两种经典的SDE参数化（Song et al., 2021）：VP SDE通过设计使整个扩散过程中数据的方差保持为1（仅改变其分布形状），VE SDE则让方差随时间指数增长。**edict（Editable Diffusion, Meng et al., 2021）**利用反向SDE的可逆性实现了图像编辑。**一致性模型（Consistency Model, Song et al., 2023）**直接在反向SDE的轨迹上学习一个将任意噪声级别映射到数据分布的映射，实现了单步生成。这些进展表明，SDE框架不仅是一个优美的数学等价描述，更是一个催生新方法论的活性研究前沿。

### 第6.6节 DALL-E系列：自回归与扩散的融合

OpenAI的DALL-E系列系统（Ramesh et al., 2021, 2022; Betker et al., 2023）代表了文本到图像生成领域的重要里程碑。DALL-E（第1代）采用了两阶段方法：首先使用dVAE（Discrete Variational Autoencoder）将图像压缩为离散token序列，然后使用自回归Transformer（类似于GPT）在文本和图像token的联合分布上进行建模。DALL-E 2则引入了CLIP（Contrastive Language-Image Pre-training, Radford et al., 2021）作为文本和图像之间的语义桥梁，结合扩散模型实现了更高质量的生成。

DALL-E 2的技术路线可以概括为以下三个阶段：

**阶段1：CLIP语义空间对齐。** DALL-E 2首先使用CLIP模型将文本和图像映射到共享的语义嵌入空间。CLIP通过对比学习在4亿个（图像，文本）对上进行预训练，学习到的文本编码器 $f_{\text{text}}$ 和图像编码器 $f_{\text{image}}$ 满足：

$$
\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\text{sim}(f_{\text{text}}(t_i), f_{\text{image}}(x_i)) / \tau)}{\sum_{j=1}^B \exp(\text{sim}(f_{\text{text}}(t_i), f_{\text{image}}(x_j)) / \tau)}
$$

其中 $\text{sim}(\mathbf{a}, \mathbf{b}) = \mathbf{a}^\top \mathbf{b} / \|\mathbf{a}\| \|\mathbf{b}\|$ 为余弦相似度，$\tau$ 为温度参数，$B$ 为批次大小。CLIP的优势在于其开放词汇能力——它能够理解任意自然语言描述，而不仅仅是预定义类别。

**阶段2：先验模型（Prior Model）。** 给定一个文本编码 $\mathbf{z}_{\text{text}} = f_{\text{text}}(t)$，需要一个先验模型来生成对应的CLIP图像嵌入 $\mathbf{z}_{\text{image}} \approx f_{\text{image}}(x)$。DALL-E 2使用两种先验模型：(a) 自回归Transformer，将CLIP图像嵌入量化为离散token后自回归生成；(b) 扩散模型，直接在连续嵌入空间中进行去噪。实验表明，扩散先验在计算效率和生成质量上都优于自回归先验。

**阶段3：解码器（Decoder）。** 解码器接收CLIP图像嵌入 $\mathbf{z}_{\text{image}}$ 和可选的原始文本 $t$ 作为条件，使用扩散模型生成高分辨率图像。解码器首先将 $\mathbf{z}_{\text{image}}$ 上采样到较低分辨率（如 $64 \times 64$），然后通过两个超分辨率扩散阶段依次生成 $256 \times 256$ 和 $1024 \times 1024$ 的图像。每个超分辨率阶段都使用文本条件进行引导，确保生成图像与文本描述的语义一致性。

### 第6.7节 Stable Diffusion与潜在扩散模型

Stable Diffusion（Rombach et al., 2022）是Stability AI于2022年8月发布的开源文本到图像生成系统，其训练使用了LAION开发的LAION-5B大规模数据集。与DALL-E 2在像素空间运行扩散模型不同，Stable Diffusion基于潜在扩散模型（Latent Diffusion Model, LDM）架构，在压缩的潜在空间而非像素空间中进行扩散和去噪，这使其能够在消费级GPU上高效运行，成为AI图像生成民主化的关键技术推动者。

**潜在空间的动机：** 在像素空间中运行扩散模型的根本挑战在于计算成本。对于 $512 \times 512 \times 3$ 的图像，像素空间中的U-Net需要处理约78万个像素点，即使使用高效的卷积架构，多步采样的计算开销也非常巨大。LDM的核心洞察是：图像的大部分"冗余"信息（如重复纹理、背景区域等）可以通过一个预训练的编码器-解码器进行压缩，而语义信息则保留在压缩后的潜在表示中。

具体地，LDM使用了一个预训练的变分自编码器（VAE, Kingma & Welling, 2014）。编码器 $\mathcal{E}$ 将图像 $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$ 映射到潜在空间：

$$
\mathbf{z} = \mathcal{E}(\mathbf{x}) \in \mathbb{R}^{h \times w \times c}
$$

其中下采样因子为 $f = H/h = W/w = 8$。对于 $512 \times 512$ 的输入，潜在空间为 $64 \times 64 \times 4$，压缩比高达48倍（$512 \times 512 \times 3 / (64 \times 64 \times 4) = 48$）。解码器 $\mathcal{D}$ 则将潜在表示重建为像素空间：

$$
\hat{\mathbf{x}} = \mathcal{D}(\mathbf{z})
$$

VAE的训练目标为标准重建损失加上KL散度正则化：

$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{\mathbf{x}} \left[ \|\mathbf{x} - \mathcal{D}(\mathcal{E}(\mathbf{x}))\|^2 + \frac{1}{2} \sum_i \left( \mu_i^2 + \sigma_i^2 - \log(\sigma_i^2) - 1 \right) \right]
$$

其中 $\boldsymbol{\mu}_{\mathcal{E}}$ 为均值，$\boldsymbol{\sigma}_{\mathcal{E}}$ 为**标准差**（对角高斯；若用对数方差参数化需相应改写），KL 项按维求和。为了减少重建模糊，LDM使用了 perceptual loss 和 patch-based adversarial loss（GAN判别器），这在原始论文中被称为LPIPS（Zhang et al., 2018）加权重建。

**潜在扩散过程：** 扩散模型的前向过程和反向过程都在潜在空间 $\mathbf{z} \in \mathbb{R}^{h \times w \times c}$ 中进行：

$$
q(\mathbf{z}_t | \mathbf{z}_{t-1}) = \mathcal{N}(\mathbf{z}_t; \sqrt{1-\beta_t}\mathbf{z}_{t-1}, \beta_t \mathbf{I})
$$

$$
p_\theta(\mathbf{z}_{t-1} | \mathbf{z}_t, \mathbf{c}) = \mathcal{N}(\mathbf{z}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{z}_t, t, \mathbf{c}), \boldsymbol{\Sigma}_\theta(\mathbf{z}_t, t))
$$

条件 $\mathbf{c}$ 通过一个预训练的CLIP文本编码器从输入文本prompt中获得。噪声预测网络 $\boldsymbol{\epsilon}_\theta$ 基于U-Net架构，并使用交叉注意力层（Cross-Attention Layer）将文本条件注入到去噪过程中：

$$
\text{CrossAttn}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V, \quad Q = W_Q \cdot \mathbf{z}_t, \quad K = W_K \cdot \mathbf{c}, \quad V = W_V \cdot \mathbf{c}
$$

其中 $\mathbf{c} \in \mathbb{R}^{L \times d_c}$ 为文本编码序列（$L$ 为文本token数），$\mathbf{z}_t \in \mathbb{R}^{N \times d}$ 为潜在特征图展平后的序列（$N = h \times w$）。交叉注意力层将文本语义信息注入到视觉去噪的每一步，使生成过程能够忠实地响应文本描述。

**无分类器引导：** Stable Diffusion采用无分类器引导（CFG），在训练时以10%的概率丢弃文本条件。推理时使用的引导公式为：

$$
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_t, t, \mathbf{c}) = \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \emptyset) + w \cdot (\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, \emptyset))
$$

引导强度 $w$ 通常在7-8之间可以获得较好的效果。过高的 $w$ 会导致图像过饱和、细节丢失和"煎饼脸"等伪影。

Stable Diffusion的成功不仅在于其技术上的优雅，更在于其开放的开源策略。Stability AI公开了模型权重和代码，使得全球的研究者和开发者能够在它的基础上进行微调（Fine-tuning）和扩展。LoRA（Low-Rank Adaptation, Hu et al., 2022）、DreamBooth（Ruiz et al., 2023）和Textual Inversion（Gal et al., 2022）等微调技术使得普通用户可以用少量个人图像训练出风格一致的角色或物体生成器，进一步推动了AI图像生成的普及。

### 第6.8节 Midjourney的技术路线

Midjourney由David Holtz于2021年创立，截至2024年已发布至V6版本，是商业文本到图像生成领域最成功的产品之一。与Stable Diffusion的开源策略不同，Midjourney采用了完全闭源的商业化路线，其技术细节并未完全公开。然而，基于其创始人的公开演讲、专利申请和社区逆向工程，可以对其技术路线进行合理的推测和分析。

Midjourney的技术栈极有可能基于扩散模型架构，且很可能从Stable Diffusion的开源社区中汲取了灵感。其独特的技术方向包括：

**1. 多阶段级联生成：** Midjourney V5和V6采用了多阶段生成流水线，首先生成一个低分辨率的粗略构图（可能为 $256 \times 256$ 或 $512 \times 512$），然后通过超分辨率扩散模型逐步提升分辨率和细节。这种级联方法相比单阶段高分辨率生成在计算效率和视觉质量上都更具优势。

**2. 美学对齐与人类反馈：** Midjourney的核心竞争力在于其生成图像的艺术品质和审美表现力。据推测，Midjourney可能使用了大量的美学评分数据来训练奖励模型（Reward Model），然后通过强化学习人类反馈（RLHF, Stiennon et al., 2020）或直接偏好优化（DPO, Rafailov et al., 2023）来对齐扩散模型的生成偏好。这种后训练（post-training）对齐策略使Midjourney的默认输出在构图、色彩、光影和艺术风格上具有高度的一致性。

**3. 内置提示词增强：** Midjourney在用户输入的文本prompt基础上自动进行扩充和增强——它可能使用了一个轻量级的语言模型来解析用户意图、推断缺失的细节（如光线方向、镜头类型、艺术媒介等），然后将增强后的prompt输入扩散模型。这一设计降低了普通用户的使用门槛，使其无需掌握复杂的prompt工程即可获得高质量的生成结果。

**4. 社区驱动的迭代：** Midjourney将其Discord社区作为核心产品迭代机制，通过分析用户在Discord中的交互数据（包括prompt、生成的图像、用户的反应等）来持续优化模型。这种反馈循环使Midjourney能够快速响应用户需求并持续改进生成质量。

Midjourney与Stable Diffusion代表了AI图像生成领域的两种不同发展路径：前者通过封闭的产品化策略追求一致的用户体验和商业成功，后者通过开放的开源生态激发社区的创新活力。两者的竞争与互补共同推动了整个领域的快速发展。

### 第6.9节 Google Imagen：级联扩散与T5语言理解

Google Research在2022年发表的Imagen（Saharia et al., 2022）代表了文本到图像生成的另一条技术路线。与DALL-E 2依赖CLIP进行文本-图像语义对齐不同，Imagen的核心设计哲学是：大规模语言模型的文本理解能力是文本到图像生成质量的关键瓶颈——一个强大的语言理解骨干网络（backbone）比增大图像生成模型或提高文本-图像对齐的粒度更能显著提升生成结果的质量。

基于这一洞察，Imagen使用了一个预训练的T5-XXL语言模型（约4.6B参数，Raffel et al., 2020）作为文本编码器，将输入文本prompt映射为语义丰富的嵌入序列。T5-XXL在C4（约750GB文本）上进行过预训练，具有强大的语言理解和常识推理能力。消融实验表明，用更强的语言模型作文本编码器（相对 CLIP 等较弱编码器）可显著改善文本对齐与生成质量；Imagen 在 COCO 上的主结果为 **FID-30K ≈ 7.27**（T5-XXL 设定），不宜写成“从 7.27 提升到 3.45”——7.27 即主结果而非待改进的基线。

Imagen采用了级联扩散架构（Cascaded Diffusion Architecture），包含三个依次递增分辨率的扩散模型：

1. **基础模型（Base Model）：** 以文本条件生成 $64 \times 64$ 分辨率的图像，使用一个条件U-Net（约2.3B参数）在像素空间中进行扩散。这个模型是生成流水线的核心，决定了生成图像的整体构图、物体布局和语义内容。

2. **超分辨率模型1（SR-64→256）：** 接收 $64 \times 64$ 的基础生成结果和文本条件，生成 $256 \times 256$ 的图像。这个模型的扩散过程也在像素空间中进行。

3. **超分辨率模型2（SR-256→1024）：** 接收 $256 \times 256$ 的结果和文本条件，最终生成 $1024 \times 1024$ 的高分辨率图像。这个模型使用了patch-based的扩散策略（将图像分成重叠的patch分别去噪），以降低单次前向传播的内存消耗。

每个超分辨率模型都使用了无分类器引导（CFG），引导强度 $w$ 在基础模型中取7.0，在两个超分辨率模型中取2.0（较低的引导强度以保持细节多样性）。Imagen 在 COCO 上的主结果为 **FID-30K ≈ 7.27**（其标准评估设定下的报告值）；不宜再拆成“无引导 7.27 / 有引导 4.0”这一对数字。人类评估中其文本对齐与图像保真度亦显著优于同期 DALL-E 2 等基线。

Imagen的一个重要技术贡献是对"深度合成文本"（deepfused text）的使用：在将文本prompt输入T5编码器时，将prompt重复拼接多次（如重复5次），使文本编码能够捕获更长距离的语义依赖关系。这一简单技巧在没有增加任何额外计算成本的情况下，显著提升了模型对复杂、描述性prompt的响应质量。

### 第6.10节 ControlNet：精确控制的条件扩散

ControlNet（Zhang et al., 2023）是2023年出现的一项关键技术，它解决了文本到图像生成中长期存在的一个核心痛点：**缺乏精确的空间控制能力**。在ControlNet出现之前，文本到图像模型的 conditioning 仅限于自然语言描述——用户只能通过修改prompt来间接影响生成结果，而无法精确指定物体的位置、姿态、边界框或深度布局。ControlNet通过引入可训练的副本网络（trainable copy），将任意用户指定的条件（如Canny边缘图、深度图、人体姿态骨架、语义分割图、涂鸦草图等）精确地注入到扩散模型的去噪过程中。

**架构设计：** ControlNet的核心架构包含两个分支：(a) **锁定分支（Locked Branch）**：冻结的原始扩散模型U-Net，保留其从大规模预训练中学到的通用生成能力；(b) **控制分支（Control Branch）**：一个可学习的副本，其结构与U-Net的编码器部分完全镜像，包含12个可训练的ControlNet块（每个块对应U-Net的一个残差块或注意力块）。

给定一个输入条件图 $\mathbf{c}$（如深度图），ControlNet首先通过一个包含四个卷积层的编码器（initial convolution layer）将其投影到与U-Net编码器相同分辨率的特征空间，然后通过12个ControlNet残差块进行处理。每个ControlNet块的输出通过零卷积（zero convolution）层与对应U-Net块的输出相加——零卷积是指权重和偏置都初始化为零的 $1 \times 1$ 卷积层，其初始输出为零，保证了在训练开始时ControlNet对U-Net的行为没有影响（即identity mapping），从而保留了预训练模型的已有能力。

形式化地，设U-Net编码器在第 $i$ 层的特征图为 $\mathbf{F}_i^{\text{UNet}}$，ControlNet在第 $i$ 层的输出为 $\mathbf{F}_i^{\text{Ctrl}}$，则修改后的U-Net前向传播为：

$$
\hat{\mathbf{F}}_i = \mathbf{F}_i^{\text{UNet}} + \mathbf{W}_i^{\text{zero}}(\mathbf{F}_i^{\text{Ctrl}})
$$

其中 $\mathbf{W}_i^{\text{zero}}$ 为零卷积层。在交叉注意力层中，ControlNet同样输出键（key）和值（value）向量，与U-Net自身的查询（query）进行交叉注意力计算：

$$
\text{CrossAttn}_{\text{controlled}}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V, \quad Q = \mathbf{F}^{\text{UNet}}, \quad [K; V] = [\mathbf{F}^{\text{Ctrl}}_K; \mathbf{F}^{\text{Ctrl}}_V]
$$

**训练策略：** ControlNet采用了一个高效的微调策略。首先，将一个小型数据集（如COCO或内部数据集，包含约数万到数十万张图像及其对应的条件标注）上的扩散模型进行端到端微调，其中只有ControlNet的参数被更新（U-Net参数冻结）。其次，使用小的batch size（如4-16）和学习率（如 $10^{-5}$）。对 Stable Diffusion 1.5 量级 UNet（约 860M），ControlNet 可训练副本通常约 **3.6×10^8** 参数量级（编码器副本+零卷积），远小于“1.5B 可训练 / 2.3B 总量”一类与 Imagen 大 UNet 混淆的数字。训练使用标准的噪声预测扩散损失（$L_{\mathrm{simple}}$ / MSE，见前文），仅在被掩码的干净图像和对应的条件图之间计算。

**实验结果：** ControlNet在多种条件类型上都取得了出色的结果。在Canny边缘图上，ControlNet能够精确生成符合边缘拓扑结构的图像；在深度图上，生成的图像具有与输入深度图一致的3D空间布局；在人体姿态骨架（OpenPose, Cao et al., 2019）上，ControlNet能够生成指定姿态的人物图像，且保持身体比例和关节角度的准确性；在语义分割图上，生成的图像在物体类别和空间分布上与分割图精确对应。更重要的是，ControlNet条件与文本prompt可以协同工作——用户可以使用文本指定物体的外观（如"一只橘色的猫"），同时使用深度图指定物体的位置（如"位于画面左侧，靠近一棵树"），两者共同引导生成过程。

ControlNet的发布被认为是文本到图像生成领域的"基础设施级"突破。截至2024年初，ControlNet已被应用于大量创意工作流中，并与Stable Diffusion、DALL-E等系统广泛集成。其技术思想——通过一个可训练的"控制副本"将外部条件注入预训练的生成模型——也被扩展到视频生成（ControlNet-Video）、3D生成（ControlNet-3D）等新领域。

## 第7节 视频理解：时空表征学习

视频理解是计算机视觉中最复杂、最具挑战性的子领域之一。与静态图像不同，视频同时包含空间（外观）和时间（运动）两个维度的信息，需要对它们进行有效的联合建模。近年来，随着Transformer架构在视觉领域的成功，视频理解领域也经历了一系列深刻的技术革新。

### 第7.1节 TimeSformer：时空注意力的可分分解

TimeSformer（Bertasius et al., 2021）是第一个将纯Transformer架构成功应用于大规模视频识别的模型。其核心贡献在于提出了一种时空注意力机制的可分分解（Factorized Spatio-Temporal Attention），在计算效率和建模能力之间取得了优异的平衡。

设输入视频为 $\mathbf{V} \in \mathbb{R}^{T \times H \times W \times C}$，其中 $T$ 为帧数（tubelet数量），$H$ 和 $W$ 为空间分辨率，$C$ 为通道数。与ViT类似，TimeSformer首先将视频划分为三维tubelet（时空块），每个tubelet的大小为 $\tau \times p \times p$（$\tau$ 为时间维度上的帧数，$p$ 为空间维度上的patch大小），然后将每个tubelet线性嵌入到隐藏维度 $D$：

$$
\mathbf{z}_0 = [\mathbf{v}_1 \mathbf{E}; \mathbf{v}_2 \mathbf{E}; \cdots; \mathbf{v}_N \mathbf{E}] + \mathbf{E}_{\text{pos}} \in \mathbb{R}^{N \times D}
$$

其中 $N = \frac{T}{\tau} \times \frac{H}{p} \times \frac{W}{p}$ 为tubelet总数。

TimeSformer的核心创新在于定义了四种不同的注意力模式：

1. **独占空间注意力（Divided Space-Time Attention）：** 在每个Transformer层中，先对所有tubelet的空间位置（固定时间帧）计算自注意力，然后对同一空间位置的不同时间帧计算自注意力。设 $\mathbf{z} \in \mathbb{R}^{N \times D}$ 为输入序列，$\mathbf{z}^{(s)} \in \mathbb{R}^{(T/\tau) \times (HW/p^2) \times D}$ 为重塑为时空网格的序列表示。空间注意力为：

$$
\mathbf{z}^{(s)'} = \text{Attention}(\text{LN}(\mathbf{z}^{(s)}), \mathbf{z}^{(s)}) + \mathbf{z}^{(s)}
$$

时间注意力为：

$$
\mathbf{z}^{(t)'} = \text{Attention}(\text{LN}(\mathbf{z}^{(s)'}), \mathbf{z}^{(s)'}) + \mathbf{z}^{(s)'}
$$

这种"先空间后时间"的分解将复杂度从 $O((THW)^2)$ 降低到 $O(T(HW)^2 + (HW)T^2)$，当 $HW \gg T$ 时（通常视频的空间分辨率远大于时间帧数），计算量大幅减少。

2. **独占时间注意力（Divided Time-Space Attention）：** 与上述相反，先计算时间注意力再计算空间注意力。
3. **联合时空注意力（Joint Space-Time Attention）：** 直接将所有tubelet展平后进行全局自注意力，复杂度为 $O((THW)^2)$，仅在计算资源充足时使用。
4. **全局空间注意力（Spatial Attention Only）：** 仅在每个时间帧内独立计算空间注意力，完全不建模时间关系。

实验表明，独占空间注意力（即TimeSformer默认配置）在Kinetics-400和Something-Something V2等基准上以较少的计算量超越了3D CNN（如SlowFast, Feichtenhofer et al., 2019）和ViViT（Arnab et al., 2021）等基线。

### 第7.2节 VideoMAE：视频领域的掩码自编码器

VideoMAE（Tong et al., 2022）将MAE的成功从静态图像扩展到了视频领域，并针对视频数据的特点进行了重要改进。与图像MAE类似，VideoMAE采用编码器-解码器架构，通过重建被掩码的tubelet来进行自监督预训练。然而，视频数据的时序冗余性使得简单的掩码策略需要重新设计。

**Tubelet掩码策略：** VideoMAE的关键创新在于其精心设计的tubelet掩码策略。设输入视频 clip 为 $\mathbf{V} \in \mathbb{R}^{3 \times T \times H \times W}$，被划分为 $N = \frac{T}{t} \times \frac{H}{p} \times \frac{W}{p}$ 个tubelet（每个tubelet大小为 $t \times p \times p$）。VideoMAE采用了两种掩码策略：

1. **随机 / tube 掩码：** 因视频存在强时序冗余，VideoMAE 反而采用**更高**的 tube 掩码比例（约 **90%–95%**），而非图像 MAE 常用的 75%。过高冗余下，过低掩码会使重建过易；更高掩码迫使模型学习跨时空结构。

2. **Tube masking：** 在时间维上对同一空间位置的 tubelet 整体掩码或保留，以强调运动与跨帧一致性。

VideoMAE v2（Tong et al., 2023）进一步引入基于帧采样的掩码策略，从原始视频中随机采样少量帧（如 4 或 8 帧）再做掩码，以提高训练效率。

**重建目标：** VideoMAE **默认重建像素**（或像素空间的归一化目标），与图像 MAE 一致；**HOG 重建**更常见于 MaskFeat 等路线，不宜写成 VideoMAE 的默认目标。重建损失可写为对被掩码 tubelet 的像素（或对应目标）均方误差：

$$
\mathcal{L}_{\text{VideoMAE}} = \frac{1}{N_{\text{mask}}} \sum_{i \in \mathcal{M}} \| \mathbf{v}_i - \text{Decoder}(\mathbf{z}_{\text{mask}, i}) \|^2
$$

实验结果表明，VideoMAE在Kinetics-400、Something-Something V2和UCF-101等视频理解基准上取得了优异的性能，且仅需少量标注数据微调即可达到甚至超越在完整数据集上监督预训练的方法。

### 第7.3节 Sora的技术原理推测

Sora是OpenAI于2024年2月发布的视频生成模型，其名称源自日语"天空"（そら）。Sora能够根据文本描述生成长达60秒的高分辨率（最高1080p）视频，展现出对复杂物理场景、多视角镜头切换和角色一致性的惊人理解。尽管OpenAI尚未公开Sora的技术细节（截至2024年7月），但其背后的技术路线可以从OpenAI的专利、技术报告和社区分析中进行合理的推测。

**1. 扩散 Transformer（Diffusion Transformer, DiT）：** 最强的推测是Sora基于扩散Transformer架构（Peebles & Xie, 2023）。DiT用Transformer块替换了传统扩散模型中的U-Net卷积块，将图像/视频的潜在特征图展平为序列，通过因果或非因果的自注意力建模全局依赖关系。DiT的生成质量随模型规模的增长表现出极强的可扩展性——从600M参数到12B参数，模型性能稳步提升。Sora很可能使用了DiT作为其核心去噪网络，且模型规模可能达到数十亿甚至上百亿参数。

**2. 时空潜在扩散：** 类似于Stable Diffusion在图像上的做法，Sora很可能在压缩的时空潜在空间中进行扩散。输入视频被编码为一个四维潜在张量（时间 × 高度 × 宽度 × 通道），扩散过程在这个压缩空间中进行，然后通过解码器重建为像素空间。这种设计使得生成高分辨率、长时序视频在计算上成为可能。

**3. 多尺度训练与长视频生成：** Sora能够生成长达60秒的视频，暗示其采用了某种形式的多尺度或级联训练策略。一种可能的方案是先在低帧率、低分辨率上进行扩散生成，然后通过时间超分辨率（Temporal Super-Resolution）和空间超分辨率（Spatial Super-Resolution）逐步提升时间分辨率和空间分辨率。另一种可能是使用了基于patch的并行生成策略（如Maska和MAR等工作中提出的），在潜在空间中以patch为单位并行生成，然后动态地组合成长视频。

**4. 语言-视频预训练：** Sora对文本prompt的理解能力远超此前任何视频生成模型，这表明它可能使用了大规模的语言-视频对比预训练（类似于CLIP，但在视频领域）。OpenAI可能在一个包含大量（文本，视频）对的大规模数据集上预训练了一个文本编码器和视频编码器，使得模型能够将复杂的文本描述（包括因果关系、空间关系、时间顺序等）精确映射到视频语义空间。

**5. 物理世界模拟器：** Sora最令人惊叹的特性是其对物理规律（如重力、碰撞、光影）的隐含理解。OpenAI将Sora描述为"世界模拟器"（world simulator）。这一描述暗示Sora可能不仅仅在学习视频的表面统计模式，而是在学习一个关于物理世界的内部模型（internal model）。从技术角度，这可能通过以下机制实现：(a) 训练数据中包含大量展示物理规律的视频（如物理实验、体育比赛、日常活动等），使模型隐含地学习了这些规律；(b) 使用了某种形式的物理一致性损失或3D感知训练目标（如深度估计、法线估计等辅助任务）；(c) 视频编码器中隐含了3D场景理解能力（如NeRF-like的神经辐射场表示）。

Sora的出现标志着视频生成领域的一个分水岭。如果其技术路线被后续研究所验证和扩展，视频生成模型可能从根本上改变电影制作、游戏开发、虚拟现实和科学模拟等领域的工作方式。

---

## 第8节 图像生成评估指标

评估生成模型的质量是确保其可靠应用的关键环节。然而，生成模型的评估本质上是一个开放性问题——"好的生成"不仅涉及图像的真实性和多样性，还涉及语义一致性、美学质量和实用性等多维度考量。在理想情况下，一个完备的评估框架应该同时衡量以下三个维度：(a) **保真度（Fidelity）**：生成图像在多大程度上与真实图像在视觉上 indistinguishable；(b) **多样性（Diversity）**：生成模型能够覆盖真实数据分布的多少模式；(c) **语义对齐（Semantic Alignment）**：在条件生成任务中，生成结果在多大程度上忠实于输入条件。目前，计算机视觉和生成式AI社区采用了一系列指标从不同角度刻画这三个维度，其中最为核心的包括Inception Score（IS）、Fréchet Inception Distance（FID）、Precision/Recall、**spatial FID（sFID，基于空间特征图而非池化向量；勿与 sliced FID 混淆）**和CLIP Score。

### 第8.1节 Inception Score（IS）

Inception Score（IS, Salimans et al., 2016）是较早被广泛采用的生成模型评估指标，其设计基于两个直觉：好的生成图像应该具有清晰可辨的类别（高置信度），且生成模型应该产生多样化的类别（高熵）。

具体地，给定一个在ImageNet上预训练的Inception-v3分类器，对于 $N$ 张生成图像，提取每张图像的类别条件分布 $p(y | \mathbf{x})$（即Softmax输出，$y \in \{1, \ldots, 1000\}$ 为ImageNet类别标签）。IS定义为：

$$
\text{IS} = \exp\left( \mathbb{E}_{\mathbf{x} \sim P_{\text{gen}}} \left[ D_{\text{KL}}(p(y | \mathbf{x}) \| p(y)) \right] \right) = \exp\left( \frac{1}{N} \sum_{i=1}^N \sum_{y=1}^K p(y | \mathbf{x}_i) \log \frac{p(y | \mathbf{x}_i)}{p(y)} \right)
$$

其中 $p(y) = \mathbb{E}_{\mathbf{x} \sim P_{\text{gen}}}[p(y | \mathbf{x})] = \frac{1}{N} \sum_{i=1}^N p(y | \mathbf{x}_i)$ 为生成的边缘类别分布，$D_{\text{KL}}$ 为KL散度。IS的取值范围通常在1到1000之间（理论上限为类别数 $K$，当生成图像每张都被完美分类到不同类别时IS达到最大值）。

IS的数学含义可以从信息论角度深入理解。KL散度 $D_{\text{KL}}(p(y | \mathbf{x}) \| p(y))$ 衡量了条件分布 $p(y | \mathbf{x})$ 与边缘分布 $p(y)$ 之间的互信息（mutual information）：$I(y; \mathbf{x}) = D_{\text{KL}}(p(y, \mathbf{x}) \| p(y)p(\mathbf{x})) = D_{\text{KL}}(p(y | \mathbf{x}) \| p(y))$。当每张生成图像都被分类器以高置信度分类到某一类别（即 $p(y | \mathbf{x})$ 接近独热分布），且不同图像覆盖了多样化的类别（即 $p(y)$ 接近均匀分布）时，互信息最大化，IS也最大化。

IS的局限性包括：

1. **无法评估与真实分布的匹配度：** IS不涉及真实图像分布 $P_{\text{real}}$，因此生成模型可能通过在少数类别上生成高质量图像而获得高分，同时完全遗漏其他类别——即IS对模式崩溃（mode collapse）不敏感。
2. **对类别不平衡敏感：** 如果生成模型的类别分布 $p(y)$ 本身不均匀（某些类别天然更易生成），IS的解释会变得模糊。
3. **依赖预训练分类器：** IS的值取决于特定Inception-v3模型的分类能力，不同模型版本或不同层的IS值会有所不同。
4. **不适合开放词汇生成：** 对于文本到图像生成，生成图像可能包含ImageNet类别系统中不存在的物体或场景，IS无法对其进行评估。

在实践中，IS通常与FID配合使用，前者提供类别级别的清晰度和多样性评估，后者提供分布级别的整体匹配度评估。

### 第8.2节 Fréchet Inception Distance（FID）

Fréchet Inception Distance（FID, Heusel et al., 2017）是当前图像生成领域最广泛使用的质量评估指标。FID通过比较真实图像分布和生成图像分布之间的Fréchet距离来衡量生成质量，其理论基础源于概率几何中的2-Wasserstein距离（2-Wasserstein Distance）。

设真实图像分布为 $P_{\text{real}}$，生成图像分布为 $P_{\text{gen}}$。FID首先使用一个在ImageNet上预训练的Inception-v3网络的pool3层（输出维度为2048）提取图像的特征表示。令 $\mathbf{x} \sim P_{\text{real}}$ 和 $\mathbf{y} \sim P_{\text{gen}}$ 分别为从两个分布中独立采样得到的特征向量。假设这些特征向量近似服从多元高斯分布：

$$
\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_{\text{real}}, \boldsymbol{\Sigma}_{\text{real}}), \quad \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_{\text{gen}}, \boldsymbol{\Sigma}_{\text{gen}})
$$

则FID定义为两个高斯分布之间的2-Wasserstein距离：

$$
\text{FID} = \|\boldsymbol{\mu}_{\text{real}} - \boldsymbol{\mu}_{\text{gen}}\|^2 + \text{Tr}\left(\boldsymbol{\Sigma}_{\text{real}} + \boldsymbol{\Sigma}_{\text{gen}} - 2\sqrt{\boldsymbol{\Sigma}_{\text{real}} \boldsymbol{\Sigma}_{\text{gen}}}\right)
$$

其中 $\|\cdot\|$ 为Frobenius范数（Frobenius Norm），$\text{Tr}(\cdot)$ 为矩阵的迹（trace），$\sqrt{\boldsymbol{\Sigma}_{\text{real}} \boldsymbol{\Sigma}_{\text{gen}}}$ 为矩阵平方根（Matrix Square Root）——若对 $\boldsymbol{\Sigma}_{\text{real}} \boldsymbol{\Sigma}_{\text{gen}}$ 进行特征值分解得到 $\mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top$，则 $\sqrt{\boldsymbol{\Sigma}_{\text{real}} \boldsymbol{\Sigma}_{\text{gen}}} = \mathbf{U} \boldsymbol{\Lambda}^{1/2} \mathbf{U}^\top$。

FID的第一项 $\|\boldsymbol{\mu}_{\text{real}} - \boldsymbol{\mu}_{\text{gen}}\|^2$ 衡量了两个分布均值向量之间的距离，反映生成图像在特征空间中的"中心位置"是否偏离了真实图像——如果生成图像整体偏亮、偏暗或风格偏差，这一项会增大。FID的第二项 $\text{Tr}(\boldsymbol{\Sigma}_{\text{real}} + \boldsymbol{\Sigma}_{\text{gen}} - 2\sqrt{\boldsymbol{\Sigma}_{\text{real}} \boldsymbol{\Sigma}_{\text{gen}}})$ 衡量了两个分布协方差矩阵之间的Frobenius距离，反映生成图像的"散布范围"是否与真实图像一致——如果模型发生模式崩溃（只生成少数几种模式），$\boldsymbol{\Sigma}_{\text{gen}}$ 会在某些方向上收缩，导致这一项增大。

FID值越低表示生成质量越好。在实践中，FID通常在10-200范围内——顶级的文本到图像模型（如SDXL, Flux）在COCO等基准上的FID约为2-5，而早期GAN模型的FID通常在20-50之间。FID的估计需要足够多的样本：真实分布 $P_{\text{real}}$ 的统计量通常从至少10000张参考图像中估计，生成分布 $P_{\text{gen}}$ 的统计量从至少10000张生成图像中估计——样本不足会导致统计量的估计偏差增大，FID值不稳定。

### 第8.3节 Precision与Recall

Kynkäänniemi等人（2019）指出，FID虽然综合了保真度和多样性信息，但无法区分两种不同的失败模式：**低劣模式覆盖（low coverage）**——生成图像只在真实分布的少数区域附近（如只生成数字"1"）和**低近邻质量（low neighbor quality）**——生成图像虽然覆盖了真实分布的大多数模式，但生成的图像质量很差（如生成模糊的、不完整的物体）。为了将这两个维度解耦，他们提出了基于流形（Manifold）的Precision和Recall指标。

设真实数据分布 $P_{\text{real}}$ 的支撑集（support）为一个 $d$ 维流形 $\mathcal{M}_{\text{real}} \subset \mathbb{R}^D$（通常 $d \ll D$，如 $d=128$ 维语义流形嵌入在 $D=2048$ 维的Inception特征空间中），生成分布 $P_{\text{gen}}$ 的支撑集为 $\mathcal{M}_{\text{gen}}$。在特征空间中，Precision衡量生成流形与真实流形的重叠质量：

$$
\text{Precision} = \mathbb{E}_{\mathbf{y} \sim P_{\text{gen}}} \left[ \mathbb{I}(\mathbf{y} \in \mathcal{M}_{\text{real}}) \right] \approx \frac{1}{N_{\text{gen}}} \sum_{i=1}^{N_{\text{gen}}} \mathbb{I}(\text{dist}(\mathbf{y}_i, \mathcal{M}_{\text{real}}) < \epsilon)
$$

其中 $\mathbb{I}(\cdot)$ 为指示函数，$\epsilon$ 为给定的邻域半径。Precision回答的问题是：生成图像在多大程度上落在真实数据流形的邻域内？Recall衡量真实流形被生成流形覆盖的程度：

$$
\text{Recall} = \mathbb{E}_{\mathbf{x} \sim P_{\text{real}}} \left[ \mathbb{I}(\mathbf{x} \in \mathcal{M}_{\text{gen}}) \right] \approx \frac{1}{N_{\text{real}}} \sum_{i=1}^{N_{\text{real}}} \mathbb{I}(\text{dist}(\mathbf{x}_i, \mathcal{M}_{\text{gen}}) < \epsilon)
$$

Recall回答的问题是：真实数据的多样性在多大程度上被生成模型所覆盖？

一个理想的生成模型应该同时具有高Precision和高Recall（即 $\mathcal{M}_{\text{gen}} \approx \mathcal{M}_{\text{real}}$）。然而，这两个指标之间存在一种权衡（trade-off）——提高引导强度 $w$ 通常会增加Precision（生成图像更接近真实模式）但降低Recall（覆盖的模式更少），而降低 $w$ 则相反。在图像生成实践中，研究者通常关注 **F1 Score**（Precision和Recall的调和平均）作为综合指标。

**密度和覆盖度（Density and Coverage）：** Naeem et al.（ICML 2020）提出了改进的Density和Coverage指标，使用k近邻（k-NN）来估计流形的局部密度，解决了原始Precision/Recall在密度不均匀流形上的偏差问题。Density衡量生成点周围真实点的局部密度（反映生成质量的局部集中度），Coverage衡量真实流形被生成点覆盖的比例。

### 第8.4节 Spatial Fréchet Inception Distance（sFID）

标准FID使用Inception网络全局池化后的2048维特征向量，主要刻画图像的整体语义统计，但对空间布局、局部结构与高频纹理不够敏感——两个语义类别相近但空间结构明显不同的图像分布，可能仍得到相近的FID。生成模型文献中的 **sFID（spatial FID）** 指的是在Inception网络**中间层保留空间维的特征图**上计算的Fréchet距离，而不是将高维特征随机投影到一维再平均的 sliced FID（Nielsen 等的一维投影方法属于另一类估计技巧，不宜与此处的sFID混用）。

具体地，设Inception某一中间层输出空间特征图 $\mathbf{F} \in \mathbb{R}^{H' \times W' \times C'}$（实践中常取较早的卷积层特征，例如保留空间分辨率的 feature map），将其在空间位置上展平为样本集合，再对真实与生成两组特征分别估计均值与协方差，按与标准FID相同的2-Wasserstein闭式公式计算：

$$
\text{sFID} = \|\boldsymbol{\mu}_{\text{real}}^{\text{sp}} - \boldsymbol{\mu}_{\text{gen}}^{\text{sp}}\|^2 + \operatorname{Tr}\!\left(\boldsymbol{\Sigma}_{\text{real}}^{\text{sp}} + \boldsymbol{\Sigma}_{\text{gen}}^{\text{sp}} - 2\bigl(\boldsymbol{\Sigma}_{\text{real}}^{\text{sp}}\boldsymbol{\Sigma}_{\text{gen}}^{\text{sp}}\bigr)^{1/2}\right)
$$

其中上标 $\text{sp}$ 表示由**空间特征图**估计的统计量。sFID的实践意义在于：它补充了全局池化FID对空间结构的盲区，常与FID一并报告以同时反映语义保真与空间/纹理质量；在StyleGAN、ADM等生成模型评估中，sFID与FID的相对高低有时并不一致，可揭示模型是否以牺牲空间一致性换取全局语义统计上的改善。

### 第8.5节 CLIP Score

CLIP Score（Ramesh et al., 2022; Hessel et al., 2021）是基于CLIP模型的文本-图像对齐评估指标，专门用于衡量生成图像与文本描述之间的语义一致性。在文本到图像生成任务中，FID和IS评估的是生成图像的整体质量（相对于训练分布），但不直接评估生成图像是否忠实地反映了文本prompt的语义内容。CLIP Score填补了这一关键空白——一个模型可能在FID上表现优异（生成图像质量高且多样化），但如果生成的图像与prompt描述的语义不一致（如prompt说"一只猫在沙发上"但生成的是一只狗在床上），FID无法捕捉到这种语义偏差。

给定一个文本描述 $t$ 和一张生成图像 $x$，CLIP Score定义为：

$$
\text{CLIP Score}(t, x) = \max(0, 100 \cdot \langle f_{\text{text}}(t), f_{\text{image}}(x) \rangle)
$$

其中 $f_{\text{text}}$ 和 $f_{\text{image}}$ 分别为CLIP的文本编码器和图像编码器，$\langle \cdot, \cdot \rangle$ 为余弦相似度，乘以100是为了将分数映射到一个更直观的范围（0-100）。CLIP Score的取值范围通常在20（语义完全不匹配）到35+（语义高度一致）之间。

对于大规模评估，给定一个包含 $N$ 个（文本，图像）对的测试集，报告平均CLIP Score：

$$
\overline{\text{CLIP Score}} = \frac{100}{N} \sum_{i=1}^N \langle f_{\text{text}}(t_i), f_{\text{image}}(x_i) \rangle
$$

对于图像描述生成（image captioning）任务，Hessel等人（2021）提出的CLIP Score可以不需要参考caption，直接衡量生成caption与图像之间的语义相似度——这在评估开放词汇图像描述时尤为有用。

CLIP Score的理论优势在于：

1. **开放词汇评估：** 由于CLIP在4亿个（图像，文本）对上进行了预训练，它能够理解任意自然语言描述，评估不局限于预定义类别。
2. **语义级评估：** CLIP Score衡量的是语义对齐而非像素级匹配——即使生成图像在像素上与参考图像不同，只要语义内容一致，CLIP Score仍可很高。
3. **无需参考图像：** 对于文本到图像任务，通常没有"标准答案"参考图像，CLIP Score仅需文本prompt即可进行评估。
4. **可微分：** CLIP Score是可微分的，因此可以直接作为损失函数用于模型微调（如在DPO或RLHF中优化生成质量）。

CLIP Score的局限性包括：

1. **与人类判断的偏差：** CLIP Score与人类对生成质量的主观判断并不完全一致——研究表明，CLIP Score与人类偏好的Spearman相关性约为0.5-0.6，仍有提升空间。有时，生成图像可能与文本具有高CLIP Score但存在明显的视觉伪影或逻辑错误（如不合比例的身体部件）。
2. **对表面特征的偏好：** CLIP可能对图像中的表面特征（颜色、纹理、物体存在性）比对复杂关系（空间布局、计数、动作）赋予更高的权重——"一只红色的苹果在蓝色盘子上"可能获得与"一只苹果在盘子上"相同的CLIP Score，尽管两者语义不同。
3. **模态瓶颈：** CLIP的嵌入空间是在预训练时固定的，它可能无法捕捉到后续生成模型的新能力或新的视觉概念——当生成模型能够产生训练时未见过的视觉概念时，CLIP可能无法准确评估。

在实践中，全面评估文本到图像生成模型通常采用"FID + CLIP Score"的组合：FID衡量生成图像的保真度和多样性（相对于参考分布），CLIP Score衡量生成图像与文本提示的语义对齐度。这两个指标从不同角度提供了对生成模型性能的多维度刻画。近期，一些研究者提出了结合人类偏好评估（human preference evaluation）的元评估（meta-evaluation）框架，将自动指标与人类判断进行校准，以获得更可靠的模型比较。

### 第8.6节 生成模型的系统比较：扩散模型、GAN与自回归模型

图像生成领域目前并存着三大技术范式：生成对抗网络（GAN）、自回归模型（Autoregressive Model）和扩散模型。理解这三者的本质差异和各自的优劣势，对于选择合适的技术路线至关重要。

**生成对抗网络（GAN）：** GAN（Goodfellow et al., 2014）通过生成器（Generator）和判别器（Discriminator）之间的对抗博弈进行训练。生成器试图生成逼真的图像以欺骗判别器，判别器试图区分真实图像和生成图像。这一极小极大博弈的优化目标为：

$$
\min_G \max_D \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]
$$

GAN的核心优势在于采样效率——一旦训练完成，生成一张图像仅需一次前向传播（几毫秒到几十毫秒），这在实时应用中具有不可替代的优势。StyleGAN系列（Karras et al., 2019, 2020）在面部生成等受限域中达到了照片级的真实感。然而，GAN面临三个根本性挑战：模式崩溃（mode collapse）——生成器只能覆盖真实分布的少数模式；训练不稳定性——生成器和判别器的平衡极难维持，容易出现模式震荡；以及难以进行灵活的条件控制。

**自回归模型：** 自回归模型（如PixelRNN, Van den Oord et al., 2016; DALL-E, Ramesh et al., 2021; Parti, Yu et al., 2022）将图像生成视为序列建模问题。其核心优势在于精确的似然度计算——由于链式法则，模型可以精确计算数据的对数似然。自回归模型的主要瓶颈是采样速度——生成一张图像需要 $N$ 次顺序前向传播。

**扩散模型：** 扩散模型的核心优势在于训练稳定性（简单的MSE损失）、生成质量（各类基准上 consistently 最优）和条件灵活性（CFG支持任意条件信号）。扩散模型的主要劣势是采样速度——即使使用DDIM加速到20步，仍需要数十次前向传播。Consistency Model（Song et al., 2023）和Distilled Diffusion Model（Meng et al., 2023）等工作正在努力缩小这一差距。

截至2024年，扩散模型已成为图像生成领域的主导范式。GAN在需要极低延迟的应用中仍有不可替代的价值。自回归模型在需要精确似然度估计的场景中具有独特优势。三种范式的融合（如GAN判别器辅助扩散训练、自回归+扩散的级联生成）是当前的研究前沿。

### 第8.7节 视频生成模型的评估

视频生成模型的评估比图像生成更具挑战性，因为它需要在时空两个维度上同时评估保真度、一致性和语义对齐。目前，视频生成社区主要采用以下评估指标：

**FVD（Fréchet Video Distance）：** FVD（Unterthiner et al., 2018）是FID在视频领域的直接推广。它使用一个在Kinetics上预训练的I3D网络（Carreira & Zisserman, 2017）提取视频的时空特征，然后计算真实视频分布和生成视频分布之间的2-Wasserstein距离：

$$
\text{FVD} = \|\boldsymbol{\mu}_{\text{real}}^{\text{I3D}} - \boldsymbol{\mu}_{\text{gen}}^{\text{I3D}}\|^2 + \text{Tr}\left(\boldsymbol{\Sigma}_{\text{real}}^{\text{I3D}} + \boldsymbol{\Sigma}_{\text{gen}}^{\text{I3D}} - 2\sqrt{\boldsymbol{\Sigma}_{\text{real}}^{\text{I3D}} \boldsymbol{\Sigma}_{\text{gen}}^{\text{I3D}}}\right)
$$

I3D网络通过将2D Inflated 3D Conv（将2D卷积核在时间维度上膨胀）预训练在Kinetics-400上，能够提取同时编码空间外观和时间运动的特征。FVD已经被广泛采用为视频生成模型的 标准基准——较低的FVD表示生成视频在时空维度上都与真实视频分布更接近。

**tFID（Temporal FID）：** 另一种方法是分别对每一帧计算FID，然后取时间上的平均值或最小值。tFID的优势在于可以直接利用成熟的图像级FID评估工具，但缺点是忽略了帧间的时间一致性和运动质量。

**文本-视频CLIP Score：** 类似于图像的CLIP Score，文本-视频CLIP Score衡量生成视频与文本描述之间的语义对齐度。具体地，使用视频编码器（如ViViT或TimeSformer提取的全局视频特征）和文本编码器（如CLIP文本编码器）计算余弦相似度。由于视频包含时间信息，视频编码器需要能够捕获整个视频clip的语义内容（如"一只猫在追一只蝴蝶"）。

Sora的评估采用了上述所有指标的组合——FVD评估视频质量，tFID评估逐帧质量，文本-视频CLIP Score评估语义对齐，加上大量的人类评估（human evaluation）来衡量物理一致性、时间连贯性和美学质量。OpenAI在其技术报告中特别强调了人类评估的重要性，因为当前自动指标与人类对视频质量的判断之间仍然存在显著差距——尤其是在评估物理规律遵循度、因果关系合理性和长期时间一致性方面。

---

## 第9节 总结与展望

本章系统地回顾了2020至2024年间计算机视觉领域最具变革性的技术进展。从Vision Transformer首次证明了纯注意力机制在视觉任务上的可行性，到DeiT通过知识蒸馏和增强策略显著降低了Transformer对数据的渴求度；从Swin Transformer通过层次化和移位窗口解决了ViT的计算复杂度和多尺度问题，到MAE以自监督学习范式重新定义了视觉预训练的标准——Transformer架构已经深刻改变了视觉理解领域的研究范式和技术路线。在生成领域，扩散模型凭借其坚实的数学基础和卓越的生成质量，从一项学术 curiosity 演变为支撑着数十亿美元产业的生成引擎。DALL-E、Stable Diffusion和Midjourney等系统不仅证明了文本到图像生成的商业可行性，更预示着人机交互方式的根本性变革。在视频理解领域，TimeSformer、VideoMAE和Sora等技术正在快速缩小机器与人类在动态视觉理解和生成上的差距。

展望未来，以下几个方向值得重点关注。第一，**多模态大模型**正在成为新的前沿——LLaVA、GPT-4V和Gemini等系统表明，视觉理解、生成和推理可以在一个统一的框架中进行。未来的视觉系统将不再局限于单一的识别或生成任务，而是能够在理解的基础上进行创造性的视觉输出。第二，**3D和4D生成**——从NeRF到3D Gaussian Splatting，结合扩散模型，三维内容生成正处于爆发期，可能彻底改变影视制作、游戏开发和虚拟现实的内容生产方式。第三，**世界模型和物理推理**——Sora所展示的"世界模拟器"方向可能引领AI系统从模式匹配走向真正的物理理解和推理，这对于自动驾驶、机器人操作和科学模拟等应用具有深远意义。第四，**效率和可持续性**——随着模型规模持续增长（从B级到T级参数），如何降低训练和推理的计算成本、减少碳足迹，将是一个兼具技术挑战和社会责任的重要课题。第五，**安全与对齐**——随着图像和视频生成模型的能力越来越强，如何防止滥用（如深度伪造、虚假信息传播）同时保持技术的创造性价值，需要技术社区、政策制定者和公众的共同参与。第六，**评估科学**——当前自动评估指标（FID、IS、CLIP Score）与人类判断之间仍存在显著差距，发展更可靠、更全面的自动评估方法——甚至通用的"视觉图灵测试"——是确保生成模型健康发展的基础性需求。

最后，值得强调的是，本章所述技术突破的大规模落地离不开开源生态系统的支撑。Hugging Face的Transformers库、Diffusers库、LAION的开源数据集、GitHub社区的无数微调脚本和教程——这些基础设施使得全球的研究者、开发者和艺术家都能够站在巨人的肩膀上继续创新。Vision Transformer、Swin Transformer、Stable Diffusion等里程碑级工作的开源发布，直接催生了数以千计的衍生研究和应用产品，充分证明了开放科学（open science）在加速技术进步中的不可替代作用。从ViT到Swin Transformer，从DDPM到Stable Diffusion，从TimeSformer到Sora——这一系列突破不仅刷新了学术基准的数字，更重要的是重新定义了人与机器视觉交互的可能性边界。站在2026年的节点回望，计算机视觉的新纪元已经全面开启，而它的下一篇章，正等待被书写。

---

## 参考文献

1. Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., & Schmid, C. (2021). ViViT: A Video Vision Transformer. ICCV 2021.
2. Betker, J., Goh, G., Zhang, L., et al. (2023). Improving Image Generation with Better Captions. OpenAI Technical Report.
3. Bertasius, G., Wang, H., & Torresani, L. (2021). Is Space-Time Attention All You Need for Video Understanding? ICML 2021.
4. Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language Models are Few-Shot Learners. NeurIPS 2020.
5. Dalal, N., & Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. CVPR 2005.
6. Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.
7. DeVries, T., & Taylor, G. W. (2017). Improved Regularization of Convolutional Neural Networks with Cutout. arXiv:1708.04552.
8. Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. NeurIPS 2021.
9. Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
10. Feichtenhofer, C., Fan, H., Malik, J., & He, K. (2019). SlowFast Networks for Video Recognition. ICCV 2019.
11. Feichtenhofer, C., Li, Y., Liu, J., et al. (2023). MAE-ST: Spectral-Temporal Masked Autoencoders for Self-Supervised Audio-Visual Pre-Training. arXiv:2212.12105.
12. Gal, R., Alaluf, Y., Atzmon, Y., et al. (2022). An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. arXiv:2208.01618.
13. Goodfellow, I., Pouget-Abadie, J., Mirza, M., et al. (2014). Generative Adversarial Networks. NeurIPS 2014.
14. He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. ICCV 2017.
15. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
16. He, K., Chen, T., Xie, S., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR 2022.
17. Heusel, M., Ramsauer, H., Unterthiner, T., et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. NeurIPS 2017.
18. Hessel, J., Holtzman, A., Forbes, M., et al. (2021). CLIPScore: A Reference-free Evaluation Metric for Image Captioning. EMNLP 2021.
19. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. NeurIPS 2014 Deep Learning Workshop.
20. Ho, J., Chan, W., Saharia, C., et al. (2022). Imagen Video: High Definition Video Generation with Diffusion Models. arXiv:2210.02303.
21. Ho, J., & Salimans, T. (2021). Classifier-Free Diffusion Guidance. NeurIPS 2021 Workshop.
22. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS 2020.
23. Hu, E. J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
24. Huang, G., Sun, Y., Liu, Z., et al. (2016). Deep Networks with Stochastic Depth. ECCV 2016.
25. Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR 2014.
26. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. NeurIPS 2012.
27. Lin, T., Dollár, P., Girshick, R., et al. (2017). Feature Pyramid Networks for Object Detection. CVPR 2017.
28. Liu, S., Qi, L., Qin, H., et al. (2018). Path Aggregation Network for Instance Segmentation. CVPR 2018.
29. Liu, Z., Lin, Y., Cao, Y., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV 2021.
30. Liu, Z., Ning, J., Cao, Y., et al. (2022). Video Swin Transformer. CVPR 2022.
31. Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR 2017.
32. Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. ICML 2021.
33. OpenAI (2024). Sora Technical Report. https://openai.com/sora
34. Peebles, W., & Xie, S. (2023). Scalable Diffusion Models with Transformers. ICCV 2023.
35. Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.
36. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving Language Understanding by Generative Pre-Training. OpenAI Technical Report.
37. Ramesh, A., Dhariwal, P., Nichol, A., et al. (2022). Hierarchical Text-Conditional Image Generation with CLIP Latents. arXiv:2204.06125.
38. Ramesh, A., Pavlov, M., Goh, G., et al. (2021). Zero-Shot Text-to-Image Generation. ICML 2021.
39. Rombach, R., Blattmann, A., Lorenz, D., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
40. Ruiz, N., Li, Y., Jampani, V., et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. CVPR 2023.
41. Salimans, T., Goodfellow, I., Zaremba, W., et al. (2016). Improved Techniques for Training GANs. NeurIPS 2016.
42. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR 2015.
43. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics. ICML 2015.
44. Song, J., Meng, C., & Ermon, S. (2021). Denoising Diffusion Implicit Models. ICLR 2021.
45. Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. A. (2017). Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning. AAAI 2017.
46. Szegedy, C., Liu, W., Jia, Y., et al. (2015). Going Deeper with Convolutions. CVPR 2015.
47. Szegedy, C., Vanhoucke, V., Ioffe, S., et al. (2016). Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
48. Tong, Z., Song, Y., Wang, J., & Wang, L. (2022). VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. NeurIPS 2022.
49. Tong, Z., Song, Y., Wang, J., & Wang, L. (2023). VideoMAE v2: Scaling Video Masked Autoencoders with Dual Masking Strategy. CVPR 2023.
50. Touvron, H., Cord, M., Douze, M., et al. (2020). Training Data-Efficient Image Transformers & Distillation through Attention. ICML 2021.
51. Van den Oord, A., Kalchbrenner, N., Kavukcuoglu, K., et al. (2016). PixelRNN: Generating Images One Pixel at a Time. ICML 2016.
51. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. NeurIPS 2017.
