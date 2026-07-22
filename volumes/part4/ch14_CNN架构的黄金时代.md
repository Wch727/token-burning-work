# 第14章 CNN架构的黄金时代：VGG、GoogLeNet、ResNet

## 第1节 引言：深度革命的序章

2012年AlexNet在ImageNet上的突破性表现标志着深度学习进入了崭新纪元，但真正将卷积神经网络从概念验证推向工程艺术的，是2014至2016年间发生的一场静默而深刻的架构革命。要理解这一革命的紧迫性，需要回顾2012-2014年间的技术背景。AlexNet（Krizhevsky等人，2012）在ILSVRC 2012上将Top-5错误率从26.1%（传统方法）降至16.4%，证明了GPU训练深度卷积网络的可行性。然而AlexNet的架构设计相当特化——两个GPU之间通过交叉连接通信，不同层使用不同大小的卷积核，全连接层占据了绝大部分参数。研究者面临的核心挑战是：AlexNet的突破究竟来自于"深度"本身，还是来自于其特定的架构选择？如果更深能带来更好的性能，那么"更深"的上限在哪里？有没有系统性的方法可以设计更深的网络？

这些问题的答案需要从架构层面寻找。VGGNet（Simonyan和Zisserman，2014）通过极致的简洁主义回答了第一个问题——仅用3×3卷积堆叠即可达到与AlexNet相当甚至更优的性能，证明深度确实是关键因素。GoogLeNet（Szegedy等人，2014）通过Inception模块回答了第二个问题——通过多尺度并行路径和高效的通道混合，可以在保持计算预算可控的前提下大幅增加网络的表征能力。ResNet（He等人，2016）则从根本上回答了第三个问题——通过残差连接，理论上可以训练无限深的网络，优化的瓶颈不再是表达能力而是计算资源。

**硬件与架构的协同演化。** CNN架构的黄金时代与GPU计算能力的爆发式增长高度同步。NVIDIA的Kepler架构（2012-2014）为AlexNet和VGG的训练提供了必要的计算能力——Tesla K40 GPU（2013）的12GB GDDR5显存使得batch size=128甚至256的训练成为可能。Tesla K80（2014，双GPU设计）将可用显存翻倍，使得VGG-19等大模型的训练更加可行。Pascal架构（2016，Tesla P100）引入了16nm FinFET工艺和NVLink互联，显存提升至16GB HBM2，带宽高达720 GB/s——这为ResNet-152等超大模型的训练提供了硬件基础。没有这些硬件进步，即使有完美的架构设计，训练也不可能完成。

这一协同演化在ImageNet训练时间上得到了直观体现：AlexNet（2012）在两个GTX 580 GPU（3GB显存）上训练约5-6天；VGG-16（2014）在4块K40 GPU上训练约2-3周；ResNet-152（2016）在8块P100 GPU上训练约1-2周。训练时间的缩短部分来自于架构优化（更高效的卷积设计减少了每epoch的计算量），部分来自于硬件加速（更快的GPU），部分来自于软件优化（更高效的CUDA内核和cuDNN库）。这三者的协同作用是深度学习快速发展的核心驱动力。

**ImageNet竞赛的演进与黄金时代的定位。** ILSVRC从2010年举办至2017年，每年吸引全球顶尖的研究团队参与竞争。2010–2011年的冠军使用传统的计算机视觉方法（SIFT特征 + Fisher向量 + SVM分类），Top-5错误率分别约为28%与25.8%。2012年AlexNet的16.4%标志着深度学习时代的到来。2014年VGGNet（7.3%）和GoogLeNet（6.67%）将错误率降至个位数。2015年ResNet集成（3.57% Top-5测试错误率）首次显著超越人类水平。2016年更强的集成方法将错误率进一步压低。2017年SENet（2.25%）标志着ImageNet竞赛的"人类水平"极限已被彻底突破。从2018年开始，ILSVRC不再举办竞赛——因为ImageNet分类任务的"基准"问题已被认为基本解决，研究焦点转向了更复杂的视觉任务（检测、分割、视频理解）和新的研究范式（自监督学习、少样本学习）。黄金时代的架构创新——VGG、GoogLeNet、ResNet——正是这一演进过程中最关键的几个里程碑。

这一时期的贡献远不止是若干更优模型的集合。VGGNet展示了极简主义美学——仅用3×3卷积堆叠即可构建出极具表现力的深度表征器；GoogLeNet以Inception模块打破了传统卷积网络的层状刚性，在保持计算预算可控的前提下极大地拓宽了网络宽度；ResNet则以革命性的残差连接驯服了梯度消失恶魔，使得训练超过一百层甚至上千层的网络成为可能。这三条路径看似殊途，实则同归——它们共同揭示了一个根本性的洞见：**深度神经网络的优化，本质上是一个关于信息流与梯度流的设计问题。**

与此同时，批归一化（Batch Normalization）为解决内部协变量偏移提供了优雅的统计学框架，深度可分离卷积为后续轻量级网络奠定了结构基础，神经架构搜索（NAS）则开启了自动化模型设计的序幕。所有这些工作，共同构成了卷积神经网络发展的"黄金时代"。

---

## 第2节 VGGNet：深度与简洁的美学

### 第2.1节 背景与动机

AlexNet的成功证明了深度卷积神经网络的巨大潜力，但其架构设计相对碎片化——不同层使用了不同大小的卷积核（11×11、5×5、3×3），全连接层与卷积层交替出现，整体结构缺乏统一的设计范式。Simonyan和Zisserman（2014）在其开创性工作中提出了一个根本性问题：**如果我们放弃一切非必要设计，仅用最基础的构件——3×3卷积和最大池化——反复堆叠，能走多远？**

答案出人意料地令人振奋。

### 第2.2节 VGG架构详解

VGGNet有两个经典变体：VGG-16（13个卷积层+3个全连接层，共16层有权重）和VGG-19（16个卷积层+3个全连接层，共19层有权重）。两者结构完全相同，仅深度不同。VGG的核心设计哲学可概括为以下三条原则：

**原则一：统一的3×3卷积核。** 整个网络中所有卷积层统一使用3×3卷积核，步长为1，padding为1，保持空间分辨率不变。这一选择绝非偶然，而是基于 receptive field（感受野）的严谨计算。两个连续的3×3卷积层的感受野为5×5，三个连续的3×3卷积层的感受野为7×7——与使用单个7×7卷积核具有相同的感受野，但参数数量大幅减少。

设输入通道数为C，输出通道数也为C（为简化分析）。单个3×3卷积层的参数量为：

$$
W_{3\times3} = 3 \times 3 \times C \times C = 9C^2
$$

两个3×3卷积层的参数量为：

$$
2 \times W_{3\times3} = 2 \times 9C^2 = 18C^2
$$

而单个7×7卷积层的参数量为：

$$
W_{7\times7} = 7 \times 7 \times C \times C = 49C^2
$$

因此，用三个3×3卷积替代一个7×7卷积，在保持相同感受野的前提下，参数量从49C²降至27C²，节省了约44.9%的参数。更重要的是，多个非线性激活层（ReLU）穿插在3×3卷积之间，提供了更多的非线性变换能力，增强了决策函数的判别力。

**原则二：通道数逐阶段倍增。** VGG将网络分为5个阶段（stage），每个阶段包含若干卷积层，阶段之间通过2×2最大池化层（步长为2）进行空间下采样。通道数从64开始，在每个池化层后翻倍：64 → 128 → 256 → 512 → 512。这一设计使得每个阶段的空间分辨率减半，但特征通道加倍，在网络深度增加的同时保持（甚至增加）每层的计算量。

**原则三：以3×3卷积替代1×1卷积。** VGG完全不使用1×1卷积（这与后来GoogLeNet的设计形成鲜明对比）。在VGG看来，1×1卷积虽然能在通道维度上进行线性组合，但缺乏空间感受能力，对于需要空间信息的视觉任务而言，纯粹的通道混合是低效的。VGG选择用多个3×3卷积层来隐式地实现通道间的非线性混合。

### 第2.3节 参数效率分析

VGG-19的全部参数分布如下：卷积层（16个conv）参数量约16.0M（在VGG-16的13个卷积层基础上，conv3_3之后增加1层、conv4_3之后增加1层、conv5_3之后增加1层，每层约0.47M参数），三个全连接层（fc6, fc7, fc8）共包含约123.7M参数，总计约143.7M可训练参数。

VGG的参数瓶颈明确地集中在全连接层，占总参数的93%以上。这一观察直接推动了后续架构设计中去掉全连接层或大幅减少其参数的趋势（GoogLeNet使用全局平均池化替代全连接层，ResNet也采用了类似策略）。

从计算量角度，VGG-16的单次前向传播约需153亿次浮点运算（FLOPs），VGG-19则约需196亿次FLOPs。在2014年的硬件条件下，这是一个相当可观的计算成本，但相较于其带来的精度提升，这一投入被认为是值得的。

**VGG的训练策略与多尺度评估。** VGG-16/19的训练使用了以下详细策略：初始学习率为0.01（对应batch size 256，分布在4块K40 GPU上，每GPU 64张图像），动量因子为0.9，权重衰减为0.0005。学习率每经过一定数量的epoch后乘以0.1的衰减因子——VGG-16使用每训练完一定比例的epoch后衰减，总计训练约370k次迭代（约74个epoch，以batch size 256、128万张训练图像计算）（这一较长的训练周期是VGG能够收敛的关键）。此外，VGG训练时使用了**尺度抖动（scale jittering）**——训练图像在缩放时，短边随机从256到512像素之间选择，然后裁剪出224×224的区域。这一数据增强策略使得模型在多种尺度下进行学习，提升了尺度不变性。

在评估阶段，VGG使用了**多尺度裁剪与密集评估（multi-scale and dense evaluation）**的融合策略：将测试图像缩放到多个尺度（短边为256、288、320、352、384、416、448、480、512），在每个尺度上以dense方式滑动裁剪多个224×224区域，加上水平翻转，共得到约120-240个裁剪，对所有裁剪的预测结果取平均。这种评估方式虽然计算成本很高，但能够更准确地估计模型的真实性能——多裁剪融合后的Top-5错误率通常比单裁剪低1-2个百分点。

### 第2.4节 历史贡献与局限

VGGNet在ILSVRC 2014上取得了Top-5错误率7.3%的成绩（使用多尺度裁剪和融合），位列亚军。虽然精度上不敌当年的冠军GoogLeNet，但其设计的统一性和简洁性具有深远的理论价值——它证明了一个设计良好的"基础构件反复堆叠"的策略可以取得极具竞争力的性能。

**VGG的迁移学习遗产。** VGGNet的一个被低估的贡献是其作为迁移学习主干网络（backbone）的卓越表现。由于VGG的架构高度规整——所有卷积层使用相同的3×3核和相同的padding策略——其特征图具有高度的空间对齐性和语义一致性。当使用VGG-16/19作为预训练主干网络，在PASCAL VOC、COCO等小规模数据集上进行微调（fine-tuning）时，VGG 在部分早期迁移设定下仍具竞争力。这一现象的解释在于：规整的架构更容易进行空间定位和像素级预测（如语义分割、目标检测），而Inception模块中不同路径产生的多尺度特征在对齐和映射到像素级输出时更加复杂。不过，ResNet原文（He等人，2015/2016）已在目标检测上显著超过VGG：以Faster R-CNN为框架时，ResNet-101在COCO上的mAP明显高于VGG-16主干。2017–2018年的FPN（特征金字塔网络）与Mask R-CNN等则进一步强化了ResNet在多尺度检测与实例分割上的优势。至今，VGG-16仍是许多迁移学习教程和研究论文的标准baseline主干网络。

**VGG与感受野的极限。** 如前所述，VGG-16最终的特征图尺寸为7×7（经过5次2×2池化，224→112→56→28→14→7）；按后文递推公式，pool5 处感受野约为 **212×212**（接近甚至覆盖整幅 224×224 输入，而非 114×114）。VGG的全连接层聚合了大范围的上下文信息，而非仅看到局部特征。然而，这种空间覆盖仍是有代价的：7×7的空间分辨率在目标检测和语义分割等需要精细空间定位的任务中显得过于粗糙。7×7的特征图对应到原始224×224图像，每个特征单元覆盖32×32像素的区域——这对于检测小物体或刻画精确边界是不足够的。这一问题直接催生了后续的特征金字塔网络（FPN, Lin等人，2017）和空洞卷积（dilated/atrous convolution）等空间分辨率增强技术。

**VGG架构的逐层解剖。** 为了更具体地理解VGG的设计，下表给出了VGG-16和VGG-19的完整层结构，包括每层的类型、卷积核大小、步长、padding和输出通道数。

| 阶段 | 层名称 | 类型 | 卷积核 | 步长 | Padding | 输出大小 | 通道数 |
|------|--------|------|--------|------|---------|----------|--------|
| 1 | conv1_1 | Conv2d | 3×3 | 1 | 1 | 224×224 | 64 |
| | conv1_2 | Conv2d+ReLU | 3×3 | 1 | 1 | 224×224 | 64 |
| | pool1 | MaxPool2d | 2×2 | 2 | 0 | 112×112 | 64 |
| 2 | conv2_1 | Conv2d+ReLU | 3×3 | 1 | 1 | 112×112 | 128 |
| | conv2_2 | Conv2d+ReLU | 3×3 | 1 | 1 | 112×112 | 128 |
| | pool2 | MaxPool2d | 2×2 | 2 | 0 | 56×56 | 128 |
| 3 | conv3_1 | Conv2d+ReLU | 3×3 | 1 | 1 | 56×56 | 256 |
| | conv3_2 | Conv2d+ReLU | 3×3 | 1 | 1 | 56×56 | 256 |
| | conv3_3 | Conv2d+ReLU | 3×3 | 1 | 1 | 56×56 | 256 |
| | pool3 | MaxPool2d | 2×2 | 2 | 0 | 28×28 | 256 |
| 4 | conv4_1 | Conv2d+ReLU | 3×3 | 1 | 1 | 28×28 | 512 |
| | conv4_2 | Conv2d+ReLU | 3×3 | 1 | 1 | 28×28 | 512 |
| | conv4_3 | Conv2d+ReLU | 3×3 | 1 | 1 | 28×28 | 512 |
| | pool4 | MaxPool2d | 2×2 | 2 | 0 | 14×14 | 512 |
| 5 | conv5_1 | Conv2d+ReLU | 3×3 | 1 | 1 | 14×14 | 512 |
| | conv5_2 | Conv2d+ReLU | 3×3 | 1 | 1 | 14×14 | 512 |
| | conv5_3 | Conv2d+ReLU | 3×3 | 1 | 1 | 14×14 | 512 |
| | pool5 | MaxPool2d | 2×2 | 2 | 0 | 7×7 | 512 |
| 全连接 | fc6 | Linear+ReLU+Dropout(0.5) | — | — | — | 4096 | — |
| | fc7 | Linear+ReLU+Dropout(0.5) | — | — | — | 4096 | — |
| | fc8 | Linear+Softmax | — | — | — | 1000 | — |

VGG-19在conv3_3之后额外增加了一个conv3_4层，conv4_3之后额外增加了一个conv4_4层，conv5_3之后额外增加了一个conv5_4层，其余完全相同。

**感受野的精确计算。** 对于VGG-16，从输入图像到conv3_3输出的感受野计算如下。每个3×3卷积层的感受野为3，池化层为2。使用递推公式$\text{RF}_l = \text{RF}_{l-1} + (k_l - 1) \cdot \prod_{i=1}^{l-1} s_i$，其中$k_l$是第$l$层的核大小，$s_i$是第$i$层的步长，$\prod s_i$即到该层为止的累积步长（cumulative stride）：

- conv1_1: RF = 3, j = 1
- conv1_2: RF = 3 + (3-1)×1 = 5, j = 1
- pool1: RF = 5 + (2-1)×1 = 6, j = 2
- conv2_1: RF = 6 + (3-1)×2 = 10, j = 2
- conv2_2: RF = 10 + (3-1)×2 = 14, j = 2
- pool2: RF = 14 + (2-1)×2 = 16, j = 4
- conv3_1: RF = 16 + (3-1)×4 = 24, j = 4
- conv3_2: RF = 24 + (3-1)×4 = 32, j = 4
- conv3_3: RF = 32 + (3-1)×4 = 40, j = 4
- pool3: RF = 40 + (2-1)×4 = 44, j = 8
- conv4_1: RF = 44 + (3-1)×8 = 60, j = 8
- conv4_2: RF = 60 + (3-1)×8 = 76, j = 8
- conv4_3: RF = 76 + (3-1)×8 = 92, j = 8
- pool4: RF = 92 + (2-1)×8 = 100, j = 16
- conv5_1: RF = 100 + (3-1)×16 = 132, j = 16
- conv5_2: RF = 132 + (3-1)×16 = 164, j = 16
- conv5_3: RF = 164 + (3-1)×16 = 196, j = 16
- pool5: RF = 196 + (2-1)×16 = 212, j = 32

因此，VGG-16经过pool5后的总感受野为212×212，但特征图分辨率仅为7×7。全连接层fc6将7×7×512的特征图展平为25,088维向量后乘以4096维权重矩阵——参数总量为$4096 \times 25088 + 4096 \approx 102.8\text{M}$（约1亿）。虽然全连接层仍能聚合全局信息，但7×7的空间分辨率在目标检测等需要精细定位的任务中仍显粗糙。这一局限性直接催生了后续的全卷积网络（FCN）等放弃全连接层的架构设计。

---

## 第3节 GoogLeNet：Inception模块与网络的宽度探索

### 第3.1节 从"一个卷积核大小够好吗？"到"为什么不全部用上？"

VGG的设计哲学是"坚持用3×3"，而GoogLeNet（Szegedy等人，2014）则提出了一个截然不同但同样深刻的问题：**在同一层中，为什么不同时使用多种大小的卷积核？** 其核心洞察是：在不同的空间尺度上，图像中的重要特征分布是不均匀的。某些物体可能在大尺度上可见，而另一些则在小尺度上才能被精确捕获。与其人工选择单一尺度的卷积核，不如在同一层中并行地使用多尺度卷积核，然后让网络自己去学习如何组合这些多尺度特征。

这一思想催生了Inception模块。

### 第3.2节 Inception模块的设计

一个标准的Inception模块包含四条并行路径（branch），每条路径对输入执行不同类型的操作：

- **路径1：** 1×1卷积，输出C₁个通道
- **路径2：** 1×1卷积（降维到C₂通道）+ 3×3卷积（保持或扩展通道）
- **路径3：** 1×1卷积（降维到C₃通道）+ 5×5卷积（保持或扩展通道）
- **路径4：** 3×3最大池化 + 1×1卷积

四条路径的输出在通道维度上拼接（concatenate），形成该模块的最终输出。

这个设计的精妙之处在于1×1卷积的"降维"角色。考虑一个输入为28×28×256的特征图，如果直接使用64个5×5卷积核，输出为28×28×64，参数量为：

$$
256 \times 5 \times 5 \times 64 = 409,600
$$

若先通过1×1卷积将通道数降维至32，再使用64个5×5卷积核：

$$
256 \times 1 \times 1 \times 32 + 32 \times 5 \times 5 \times 64 = 2,048 + 51,200 = 53,248
$$

参数量从409,600骤降至53,248，压缩比约为5.8倍。同时，1×1卷积本身增加了网络的非线性能力（经过ReLU激活后），而池化路径保留了平移不变性的最强信号。所有路径的输出在通道维拼接后，网络可以自行学习不同尺度特征的权重分配。

### 第3.3节 GoogLeNet完整架构

GoogLeNet的完整架构可概括为以下序列：

**Stem部分（网络"主干"）：** 输入224×224×3的图像，首先经过一个7×7卷积（步长2，输出64通道，空间分辨率降至112×112），接ReLU和3×3最大池化（步长2，输出56×56×64），再接一个局部响应归一化（LRN）层。随后是一个1×1卷积（输出64通道）和一个3×3卷积（输出192通道），再接一个3×3最大池化（步长2，输出28×28×192）。这部分被称为"stem"，是后续所有Inception模块的基础特征提取器。

Stem的设计选择值得仔细分析。7×7卷积核配合步长2，将空间分辨率从224×224快速降至112×112——这在当时被认为是大卷积核快速下采样的标准做法（AlexNet使用了11×11步长4的卷积达到类似效果）。然而，GoogLeNet的7×7卷积设计在后来的Inception v3中被分解为三个3×3卷积的级联，反映了架构设计从"大核"向"小核级联"的范式转变。LRN层（局部响应归一化，一种早期的归一化技术，模仿侧抑制机制）在GoogLeNet中仅用于stem部分，后续的Inception模块不再使用——LRN的实际效果有限，在ResNet和后续架构中被完全抛弃。从stem输出的28×28×192特征图，是Inception模块堆叠的起点，其通道数192为各Inception模块提供了丰富的低层特征基础（边缘、纹理、简单形状）。

**Inception模块堆叠：** Stem之后依次堆叠两组Inception模块（记为3a和3b）、一组Inception模块与降维模块的组合（4a、4b、4c、4d、4e），以及最后两组Inception模块（5a和5b）。这些模块通过调整通道数实现参数效率的最优化。

**辅助分类器（Auxiliary Classifiers）：** 在Inception 4a和4d的输出之后，GoogLeNet各插入了一个辅助分类器。每个辅助分类器由一个5×5平均池化层（步长3）、一个1×1卷积（128通道）、两个全连接层（1024和1000节点，1000对应ImageNet类别数）以及一个Softmax层组成。辅助分类器的损失以0.3的权重加权加入总损失：

$$
L_{\text{total}} = L_{\text{main}} + 0.3 \times (L_{\text{aux1}} + L_{\text{aux2}})
$$

辅助分类器有两个作用：第一，在网络的中间层提供额外的梯度信号，缓解梯度消失问题——这在深层网络的训练中尤为关键；第二，通过中间层的分类约束，促使浅层特征具有更强的判别能力，起到正则化的效果。实验表明，辅助分类器在训练阶段确实提升了主分类器的收敛速度和最终精度，但在测试阶段被丢弃。

**分类头：** 经过Inception 5b后，是一个平均池化层（输出1×1×1024），一个Dropout层（dropout率0.4），以及一个全连接层（1000输出），最终接Softmax。

**GoogLeNet的计算效率分析。** GoogLeNet在Inception模块设计上所做的每一个参数削减，都直接转化为训练和推理速度的提升。以Inception 3a模块为例，如果不使用1×1降维（即直接使用128个5×5卷积处理192通道输入），参数量为$192 \times 5 \times 5 \times 128 = 614,400$；使用1×1降维至32通道后，参数量仅为$192 \times 1 \times 1 \times 32 + 32 \times 5 \times 5 \times 128 = 106,496$，减少了约83%。考虑到GoogLeNet中所有Inception模块累积的参数量，1×1降维总共节省了约80%的参数——这正是GoogLeNet能以6.6M参数达到VGG-19级精度的核心原因。

从计算量角度，GoogLeNet的设计预算约为15亿次乘加（约1.5G Mult-Adds，单尺度前向传播），远低于VGG-16（约15.3G，ResNet论文口径）。如果使用多尺度评估（如VGG的多裁剪融合），GoogLeNet的总计算量仍远低于VGG。这一效率优势在移动端和嵌入式设备上尤为宝贵——虽然GoogLeNet并未专门为移动端设计，但其参数效率已经为后续MobileNet等移动端架构铺平了道路。

GoogLeNet在ILSVRC 2014上的Top-5错误率为6.67%（7模型×144裁剪集成），优于VGG团队提交的7.3%（7模型集成）。单模型单裁剪下，GoogLeNet约为10.07% Top-5；VGG-16单尺度（$S=Q=256$）约为8.8% Top-5。GoogLeNet以更少的参数（约6.8M vs. 138M）和更低的计算预算达到了接近的集成性能。GoogLeNet的优势主要体现在：（1）参数量约为VGG-16的5%；（2）推理乘加次数显著低于VGG；（3）更深的网络结构（22层 vs. 16层）提供了更强的表征能力潜力。

### 第3.4节 Inception v2与v3：尺寸分解的思想

虽然GoogLeNet（通常被称为Inception v1）奠定了Inception系列的基础，Szegedy等人并未止步。Inception v2（2015）引入了一个关键改进：**将大卷积核分解为小卷积核的级联**。具体而言，5×5卷积被两个3×3卷积替代，这在保持感受野的同时进一步降低了参数量。更重要的是，Inception v2引入了批归一化（下一节详述），显著提升了训练速度和稳定性。

Inception v3（2015）进一步将n×n卷积分解为1×n和n×1两个卷积的级联。例如，一个7×7卷积被1×7卷积接7×1卷积替代。这种"非对称卷积"的设计在保持空间混合能力的同时，进一步减少了参数量，并在某些任务上表现出优于对称大卷积核的性能。

**Inception系列的完整演化：从v1到Inception-ResNet。** Inception v1（GoogLeNet, 22层）证明了多尺度并行路径的有效性，但其设计中存在一些不一致之处——例如，Inception模块中3×3池化和5×5卷积使用了不同的padding策略，导致不同路径的输出尺寸需要仔细对齐。Inception v2（2015）解决了这些问题，并将5×5卷积替换为两个连续的3×3卷积，同时引入了批归一化——这是BN在大型视觉模型中的首次成功应用。Inception v2在ImageNet上取得了4.9%的Top-5错误率（使用多裁剪），优于GoogLeNet的6.67%。

Inception v3（2015）进一步将n×n卷积分解为1×n和n×1的非对称卷积，并将网络深度增加到48层（含参数层）。Inception v3引入了**标签平滑（label smoothing）**——将hard one-hot标签替换为soft标签（例如，将[0, 1, 0, ...]替换为[0.1/999, 0.9, 0.1/999, ...]），减少了模型对错误标签的过度置信，提升了泛化能力。Inception v3在ImageNet上单裁剪Top-5错误率约为5.6%，单模型多裁剪Top-5约为3.58%。

Inception-v4（2016）和Inception-ResNet-v2（2016）将Inception模块与残差连接深度融合。Inception-v4进一步加深了网络（70层），优化了Stem设计（使用三个3×3卷积替代7×7卷积），并引入了Reduction模块替代部分Inception模块以实现更高效的下采样。Inception-ResNet-v2则在每个Inception模块后添加残差连接——与ResNet的恒等快捷连接不同，Inception-ResNet的残差连接使用了1×1卷积（步长1）来匹配通道数，而非简单的恒等映射。这种设计在处理多路径拼接后的通道数变化时更加灵活。Inception-ResNet-v2在ImageNet上取得了约19.9%的Top-1 / 4.9%的Top-5单模型错误率，优于同期ResNet-152的单模型结果（约21.4% / 5.7%，10裁剪），证明了Inception模块和残差连接的协同效应。需要注意，ResNet的3.57% Top-5是集成结果，不宜与Inception-ResNet-v2的单模型数字直接比较。

**Inception模块的设计哲学与当代 relevance。** Inception模块的核心思想——"在同一层中并行处理多种尺度，让网络自行学习组合"——在当代架构中仍然活跃。SENet（Hu等人，2018）的Squeeze-and-Excitation模块可以被视为Inception思想的继承者——它不改变空间维度，而是在通道维度上学习注意力权重，让网络自行决定每个通道的重要性。SKNet（Li等人，2019）的Selective Kernel进一步将Inception的多路径思想与注意力机制结合——通过动态选择不同大小的卷积核，网络可以根据输入内容自适应地调整感受野。这些后续工作表明，Inception模块的设计哲学——"并行探索，自适应融合"——是CNN架构设计中的一个持久性原则。

**Inception模块的通道维度详解。** 为了更具体地理解GoogLeNet的信息流，以下以Inception 3a模块为例，逐步追踪各路径的通道变化。输入为28×28×192的特征图（来自stem的输出）。路径1使用64个1×1卷积，输出28×28×64；路径2先通过96个1×1卷积降维至28×28×96，再通过128个3×3卷积输出28×28×128；路径3先通过16个1×1卷积降维至28×28×16，再通过32个5×5卷积输出28×28×32；路径4先经过3×3最大池化（步长1，padding为1以保持分辨率），再通过32个1×1卷积输出28×28×32。四条路径在通道维拼接后，输出为28×28×256（64+128+32+32=256）。不计偏置时，Inception 3a 参数量约为 $192\times64 + (192\times96 + 96\times3\times3\times128) + (192\times16 + 16\times5\times5\times32) + 192\times32 = 163{,}328$（约16.3万）——在提供256通道的多尺度特征的同时保持了极高的参数效率。

GoogLeNet中9个Inception模块的通道配置各不相同，但都遵循了"1×1降维 → 空间卷积 → 拼接"的基本模式。各模块的通道数从浅层的64通道逐步增加到深层的384通道甚至512通道（Inception 4e使用4条路径各128通道的1×1降维，加192通道的5×5路径），反映了网络对特征抽象程度逐渐提高的需求。

**"网络层数"的计数惯例。** GoogLeNet被描述为"22层"，这一数字仅包含具有可学习参数的层（卷积层和全连接层），不包括池化层和激活函数层（ReLU、LRN等）。按照这一惯例，AlexNet为8层（5个卷积层+3个全连接层），VGG-16为16层（13个卷积层+3个全连接层）。如果将所有层（包括池化和激活）都计入，VGG-16的实际深度约为41层。这一计数差异在比较不同架构时需要特别注意。

**Network in Network：1×1卷积的思想前驱。** GoogLeNet中大量使用的1×1卷积并非凭空发明，其思想根源可追溯到Lin等人（2013）提出的Network in Network（NiN）结构。NiN的核心思想是：用"微多层感知机"（mlpconv）替代传统卷积核。具体而言，mlpconv在每个空间位置用一个1×1卷积（作为微型全连接网络）替代单一标量卷积核，从而在每个空间位置进行通道间的非线性混合。NiN还引入了全局平均池化（global average pooling）替代全连接层，这一设计被GoogLeNet和后续几乎所有架构继承。从这一谱系可以看出，GoogLeNet的1×1降维、全局平均池化和辅助分类器等设计，都建立在NiN奠定的概念基础之上。

---

## 第4节 残差连接：ResNet的数学原理

### 第4.1节 退化问题：为什么更深的网络反而更差？

2015年之前，一个普遍的认知是：网络越深，表征能力越强，性能应该越好或至少持平。然而，He等人（2016）的实验揭示了一个反直觉的现象：**当网络深度超过某个阈值后，继续增加深度不仅没有提升性能，反而导致训练误差和测试误差同时上升。** 这不是过拟合——过拟合表现为训练误差下降而测试误差上升——而是更深网络在训练集上就已经表现更差了。He等人将这一现象命名为**退化问题（degradation problem）**。

退化问题的存在说明：深度神经网络的优化存在根本性的困难。问题不在于表达能力（一个深层网络完全可以"模拟"一个浅层网络——只需将多余的层学成恒等映射），而在于**优化算法能否找到好的参数使得多余层实现恒等映射**。随机初始化的深层网络很难通过梯度下降逼近恒等映射。

### 第4.2节 残差块的数学形式化

ResNet的核心思想直截了当：与其让每一层直接学习一个映射H(x)，不如让它学习一个相对于输入的残差映射F(x) = H(x) - x。原始映射则通过以下方式恢复：

$$
H(x) = F(x) + x
$$

这里的关键是"+ x"操作——**恒等快捷连接（identity shortcut connection）**。在残差网络中，一个基本的残差块定义为：

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)
$$

其中$\mathbf{x}_l$和$\mathbf{x}_{l+1}$分别是第$l$个残差块的输入和输出，$\mathcal{F}$是残差函数（由若干卷积层、BN层和ReLU组成），$\mathcal{W}_l$是$\mathcal{F}$的可学习权重。

**恒等映射的选择并非偶然。** 如果最优映射恰好接近恒等映射（这在深层网络中很常见——许多层的贡献是微调而非根本性转变），那么残差映射F(x)会趋近于零矩阵。通过权重衰减等正则化手段，网络倾向于让F(x)的权重保持较小的范数，从而自然地将F(x)推向零，使得$\mathbf{x}_{l+1} \approx \mathbf{x}_l$。换句话说，一个设计良好的残差网络在初始化时已经"近似最优"——恒等快捷连接提供了对恒等映射的强归纳偏置。

### 第4.3节 完整前向传播推导

考虑一个包含L个残差块的深度网络，第$l$个块的输入为$\mathbf{x}_l$。令$\mathbf{x}_L$为第L个块的输出，即网络的最终特征表示。对于任意浅层$l$和深层$L$（$l < L$），有：

$$
\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)
$$

这一递归关系式的意义是深远的：**任何深层的特征都可以被表达为任意浅层特征加上一系列残差修正的总和。** 这与信号处理中叠加原理的形式高度相似。

### 第4.4节 反向传播与梯度流

ResNet最核心的理论贡献在于其对梯度流的优雅处理。考虑损失函数$\mathcal{L}$对第$l$个残差块输入$\mathbf{x}_l$的梯度。根据链式法则：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \frac{\partial}{\partial \mathbf{x}_l} \left[ \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i) \right]
$$

展开后得到：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \left( \mathbf{I} + \frac{\partial}{\partial \mathbf{x}_l} \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i) \right)
$$

进一步展开链式法则到具体梯度流：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \left( \mathbf{I} + \frac{\partial \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)}{\partial \mathbf{x}_l} + \sum_{i=l+1}^{L-1} \frac{\partial \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)}{\partial \mathbf{x}_l} \right)
$$

这一等式的物理意义极其重要。第一项中的单位矩阵$\mathbf{I}$是恒等快捷连接带来的"梯度高速公路"——**无论后面的残差函数$\mathcal{F}$的雅可比矩阵$\partial \mathcal{F}/\partial \mathbf{x}_l$的谱半径如何，恒等项始终保证梯度至少以幅度1进行前传。** 如果残差映射的雅可比矩阵的奇异值小于1（这在深度网络中很常见，会导致梯度指数衰减），恒等项提供了一个稳定的下界，确保梯度永远不会完全消失。

换句话说，恒等快捷连接使得梯度传播被分解为两条路径：
1. **短路径（shortcut）：** 梯度直接通过$\mathbf{I}$前传，不经过任何非线性变换层
2. **长路径（residual）：** 梯度通过残差函数$\mathcal{F}$前传，经历正常的链式法则

即使长路径的梯度衰减严重，短路径始终存在。这种"冗余的梯度传播路径"是ResNet能够训练数百层网络的根本原因。为了更直观地理解这一机制，考虑一个简化的数值例子。假设一个残差块的残差函数雅可比矩阵的特征值为$\lambda = 0.9$（即梯度通过残差路径传播时衰减为原来的90%）。对于20层深的网络，不使用残差连接的标准网络，梯度从第20层传播到第1层的衰减因子为$0.9^{20} \approx 0.122$——梯度衰减到原来的约12%，虽未完全消失但已经相当微弱。对于100层网络，衰减因子为$0.9^{100} \approx 0.000026$——梯度几乎完全消失。而使用残差连接时，每一层的反向传播雅可比都含有恒等项$\mathbf{I}$，链式乘积中始终保留一条不经过反复$\lambda\ll 1$因子的通路，因此不会像标准深度网络那样被$0.9^{100}$这类连乘因子反复压扁。当然，实际网络中$\lambda$的值在不同层之间变化，有些层的$\lambda$可能大于1（导致梯度膨胀），有些可能远小于1（导致梯度消失）。但由于恒等项$\mathbf{I}$的存在，每一层的梯度传播至少保留一个单位的"基准流量"，使得优化器始终能够接收到来自损失函数的有效信号。正是这种"有保证的最小梯度流"使得ResNet的优化 landscape 远比标准深度网络平坦。

### 第4.5节 两种快捷连接：恒等映射 vs. 投影映射

当输入和输出的通道数或空间分辨率不匹配时（例如经过步长为2的卷积后），恒等映射无法直接相加。ResNet原始论文提出了两种解决方案：

**方案A（恒等映射 + 零填充）：** 保持恒等映射，对通道数不足的部分用零填充。优点是无额外参数，但零填充意味着增加的通道在信息传递中被"静默"，可能影响表达。

**方案B（投影映射/投影快捷连接）：** 使用1×1卷积（步长2）匹配输出通道数：

$$
\mathbf{x}_{l+1} = \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l) + \mathcal{W}_s \mathbf{x}_l
$$

其中$\mathcal{W}_s$是一个可学习的1×1卷积（步长2），用于匹配维度和下采样。实验表明，方案B略优于方案A，尤其在极深网络（如1001层）中。

ResNet-50及更深的变体使用了**瓶颈结构（bottleneck block）**，由三个卷积层组成：1×1卷积降维（如从256降至64）、3×3卷积（64通道）、1×1卷积升维（64→256）。这种设计大幅减少了3×3卷积层的计算量，使得构建50层、101层甚至152层的网络成为可能。

**ResNet变体：从ResNet到ResNeXt。** 在原始ResNet之后，研究者提出了多种改进变体。ResNeXt（Xie等人，2017）将Inception模块的多路径并行思想与残差连接相结合，提出了"基数（cardinality）"的概念——一个残差块由多个并行但结构相同的"路径"组成，每条路径的输出在通道维拼接后通过1×1卷积融合。设基数为C，每条路径的输出通道为d，则残差块的总输出通道为C×d。实验表明，增大基数C（同时保持总参数量不变）比加深网络或加宽网络更有效——ResNeXt-101（基数32）在ImageNet上的Top-1错误率（19.8%）优于ResNet-101（21.3%）和同等宽度的ResNet（约21.0%）。

预激活ResNet（Pre-activation ResNet, He等人，2016）将BN和ReLU置于卷积层之前（而非之后），残差块的公式变为：

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{W}_2 \cdot \text{ReLU}(\text{BN}(\mathcal{W}_1 \cdot \text{ReLU}(\text{BN}(\mathbf{x}_l))))
$$

预激活结构使得梯度在反向传播中经过残差块时只需经过加法节点和权重矩阵，不经过任何ReLU非线性——这保证了梯度的无损传播，对于训练超过1000层的极深网络至关重要。He等人使用预激活ResNet成功训练了1001层的网络，在CIFAR-10上测试错误率约**4.62%**（Identity Mappings in Deep Residual Networks, He et al., 2016），说明极深残差网络在小数据集上仍可达到很强精度，而非退化到失败。

Wide Residual Networks（WRN, Zagoruyko和Komodakis，2016）则从另一个角度拓展了残差网络：不增加深度，而是增加每层的通道数。实验表明，一个宽度为原来的k倍、深度仅为原始ResNet二分之一的网络，可以取得更优的性能——例如WRN-28-10（28层，宽度因子10）的CIFAR-10错误率约**4.00%**，优于原始ResNet-1001的约4.92%（后者深度约为其36倍）。这一发现挑战了"深度是最关键因素"的假设，表明宽度和深度之间存在可以互补的权衡。

**瓶颈结构的参数效率量化分析。** ResNet-50的瓶颈块包含三个卷积层：1×1卷积将256通道降维至64通道（参数量$256 \times 1 \times 1 \times 64 = 16,384$），3×3卷积在64通道上进行空间特征提取（参数量$64 \times 3 \times 3 \times 64 = 36,864$），1×1卷积将64通道升维回256通道（参数量$64 \times 1 \times 1 \times 256 = 16,384$）。总计每个瓶颈块约69,632个可训练参数（不含BN的γ和β）。如果不使用瓶颈结构，直接使用两个3×3卷积处理256通道（如VGG风格），每个块的参数量为$2 \times 256 \times 3 \times 3 \times 256 = 1,179,648$——瓶颈结构的参数效率是后者的约17倍。考虑到ResNet-50包含16个瓶颈块（各组块数为[3, 4, 6, 3]），这一参数节省的总量极为可观。正是这种高效的参数利用使得ResNet-50仅用约25.6M参数就达到了约22.9%的Top-1错误率（10裁剪，He等人，2016，Table 3）。

**Highway Networks：残差连接的思想先驱。** 在ResNet之前，Srivastava等人（2015）提出的Highway Networks已经探索了在深层网络中使用"信息高速公路"的思想。Highway Networks引入了门控机制（gating mechanism）来控制信息在 shortcuts 上的流动：

$$
\mathbf{y} = \mathcal{H}(\mathbf{x}, \mathcal{W}_{\mathcal{H}}) \cdot T(\mathbf{x}, \mathcal{W}_T) + \mathbf{x} \cdot C(\mathbf{x}, \mathcal{W}_C)
$$

其中$T$是变换门（transform gate），$C$是携带门（carry gate），通常设为$C = 1 - T$。当变换门$T$的输出接近0时，信息直接通过恒等映射前传（携带）；当$T$接近1时，信息通过非线性变换$\mathcal{H}$前传（变换）。这种门控残差连接允许网络自适应地决定每一层的信息流策略。

Highway Networks与ResNet的核心区别在于门的引入。ResNet的恒等快捷连接是无门控的——梯度始终以完整幅度通过快捷连接前传，没有任何衰减机制。这种简化使得ResNet的数学分析更加清晰（梯度流中存在确定的恒等项$\mathbf{I}$），训练也更加稳定。然而，门控机制的灵活性在高层次语义推理任务中可能具有优势——Srivastava等人报告Highway Networks在语言建模任务上优于同等深度的残差网络。对于视觉任务，ResNet的无门控设计在ImageNet上取得了更好的结果，但门控思想在后续的LSTM和GRU（以及Transformer中的门控注意力机制）中得到了广泛的应用。

从历史角度看，Highway Networks → ResNet → 神经ODE的演进路径清晰地展示了"残差连接"这一思想从简单到复杂、再从复杂回归简单的辩证发展过程。Highway Networks的门控机制是复杂的，但ResNet证明：去掉门控、保留最简单的恒等映射，已经足以解决优化问题。神经ODE则将残差连接嵌入到连续动力系统的理论框架中，为理解残差网络提供了最高层次的抽象。

### 第4.6节 ResNet在ImageNet上的表现

ResNet在ILSVRC 2015上取得了Top-5错误率3.57%的成绩（**6个不同深度残差网络的集成**，其中仅包含两个152层模型）——首次在ImageNet测试集上显著超越人类水平（人类Top-5错误率约为5%）。单模型ResNet-152在验证集上的多尺度评估结果为Top-1 19.38%、Top-5 4.49%（He等人，2016，Table 4），10裁剪评估则为Top-1 21.43%、Top-5 5.71%（Table 3）。这一里程碑事件证明了深度残差学习的强大威力，也标志着CNN架构研究进入了一个全新的阶段。

**ResNet的工程实现细节。** 原始ResNet-50/101/152的训练使用了以下参数：初始学习率0.1（对应batch size 256），动量0.9，权重衰减0.0001，最多训练约$60\times 10^{4}$次迭代（按ImageNet约128万训练图像、batch size 256估算，约合120个epoch），学习率在验证误差进入平台期时除以10。值得注意的是，ResNet训练时使用了短边在$[256,480]$上随机采样的尺度增强（scale augmentation），而非完全固定尺度。这一数据增强策略配合残差连接的有效优化，已经足以达到超越人类水平的性能。

在测试阶段，为获得最佳结果，ResNet采用了**全卷积多尺度评估**——将图像短边缩放到$\{224,256,384,480,640\}$等多个尺度并平均得分（He等人，2016）。比较实验中则采用标准10裁剪测试。需要强调的是，ILSVRC 2015夺冠的3.57% Top-5测试错误率来自**多模型集成**，而非单模型多裁剪；单模型ResNet-152的多尺度Top-5验证错误率约为4.49%。

ResNet的另一个重要实践发现是：**预训练ResNet在下游任务上的迁移性能优于其ImageNet单精度所暗示的水平。** 在COCO目标检测任务上，以ResNet-50为主干网络的Faster R-CNN取得了37.8%的mAP（在minival上），而以VGG-16为主干网络的Faster R-CNN仅为33.7%。ResNet-101进一步提升至40.8% mAP。这种迁移优势部分来自于ResNet的多尺度特征表达能力（深层特征图虽然空间分辨率低，但语义信息丰富），部分来自于残差连接对梯度流的保护——在fine-tuning初期，大学习率更新可能导致特征分布剧烈变化，而残差结构提供了更稳定的优化路径。

---

## 第5节 ResNet的理论解释：离散Hamilton-Jacobi-Bellman方程与神经ODE

### 第5.1节 残差连接与动力系统

从动力系统（dynamical systems）的视角审视，一个残差块可以被理解为对一个连续时间动力系统的**欧拉离散化**。

考虑以下常微分方程（ODE）：

$$
\frac{d\mathbf{x}(t)}{dt} = f(\mathbf{x}(t), t, \theta)
$$

其中$\mathbf{x}(t) \in \mathbb{R}^d$是系统的状态向量，$f$是一个由神经网络参数化的向量场，$t \in [0, T]$是时间参数。这个ODE的解描述了状态从初始值$\mathbf{x}(0)$到最终状态$\mathbf{x}(T)$的演化轨迹。

使用前向欧拉方法（Forward Euler method），对上述ODE进行时间步长为$h$的离散化：

$$
\mathbf{x}(t + h) = \mathbf{x}(t) + h \cdot f(\mathbf{x}(t), t, \theta)
$$

将$h \cdot f(\mathbf{x}(t), t, \theta)$对应到残差块的残差映射$\mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$，将$\mathbf{x}(t)$对应到$\mathbf{x}_l$，我们得到精确的对应关系：

$$
\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)
$$

这正是残差块的数学形式！在这一视角下，**网络的每一层对应ODE求解器的一个时间步**，整个深度网络（从输入到输出）对应ODE求解器从$t=0$到$t=T$的轨迹积分。更深层的网络意味着更小的时间步长$h$和更多的积分步——更精确地逼近连续动力系统的解。

这一洞见（Chen等人，2018；Haber和Ruthotto，2017）具有深远的理论意义：它将深度学习的"深度"参数化为积分步长，而非网络层数。步长$h$的选取需要权衡精度（小步长）与效率（大步长），这与数值分析中的经典权衡完全一致。

### 第5.2节 Hamilton-Jacobi-Bellman方程的离散化

最优控制理论中的**值函数（value function）**满足Hamilton-Jacobi-Bellman（HJB）方程，其离散化形式与残差网络有着引人注目的结构相似性。

考虑一个离散时间最优控制问题，其中值函数$V_t(x)$表示从状态$x$和时间步$t$开始的最优累积折扣回报。HJB方程的离散形式为：

$$
V_t(x) = \max_u \left[ r(x, u) + \gamma V_{t+1}(x + f(x, u)) \right]
$$

或者写成贝尔曼残差（Bellman residual）的形式：

$$
V_t(x) - \max_u \left[ r(x, u) + \gamma V_{t+1}(x + f(x, u)) \right] = 0
$$

如果我们从另一个角度解读——不将上式视为需要精确满足的方程，而是将其视为一个迭代更新规则——那么价值函数的更新可以写为：

$$
V_t^{\text{new}}(x) = V_t(x) + \Delta V_t(x)
$$

其中$\Delta V_t(x)$是贝尔曼残差项，衡量当前价值函数与最优价值函数之间的差距。这种"当前值 + 更新量"的结构与残差块的"输入 + 残差"结构完全相同。

更具体地说，考虑价值函数的参数化近似$V_t(x; \theta)$。在强化学习中，时序差分（TD）学习的目标是最小化贝尔曼残差：

$$
\mathcal{L}(\theta) = \mathbb{E}_{(x,a) \sim \mathcal{D}} \left[ \left( r(x,a) + \gamma \max_{a'} V_{t+1}(x'; \theta') - V_t(x; \theta) \right)^2 \right]
$$

最小化这一损失的过程，等价于让网络学习残差更新$\delta = r + \gamma V_{t+1} - V_t$。神经网络的每一层因此可以被理解为贝尔曼方程的一次"价值修正"迭代。

当然，这种类比对ResNet与强化学习之间关系的精确表述需要谨慎。最初的对应关系更多是结构和哲学层面的类比——残差连接提供了一种通用的"增量修正"机制，而贝尔曼方程恰好也以"增量更新"为核心。Munos和Scalia等人（2021）进一步将这一类比严格化，证明了深度残差网络可以模拟值迭代（value iteration）过程，并分析了收敛条件。

### 第5.3节 神经ODE：连续深度学习的框架

ResNet与动力系统的对应关系直接引出了Chen等人（2018）提出的**神经ODE（Neural ODE）**框架。其核心思想是：与其将深度网络视为一系列离散层，不如将其视为一个连续的ODE求解过程。网络的深度不再是一个离散的超参数，而是ODE的终止时间$T$。

给定初始状态$\mathbf{x}(0) = \mathbf{x}_0$，定义：

$$
\mathbf{x}(t) = \mathbf{x}_0 + \int_0^t f(\mathbf{x}(\tau), \tau, \theta) \, d\tau
$$

输出为$\mathbf{x}(T)$。前向传播不再是逐层的矩阵运算，而是ODE求解器（如欧拉法、龙格-库塔法）对上述ODE的数值积分。反向传播则通过伴随敏感度方法（adjoint sensitivity method）计算，无需在内存中存储完整的中间状态轨迹。

神经ODE的一个核心优势是**参数效率**。对于同一个任务，神经ODE可以用更少的参数达到与离散深度网络相当的性能，因为连续参数化$f(\mathbf{x}, t, \theta)$天然地共享了所有时间步的参数。此外，神经ODE提供了一种统一的框架来处理**连续时间序列数据**——输入可以是任意时间点的状态，输出同样如此，无需离散采样。

神经ODE与ResNet的关系是渐进的：当ODE求解器使用前向欧拉方法且步长固定时，神经ODE退化为ResNet。因此，ResNet可以被视为神经ODE的一种特殊（欧拉离散）情况。这为理解ResNet的泛化行为提供了新的工具——连续动力系统的稳定性理论（如Lyapunov稳定性）可以直接应用于分析残差网络的梯度流。

**ODE稳定性与梯度消失的深层联系。** 从动力系统视角，残差网络的前向传播等价于对ODE $\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t, \theta)$的欧拉离散化。如果向量场$f$的雅可比矩阵$\partial f/\partial \mathbf{x}$的所有特征值的实部均为负（即系统是**渐近稳定**的），那么ODE的解会指数收敛到平衡点。在神经网络的语境下，这意味着深层特征会"坍缩"到一个固定的表征——网络的深度无法提供额外的表征能力。反之，如果雅可比矩阵存在特征值的实部接近零（即系统接近**临界稳定**），则状态可以在不同层次之间维持丰富的变化，网络的深度才能充分发挥作用。

残差连接的设计恰好将系统推向临界稳定区域。考虑残差块的雅可比矩阵：

$$
\frac{\partial \mathbf{x}_{l+1}}{\partial \mathbf{x}_l} = \mathbf{I} + \frac{\partial \mathcal{F}}{\partial \mathbf{x}_l}
$$

如果没有残差连接（即标准卷积网络），雅可比矩阵就是$\partial \mathcal{F}/\partial \mathbf{x}_l$本身，其特征值的实部通常为负（经过ReLU激活后），导致梯度指数衰减。有了恒等项$\mathbf{I}$，雅可比矩阵的特征值整体向右平移了1个单位——原本接近零的负特征值变为接近零的正值或零值，原本远小于-1的特征值也可能变为负值（仍会导致梯度衰减，但程度减轻）。这种"特征值平移"机制使得梯度既不会完全消失也不会完全爆炸，而是在一个可控的范围内传播——这正是深度网络能够优化的动力学根源。

Haber和Ruthotto（2017）进一步将这一分析严格化：他们证明了残差连接等价于对ODE求解器的时间步长进行**反向传播的稳定性约束**——当步长$h$足够小时，欧拉方法是稳定的；当$h$过大时，数值解会振荡或发散。在神经网络中，"步长过大"对应于学习率过高或残差映射的权重过大——梯度更新导致参数越过最优解。残差连接通过恒等项提供了一个"阻尼"，使得即使学习率较高，参数的更新仍然在合理范围内。

**从ResNet到Neural ODE：连续控制视角。** Chen等人（2018）的神经ODE框架将上述动力系统视角推进到了一个全新的高度。在神经ODE中，前向传播定义为：

$$
\mathbf{x}(t_1) = \mathbf{x}(t_0) + \int_{t_0}^{t_1} f(\mathbf{x}(t), t, \theta) \, dt
$$

其中向量场$f$由一个深度神经网络参数化。与ResNet的关键区别在于：神经ODE的"深度"（即$t_1 - t_0$）是连续可调的，而ResNet的深度是离散固定的。这意味着神经ODE可以在推理时动态调整积分精度——对于简单输入，使用大步长快速求解；对于复杂输入，使用小步长精细求解。此外，神经ODE天然地处理**不规则时间序列**——输入和输出可以是不规则采样的时间点，不需要插值或填充。

神经ODE的反向传播使用**伴随方法（adjoint method）**，避免了存储完整的中间状态轨迹。定义伴随变量$\mathbf{a}(t) = \partial \mathcal{L}/\partial \mathbf{x}(t)$，其满足另一个ODE：

$$
\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^T \frac{\partial f(\mathbf{x}(t), t, \theta)}{\partial \mathbf{x}}
$$

从$t_1$到$t_0$积分该ODE，即可得到$\mathbf{a}(t_0)$，然后计算：

$$
\frac{\partial \mathcal{L}}{\partial \theta} = -\int_{t_1}^{t_0} \mathbf{a}(t)^T \frac{\partial f(\mathbf{x}(t), t, \theta)}{\partial \theta} \, dt
$$

伴随方法的内存复杂度从O(L)（L为层数/步数）降至O(1)，使得训练极深（或等效极深）的连续深度网络成为可能。

**ODE求解器的选择与泛化。** 神经ODE框架的一个重要自由度是ODE求解器的选择。前向欧拉方法（Euler method）对应ResNet的离散残差结构——简单但精度较低，需要小步长才能保持稳定。中点法（midpoint method）在每个步长内评估两次向量场（中点预测和端点修正），精度为二阶——对应一种"两步残差"结构，在相同步长下比欧拉法更精确。四阶龙格-库塔法（RK4）评估四次向量场，精度为四阶，步长可以更大——对应一种"四路残差"结构，在相同精度下比欧拉法使用更少的评估次数。实际上，Ruiz-Bolado等人（2020）的实验表明，使用高阶求解器的神经ODE在ImageNet上取得了优于使用欧拉法（即等价ResNet）的性能，同时使用更少的函数评估次数（即等效更少的"层"）。这一发现暗示：**网络的"有效深度"不仅取决于层数，还取决于每层的"质量"——高阶求解器的每步更新更加精确，因此需要更少的步数达到相同的精度。** 这为理解"为什么ResNet-50比VGG-16好"提供了一个新的视角——ResNet的残差结构相当于一个隐式的自适应步长ODE求解器，它在困难区域使用小步长（通过更多层进行精细调整），在简单区域使用大步长（通过恒等映射快速通过）。

---

## 第6节 批归一化：完整数学推导与统计解释

### 第6.1节 内部协变量偏移

Ioffe和Szegedy（2015）提出批归一化（Batch Normalization, BN）的原始动机是**内部协变量偏移（Internal Covariate Shift）**。在深度神经网络的训练过程中，每一层的参数更新都会改变其后续层的输入分布。当网络很深时，这种分布漂移会逐层累积，导致后续层需要不断适应新的输入分布，严重拖慢了收敛速度。

以第$l$层为例，其输入为$\mathbf{u}^{(l)}$，输出为$\mathbf{x}^{(l)} = \text{ReLU}(W^{(l)}\mathbf{u}^{(l)} + \mathbf{b}^{(l)})$。在SGD的一次迭代中，$W^{(l)}$被更新为$W^{(l)} + \Delta W^{(l)}$，第$l+1$层的输入分布随之改变。在深层网络中，这种改变通过链式法则逐层放大，使得网络深层参数的优化 landscape 不断变化，如同在移动的沙地上建房。

### 第6.2节 BN的前向传播

批归一化对每个特征通道独立进行。对于一个包含$m$个样本的小批量$\mathcal{B} = \{x_1, x_2, \ldots, x_m\}$（每个$x_i$是一个特征通道上的标量激活值），BN的前向传播分为两步。

**第一步：归一化。** 计算小批量的均值和方差：

$$
\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i
$$

$$
\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2
$$

对小批量中的每个样本进行归一化：

$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}
$$

其中$\epsilon$是一个极小常数（通常取$10^{-5}$），用于数值稳定性，防止分母为零。

**第二步：缩放与平移（scale and shift）。** 归一化后的激活值$\hat{x}_i$被限制在零均值单位方差的分布中，这可能过度约束了模型的表达容量——某些任务可能需要非零均值或非单位方差的激活分布。BN引入了两个可学习参数$\gamma$和$\beta$（每个通道各一对）来恢复这种表达能力：

$$
y_i = \gamma \hat{x}_i + \beta = \text{BN}_{\gamma,\beta}(x_i)
$$

参数$\gamma$控制缩放（scale），$\beta$控制平移（shift）。当$\gamma = \sigma_{\mathcal{B}}$且$\beta = \mu_{\mathcal{B}}$时，BN层退化为恒等变换，网络可以自行选择是否利用归一化。这一设计是BN能够无缝集成到各种网络架构中的关键。

在卷积层中，BN通常应用于每个输出通道。对于一个小批量形状为$(N, C, H, W)$的卷积特征（$N$为样本数，$C$个通道，每个通道是$H\times W$的空间特征图），BN对每个通道独立计算统计量：将该通道上全部$N\times H\times W$个激活视为一个集合，求其$\mu_{\mathcal{B}}$和$\sigma_{\mathcal{B}}^2$，然后对该通道在所有样本与空间位置上应用相同的归一化变换。

### 第6.3节 BN的反向传播推导

BN层的反向传播是整个推导中最关键也最复杂的部分。我们需要计算损失$\mathcal{L}$对BN层输入$x_i$、可学习参数$\gamma$和$\beta$、以及小批量统计量$\mu_{\mathcal{B}}$和$\sigma_{\mathcal{B}}^2$的梯度。完整的链式法则路径如下：

定义中间变量：
$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}, \quad s = \sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}
$$

**步骤1：对$\gamma$和$\beta$的梯度（直接计算，无需链式法则展开）。**

$$
\frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \cdot \frac{\partial y_i}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \cdot 1 = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i}
$$

$$
\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \cdot \frac{\partial y_i}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \cdot \hat{x}_i = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \hat{x}_i
$$

令$\frac{\partial \mathcal{L}}{\partial y_i}$从后续层反向传播进来，记为$\delta_i$（通常写作$\frac{\partial \mathcal{L}}{\partial y_i}$本身）。

**步骤2：对归一化值$\hat{x}_i$的梯度。**

$$
\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \frac{\partial y_i}{\partial \hat{x}_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \gamma = \gamma \delta_i
$$

**步骤3：对标准差$s$的梯度。**

$$
\frac{\partial \mathcal{L}}{\partial s} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial s} = \sum_{i=1}^{m} \gamma \delta_i \cdot \left( -\frac{x_i - \mu_{\mathcal{B}}}{s^2} \right) = -\frac{\gamma}{s} \sum_{i=1}^{m} \delta_i \hat{x}_i
$$

注意到$\sum_{i=1}^{m} \delta_i \hat{x}_i = \frac{\partial \mathcal{L}}{\partial \gamma}$（来自步骤1），因此：

$$
\frac{\partial \mathcal{L}}{\partial s} = -\frac{\gamma}{s} \cdot \frac{\partial \mathcal{L}}{\partial \gamma}
$$

**步骤4：对方差$\sigma_{\mathcal{B}}^2$的梯度。**

$$
\frac{\partial \mathcal{L}}{\partial \sigma_{\mathcal{B}}^2} = \frac{\partial \mathcal{L}}{\partial s} \cdot \frac{\partial s}{\partial \sigma_{\mathcal{B}}^2} = \frac{\partial \mathcal{L}}{\partial s} \cdot \frac{1}{2s}
$$

代入步骤3的结果：

$$
\frac{\partial \mathcal{L}}{\partial \sigma_{\mathcal{B}}^2} = -\frac{\gamma}{2s^2} \cdot \frac{\partial \mathcal{L}}{\partial \gamma} = -\frac{\gamma}{2(\sigma_{\mathcal{B}}^2 + \epsilon)} \cdot \frac{\partial \mathcal{L}}{\partial \gamma}
$$

**步骤5：对均值$\mu_{\mathcal{B}}$的梯度。**

$$
\frac{\partial \mathcal{L}}{\partial \mu_{\mathcal{B}}} = \sum_{i=1}^{m} \left[ \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial \mu_{\mathcal{B}}} + \frac{\partial \mathcal{L}}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{\partial \sigma_{\mathcal{B}}^2}{\partial \mu_{\mathcal{B}}} \right]
$$

其中：
$$
\frac{\partial \hat{x}_i}{\partial \mu_{\mathcal{B}}} = -\frac{1}{s}, \quad \frac{\partial \sigma_{\mathcal{B}}^2}{\partial \mu_{\mathcal{B}}} = -\frac{2}{m} \sum_{j=1}^{m} (x_j - \mu_{\mathcal{B}}) = 0
$$

（方差对均值的导数中$\sum(x_j - \mu_{\mathcal{B}}) = 0$，因此第二项为零。）

$$
\frac{\partial \mathcal{L}}{\partial \mu_{\mathcal{B}}} = \sum_{i=1}^{m} \gamma \delta_i \cdot \left(-\frac{1}{s}\right) = -\frac{\gamma}{s} \sum_{i=1}^{m} \delta_i
$$

**步骤6：对原始输入$x_i$的梯度（最终目标）。**

$$
\frac{\partial \mathcal{L}}{\partial x_i} = \underbrace{\frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i}}_{\text{路径1}} + \underbrace{\frac{\partial \mathcal{L}}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{\partial \sigma_{\mathcal{B}}^2}{\partial x_i}}_{\text{路径2}} + \underbrace{\frac{\partial \mathcal{L}}{\partial \mu_{\mathcal{B}}} \cdot \frac{\partial \mu_{\mathcal{B}}}{\partial x_i}}_{\text{路径3}}
$$

逐项计算：
- 路径1：$\gamma \delta_i \cdot \frac{1}{s} = \frac{\gamma}{s} \delta_i$
- 路径2：$\frac{\partial \sigma_{\mathcal{B}}^2}{\partial x_i} = \frac{2}{m}(x_i - \mu_{\mathcal{B}}) = \frac{2s}{m} \hat{x}_i$，所以$\frac{\partial \mathcal{L}}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{\partial \sigma_{\mathcal{B}}^2}{\partial x_i} = -\frac{\gamma}{2s^2} \cdot \frac{\partial \mathcal{L}}{\partial \gamma} \cdot \frac{2s}{m} \hat{x}_i = -\frac{\gamma}{s m} \hat{x}_i \cdot \frac{\partial \mathcal{L}}{\partial \gamma}$
- 路径3：$\frac{\partial \mu_{\mathcal{B}}}{\partial x_i} = \frac{1}{m}$，所以$\frac{\partial \mathcal{L}}{\partial \mu_{\mathcal{B}}} \cdot \frac{\partial \mu_{\mathcal{B}}}{\partial x_i} = -\frac{\gamma}{s} \sum_{j=1}^{m} \delta_j \cdot \frac{1}{m} = -\frac{\gamma}{ms} \sum_{j=1}^{m} \delta_j$

合并三条路径：

$$
\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma}{s} \left[ \delta_i - \frac{1}{m} \sum_{j=1}^{m} \delta_j - \frac{\hat{x}_i}{m} \sum_{j=1}^{m} \delta_j \hat{x}_j \right]
$$

这个表达式可以进一步简化。定义：
$$
\Delta = \frac{1}{m} \sum_{j=1}^{m} \delta_j, \quad \Delta_{\hat{x}} = \frac{1}{m} \sum_{j=1}^{m} \delta_j \hat{x}_j
$$

则：
$$
\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma}{s} \left[ \delta_i - \Delta - \hat{x}_i \cdot \Delta_{\hat{x}} \right]
$$

**这一形式具有深刻的统计意义。** 梯度被分解为三个部分：
1. $\delta_i$：未经修正的原始梯度
2. $\Delta$：所有样本梯度的均值（偏置修正项）
3. $\hat{x}_i \cdot \Delta_{\hat{x}}$：归一化激活与梯度协方差相关的修正项

BN的反向传播公式表明：**梯度在传播过程中被"重缩放"了——其有效方差被约束在与激活值统计量相关的范围内。** 这种重缩放机制避免了梯度的指数级衰减或爆炸，是BN加速收敛的数学根源之一。

### 第6.4节 推理模式：运行统计量

在训练阶段，BN使用小批量统计量$\mu_{\mathcal{B}}$和$\sigma_{\mathcal{B}}^2$进行归一化。但在推理（测试）阶段，通常只有一个样本（$m=1$），无法计算有意义的均值和方差。为此，BN在训练过程中维护**运行均值（running mean）**和**运行方差（running variance）**的移动平均：

$$
\mu_{\text{running}} \leftarrow m \cdot \mu_{\text{running}} + (1-m) \cdot \mu_{\mathcal{B}}
$$

$$
\sigma_{\text{running}}^2 \leftarrow m \cdot \sigma_{\text{running}}^2 + (1-m) \cdot \sigma_{\mathcal{B}}^2
$$

其中$m$是动量参数（通常取0.9或0.99）。在推理阶段，使用运行统计量替代小批量统计量：

$$
\hat{x} = \frac{x - \mu_{\text{running}}}{\sqrt{\sigma_{\text{running}}^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta
$$

这一推理模式使得BN层的行为完全确定性，不依赖于输入数据，这对于模型部署和实时推理至关重要。

### 第6.5节 BN加速训练的统计解释

BN加速训练的原因，在原始论文发表后的数年间经历了深入的再审视。Ioffe和Szegedy的原始假设——BN通过减少内部协变量偏移来加速训练——在Santurkar等人（2018）的工作中受到了挑战。Santurkar等人的实验表明：即使人为地在BN层引入人为的协变量偏移，BN依然能够加速收敛并提升最终性能。这说明**减少内部协变量偏移并非BN加速训练的根本原因**。

Santurkar等人提出了一种更精确的解释：**BN通过平滑优化 landscape 来加速训练。** 具体而言，BN使得损失函数在其参数空间中的Lipschitz常数变小——损失函数的变化更加平缓，梯度方向更加可靠，从而允许使用更大的学习率而不会导致发散。

从数学角度，考虑一个mini-batch中的损失$\mathcal{L}_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^{m} \mathcal{L}(x_i; \theta)$。BN通过归一化每层的输入，使得损失对参数的梯度满足：

$$
\left\| \frac{\partial \mathcal{L}_{\mathcal{B}}}{\partial \theta} \right\| \leq \frac{\gamma}{s} \cdot \left\| \delta - \Delta - \hat{x} \cdot \Delta_{\hat{x}} \right\|
$$

由于$\hat{x}_i$的均值为0、方差为1，梯度中的各项被自然地尺度化（rescaled），避免了极端值。这种**长度-方向解耦（length-direction decoupling）**——梯度的大小和方向被分别控制——是BN优化效果的核心机制（Köhler等人，2019）。

### 第6.6节 BN的实践考量

BN在实际应用中面临若干挑战。首先，BN要求小批量具有足够的统计代表性——当batch size过小时（如1或2），小批量均值和方差的估计变得不可靠。这在以下场景中尤为突出：（1）大规模语义分割——高分辨率图像消耗大量显存，batch size被迫缩小；（2）迁移学习——fine-tuning时batch size受限于GPU内存；（3）在线学习——逐样本更新。

其次，BN将训练过程中的计算图与 batch 维度耦合，这使得某些任务变得复杂——例如，在序列模型中，不同序列的长度各不相同，padding token 会污染统计量。层归一化（Layer Normalization）、实例归一化（Instance Normalization）和组归一化（Group Normalization）正是为了解决这些问题而提出的替代方案。

**层归一化（Layer Normalization, LN）。** Ba等人（2016）提出的层归一化对每个样本的所有特征进行归一化，而非跨样本对每个特征通道进行归一化。对于一个形状为$(C, H, W)$的特征图，LN计算所有$C \times H \times W$个元素的均值和方差，然后独立地对每个样本进行归一化。由于LN不依赖于batch维度，它在batch size为1时（如在线学习和小批量推理）同样有效。LN在循环神经网络（如LSTM、Transformer）中特别有用——RNN的隐状态是沿时间维度序列化的，batch维度上的统计量不具有统计意义，而LN可以在每个时间步独立地进行归一化。LN的公式与BN完全相同，只是将小批量维度替换为特征维度。

**内部协变量偏移的再审视：从因果到相关。** BN最初被提出的动机——减少内部协变量偏移——在后续研究中经历了重要的修正和完善。虽然Santurkar等人（2018）的实验表明BN的加速效果不依赖于协变量偏移的减少，但Bjorck等人（2018）发现：在ResNet中，BN确实显著减少了层输入分布的漂移，只是这种减少是加速训练的**伴随效应（correlate）**而非**因果原因（cause）**。更准确的理解是：BN通过约束每层输入的Lipschitz常数，使损失 landscape 更加平滑，从而允许更大的学习率和更快的收敛。这一"平滑 landscape "解释与"减少协变量偏移"解释并不矛盾——它们描述的是同一现象的不同方面，只是前者更精确地指向了优化动力学的根本机制。

Köhler等人（2019）从理论上严格证明了BN的收敛加速效果：在满足一定条件下（梯度有界、Lipschitz连续），使用BN的随机梯度下降的收敛速度为$O(1/\sqrt{T})$，与不使用BN的$O(1/\sqrt{T})$相同阶但常数因子更小——BN缩小了收敛速率中的隐含常数。这一理论结果为BN的实践效果提供了数学保障。

**实例归一化（Instance Normalization, IN）。** Ulyanov等人（2017）提出的实例归一化是LN在风格迁移任务上的特例——对每个样本的每个通道独立计算均值和方差。对于一个形状为$(C, H, W)$的特征图，IN产生C组独立的归一化统计量（每组H×W个元素）。IN完全去除了图像的对比度信息（因为每个通道被独立归一化为零均值单位方差），而保留了风格信息——这解释了为什么IN在风格迁移任务上优于BN：它强制网络关注空间结构和纹理模式，而非绝对颜色值。

**组归一化（Group Normalization, GN）。** Wu和He（2018）提出的组归一化是BN、LN和IN的统一泛化。GN将通道分为G个组，在每个组内（跨空间位置但跨样本）计算均值和方差。当G = 1时，GN退化为LN（所有通道在一个组中）；当G = C（通道数）时，GN退化为IN（每个通道独立为一组）；当G的值介于两者之间时，GN在通道组之间进行归一化。实验表明，在batch size较小时（如每GPU 1-2张图像），GN consistently 优于BN——在ImageNet上，使用GN的ResNet-50在batch size为2时取得了26.0%的Top-1错误率，而相同条件下的BN仅为29.5%。GN已经成为小批量训练和实例分割（Mask R-CNN）的标准配置。

---

## 第7节 深度可分离卷积：从理论到MobileNet

### 第7.1节 标准卷积的参数瓶颈

标准卷积操作是卷积神经网络的基本计算单元。对于一个输入特征图$\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times H \times W}$和输出通道数为$C_{\text{out}}$、卷积核大小为$D_k \times D_k$的卷积层，标准卷积的参数量为：

$$
\text{Params}_{\text{std}} = D_k^2 \cdot C_{\text{in}} \cdot C_{\text{out}}
$$

计算量（FLOPs）为：

$$
\text{FLOPs}_{\text{std}} = D_k^2 \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot H \cdot W
$$

当$C_{\text{in}}$和$C_{\text{out}}$较大时（如512或1024），标准卷积的参数量迅速膨胀。例如，一个3×3、输入512通道、输出512通道的卷积层，参数量为$9 \times 512 \times 512 = 2,359,296$（约236万）。一个包含数十个此类层的网络，参数量将非常可观。

### 第7.2节 深度可分离卷积的分解

深度可分离卷积（Depthwise Separable Convolution）将一个标准卷积分解为两个更简单的操作：

**第一步：深度卷积（Depthwise Convolution）。** 对每个输入通道独立应用一个空间卷积核。输入有$C_{\text{in}}$个通道，就有$C_{\text{in}}$个卷积核，每个卷积核大小为$D_k \times D_k \times 1$。输出通道数等于输入通道数$C_{\text{in}}$。

$$
\text{Params}_{\text{depthwise}} = D_k^2 \cdot C_{\text{in}}
$$

深度卷积负责在每个通道内部进行空间特征提取，但它不混合不同通道的信息。

**第二步：逐点卷积（Pointwise Convolution）。** 使用1×1卷积在通道维度上进行线性组合，将$C_{\text{in}}$个通道混合为$C_{\text{out}}$个通道。

$$
\text{Params}_{\text{pointwise}} = 1 \times 1 \cdot C_{\text{in}} \cdot C_{\text{out}} = C_{\text{in}} \cdot C_{\text{out}}
$$

深度可分离卷积的总参数量为：

$$
\text{Params}_{\text{dsep}} = D_k^2 \cdot C_{\text{in}} + C_{\text{in}} \cdot C_{\text{out}} = C_{\text{in}} \cdot (D_k^2 + C_{\text{out}})
$$

与标准卷积相比的压缩比为：

$$
\frac{\text{Params}_{\text{dsep}}}{\text{Params}_{\text{std}}} = \frac{D_k^2 \cdot C_{\text{in}} + C_{\text{in}} \cdot C_{\text{out}}}{D_k^2 \cdot C_{\text{in}} \cdot C_{\text{out}}} = \frac{1}{C_{\text{out}}} + \frac{1}{D_k^2}
$$

以3×3卷积核、$C_{\text{in}} = C_{\text{out}} = 512$为例：

$$
\frac{\text{Params}_{\text{dsep}}}{\text{Params}_{\text{std}}} = \frac{1}{512} + \frac{1}{9} \approx 0.113
$$

即深度可分离卷积仅使用标准卷积约11.3%的参数。当通道数更大时，压缩比进一步提高。对于3×3核、512通道，参数量从约236万降至约29万，压缩约8.8倍。

**分组卷积：从AlexNet到深度可分离的桥梁。** 深度可分离卷积可以被理解为分组卷积（Grouped Convolution）的一个极端特例。在标准分组卷积中，输入通道被分为G个组，每个组独立进行卷积，组间不混合信息：

$$
\text{Params}_{\text{grouped}} = D_k^2 \cdot \frac{C_{\text{in}}}{G} \cdot C_{\text{out}}
$$

当$G = C_{\text{in}}$时，每个通道独立作为一个组，这正是深度卷积——每个通道只有一个空间滤波器，输出通道数等于输入通道数。深度可分离卷积在此基础上增加了一个1×1逐点卷积来混合通道信息。

AlexNet（2012）是分组卷积的早期实践者——将网络分为2个组，分别在不同的GPU上计算，以解决单GPU显存不足的问题。这一工程折中意外地揭示了分组卷积的价值：减少参数量的同时保持（甚至提升）特征表达能力。下表对比了三种卷积方式在3×3核、$C_{\text{in}} = C_{\text{out}} = 512$条件下的参数量：

| 卷积类型 | 分组数 | 参数量 | 相对于标准卷积 |
|----------|--------|--------|--------------|
| 标准卷积 | 1 | 2,359,296 | 100% |
| 分组卷积（4组） | 4 | 589,824 | 25.0% |
| 分组卷积（16组） | 16 | 147,456 | 6.25% |
| 深度可分离卷积 | 512 | 294,912 | 12.5% |

**ShuffleNet：通道混洗与分组卷积的优化。** 深度可分离卷积和分组卷积的一个潜在问题是：当分组数较多时，不同组之间的信息无法流通，每个组只能学习到输入通道子集的特征，导致表征能力受限。Zhang等人（2018）提出的ShuffleNet通过**通道混洗（channel shuffle）**操作解决了这一问题：在进行分组卷积后，将输出通道按组重新排列，使得下一层的分组卷积能够访问上一层所有组的信息。具体而言，对于一个包含G个组、每组C个通道的输出，将通道重新排列为$(g_1, g_2, \ldots, g_G, g_{G+1}, \ldots)$——即第1个通道属于组1的样本与第2个通道属于组2的样本相邻——然后在下一层以不同的分组划分进行分组卷积。这种跨组信息流通机制使得ShuffleNet在保持分组卷积参数效率的同时，不损失特征表达能力。

ShuffleNet v2（2018）进一步指出，分组卷积中的通道混洗操作和逐点分组卷积的通道数瓶颈（1×1卷积占据大部分计算量）限制了效率。ShuffleNet v2通过通道分裂（channel split）策略——在残差块的每个分支中，将输入通道分为两部分，一部分保持恒等映射，另一部分进行卷积——在保持精度的同时进一步降低了计算量。在ImageNet上，ShuffleNet v2在仅0.5G FLOPs的计算预算下取得了约72.6%的Top-1准确率，证明了极低计算量下的有效特征学习是可行的。

### 第7.3节 理论直觉与经验验证

深度可分离卷积的参数效率源于一个核心假设：**空间特征提取和通道特征混合可以解耦为两个独立的步骤。** 标准卷积同时在这两个维度上进行操作，而深度可分离卷积将它们顺序执行。这种解耦的合理性在于：（1）大多数视觉特征的空间相关性和通道相关性具有不同的统计特性——边缘、纹理等空间模式可以在通道级别共享；（2）通道混合操作（1×1卷积）本身已经具有充分的表达能力。

Chollet（2017）提出的Xception网络将深度可分离卷积推向了极致——几乎完全用深度可分离卷积替代了Inception模块中的所有标准卷积。在ImageNet上，Xception与Inception v3取得了相近的Top-1准确率（约79.0% vs. 78.0%），但参数更少（约2300万 vs. 2300万——Inception v3为2320万，Xception为2290万），证明了深度可分离卷积在资源受限场景下的实用性。

Howard等人（2017）的MobileNet进一步验证了这一结论。MobileNet主要由深度可分离卷积构成，在ImageNet上达到70.6%的Top-1准确率（约 **4.2M 参数**、约 **0.57G Mult-Adds**），相比同等计算预算下的标准卷积网络，精度损失极小但速度大幅提升。MobileNet v2（2018）通过引入倒残差结构（inverted residual）和线性瓶颈（linear bottleneck），进一步优化了深度可分离卷积的表达效率。

---

## 第8节 神经架构搜索的早期探索

### 第8.1节 从手工设计到自动化设计

截至2016年底，卷积神经网络的设计完全依赖于人类专家的经验和直觉——选择卷积核大小、网络深度、通道数、跳跃连接的位置等。这种手工设计范式虽然取得巨大成功，但面临两个根本性局限：

1. **设计空间爆炸。** 对于包含N层的网络，每层需要在卷积核大小、通道数、步长等选项中做出选择，总设计空间随N呈指数增长，穷举搜索不可行。
2. **次优性。** 人类设计师的偏好（如偏好3×3卷积、避免过大的核）可能限制了对更优架构的发现。

Zoph和Le（2017）的工作首次系统性地提出了**神经架构搜索（Neural Architecture Search, NAS）**框架，使用强化学习自动发现高性能的卷积网络架构。

### 第8.2节 NAS的强化学习框架

Zoph和Le将网络架构的设计建模为一个序列生成问题。一个循环神经网络（RNN）控制器（controller）按序生成网络的每一层结构描述。对于CIFAR-10数据集上的卷积网络，控制器在每个时间步输出以下离散 token：

- 一个卷积核大小（从{1, 3, 5, 7}中选择）
- 一个滤波器数量（从{24, 36, 48, 64}中选择）
- 一个步长（从{1, 2}中选择）

控制器在20个时间步内生成20个层的描述，定义一个完整的卷积网络。生成架构后，在训练集上训练该网络并在验证集上评估其精度。验证精度作为奖励信号，通过**策略梯度方法（REINFORCE算法）**更新控制器参数：

$$
\nabla_{\theta} J(\theta) \approx \sum_{t=1}^{T} \nabla_{\theta} \log P(a_t|\theta) \cdot R
$$

其中$P(a_t|\theta)$是控制器在第$t$步选择动作$a_t$的概率，$R$是训练好的子网络的验证精度。这一更新规则使得控制器逐渐学会生成具有更高验证精度的架构。

**搜索空间与训练 pipeline 的细节。** Zoph和Le的搜索空间定义了每个可能层的结构：在每个时间步，控制器首先在一个包含五种操作类型的softmax层上采样——1×1卷积、3×3卷积、5×5卷积、3×3深度卷积（depthwise convolution）和3×3平均池化。如果选择卷积操作，控制器再独立采样滤波器数量（从{24, 36, 48, 64}中选择）和是否添加快捷连接。每个生成的架构在CIFAR-10训练集上训练50个epoch（初始学习率0.025，动量0.9，权重衰减3×10⁻⁴，batch size为128），然后在验证集上评估精度。为了减少方差，奖励信号使用验证精度的移动平均作为基线（baseline），使得策略梯度的估计更加稳定。

值得注意的是，NAS搜索到的架构展现出与人类设计师不同的结构偏好：它倾向于使用更多的跳跃连接（skip connections）——在不同层之间建立直接连接——以及在高层使用更多3×3深度卷积。这种偏好与人类的直觉设计（偏好3×3卷积和5×5卷积交替）有所不同，表明自动化搜索能够发现人类直觉可能忽略的有效结构模式。

### 第8.3节 关键结果与影响

Zoph和Le的NAS框架在CIFAR-10上取得了3.65%的测试错误率，优于当时手工设计的所有模型（包括Inception v2的4.0%和残差网络的4.3%）。更重要的是，NAS发现的架构展现出与人类设计师不同的偏好——例如，NAS倾向于使用更多跳跃连接（skip connections）和更少的3×3卷积，这表明自动化搜索能够发现人类直觉可能忽略的结构模式。

**NAS的后续发展：从搜索空间到权重共享。** Zoph和Le的原始工作开启了一系列后续研究。Zoph等人（2018）的PNAS（Progressive Neural Architecture Search）使用逐步扩展的搜索空间和基于准确率预测的排序模型（surrogate model），将搜索成本降至约225 GPU-days（仅为原始NAS的1%），同时在ImageNet上取得了80.3%的Top-1准确率。Liu等人（2019）的DARTS通过将离散的架构选择松弛为连续的架构参数，使用梯度下降同时优化架构参数和网络权重——搜索成本仅为4块GPU、4天。DARTS的后续工作（如P-DARTS和PC-DARTS）进一步解决了可微架构搜索中的泛化差距问题（architecture collapse），使得搜索到的架构在ImageNet上取得了与手工设计架构相当甚至更优的性能。Fang等人（2020）的AlphaNet将可微分搜索与进化算法相结合，在GPU上实现了实时架构搜索（约1小时），使得针对特定硬件平台的定制化架构设计成为可能。

NAS的另一个重要发展方向是**once-for-all网络**（Cai等人，2020）——训练一个超网络，包含所有可能的子架构，使得在部署时可以根据不同的延迟约束（如移动端20ms vs. 服务器端50ms）直接从超网络中提取最优子网络，无需重新训练。这一范式将NAS从"一次性搜索"转变为"一次训练，多次部署"，大大降低了架构搜索的实际部署成本。

然而，NAS的原始框架面临巨大的计算成本。Zoph和Le使用了800块GPU，训练了大约28天（总计22400 GPU-days）来完成一次搜索。这一成本在随后的研究中得到了大幅降低——Liu等人（2019）的DARTS通过可微分架构搜索将搜索成本降至4块GPU、4天；Pham等人（2018）的ENAS通过权重共享将成本降至16块GPU、7小时。但NAS的种子——将架构设计建模为搜索问题——正是Zoph和Le在2017年奠定的。

### 第8.4节 对黄金时代的桥梁作用

NAS标志着CNN架构研究的一个重要转折。VGG、GoogLeNet和ResNet代表了人类设计师通过直觉和实验驱动发现的架构，而NAS则预示了架构设计的自动化时代。从黄金时代到现代架构搜索的演进，其逻辑脉络清晰可辨：手工设计的洞察（残差连接的有效性、1×1卷积的降维作用、多尺度特征融合的价值）为NAS搜索空间的设计提供了先验知识，而NAS则对这些设计选择进行了系统性的验证和超越。

**ENAS与权重共享的效率革命。** Zoph和Le的原始NAS框架的核心瓶颈在于每个生成架构都需要从头训练——控制器生成一个架构，在CIFAR-10上训练该架构50个epoch评估精度，然后根据精度更新控制器。这意味着每评估一个架构就需要消耗大量计算资源。Pham等人（2018）提出的Efficient NAS（ENAS）通过**权重共享（weight sharing）**机制解决了这一问题：所有由控制器生成的子架构共享同一个超网络（supernet）的权重。具体而言，首先训练一个包含所有可能架构搜索空间的大网络（supernet），然后在控制器生成子架构时，直接从supernet中提取对应的子图权重进行评估，无需从头训练。这一机制将NAS的搜索成本从22400 GPU-days（Zoph & Le, 2017）降至约16块GPU、7小时——压缩了超过10000倍。

ENAS的权重共享机制有两个关键假设：（1）不同架构之间存在"可转移性"——一个在特定架构上学到的权重可以在相似架构中复用；（2）supernet的权重足够好，能够提供有意义的架构评估信号。实验表明，这两个假设在实践中基本成立——ENAS在CIFAR-10上搜索到的架构仅用3.2M参数就取得了2.89%的错误率，优于Zoph和Le的3.65%（使用了4.7M参数）。权重共享的思想后来被DARTS（Liu等人，2019）进一步发扬——DARTS使用可微分的架构参数（architecture parameters）替代离散的架构选择，将搜索空间从离散组合优化转换为连续的优化问题，搜索成本进一步降至4块GPU、4天。

---

## 第9节 ImageNet基准对比：黄金时代的完整图景

### 第9.1节 ImageNet数据集背景

ImageNet Large Scale Visual Recognition Challenge（ILSVRC）使用包含128万张训练图像和5万张验证图像的ImageNet子集，涵盖1000个类别。Top-1错误率衡量单标签预测的准确率，Top-5错误率衡量真实标签是否出现在预测概率最高的5个类别中。2012至2017年间，ILSVRC Top-5错误率从16.4%（AlexNet）降至2.25%（SENet），七年内下降了14个百分点，这一进步在很大程度上归功于架构创新。

### 第9.2节 完整Benchmark对比表

下表汇总了CNN黄金时代各代表性架构在ImageNet验证集上的性能，包括Top-1/Top-5错误率、参数量和计算量（FLOPs）。所有数据均来自各论文的原始报告，使用单裁剪（single-crop）或10裁剪评估。

| 架构 | 年份 | 深度 | 参数量（M） | FLOPs（G） | Top-1错误率（%） | Top-5错误率（%） |
|------|------|------|------------|-----------|----------------|----------------|
| AlexNet | 2012 | 8 | 60 | 1.4 | 40.7 | 18.2 |
| VGG-16 | 2014 | 16 | 138 | 15.3 | 27.0 | 8.8 |
| VGG-19 | 2014 | 19 | 144 | 19.6 | 27.3 | 9.0 |
| GoogLeNet (v1) | 2014 | 22 | 6.8 | 1.5 | — | 6.67† |
| ResNet-18 | 2016 | 18 | 11.7 | 1.8 | 27.9 | — |
| ResNet-34 | 2016 | 34 | 21.8 | 3.6 | 25.0 | 7.8 |
| ResNet-50 | 2016 | 50 | 25.6 | 3.8 | 22.9 | 6.7 |
| ResNet-101 | 2016 | 101 | 44.5 | 7.6 | 21.8 | 6.1 |
| ResNet-152 | 2016 | 152 | 60.2 | 11.3 | 21.4 | 5.7 |
| Inception-v3 | 2015 | 48 | 23.8 | 5.0 | 21.2 | 5.6 |
| Inception-v4 | 2016 | — | 43 | — | 20.0 | 5.0 |
| Inception-ResNet-v2 | 2016 | — | 55.8 | — | 19.9 | 4.9 |
| MobileNet-v1 | 2017 | 28 | 4.2 | 0.57 | 29.4 | — |
| MobileNet-v2 | 2018 | — | 3.4 | 0.30 | 28.0 | — |

*注：除非另有说明，错误率为单模型结果。AlexNet为单模型中心裁剪（Krizhevsky等人，2012）；VGG为单尺度$S=Q=256$（Simonyan和Zisserman，2015，Table 3，配置D/E）；ResNet为10裁剪验证集结果（He等人，2016，Table 2–3），其FLOPs取自原文Table 1（乘加次数）；GoogLeNet的†6.67%为7模型×144裁剪集成测试结果，单模型单裁剪约10.07%（Szegedy等人，2015，Table 3）；Inception-v3为单帧评估（Szegedy等人，2016）；MobileNet为Top-1准确率换算的错误率，FLOPs按原文Mult-Adds计（Howard等人，2017；Sandler等人，2018）。不同论文对FLOPs/Mult-Adds计数约定不一，横向比较时需注意。*

**训练超参数对比。** 各架构在ImageNet上的训练设置存在显著差异，这些差异对最终性能有不可忽视的影响。下表汇总了各论文报告的关键训练超参数：

| 架构 | 优化器 | 初始学习率 | Batch Size | 训练长度 | 学习率衰减策略 | 权重衰减 | 数据增强 |
|------|--------|-----------|------------|----------|--------------|---------|----------|
| AlexNet | SGD | 0.01 | 128 | 90 epoch | 验证误差停滞时×0.1 | 0.0005 | 随机裁剪+水平翻转+PCA颜色增强 |
| VGG-16/19 | SGD | 0.01 | 256 | 约370k迭代（~74 epoch） | 验证误差停滞时×0.1 | 0.0005 | 随机裁剪+水平翻转+尺度抖动 |
| GoogLeNet | ASGD | — | — | — | 每8 epoch×0.96 | — | 随机裁剪+光度扭曲 |
| ResNet-50 | SGD | 0.1 | 256 | 约$60\times 10^{4}$迭代（~120 epoch） | 误差平台期÷10 | 0.0001 | 短边$[256,480]$尺度增强+随机裁剪 |
| Inception-v3 | RMSProp | 0.045 | 32×50 | 100 epoch | 每2 epoch×0.94 | — | 随机裁剪+水平翻转 |

值得注意的是，VGG-16/19的训练使用了约370k次迭代（batch size 256，约合74个epoch），这是VGG能够在较深网络上收敛的关键条件之一。GoogLeNet使用异步SGD与"每8个epoch学习率衰减4%"（即×0.96）的固定策略。ResNet则训练至约$60\times 10^{4}$次迭代（batch size 256，约120个epoch），学习率在误差进入平台期时除以10，而非固定在第30/60/90 epoch衰减。

**评估协议的影响。** ILSVRC评估中存在多种协议，直接比较不同论文报告的错误率时需要谨慎。主要差异包括：
- **单裁剪（single-crop）：** 从图像中裁剪一个224×224区域进行评估，速度快但精度估计偏乐观。
- **十裁剪（10-crop）：** 从图像的四角和中心各裁剪一个224×224区域，加上水平翻转共10个裁剪，取平均预测结果。GoogLeNet和VGG使用了多裁剪融合（multi-crop + dense evaluation），通常比单裁剪低1-2个百分点。
- **多尺度密集评估（multi-scale dense evaluation）：** 先对图像进行多个尺度缩放，然后在每个缩放后的图像上进行密集滑动窗口裁剪，综合所有裁剪的预测结果。Inception-v3和ResNet论文部分使用了这一方法。

由于GoogLeNet的6.67%是7模型×144裁剪集成结果，而ResNet-152的10裁剪单模型Top-5约为5.71%、多尺度约为4.49%，二者评估协议不同，不宜直接横向比较。集成与单模型、多裁剪与单裁剪之间通常可差1–3个百分点。

### 第9.3节 数据分析与趋势解读

从表中可以观察到几个关键趋势：

**趋势一：深度与精度的正相关性（在残差连接出现之前）。** 在VGG和GoogLeNet时代，更深的网络确实倾向于取得更低的错误率——但边际收益递减明显。VGG-19与VGG-16在单尺度评估下Top-1接近（约27.3% vs. 27.0%），却增加了约28%的计算量（19.6G vs. 15.3G）。GoogLeNet在仅22层的情况下，以远少于VGG的参数达到了有竞争力的集成结果（Top-5 6.67%），靠的是Inception模块的参数效率而非单纯的深度。值得注意的是，AlexNet仅有8层（含参数层），其单模型Top-1错误率（约40.7%）与VGG-16的约27.0%之间存在超过13个百分点的差距——这说明从"浅层网络"到"深层网络"的跨越是最具决定性的单一步骤，而在此基础上的深度增加（16层→19层→50层→152层）带来的是渐进式的精细改进。

**趋势二：残差连接打破了深度-精度关系的瓶颈。** ResNet的出现使得深度与精度之间恢复了稳健的正相关关系。ResNet-50（50层）的10裁剪Top-1错误率约22.9%，已明显优于VGG-19的单尺度约27.3%，而参数量仅为VGG-19的约18%。ResNet-152的10裁剪Top-1约21.4%、Top-5约5.7%。Inception-ResNet-v2报告约19.9% Top-1 / 4.9% Top-5，略优于同期ResNet-152单模型，说明Inception模块与残差连接的结合比单独使用残差连接更有效。这一结果直接推动了后续Inception-ResNet系列和SENet（Hu等人，2018）的发展。

**趋势三：参数效率的代际跃迁。** GoogLeNet以约6.8M参数达到与VGG-16（138M参数）有竞争力的结果，参数效率提升约20倍。ResNet系列延续了这一趋势——ResNet-50用约25.6M参数超过了VGG-19的144M参数。这一趋势在MobileNet中达到顶峰——仅用4.2M参数和约0.57G Mult-Adds，MobileNet-v1就实现了约70.6%的Top-1准确率（错误率约29.4%），为移动端部署铺平了道路。从VGG到MobileNet，参数量约从138M降至4.2M，Mult-Adds从约15.3G降至约0.57G，压缩比约27倍，精度损失仍在可控范围内。

**趋势四：深度可分离卷积的极端参数效率。** MobileNet-v1约0.57G Mult-Adds，约为VGG-16（15.3G）的4%，在移动设备上实现了实时推理。MobileNet-v2进一步将计算量降至约0.30G Mult-Adds、参数约3.4M，同时保持约72%的Top-1准确率（错误率约28%），证明了深度可分离卷积在极端资源约束下的可行性。

**从单一指标到综合评估。** 表格中的Top-1/Top-5错误率只是评估网络性能的一个维度。实际部署中还需要考虑：推理延迟（latency）、内存占用（model size）、能耗（energy consumption，对移动端尤为重要）和可训练性（training stability）。例如，ResNet-152虽然精度很高，但其约11.3G FLOPs和约60M参数在实时应用中往往不可接受；VGG-19的计算量中很大一部分来自全连接层——如果去掉全连接层（如使用全局平均池化），VGG的计算量可以大幅降低，精度损失极小。这些"纸上指标"与"实际性能"之间的差距，正是架构工程师需要在实际部署中解决的核心问题。

### 第9.4节 推理速度与模型压缩

错误率只是评估的维度之一。在真实部署场景中，推理延迟（inference latency）和模型大小（model size）同样关键。下表列出了各架构在单块NVIDIA Tesla K40 GPU上的典型单图像推理时间（forward pass only）：

| 架构 | 模型大小（MB） | 推理时间（ms/image） | FLOPs/Mult-Adds（G） | 精度-速度权衡比 |
|------|--------------|-------------------|---------------------|--------------|
| AlexNet | 约240 | 1.4 | 约1.4 | 基准 |
| VGG-16 | 约528 | 4.2 | 15.3 | 低（大而慢） |
| GoogLeNet | 约27 | 2.8 | 约1.5 | 高 |
| ResNet-50 | 约98 | 4.1 | 3.8 | 中 |
| ResNet-152 | 约230 | 9.1 | 11.3 | 低 |
| MobileNet-v1 | 约17 | 3.2 | 0.57 | 极高 |

从表中可以看出，GoogLeNet在精度和速度之间取得了很好的平衡——约27MB模型大小和较低推理延迟，集成Top-5可达6.67%，参数量远少于VGG-16（约528MB）。ResNet-50单模型精度更高（10裁剪Top-5约6.7%），但模型体积更大。MobileNet-v1以约17MB和0.57G Mult-Adds实现了移动端实时推理，证明了极低计算预算下依然可以保持可接受的精度。

**批处理（batch inference）的影响。** 在实际服务场景中，图像通常以batch形式批量推理。GPU的并行计算能力使得大批量推理的吞吐量（throughput）远高于单图像推理。例如，在Tesla V100 GPU上，ResNet-50处理256张图像的batch仅需约12ms（折合约0.047ms/image），而MobileNet-v1处理相同batch仅需约3ms（折合约0.012ms/image）。大批量推理下，精度-速度权衡的天平向计算量更大的模型倾斜——ResNet-50的高FLOPs被GPU的并行能力摊销后，单张图像的等效延迟大幅降低。这一现象解释了为什么大规模推荐系统和搜索引擎中的视觉特征提取仍然倾向于使用ResNet-50而非MobileNet——在批量处理场景中，ResNet-50的吞吐量优势可以弥补其单张延迟的劣势。

**模型压缩技术。** 除了架构层面的参数效率优化外，模型压缩技术（量化、剪枝、知识蒸馏）是降低部署成本的另一条重要途径。量化将32位浮点权重替换为8位整数甚至更低精度，模型大小可压缩4-8倍，推理速度提升2-4倍，精度损失通常控制在1%以内。Han等人（2015）的深度压缩（deep compression） pipeline 结合剪枝、量化和霍夫曼编码，将AlexNet的模型大小从233MB压缩至6.9MB（压缩比约34倍），精度损失几乎为零。知识蒸馏（Hinton等人，2015）通过让学生网络模仿教师网络的软标签（soft labels），使得小模型可以继承大模型的知识——例如，用ResNet-152作为教师网络训练MobileNet，可以在不改变学生架构的前提下进一步降低其错误率。

**下游任务：检测、分割与关键点估计。** ImageNet分类精度只是衡量架构性能的一个维度。在目标检测（COCO）、语义分割（Cityscapes, ADE20K）和人体关键点估计（MPII Human Pose）等下游任务上，不同架构的表现差异更加显著。以COCO目标检测为例（使用Faster R-CNN框架）：

| 主干网络 | COCO mAP (minival) | 参数量 | 推理速度 |
|----------|-------------------|--------|---------|
| ResNet-50 | 37.8 | 41M | 0.14s/image |
| ResNet-101 | 40.8 | 62M | 0.20s/image |
| ResNet-152 | 42.0 | 83M | 0.26s/image |
| VGG-16 | 33.7 | 136M | 0.38s/image |

从表中可见，ResNet系列在检测任务上全面超越了VGG-16——ResNet-50的mAP（37.8%）比VGG-16（33.7%）高出4.1个百分点，同时参数更少、速度更快。这一优势主要来自于两个因素：（1）ResNet的深层特征具有更强的语义表达能力，有利于分类子网络（classification head）的决策；（2）残差连接使得深层特征图（如ResNet的C4和C5阶段）保留了更多的空间结构信息，有利于区域提议网络（RPN）和边界框回归（bbox regression）。值得注意的是，在实例分割任务上，使用ResNet-50-C4的Mask R-CNN取得了37.1%的mask mAP，而使用ResNet-50-FPN（特征金字塔网络，在ResNet-50上添加FPN）进一步提升至38.2%——FPN通过融合多尺度特征，显著改善了小物体分割的性能。

在人体关键点估计任务上，Cao等人（2017）的OpenPose使用VGG-19作为初始特征提取主干网络，在MPII数据集上取得了87.5%的PCKh@0.5。然而，后续使用ResNet-50的工作迅速超越了这一结果——He等人（2017）的Mask R-CNN使用ResNet-50-FPN在COCO关键点任务上取得了65.1%的AP，证明了残差连接在精确定位任务上的优势。

这些下游任务的性能差异进一步验证了黄金时代架构改进的实际价值——ImageNet Top-1错误率降低1个百分点，在COCO检测上可能对应2-3个mAP的提升，在Cityscapes分割上可能对应3-4个mIoU的提升。架构创新在金字塔式传递中得到了放大。

回望2014-2018年的架构演化，可以梳理出一条清晰的逻辑主线：

**VGG（2014）→ GoogLeNet（2014）→ ResNet（2016）→ MobileNet（2017-2018）→ NAS（2017-）**

VGG证明了"简单构件深度堆叠"的可行性，但也暴露了全连接层和连续大卷积层的参数瓶颈。GoogLeNet通过多尺度并行路径和1×1降维解决了瓶颈，引入了辅助分类器和全局平均池化。ResNet通过残差连接解决了优化问题，使得构建超深度网络成为可能——这是整个黄金时代最重要的贡献。MobileNet将深度可分离卷积推向实用，开启了高效网络的新方向。NAS则将架构设计从手工劳动中解放出来，开启了自动化研究的序幕。

每一个架构都不是凭空出现的。它们站在前人的肩膀上，针对前人的局限做出了回应，同时又开启了新的可能性。理解这一脉络，对于把握深度学习的发展方向至关重要。

---

## 第10节 训练实践：架构创新背后的工程智慧

### 第10.1节 学习率与批大小的协同演化

黄金时代的架构创新与训练实践的演进是密不可分的。VGG的训练使用了初始学习率0.01（对应batch size 256）配合验证误差停滞时×0.1的学习率调度，而GoogLeNet采用异步SGD，并以**每8个epoch将学习率乘以0.96**的固定指数衰减策略（与上文训练超参数表一致）。这些差异并非随意选择，而是与架构特性紧密相关。

GoogLeNet之所以能在较平滑的优化路径上稳定收敛，部分归功于Inception模块中的批归一化和辅助分类器。批归一化通过对每层输入的归一化，使得损失 landscape 更加平滑，允许使用更大的步长而不会发散。Ioffe和Szegedy在原始BN论文中报告了一个关键数据：使用BN后，VGG-style网络的训练速度提升了14倍——从13天缩短至不到1天。这一加速效果部分来自于更大的有效学习率，部分来自于减少了对手动学习率调度的依赖。

### 第10.2节 权重初始化的范式转换

在BN普及之前，权重初始化是深度网络训练中最关键的工程决策之一。Xavier/Glorot初始化（2010）基于保持前向传播和反向传播中方差不变的原理，推导出均匀分布$U(-\sqrt{6/(n_{\text{in}}+n_{\text{out}})}, \sqrt{6/(n_{\text{in}}+n_{\text{out}})})$作为权重初始值。然而，Xavier初始化的推导假设激活函数关于零对称（如tanh），对于ReLU及其变体并不适用——ReLU将负值截断为零，导致前向传播的方差减半。

He等人（2015）针对ReLU激活函数推导了新的初始化方案：权重从$N(0, \sqrt{2/n_{\text{in}}})$的正态分布中采样，其中$n_{\text{in}}$是输入单元的数量。这一"He初始化"确保在前向传播中，每一层的输出方差保持与输入方差一致（考虑到ReLU只激活一半的单元，方差补偿因子为2）。实验表明，使用He初始化的网络比Xavier初始化收敛更快，最终精度也更高——对于VGG-16，He初始化将Top-1错误率降低了约1-2个百分点。

对于残差网络，初始化策略需要进一步调整。He等人（2016）提出：残差块中的最后一个卷积层（将通道从瓶颈维度恢复到原始维度的层）应初始化为零，使得残差映射$\mathcal{F}(x)$在训练开始时为零，整个残差块退化为恒等映射$\mathbf{x}_{l+1} = \mathbf{x}_l$。这一初始化策略确保了深层残差网络在训练初期的梯度流动是稳定的，避免了"坏初始化"导致的梯度消失或爆炸。

### 第10.3节 学习率预热（Warm-up）

当使用大批量训练（如batch size ≥ 256）时，即使有BN和合适的权重初始化，训练初期的大学习率仍可能导致梯度不稳定。Goyal等人（2017）在训练ImageNet上的ResNet-50时发现：如果直接使用初始学习率0.1（对应batch size 256），训练损失在前几个epoch内会剧烈震荡甚至发散。他们提出**线性学习率预热（linear warm-up）**策略——在训练开始的5个epoch内，学习率从0线性增长到目标初始学习率0.1，然后再按照正常的衰减策略（每30epoch×0.1）进行衰减。

预热策略的统计解释是：在训练初期，BN的运行均值（running mean）和运行方差（running variance）尚未收敛到真实值，归一化效果不稳定。此时使用过大的学习率更新参数，会导致归一化的统计估计进一步偏离，形成恶性循环。预热策略通过逐步增大学习率，让BN的统计量先稳定下来，再进入正常的优化轨道。这一策略在大批量训练中已成为标配——从ResNet到ViT，几乎所有大规模训练的流程都包含学习率预热。

### 第10.4节 数据增强的协同演进

数据增强是ImageNet训练中精度提升的关键因素之一，与架构创新形成了协同效应：

- **随机裁剪（Random Resized Crop）：** 从随机位置和比例裁剪224×224区域。GoogLeNet和VGG使用了尺度抖动（scale jittering）——短边在256到512之间随机缩放后再裁剪，迫使模型在多种尺度下识别物体。
- **水平翻转（Random Horizontal Flip）：** 以0.5概率水平翻转图像。这是一个几乎无成本的增强，对所有架构都有约1-2个百分点的精度提升。
- **颜色增强：** AlexNet使用了PCA颜色增强（PCA jittering）——对RGB通道进行PCA，在主方向上按一定幅度添加随机扰动。GoogLeNet则采用光度扭曲（photometric distortions）等颜色扰动；VGG-16/19使用了更简单的随机颜色抖动（随机调整亮度、对比度、饱和度）。
- **Cutout / Random Erasing：** 在训练图像上随机遮挡矩形区域，迫使模型不依赖于局部的、可能具有误导性的特征。DeVries和Taylor（2017）证明Cutout可以将CIFAR-10上的错误率从3.73%（标准训练）降至3.31%。

这些数据增强策略与架构设计之间存在协同作用。例如，批归一化通过对输入分布的归一化，使得模型对输入颜色的微小变化更加鲁棒，从而放大了颜色增强的正面效果。残差连接通过提供恒等映射路径，使得模型在信息经过被遮挡区域时仍能保持完整，增强了Cutout的有效性。

---

## 第11节 结语：黄金时代遗产与未来方向

CNN架构的黄金时代（2014-2017）奠定了现代深度学习的工程基础。VGGNet的3×3卷积哲学、GoogLeNet的Inception模块、ResNet的残差连接、批归一化的统计学框架、深度可分离卷积的分解思想、神经架构搜索的自动化范式——这些贡献共同塑造了当前卷积神经网络的设计语言。

从遗产角度看，残差连接已成为几乎所有现代深度架构的标配——从Transformer的残差连接到扩散模型的残差去噪，$\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l)$这一形式已经超越了卷积网络的范畴，成为深度生成模型的标准组件。批归一化及其变体（层归一化、组归一化）同样广泛存在于大语言模型的训练 pipeline 中。Inception模块中"多尺度并行处理"的思想在特征金字塔网络（FPN）和U-Net的跳跃连接中得到了延续。深度可分离卷积是移动端和边缘端AI的事实标准。神经架构搜索则已经发展为一个独立的研究领域，DARTS、ENAS、PNAS、Once-for-All等后续工作不断拓展着搜索空间和效率的边界。

黄金时代确立的设计原则——"先归一化，再加残差"——在Transformer架构中获得了最有力的验证。Vaswani等人（2017）的原始Transformer论文中，每个注意力层和前馈网络层之后都跟随残差连接和层归一化（Layer Normalization），这与ResNet中"卷积层 + BN + 残差连接"的模式如出一辙。这种跨架构的共性表明：**残差连接和归一化不是卷积网络的特殊技巧，而是训练任意深度非线性模型的通用原则。** 无论模型的底层操作是卷积、注意力还是全连接，优化问题的本质——在深层网络中保持有效的梯度流——是相同的。

**集成学习（Ensemble Learning）与黄金时代架构。** 在ILSVRC竞赛中，集成多个模型的预测结果是提升精度的标准策略。2015年ILSVRC分类任务的获胜方案是**6个不同深度残差网络**的集成（其中仅含两个152层模型），将测试集Top-5错误率降至3.57%（He等人，2016，Table 5）。集成策略包括：（1）多裁剪/多尺度集成——对同一图像的不同裁剪位置和尺度分别预测后取平均；（2）多模型集成——将不同深度或架构的预测概率取平均；（3）测试时增强（test-time augmentation, TTA）。多尺度评估本身可以将单模型错误率降低约1–2个百分点，而多模型集成可以进一步降低0.5–1个百分点。Inception-v4与Inception-ResNet等工作随后将集成Top-5进一步压至约3.08%（Szegedy等人，2016）。然而，集成策略的成本是推理延迟和模型存储的成倍增加——多个ResNet-152量级模型的推理时间与内存占用在真实部署中往往不可接受。

**对抗鲁棒性与架构选择。** 黄金时代的架构在对抗样本（adversarial examples）面前表现出不同的鲁棒性。Goodfellow等人（2015）发现，VGG-16和GoogLeNet对FGSM（Fast Gradient Sign Method）攻击的鲁棒性相近——干净样本上约27%的Top-1错误率在攻击下会急剧上升。ResNet由于使用了更多的非线性层（瓶颈结构中的三层堆叠），在某些攻击下的鲁棒性略优于VGG，但差距不大。这一观察表明：架构设计主要影响"干净精度"（clean accuracy），而对抗鲁棒性更多地取决于训练策略（对抗训练、数据增强）而非架构本身。Szegedy等人（2014）在GoogLeNet相关工作中报告了对抗样本对深度网络的影响——即使使用多尺度裁剪和融合，模型仍会被微小扰动（人眼不可见的像素级修改）欺骗，输出完全错误的分类结果。这一发现引发了深度学习安全领域的广泛关注。

从未来方向看，黄金时代确立的几个核心原则——残差连接、归一化、多尺度特征融合——在Vision Transformer（ViT, 2020）及其后续变体（Swin Transformer、ConvNeXt）中获得了新的诠释。ConvNeXt尤其有趣——它通过将ResNet"现代化"（引入大型卷积核、LayerNorm、GELU激活等），在ImageNet上达到了与ViT相当甚至更优的性能，证明了黄金时代架构的强大生命力和可演进性。Dosovitskiy等人（2020）的ViT将图像切分为16×16的patch序列，直接用Transformer编码器处理——这一做法放弃了卷积归纳偏置（inductive bias），却在大规模预训练（JFT-300M数据集）下取得了超越ResNet的性能。然而，当训练数据规模缩小时，ViT的性能急剧下降——Liu等人（2021）的Swin Transformer通过引入分层特征表示和局部注意力窗口，将卷积归纳偏置重新引入Transformer，使得模型在小规模数据集上也能有效训练。这一来回摆动（从CNN到Transformer，再回到类CNN结构）恰恰说明了黄金时代架构的根基之深——即使是最激进的新架构，也往往需要在某种程度上回归那些已被证明有效的设计原则。

**架构设计的"无免费午餐"定理。** 回顾黄金时代的架构演进，一个重要的教训是：不存在 universally optimal 的架构。VGG在规整性任务（如细粒度分类）上表现优异，但在需要大感受野或高效参数利用的任务中力不从心。GoogLeNet在参数效率和精度之间取得了出色平衡，但Inception模块的复杂实现和调试难度限制了其普及。ResNet在几乎所有视觉任务上都是可靠的基线选择，但其瓶颈结构中的1×1卷积在低算力设备上仍然是瓶颈。MobileNet在移动端实现了革命性的效率提升，但在精度要求高的服务器端场景中不适用。NAS能够发现针对特定任务和硬件平台的最优架构，但搜索成本高昂且搜索到的架构可解释性差。

这种"没有银弹"的格局实际上是一种健康的信号——它表明CNN架构设计仍然是一个充满创造力的开放领域，而非已被完全解决的工程问题。每个架构的优劣取决于具体的使用场景（数据集规模、计算预算、延迟要求、精度要求、硬件平台），而架构工程师的核心能力正是在这些约束条件下做出最优权衡。黄金时代留下的最重要遗产或许不是具体的架构设计（3×3卷积、残差连接、Inception模块），而是**架构设计的系统性方法论**——从理解优化动力学的角度出发，通过数学分析和实验验证相结合的方式，逐步推进网络的设计边界。

黄金时代的另一个未竟主题是**架构的鲁棒性和泛化边界**。ResNet的成功引发了大量关于"为什么深度网络能够泛化"的理论研究——从PAC-Bayes框架到神经切向核（NTK）理论，从信息瓶颈理论到最小描述长度原则。这些理论研究为理解深度学习的泛化行为提供了重要的理论工具，但至今仍未有一个统一的理论框架能够完全解释为什么ResNet-152在ImageNet上能够达到21.3%的Top-1错误率。黄金时代留给我们的，不仅是实用的架构设计，还有一系列深刻的、尚未完全解答的理论问题。

深度学习的研究永不停歇，但黄金时代的架构创新——那种通过深刻的数学洞察和工程直觉推动领域前进的方式——将作为典范，持续激励着后来的研究者。

**自监督学习：黄金时代架构的新使命。** 进入2020年代，当ImageNet分类的精度提升进入瓶颈期后，研究焦点从"监督学习的最佳架构"转向了"自监督学习的最佳架构"。SimCLR（Chen等人，2020）和MoCo（He等人，2019）等工作使用对比学习（contrastive learning）在无标签的ImageNet数据上进行预训练，然后在下游任务上微调——这些工作的骨干网络（backbone）几乎无一例外地选择了ResNet-50。这一选择并非偶然：ResNet的残差结构提供了稳定的梯度流，使得网络在大量迭代的对比学习训练中不会退化；批归一化提供了对特征分布的约束，使得对比损失能够在规范化的特征空间中进行有效的度量学习。黄金时代架构的工程遗产——残差连接、归一化、全局平均池化——在自监督学习时代继续发挥着核心作用。

从AlexNet到ConvNeXt，从监督预训练到对比学习，CNN架构的演化始终围绕着一个核心主题：**如何让信息在网络中高效、稳定地流动？** VGG用统一的3×3卷积回答了这一问题，GoogLeNet用Inception模块回答了这一问题，ResNet用残差连接回答了这一问题。每一个答案都是对前一个答案的补充和完善，而非简单的替代。理解这一演化脉络，对于把握深度学习未来发展的方向具有重要的指导意义。

**黄金时代架构在生成模型中的延伸。** 值得特别指出的是，ResNet的残差结构在生成模型领域找到了意想不到的应用。扩散模型（diffusion models, Ho等人，2020）的每一步去噪过程本质上是一个残差更新：

$$
\mathbf{x}_{t-1} = \mathbf{x}_t + \mathcal{F}(\mathbf{x}_t, t, \theta)
$$

其中$\mathcal{F}$预测噪声分量，$\mathbf{x}_t$是当前噪声图像。这一结构与ResNet的残差块完全相同——Stable Diffusion、DALL-E 2、Midjourney等主流生成模型的骨干网络无一例外地使用了基于ResNet的U-Net架构。批归一化在扩散模型的训练中也扮演着关键角色，确保了去噪过程的稳定性。可以说，黄金时代架构的遗产不仅体现在分类、检测、分割等判别任务中，更在2020年代最引人注目的生成式AI革命中发挥了不可替代的作用。

**黄金时代架构对AI研究的深远影响。** 从更广阔的视角审视，VGG、GoogLeNet和ResNet所确立的设计原则已经超越了计算机视觉领域。在自然语言处理中，Transformer的残差连接和层归一化直接继承了ResNet和BN的设计理念。在语音识别中，Conformer架构将CNN的局部特征提取能力与Transformer的全局建模能力相结合，其中CNN部分使用了深度可分离卷积。在强化学习中，价值函数的迭代更新与贝尔曼方程的残差结构高度相似。在科学计算中，神经算子（neural operators）使用残差网络学习偏微分方程的算子映射。这些跨领域的应用表明：黄金时代架构创新所解决的问题——如何在深层非线性网络中实现稳定、高效的优化——是一个具有普遍意义的科学问题，其解决方案具有超越图像分类的广泛适用性。

**VGG风格的持续生命力。** 尽管ResNet在几乎所有指标上超越了VGG，VGG-16和VGG-19至今仍是学术研究和工程实践中最常用的特征提取器之一。在风格迁移（Gatys等人，2016）中，VGG-16被用作内容特征和风格特征的主干网络——其规整的多层结构使得不同层的特征图可以方便地用于内容损失和风格损失的计算。在超分辨率（Super-Resolution）中，SRGAN（Ledig等人，2017）使用VGG-19的conv5_4层特征计算感知损失（perceptual loss），在视觉效果上显著优于仅使用MSE损失的模型。在医学图像分析中，VGG-16因其简单的架构和良好的可解释性（3×3卷积的叠加具有明确的感受野计算方式），在小规模医学数据集上仍然是首选的预训练主干网络。这些应用说明：架构的"过时"往往是相对于特定任务和数据集而言的——VGG的规整性和可解释性在某些场景中仍然是宝贵的优势。

**从ImageNet到更大规模预训练。** ILSVRC竞赛结束后，研究的焦点从ImageNet-1K（1000类，128万张图像）扩展到了更大规模的预训练数据集。ImageNet-21K（约1400万张图像，21000类）为预训练提供了更丰富的视觉概念覆盖。在ImageNet-21K上预训练的ResNet-50（使用渐进式缩放训练策略）在下游任务上取得了显著优于ImageNet-1K预训练的性能——在COCO检测上，mAP从37.8%提升至约41.0%。这一提升直接归因于更大规模预训练带来的更丰富的特征学习。黄金时代架构的另一个关键遗产——全局平均池化（GAP）——在这一规模扩展中发挥了关键作用：GAP使得模型可以在任意输入分辨率下进行推理（因为GAP不依赖于固定的空间尺寸），为大分辨率预训练和多尺度微调提供了灵活性。

**黄金时代的终结与延续。** 2017年ILSVRC竞赛结束后，ImageNet分类任务作为一个独立的基准研究逐渐降温——不是因为问题被完全解决，而是因为研究社区认识到：在ImageNet上再提升0.1个百分点的Top-1错误率，对实际应用的价值有限。研究焦点转向了更具挑战性的任务：开放词汇识别（open-vocabulary recognition）、少样本学习（few-shot learning）、视频理解（video understanding）、三维视觉（3D vision）等。然而，这些新任务的基线模型（baseline）几乎无一例外地使用了ResNet-50或ResNet-101作为特征提取器。黄金时代架构的遗产不仅体现在它们自身的性能数据上，更体现在它们作为后续一切视觉研究的基础设施（infrastructure）所发挥的作用——如同电力或互联网，优秀的架构设计在达到一定成熟度后会"融入背景"，成为研究者习以为常的默认选择，而不再被显式地讨论。这正是VGG、GoogLeNet和ResNet等架构所达到的境界。

---

## 参考文献

[1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 25.

[2] Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition. *arXiv:1409.1556*.

[3] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., & Rabinovich, A. (2015). Going Deeper with Convolutions. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.

[4] Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *International Conference on Machine Learning (ICML)*, 37, 448-456.

[5] Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How Does Batch Normalization Help Optimization? *Advances in Neural Information Processing Systems (NeurIPS)*, 31.

[6] Köhler, J., Daneshmand, H., Lucchi, A., Ramesh, R., Neymeyr, M., & Zhou, M. (2019). Exponential Convergence Rates for Batch Normalization: The Power of Length-Direction Decoupling in Non-Convex Optimization. *International Conference on Artificial Intelligence and Statistics (AISTATS)*, 22, 806-815.

[7] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR*.

[8] Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. (2017). Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning. *AAAI Conference on Artificial Intelligence*, 31(1).

[9] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.

[10] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. *European Conference on Computer Vision (ECCV)*, 630-646.

[11] Haber, E., & Ruthotto, L. (2017). Stable Architectures for Deep Neural Networks. *Inverse Problems*, 34(1), 014004.

[12] E, W. (2017). A Proposal on Machine Learning via Dynamical Systems. *Communications in Mathematics and Statistics*, 5(1), 1-11.

[13] Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. *Advances in Neural Information Processing Systems (NeurIPS)*, 31.

[14] Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. *CVPR*.

[15] Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Sandler, M., Lee, H., Liu, W., & Adam, H. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. *arXiv:1704.04861*.

[16] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*.

[17] Zoph, B., & Le, Q. V. (2017). Neural Architecture Search with Reinforcement Learning. *International Conference on Learning Representations (ICLR)*.

[18] Liu, H., Simonyan, K., & Yang, Y. (2019). DARTS: Differentiable Architecture Search. *ICLR*.

[19] Pham, H., Guan, M. Y., Zoph, B., Le, Q. V., & Dean, J. (2018). Efficient Neural Architecture Search via Parameter Sharing. *International Conference on Machine Learning (ICML)*, 80, 4092-4101.

[20] Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely Connected Convolutional Networks. *CVPR*.