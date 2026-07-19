export const meta = {
  name: 'ai-history-book',
  description: 'AI通史：从图灵到2024，覆盖所有领域、公式、原理、故事',
  phases: [
    { title: 'Part 1', detail: '起源到1980s (4并行章节)' },
    { title: 'Part 2', detail: '1990s 统计学习时代 (4并行章节)' },
    { title: 'Part 3', detail: '2000s 深度学习萌芽 (4并行章节)' },
    { title: 'Part 4', detail: '2012-2017 深度学习爆发 (4并行章节)' },
    { title: 'Part 5', detail: '2018-2024 Transformer时代 (4并行章节)' },
    { title: 'Part 6', detail: '各领域专题深潜 (4并行)' },
    { title: 'Part 7', detail: '综合编年史+索引' },
  ],
}

// ===== Helper =====
async function writeChapter(label, prompt) {
  return await agent(prompt, { label, effort: 'high' })
}

async function batch4(labels, prompts) {
  return await parallel(labels.map((label, i) => () =>
    agent(prompts[i], { label, effort: 'high' })
  ))
}

// ===== Part 1: 起源到1980s =====
phase('Part 1: 起源与黄金时代 (1940-1980)')
const p1 = await batch4(
  ['ch01', 'ch02', 'ch03', 'ch04'],
  [
    // Ch1: 图灵与AI的诞生
    `写一篇20000字章节"图灵与人工智能的诞生"。
内容要求：
- 图灵的生平、图灵机的设计思想、可计算性理论
- 1950年《计算机器与智能》的完整解读（包括图灵测试的原始表述、常见误解）
- 1956年达特茅斯会议的完整经过：参会人员名单、每人贡献、会议提案原文解读
- "人工智能"这个词的命名故事（ McCarthy vs Minsky 的争议）
- 早期 optimism：Newell和Simon的Logic Theorist、GPS通用解题系统
- 当时的计算资源限制（IBM 701、内存大小、运算速度）
- 历史照片描述和时间线
必须完整输出，包含所有细节。`,

    // Ch2: 感知机与符号主义
    `写一篇20000字章节"感知机与连接主义的兴衰"。
内容要求：
- Rosenblatt感知机的完整数学原理：权重更新规则、几何解释、收敛性证明（Cover定理）
- 感知机能解决什么不能解决什么问题（AND、OR、XOR的完整分析）
- Minsky和Papert《Perceptrons》一书的出版背景、核心结论（线性可分性的严格证明）
- 连接主义的第一次冬天：资金削减、人员流失、研究转向
- 符号主义AI的崛起：启发式搜索、专家系统、知识工程
- Newell和Simon的物理符号系统假说（PSSH）的完整陈述和争议
- 历史故事：Rosenblatt的早逝、Minsky的学术影响
必须完整输出。`,

    // Ch3: 第一次AI冬天
    `写一篇15000字章节"第一次AI冬天与专家系统的兴衰"。
内容要求：
- 1970年代AI研究的过度承诺和实际差距
- Lighthill报告（1973年）的详细内容和影响
- 英国、美国政府对AI资金的削减过程
- 专家系统的崛起：DENDRAL、MYCIN、XCON的完整技术细节
- 知识工程方法论：知识获取瓶颈、知识表示（产生式规则、语义网络、框架）
- 专家系统的商业成功和最终局限性
- 从冬天到春天的转折信号
必须完整输出。`,

    // Ch4: 统计学习方法的萌芽
    `写一篇15000字章节"统计学习思想的萌芽"。
内容要求：
- 从符号主义到统计学习的范式转换背景
- Bayes统计在AI中的应用早期探索
- 信息论与AI：Shannon熵、信息增益、决策树的思想源头
- 早期模式识别：Fisher线性判别、最近邻方法、聚类分析
- 感知机的统计视角：随机梯度下降的早期形式
- 控制论与早期神经网络：Ashby、Wiener的贡献
- 这些早期思想如何为1990年代的统计学习革命埋下伏笔
必须完整输出。`,
  ]
)

// ===== Part 2: 1990s 统计学习时代 =====
phase('Part 2: 统计学习革命 (1990-2009)')
const p2 = await batch4(
  ['ch05', 'ch06', 'ch07', 'ch08'],
  [
    `写一篇20000字章节"统计机器学习的基础奠定"。
内容要求：
- Vapnik和Chervonenkis的VC维理论：完整数学定义、几何解释、泛化界
- 结构风险最小化（SRM）原理
- 支持向量机（SVM）的完整推导：最大间隔分类器、对偶问题、KKT条件、核技巧（Mercer定理）
- Softmax回归（逻辑回归）的完整推导
- 最大熵模型和最大熵马尔可夫模型（MEMM）
- Boosting思想的诞生：Schapire的原始证明、AdaBoost算法完整推导和证明
- Bagging和随机森林：Bootstrap聚合、OOB误差估计
- 每个算法的完整伪代码和复杂度分析
必须完整输出。`,

    `写一篇20000字章节"概率图模型与贝叶斯革命"。
内容要求：
- 贝叶斯统计的完整框架：先验、似然、后验、共轭先验
- 隐马尔可夫模型（HMM）：前向后向算法、Viterbi解码、Baum-Welch训练
- 条件随机场（CRF）：Lafferty 2001年论文完整解读、与HMM和MEMM的对比、结构化预测
- 贝叶斯网络：d-分离、条件独立性、推理算法（变量消除、信念传播）
- 马尔可夫随机场（MRF）：Gibbs分布、Ising模型、能量函数
- 主题模型：pLSA、LDA的完整推导（Dirichlet分布、共轭先验、变分推断）
- EM算法的完整推导：Jensen不等式、E步M步、收敛性证明
- 概率图模型在NLP中的早期应用：词性标注、命名实体识别、句法分析
必须完整输出。`,

    `写一篇20000字章节"神经网络的下沉期：反向传播的完善与局限"。
内容要求：
- Werbos的反向传播原始论文（1974年）的完整解读
- Rumelhart、Hinton和Williams的BP重新发现（1986年）的历史背景
- BP算法的完整数学推导：链式法则、计算图、梯度消失/爆炸的数学分析
- LeCun的LeNet（1998年）：CNN架构完整设计、反向传播在CNN上的实现、MNIST上的突破
- 为什么CNN在1990年代没有大规模应用：计算资源、数据集规模、SVM的竞争
- Hopfield网络和Boltzmann机的完整数学分析：能量函数、收敛性
- Kohonen的自组织映射（SOM）
- 这个时期少数坚持者的故事：Hinton在加拿大的孤立研究、LeCun在贝尔实验室的坚持
必须完整输出。`,

    `写一篇20000字章节"语言模型的进化：从n-gram到神经网络"。
内容要求：
- n-gram语言模型的完整数学框架：马尔可夫假设、最大似然估计、平滑技术（Laplace、Good-Turing、Kneser-Ney的完整推导）
- 信息论语言模型：熵、交叉熵、困惑度（Perplexity）的数学定义和物理意义
- 词表示的学习：Brown聚类、NNLM（Bengio 2003）的完整架构和训练
- Word2Vec：CBOW和Skip-gram的完整推导、负采样、层次Softmax、与矩阵分解的关系（Levy和Goldberg的证明）
- GloVe：全局词频统计和局部上下文窗口的统一（矩阵分解视角）
- 早期神经机器翻译：Encoder-Decoder架构的雏形、注意力机制的早期探索（Bahdanau 2014的完整解读）
- 评估方法：BLEU、ROUGE、METEOR的数学定义和性质
必须完整输出。`,
  ]
)

// ===== Part 3: 2000s 深度学习萌芽 =====
phase('Part 3: 深度学习的黎明 (2000-2011)')
const p3 = await batch4(
  ['ch09', 'ch10', 'ch11', 'ch12'],
  [
    `写一篇20000字章节"深度信念网络与预训练的革命"。
内容要求：
- Hinton的信念网络（Belief Network）和受限玻尔兹曼机（RBM）的完整数学推导
- 对比散度（CD-k）算法的完整推导：Hinton的原始论文解读
- 深度信念网络（DBN）的逐层预训练策略：贪心逐层训练的理论依据
- 自编码器（Autoencoder）的完整数学：编码器-解码器、重建损失、过完备表示
- 稀疏自编码器：稀疏性约束的数学形式化（KL散度）
- 去噪自编码器和堆叠去噪自编码器
- 2006年Science论文的完整解读：为什么这篇论文被认为是深度学习革命的起点
- 预训练-微调范式的理论解释：正则化视角、优化 Landscape 分析
必须完整输出。`,

    `写一篇20000字章节"GPU计算与大数据：深度学习的燃料"。
内容要求：
- GPU架构的演变：从图形渲染到通用计算（GPGPU）
- CUDA的诞生和发展：并行计算模型、线程层次结构
- 为什么神经网络适合GPU：矩阵运算的并行性、SIMD架构
- ImageNet数据集的构建：Fei-Fei Li团队的完整故事、数据收集过程、规模里程碑
- 数据规模对深度学习的影响：样本复杂度理论、VC维在现代视角下的解读
- 云计算与深度学习的共生关系：AWS、Google Cloud的GPU实例发展
- 分布式训练的理论基础：数据并行、模型并行、梯度聚合
- 这个时期的硬件竞赛：NVIDIA vs AMD vs Google TPU的前夜
必须完整输出。`,

    `写一篇20000字章节"深度学习的关键理论突破（2000-2011）"。
内容要求：
- 不收敛的问题 revisited：2000年代对BP问题的新理解
- Hessian自由优化（Hessian-Free Optimization）：Martens的贡献
- 逐层预训练的理论分析：为什么它能帮助优化（局部最优的规避）
- ReLU激活函数的重新发现：Nair和Hinton 2010、Glorot和Bengio 2011
- Xavier初始化：Glorot和Bengio 2010的完整推导
- Batch Normalization的前奏：内余弦变换（Whitening）在神经网络中的应用
- 正则化方法的演进：L1/L2正则化、Dropout（Srivastava 2014的前期工作）、数据增强
- 优化理论的进展：SGD的收敛性分析、学习率调度策略
必须完整输出。`,

    `写一篇20000字章节"计算机视觉的演进（2000-2011）"。
内容要求：
- SIFT和SURF：尺度不变特征变换的完整数学推导（DoG、关键点检测、描述子）
- HOG特征：方向梯度直方图、Dalal和Triggs的完整工作
- 特征工程的时代：Haar特征、LBP、HOG、SIFT的对比分析
- 目标检测的演进：Viola-Jones实时人脸检测（AdaBoost + Haar）、HOG + SVM行人检测（Dalal）
- 语义分割的早期方法：Normalized Cuts、GrabCut
- 图像检索：词袋模型（Bag of Visual Words）、费雪向量（Fisher Vectors）
- 为什么特征工程最终被端到端学习取代：特征表达能力的理论极限
- LeCun的不懈努力：从LeNet到数字识别竞赛的连胜
必须完整输出。`,
  ]
)

// ===== Part 4: 深度学习爆发 (2012-2017) =====
phase('Part 4: 深度学习爆发 (2012-2017)')
const p4 = await batch4(
  ['ch13', 'ch14', 'ch15', 'ch16'],
  [
    `写一篇25000字章节"ImageNet时刻：AlexNet与深度学习的崛起"。
内容要求：
- AlexNet完整架构分析：8层CNN、ReLU、Dropout、Data Augmentation、GPU训练
- ImageNet 2012 ILSVRC竞赛的完整故事：Hinton的学生Krizhevsky和Sutskever
- AlexNet vs 传统方法：top-5 error从26.1%到15.3%，对比SVM+特征工程的结果
- GPU训练的技术细节：两块GTX 580、3天训练、数据并行
- ReLU激活函数为何有效： dying ReLU问题、Leaky ReLU、Parametric ReLU
- 2012年后的连锁反应：Google Brain项目的启动、DeepMind的成立
- 学术界的反应： skepticism → curiosity → gold rush
- AlexNet每层的特征可视化：从边缘到纹理到物体部件
必须完整输出，包含所有数学推导。`,

    `写一篇25000字章节"CNN架构的黄金时代：VGG、GoogLeNet、ResNet"。
内容要求：
- VGGNet（Simonyan和Zisserman 2014）：19层深度、3x3卷积核的设计哲学、参数效率分析
- GoogLeNet/Inception（Szegedy 2014）：1x1卷积降维、Inception模块、辅助分类器、7x7卷积 stem
- 残差连接（ResNet, He et al. 2016）：恒等映射的数学原理、退化问题、ResNet完整推导
- ResNet的理论解释：残差作为Hamilton-Jacobi-Bellman方程的离散化、神经网络作为ODE求解器（后续Euler方法的前奏）
- 批归一化（Ioffe和Szegedy 2015）：完整数学推导、训练加速的统计解释、与Covariate Shift的关系
- 深度可分离卷积（MobileNet、Xception的 precursor）
- 架构搜索的早期探索：Zoph和Le 2017的NAS框架
- 各架构在ImageNet上的完整 benchmark 对比表
必须完整输出。`,

    `写一篇25000字章节"Seq2Seq革命：注意力机制与Transformer"。
内容要求：
- Encoder-Decoder架构的演进：Cho 2014、Sutskever 2014、Bahdanau注意力 2015（完整数学推导）
- Luong注意力（2015）：全局vs局部注意力、得分函数对比
- Transformer（Vaswani et al. 2017）的完整数学推导：
  - 自注意力（Self-Attention）的矩阵形式
  - 多头注意力的设计动机和数学
  - 位置编码：正弦编码的完整推导和直觉解释
  - 层归一化（Layer Normalization）vs 批归一化
  - 前馈网络（Feed-Forward Network）的设计
  - 编码器-解码器的掩码机制
- 注意力机制的可视化分析：对齐矩阵、注意力头分工
- Transformer相对于RNN的理论优势：并行化、长程依赖、梯度流
- "Attention is All You Need"论文的发表故事
必须完整输出。`,

    `写一篇20000字章节"生成模型的演进（2012-2017）"。
内容要求：
- 生成对抗网络（GAN, Goodfellow 2014）：Minimax博弈的数学形式化、纳什均衡、JS散度问题
- DCGAN：卷积GAN的架构设计、训练稳定性技巧
- GAN的变体探索：LAPGAN、Conditional GAN、InfoGAN、LSGAN、WGAN（Arjovsky 2017）
- Wasserstein距离和WGAN：Earth Mover距离、Lipschitz约束、权重剪枝
- 变分自编码器（VAE, Kingma和Welling 2013）：变分推断、ELBO的完整推导、重参数化技巧
- VAE vs GAN：生成质量vs模式覆盖、似然 vs 无似然训练
- PixelRNN/PixelCNN：自回归生成模型、 masked卷积
- 早期图像生成的质量问题和模式崩溃现象
必须完整输出。`,
  ]
)

// ===== Part 5: Transformer时代 (2018-2024) =====
phase('Part 5: Transformer时代 (2018-2024)')
const p5 = await batch4(
  ['ch17', 'ch18', 'ch19', 'ch20'],
  [
    `写一篇30000字章节"预训练语言模型：从BERT到ChatGPT"。
内容要求：
- BERT（Devlin 2018）：MLM和NSP任务的完整数学形式化、Transformer Encoder架构、WordPiece分词
- GPT系列演进：
  - GPT-1（Radford 2018）：生成式预训练、Zero-shot能力
  - GPT-2（Radford 2019）：大规模语言模型、Zero-shot转换、发布策略争议
  - GPT-3（Brown 2020）：1750亿参数、Few-shot/One-shot/Zero-shot对比、上下文学习（In-context Learning）的涌现现象
  - GPT-3.5/4：RLHF、多模态扩展
- T5和统一文本到文本框架（Raffel 2020）
- 中文大模型的发展：ERNIE（百度）、GLM（清华）、通义千问（阿里）、文心一言、讯飞星火
-  scaling Laws：Kaplan et al. 2020、Chinchilla 2022的完整分析
- 对齐技术：RLHF（Christiano 2017 → Ouyang 2022）、DPO、Constitutional AI
- 涌现能力（Emergent Abilities）：阈值现象、相变类比、争议
- 大模型的哲学含义：理解 vs 模仿、意识问题、中文房间论战
必须完整输出。`,

    `写一篇25000字章节"计算机视觉的新纪元：ViT、扩散模型、视频理解"。
内容要求：
- Vision Transformer（ViT, Dosovitskiy 2021）：图像块（Patch）的数学表示、位置编码、纯Transformer在CV上的可行性证明
- DeiT：数据高效的Vision Transformer、知识蒸馏
- Swin Transformer：分层特征图、移位窗口、多尺度建模
- Masked Autoencoder（MAE, He 2022）：自监督视觉预训练、高比例掩码
- 扩散模型（Diffusion Models）的完整数学推导：
  - 前向扩散过程：高斯马尔可夫链、方差调度
  - 反向去噪过程： score matching、DDPM、DDIM
  - 条件生成：Classifier Guidance、Classifier-Free Guidance
- DALL-E、Midjourney、Stable Diffusion：文本到图像的完整技术路线
- 视频理解：TimeSformer、VideoMAE、Sora的技术原理推测
- 图像生成的评估指标：FID、IS、CLIP Score的数学定义
必须完整输出。`,

    `写一篇20000字章节"深度学习的理论难题"。
内容要求：
- 优化 Landscape：损失曲面的几何结构、局部最优 vs 全局最优、鞍点问题
- 泛化理论：为什么过参数化网络不出现过拟合？PAC-Bayes界、Rademacher复杂度
- 神经 tangent kernel（NTK）：无限宽网络的线性化、训练动态的精确描述
- 特征学习的理论：表示坍塌（Representation Collapse）、信息瓶颈理论
- 梯度消失/爆炸的现代理解：初始化设计、残差连接的理论保证
- 深度学习的可解释性：Saliency Map、Grad-CAM、Integrated Gradients、SHAP、LIME
- 对抗样本：Goodfellow的Fast Gradient Sign Method（FGSM）、对抗训练的博弈论解释
- 因果推理与深度学习的融合：do-calculus、结构因果模型
必须完整输出。`,

    `写一篇20000字章节"强化学习的复兴：从DQN到AlphaGo到ChatGPT"。
内容要求：
- 强化学习的数学基础：MDP的完整形式化、Bellman方程、值函数、策略梯度定理（Sutton et al. 2000的完整证明）
- 深度Q网络（DQN, Mnih 2013/2015）：经验回放、目标网络、Q-learning的稳定性分析
- 策略梯度方法：REINFORCE、Actor-Critic、A3C、PPO的完整推导
- AlphaGo的完整技术拆解：蒙特卡洛树搜索（MCTS）的数学、策略网络+价值网络、自我对弈
- AlphaZero：不依赖人类知识的通用强化学习
- 开放世界游戏：OpenAI Five、Dota 2的规模化挑战
- 机器人学习：Sim2Real、领域随机化、MuJoCo模拟器
- RLHF：从人类反馈中学习（Christiano 2017）、 InstructGPT（Ouyang 2022）、PPO在人类偏好优化中的应用
必须完整输出。`,
  ]
)

// ===== Part 6: 各领域专题深潜 =====
phase('Part 6: 各领域专题深潜')
const p6 = await batch4(
  ['ch21', 'ch22', 'ch23', 'ch24'],
  [
    `写一篇20000字章节"NLP的精细化：Transformer之后的语言理解"。
内容要求：
- 高效Transformer：Linformer、Performer、FlashAttention的完整数学推导
- 大语言模型的推理能力：Chain-of-Thought、Tree-of-Thought、Graph-of-Thought
- 代码大模型：Codex、GitHub Copilot、StarCoder的技术架构
- 多语言大模型：mBERT、XLM-R、mT5、PaLM的技术细节
- RAG（检索增强生成）：密集检索、稀疏检索的融合
- Agent与工具使用：ReAct、Toolformer、AutoGPT的技术原理
- NLP的评估体系：GLUE/SuperGLUE基准、大规模多任务语言理解（MMLU）
- 大语言模型的幻觉问题：事实性与一致性的数学刻画
必须完整输出。`,

    `写一篇20000字章节"计算机视觉的精细化：检测、分割、生成"。
内容要求：
- 目标检测的两阶段方法：Faster R-CNN（区域提议网络RPN的完整推导）、Mask R-CNN
- 单阶段检测器：YOLO系列演进（Redmon 2016→2023）、SSD、RetinaNet（Focal Loss的完整推导）
- 实例分割和语义分割：FCN、U-Net、DeepLab系列（Atrous卷积、ASPP模块）
- 目标跟踪：SORT、DeepSORT、ByteTrack
- 三维视觉：NeRF的完整数学推导（体渲染、位置编码、MLP架构）、3D Gaussian Splatting
- 图像生成的工业应用：ControlNet、IP-Adapter、LoRA
- 视频生成：Sora的DiT架构、 spacetime patches、世界模型
- CV的评估体系：COCO基准、mAP的定义和计算
必须完整输出。`,

    `写一篇20000字章节"图神经网络与结构数据"。
内容要求：
- GCN（Kipf和Welling 2017）：谱图卷积的简化、Chebyshev多项式近似、消息传递的矩阵形式
- GAT（Veličković 2018）：注意力机制的引入、多头注意力在图上的实现
- GraphSAGE：归纳式学习、邻域采样策略
- GIN（Xu 2019）： WL test 等价性、图同构的理论保证
- 图 Transformer：Graphormer、SAN、GraphGPS
- 分子图神经网络：分子性质预测、药物发现、AlphaFold 2的Evoformer模块
- 社交网络分析：图表示学习、链路预测、社区检测
- 知识图谱：TransE、RotatE、 ComplEx的完整数学推导
必须完整输出。`,

    `写一篇20000字章节"AI安全、对齐与伦理"。
内容要求：
- AI对齐问题（Alignment Problem）的完整形式化： specification gaming、 reward hacking
- 可解释AI（XAI）：SHAP值（Shapley值的定义和计算）、LIME、Integrated Gradients
- AI偏见与公平性：群体公平性指标（ demographic parity、equalized odds）、公平性的理论冲突
- 隐私与机器学习：差分隐私（Dwork 2006）的完整数学、Rényi差分隐私、梯度剪裁
- 对抗鲁棒性：认证鲁棒性（Cohen 2019）、 randomized smoothing
- AI治理：欧盟AI法案、NIST AI RMF、中国AI监管框架
- AGI安全：Orthogonality Thesis、Instrumental Convergence、 corrigibility
- AI评估体系：大规模语言模型元评估、 HELM、BigBench
必须完整输出。`,
  ]
)

// ===== Part 7: 综合编年史 =====
phase('Part 7: 综合编年史')
const chronicle = await agent(`以下是AI历史各章节的完整内容：

${[
  ['图灵与AI的诞生', p1[0]],
  ['感知机与连接主义', p1[1]],
  ['第一次AI冬天', p1[2]],
  ['统计学习萌芽', p1[3]],
  ['统计机器学习', p2[0]],
  ['概率图模型', p2[1]],
  ['神经网络下沉期', p2[2]],
  ['语言模型进化', p2[3]],
  ['深度信念网络', p3[0]],
  ['GPU与大数据', p3[1]],
  ['深度学习理论突破', p3[2]],
  ['计算机视觉演进', p3[3]],
  ['ImageNet时刻', p4[0]],
  ['CNN黄金时代', p4[1]],
  ['Seq2Seq革命', p4[2]],
  ['生成模型演进', p4[3]],
  ['预训练语言模型', p5[0]],
  ['CV新纪元', p5[1]],
  ['深度学习理论难题', p5[2]],
  ['强化学习复兴', p5[3]],
  ['NLP精细化', p6[0]],
  ['CV精细化', p6[1]],
  ['图神经网络', p6[2]],
  ['AI安全与对齐', p6[3]],
].map(([title, content]) => `===== ${title} =====\n${content.slice(0, 2000)}\n`).join('\n\n')}

请撰写一部完整的《人工智能通史》综合编年史（50000字）：

要求：
1. 严格按时间顺序编排所有内容
2. 在每个关键节点添加"历史 sidebar"：当时的政治经济背景、关键人物逸事、未成功的技术路线
3. 绘制完整的技术演进脉络图（用文字描述节点和连线）
4. 分析每次范式转移的驱动因素（理论突破、硬件进步、数据可用性、资金流向）
5. 预测未来10年的可能发展方向
6. 包含至少500个人名、200个算法/模型名称、100个重要日期
7. 包含中英双语术语索引

输出必须完整。`, { label: 'chronicle', effort: 'high' })

return {
  totalPhases: 7,
  totalChapters: 24,
  total_chars: chronicle.length,
}
