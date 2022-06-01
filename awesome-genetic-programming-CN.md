##  遗传编程入门指南

### 必读论文

下面的必读论文列表包含了一些近五年对遗传编程领域有重要贡献的文章，以下是阅读建议：

* **机器学习理论**系列论文建议所有研究GP机器学习的同学阅读
* **符号回归、图像分类、无监督学习**系列相互独立，可按需阅读
* **特征工程**系列论文建议希望使用GP应用到现实回归/分类问题的同学阅读
* **运筹优化、强化学习**系列文章自成体系，不需要预先阅读其他系列论文
* **程序合成**系列文章建议计算机背景同学阅读

##### **机器学习理论**

* **VC-Dimension**：Chen Q, Zhang M, Xue B. Structural risk minimization-driven genetic programming for enhancing generalization in symbolic regression[J]. IEEE Transactions on Evolutionary Computation, 2018.
* **Rademacher complexity**：Chen Q, Xue B, Zhang M. Rademacher complexity for enhancing the generalization of genetic programming for symbolic regression[J]. IEEE Transactions on Cybernetics, 2020.
* **GP-based Machine Learning Survey**：Agapitos A, Loughran R, Nicolau M, et al. A survey of statistical machine learning elements in genetic programming[J]. IEEE Transactions on Evolutionary Computation, 2019.
* **偏差方差分解**：Owen C A, Dick G, Whigham P A. Characterizing genetic programming error through extended bias and variance decomposition[J]. IEEE Transactions on Evolutionary Computation, 2020.

##### **特征工程**

Wrapper Method（基于特定模型的特征工程），该类方法速度较慢，准确率较高

* **SVM增强**：Nag K, Pal N R. Feature extraction and selection for parsimonious classifiers with multiobjective genetic programming[J]. IEEE Transactions on Evolutionary Computation, 2019.
* **线性模型增强**：La Cava W, Singh T R, Taggart J, et al. Learning concise representations for regression by evolving networks of trees[C]//International Conference on Learning Representations. 2018.
* **KNN增强**：La Cava W, Silva S, Danai K, et al. Multidimensional genetic programming for multiclass classification[J]. Swarm and Evolutionary Computation, 2019.
* **随机森林增强**：H. Zhang, A. Zhou and H. Zhang, An Evolutionary Forest for Regression[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **决策树增强**：Zhang H, Zhou A, Qian H, et al. PS-Tree: A piecewise symbolic regression tree[J]. Swarm and Evolutionary Computation, 2022.

Filter Method（基于统计指标的特征工程），该类方法速度较快，准确率较低

* **信息增益**：Tran B, Xue B, Zhang M. Genetic programming for multiple-feature construction on high-dimensional classification[J]. Pattern Recognition, 2019.

应用

* **故障检测**：Peng B, Wan S, Bi Y, et al. Automatic feature extraction and construction using genetic programming for rotating machinery fault diagnosis[J]. IEEE transactions on Cybernetics, 2020.
* **低耗能EEG分类**：Lu J, Jia H, Verma N, et al. Genetic programming for energy-efficient and energy-scalable approximate feature computation in embedded inference systems[J]. IEEE Transactions on Computers, 2017.
* **地震分析**：Gandomi A H, Roke D. A Multi-Objective Evolutionary Framework for Formulation of Nonlinear Structural Systems[J]. IEEE Transactions on Industrial Informatics, 2021.
* **QoS预测**：FanJiang Y Y, Syu Y, Huang W L. Time series QoS forecasting for Web services using multi-predictor-based genetic programming[J]. IEEE Transactions on Services Computing, 2020.

##### **符号回归**

* **领域自适应**：Chen Q, Xue B, Zhang M. Genetic programming for instance transfer learning in symbolic regression[J]. IEEE Transactions on Cybernetics, 2020.
* **跨领域数据补全**：Al-Helali B, Chen Q, Xue B, et al. Multitree Genetic Programming With New Operators for Transfer Learning in Symbolic Regression With Incomplete Data[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **特征选择**：Chen Q, Zhang M, Xue B. Feature selection to improve generalization of genetic programming for high-dimensional symbolic regression[J]. IEEE Transactions on Evolutionary Computation, 2017.
* **多任务学习**：Zhong J, Feng L, Cai W, et al. Multifactorial genetic programming for symbolic regression problems[J]. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2018.

交叉/变异算子

* **Linkage Learning**：Virgolin M, Alderliesten T, Witteveen C, et al. Improving model-based genetic programming for symbolic regression of small expressions[J]. Evolutionary Computation, 2021.
* **贝叶斯网络分布学习**：Wong P K, Wong M L, Leung K S. Probabilistic Contextual and Structural Dependencies Learning in Grammar-Based Genetic Programming[J]. Evolutionary Computation, 2021.
* **语义交叉**：Chen Q, Xue B, Zhang M. Improving generalization of genetic programming for symbolic regression with angle-driven geometric semantic operators[J]. IEEE Transactions on Evolutionary Computation, 2018.
* **语义交叉+混合整数规划**：Huynh Q N, Chand S, Singh H K, et al. Genetic programming with mixed-integer linear programming-based library search[J]. IEEE Transactions on Evolutionary Computation, 2018.

评估算子

* **自适应评估**：Drahosova M, Sekanina L, Wiglasz M. Adaptive fitness predictors in coevolutionary Cartesian genetic programming[J]. Evolutionary Computation, 2019.

选择算子

* **多样性选择算子**：La Cava W, Helmuth T, Spector L, et al. A probabilistic and multi-objective analysis of lexicase selection and ε-lexicase selection[J]. Evolutionary Computation, 2019.
* **多样性选择算子**：Chen Q, Xue B, Zhang M. Preserving Population Diversity Based on Transformed Semantics in Genetic Programming for Symbolic Regression[J]. IEEE Transactions on Evolutionary Computation, 2020.

##### **图像分类**

* **集成学习**：Bi Y, Xue B, Zhang M. Genetic programming with a new representation to automatically learn features and evolve ensembles for image classification[J]. IEEE Transactions on Cybernetics, 2020.
* **数据分治**：Bi Y, Xue B, Zhang M. A divide-and-conquer genetic programming algorithm with ensembles for image classification[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **低质量图像**：Bi Y, Xue B, Zhang M. Genetic programming-based discriminative feature learning for low-quality image classification[J]. IEEE Transactions on Cybernetics, 2021.
* **少样本学习**：Bi Y, Xue B, Zhang M. Dual-tree genetic programming for few-shot image classification[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **多任务学习**：Bi Y, Xue B, Zhang M. Learning and sharing: A multitask genetic programming approach to image feature learning[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **样本选择**：Bi Y, Xue B, Zhang M. Instance Selection-Based Surrogate-Assisted Genetic Programming for Feature Learning in Image Classification[J]. IEEE Transactions on Cybernetics, 2021.
* **Skip-Connection**：Fan Q, Bi Y, Xue B, et al. Genetic Programming for Image Classification: A New Program Representation with Flexible Feature Reuse[J]. IEEE Transactions on Evolutionary Computation, 2022.
* **纹理分类**：Al-Sahaf H, Al-Sahaf A, Xue B, et al. Automatically Evolving Texture Image Descriptors Using the Multitree Representation in Genetic Programming Using Few Instances[J]. Evolutionary Computation, 2021.
* **演化神经网络**：Suganuma M, Kobayashi M, Shirakawa S, et al. Evolution of deep convolutional neural networks using cartesian genetic programming[J]. Evolutionary Computation, 2020.

##### **无监督学习**

* **Manifold Learning**：Lensen A, Xue B, Zhang M. Genetic Programming for Manifold Learning: Preserving Local Topology[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **可解释性降维**：Lensen A, Xue B, Zhang M. Genetic programming for evolving a front of interpretable models for data visualization[J]. IEEE Transactions on Cybernetics, 2020.
* **聚类**：Lensen A, Xue B, Zhang M. Genetic programming for evolving similarity functions for clustering: Representations and analysis[J]. Evolutionary Computation, 2020.

##### **运筹优化**

GP在运筹优化领域的研究主要集中于车间调度问题（Job-shop Scheduling）中的调度规则自动设计。

* **Multifidelity（多保真度）**：Zhang F, Mei Y, Nguyen S, et al. Collaborative multifidelity-based surrogate models for genetic programming in dynamic flexible job shop scheduling[J]. IEEE Transactions on Cybernetics, 2021.
* **Feature Selection**：Zhang F, Mei Y, Nguyen S, et al. Evolving scheduling heuristics via genetic programming with feature selection in dynamic flexible job-shop scheduling[J]. IEEE Transactions on Cybernetics, 2020.
* **Multitask（多任务）**：Zhang F, Mei Y, Nguyen S, et al. Multitask genetic programming-based generative hyperheuristics: A case study in dynamic scheduling[J]. IEEE Transactions on Cybernetics, 2021.
* **Surrogate Model（代理模型）**：Zhang F, Mei Y, Nguyen S, et al. Surrogate-assisted evolutionary multitask genetic programming for dynamic flexible job shop scheduling[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **Multigene（多基因）**：Nguyen S, Thiruvady D, Zhang M, et al. Automated design of multipass heuristics for resource-constrained job scheduling with self-competitive genetic programming[J]. IEEE transactions on Cybernetics, 2021.
* **重组算子**：Zhang F, Mei Y, Nguyen S, et al. Correlation coefficient-based recombinative guidance for genetic programming hyperheuristics in dynamic flexible job shop scheduling[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **改进约束规划求解器**：Nguyen S, Thiruvady D, Zhang M, et al. A genetic programming approach for evolving variable selectors in constraint programming[J]. IEEE Transactions on Evolutionary Computation, 2021.

容量受限的路径规划问题（Uncertain Capacitated Arc Routing）也是GP在运筹优化方向的一个关注热点。

* **集成学习**：Wang S, Mei Y, Zhang M, et al. Genetic programming with niching for uncertain capacitated arc routing problem[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **Multitask（多任务）**：Ardeh M A, Mei Y, Zhang M, et al. Knowledge Transfer Genetic Programming with Auxiliary Population for Solving Uncertain Capacitated Arc Routing Problem[J]. IEEE Transactions on Evolutionary Computation, 2022.
* **知识迁移**：Ardeh M A, Mei Y, Zhang M. Genetic Programming with Knowledge Transfer and Guided Search for Uncertain Capacitated Arc Routing Problem[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **协同演化**：Liu Y, Mei Y, Zhang M, et al. A predictive-reactive approach with genetic programming and cooperative coevolution for the uncertain capacitated arc routing problem[J]. Evolutionary Computation, 2020.
* **实时路由**：Xu B, Mei Y, Wang Y, et al. Genetic programming with delayed routing for multiobjective dynamic flexible job shop scheduling[J]. Evolutionary Computation, 2021.

除了车间调度和路径规划问题，GP在其他运筹优化问题上也有偶有应用。

* **容器调度**：Tan B, Ma H, Mei Y, et al. A cooperative coevolution genetic programming hyper-heuristic approach for on-line resource allocation in container-based clouds[J]. IEEE Transactions on Cloud Computing, 2020.
* **多智能体调度**：Gao G, Mei Y, Xin B, et al. Automated Coordination Strategy Design Using Genetic Programming for Dynamic Multipoint Dynamic Aggregation[J]. IEEE Transactions on Cybernetics, 2021.
* **定价问题**：Kieffer E, Danoy G, Brust M R, et al. Tackling large-scale and combinatorial bi-level problems with a genetic programming hyper-heuristic[J]. IEEE Transactions on Evolutionary Computation.
* **移动网络**：Fenton M, Lynch D, Kucera S, et al. Multilayer optimization of heterogeneous networks using grammatical genetic programming[J]. IEEE Transactions on Cybernetics, 2017.
* **约束生成**：Pawlak T P, Krawiec K. Synthesis of constraints for mathematical programming with one-class genetic programming[J]. IEEE Transactions on Evolutionary Computation, 2018.

##### **强化学习**

* **多任务学习**：Kelly S, Heywood M I. Emergent solutions to high-dimensional multitask reinforcement learning[J]. Evolutionary Computation, 2018.
* **迁移学习**：Kelly S, Heywood M I. Discovering agent behaviors through code reuse: Examples from half-field offense and ms. pac-man[J]. IEEE Transactions on Games, 2017.
* **MCTS增强**：Holmgård C, Green M C, Liapis A, et al. Automated playtesting with procedural personas through MCTS with evolved heuristics[J]. IEEE Transactions on Games, 2018.

##### **程序合成**

* **Genetic Improvement（程序改进）**：Yuan Y, Banzhaf W. Arja: Automated repair of java programs via multi-objective genetic programming[J]. IEEE Transactions on Software Engineering, 2018.
* **正则表达式**：Bartoli A, De Lorenzo A, Medvet E, et al. Automatic search-and-replace from examples with coevolutionary genetic programming[J]. IEEE transactions on Cybernetics, 2019.
* **实体抽取**：Bartoli A, De Lorenzo A, Medvet E, et al. Active learning of regular expressions for entity extraction[J]. IEEE transactions on Cybernetics, 2017.
* **形式验证**：Błądek I, Krawiec K, Swan J. Counterexample-driven genetic programming: heuristic program synthesis from formal specifications[J]. Evolutionary Computation, 2018.
* **游戏AI合成**：Martinez-Arellano G, Cant R, Woods D. Creating AI characters for fighting games using genetic programming[J]. IEEE Transactions on Computational Intelligence and AI in Games, 2016.
* **全局优化算法**：Fajfar I, Puhan J, Bűrmen Á. Evolving a Nelder–Mead algorithm for optimization with genetic programming[J]. Evolutionary Computation, 2017.
* **Linear GP 冗余分析**：Sotto L F D P, Rothlauf F, de Melo V V, et al. An Analysis of the Influence of Noneffective Instructions in Linear Genetic Programming[J]. Evolutionary Computation, 2022.

### 领域专家

参考排名：http://gpbib.cs.ucl.ac.uk/gp-html/index.html

* Mengjie Zhang
  * Victoria University of Wellington
  * 方向：演化机器学习
  * 代表作：MOGP
* Michael O'Neill
  * University College Dublin
  * 方向：程序合成
  * 代表作：Grammatical Evolution
* Wolfgang Banzhaf
  * Michigan State University
  * 方向：程序合成
  * 代表作：Genetic Improvement
* Lee Spector 
  * University of Massachusetts Amherst
  * 方向：程序合成
  * 代表作：Lexicase Selection，PyshGP
* Krzysztof Krawiec
  * Poznan University of Technology
  * 方向：符号回归
  * 代表作：GSGP(Geometric Semantic Genetic Programming)

* Una-May O'Reilly
  * Massachusetts Institute of Technology
  * 方向：符号回归
  * 代表作：M4GP，EFS

### 开源框架

##### **AutoML**

* **TPOT:** https://github.com/EpistasisLab/tpot
  * 8000+ Star，最流行的AutoML框架之一
* **GAMA:** https://github.com/openml-labs/gama
  * 支持模型集成的AutoML框架

##### **GP研究**

* **DEAP:** https://github.com/DEAP/deap
  * 最灵活的GP框架，基于Python开发
* **ECJ:** https://github.com/GMUEClab/ecj
  * 基于Java的GP框架

##### **符号回归**

* **gplearn:** https://github.com/trevorstephens/gplearn
  * 1000+Star，GP领域最成熟的符号回归框架
* **PS-Tree:** https://github.com/hengzhe-zhang/PS-Tree/
  * 支持分段符号回归的符号回归算法
* **其他符号回归算法参考SRBench:** https://github.com/cavalab/srbench/

##### **特征工程**

* **Evolutionary Forest:** https://github.com/hengzhe-zhang/EvolutionaryForest/
  * 基于集成学习思想的特征工程框架
* **FEW:** https://github.com/lacava/few
  * 基于特征重要性的特征工程框架


**运筹优化**

* **GPJSS:** https://github.com/meiyi1986/GPJSS
  * 基于遗传编程的车间调度（Job-shop Scheduling）算法

### 测试套件

##### **回归**

* SRBench
  * 机器学习套件，包含真实机器学习问题和合成问题
  * https://github.com/cavalab/srbench/

##### **分类**

* DIGEN
  * 基于遗传编程算法合成的分类问题
  * https://github.com/EpistasisLab/digen/
* PMLB
  * 包含大量真实机器学习问题
  * https://github.com/EpistasisLab/pmlb

### 必看期刊/会议

##### **期刊**

* **《IEEE Transactions on Evolutionary Computation》（JCR一区，CCF B）**
  * 演化计算领域最佳期刊，GP领域开创性成果的第一选择
* **《IEEE Transactions on Cybernetics》（JCR一区，CCF B）**
  * 机器学习/运筹优化方向GP论文较多
* **《Evolutionary Computation》（CCF B）**
  * 论文水平高，但由于论文数量少，期刊影响力不高
* **《Swarm and Evolutionary Computation》（JCR 一区）**
  * 演化计算领域综合期刊，GP方向论文质量较高
* **《IEEE Transactions on Emerging Topics in Computational Intelligence》**
  * 综合类期刊，偶有GP方向论文

* **《Genetic Programming and Evolvable Machines》**
  * GP领域唯一的专业期刊，论文质量高

##### **会议**

* **The Genetic and Evolutionary Computation Conference (*GECCO*, CCF C)**
  * 演化计算领域的顶级年度会议，GP论文质量高
* **Parallel Problem Solving from Nature (*PPSN*, CCF B)**
  * 演化计算领域的顶级双年度会议，GP论文质量高
* **European Conference on Genetic Programming (*EuroGP*)** 
  * GP领域唯一的专业会议，主要以领域新人练手为主
* **IEEE Congress on Evolutionary Computation (*IEEE CEC*)**
  * 演化计算领域会议，主要以领域新人练手为主，论文质量参差不齐

### 相关学习资料

* DEAP框架GP开发文档
  * https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html
* GP领域研究者排名
  * http://gpbib.cs.ucl.ac.uk/gp-html/index.html
* GP领域概述
  * https://geneticprogramming.com/

### 遗传编程中的技巧（七种武器）

此处重点介绍GP机器学习方向的七种技巧，以下七种技巧均在开源框架[Evolutionary Forest]()中验证了有效性。

1. **交叉变异率**：尽量选用大的Crossover Rate，小的Mutation Rate

   原因：只有大的Crossover Rate才能充分组合Building Blocks，而Building Blocks是演化算法演进的基石

2. **运算符**：除法运算符应使用Analytical Quotient而非Protected Division

   原因：Analytical Quotient可以规避Protected Division造成的运算结果阶跃问题

   论文：Ni J, Drieberg R H, Rockett P I. The use of an analytic quotient operator in genetic programming[J]. IEEE Transactions on Evolutionary Computation, 2012.

3. **选择算子**：Lexicase Selection可以有效增强种群多样性

   提示：Lexicase Selection需要配合大种群使用，在原论文中作者的种群大小为1000

   论文：La Cava W, Helmuth T, Spector L, et al. A probabilistic and multi-objective analysis of lexicase selection and ε-lexicase selection[J]. Evolutionary Computation, 2019.

4. **预处理**：特征归一化是提升模型拟合能力的重要手段

   原因：GP擅长结构优化，不擅长优化缩放系数

   论文：C. A. Owen, G. Dick and P. A. Whigham, Standardisation and Data Augmentation in Genetic Programming[J]. IEEE Transactions on Evolutionary Computation, 2022.

5. **后处理**：线性缩放是提升拟合性能的一大利器

   原因：GP不擅长系数拟合，线性缩放可以大幅度缓解系数拟合困境

   论文：Keijzer M. Improving symbolic regression with interval arithmetic and linear scaling[C]//European Conference on Genetic Programming, 2003.

6. **模型表征**：多Gene的遗传编程算法往往可以取得更好的效果

   原因：多Gene的GP系统可以更充分利用线性缩放的优势，降低系数拟合的复杂度

 7. **集成学习**：集成学习可以降低遗传编程算法的方差

    原因：种群中可能存在多个表现良好的遗传编程个体，集成这些个体的预测结果可以有效降低预测随机性

    论文：H. Zhang, A. Zhou and H. Zhang, An Evolutionary Forest for Regression[J]. IEEE Transactions on Evolutionary Computation, 2021.
