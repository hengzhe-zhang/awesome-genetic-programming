# Awesome Genetic Programming
[Chinese Version (中文版)](https://zhuanlan.zhihu.com/p/519338969)

### Papers

The following papers list contains some papers that have made significant contributions to the field of genetic programming in the past five years. Here are some reading suggestions:

* **Machine Learning Theory:** This series of papers are recommended for all students studying GP machine learning
* **Symbolic regression, image classification, unsupervised learning:** These series are independent of each other and can be read on demand
* **Feature Engineering:** This series of papers are recommended for students who want to use GP to apply to real-world regression/classification problems
* **Operation research optimization, reinforcement learning:** This series of papers are self-contained, no need to read other series of papers in advance
* **Program synthesis:** This series of papers are recommended for students with computer background to read

##### **Machine Learning Theory**

* **VC-Dimension**：Chen Q, Zhang M, Xue B. Structural risk minimization-driven genetic programming for enhancing generalization in symbolic regression[J]. IEEE Transactions on Evolutionary Computation, 2018.
* **Rademacher complexity**：Chen Q, Xue B, Zhang M. Rademacher complexity for enhancing the generalization of genetic programming for symbolic regression[J]. IEEE Transactions on Cybernetics, 2020.
* **GP-based Machine Learning Survey**：Agapitos A, Loughran R, Nicolau M, et al. A survey of statistical machine learning elements in genetic programming[J]. IEEE Transactions on Evolutionary Computation, 2019.
* **Bias-Variance Decomposition**：Owen C A, Dick G, Whigham P A. Characterizing genetic programming error through extended bias and variance decomposition[J]. IEEE Transactions on Evolutionary Computation, 2020.

##### **Feature Engineering**

Wrapper Method (Feature engineering based on a specific model), this type of method is slow but accurate

* **SVM Enhancement**：Nag K, Pal N R. Feature extraction and selection for parsimonious classifiers with multiobjective genetic programming[J]. IEEE Transactions on Evolutionary Computation, 2019.
* **Linear Model Enhancement**：La Cava W, Singh T R, Taggart J, et al. Learning concise representations for regression by evolving networks of trees[C]//International Conference on Learning Representations. 2018.
* **KNN Enhancement**：La Cava W, Silva S, Danai K, et al. Multidimensional genetic programming for multiclass classification[J]. Swarm and Evolutionary Computation, 2019.
* **Random Forest Enhancement**：H. Zhang, A. Zhou and H. Zhang, An Evolutionary Forest for Regression[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **Decision Tree Enhancement**：Zhang H, Zhou A, Qian H, et al. PS-Tree: A piecewise symbolic regression tree[J]. Swarm and Evolutionary Computation, 2022.

Filter Method (Feature engineering based on statistical information), this type of method is fast but less accurate

* **Information Gain**：Tran B, Xue B, Zhang M. Genetic programming for multiple-feature construction on high-dimensional classification[J]. Pattern Recognition, 2019.

Application

* **Fault Diagnosis**：Peng B, Wan S, Bi Y, et al. Automatic feature extraction and construction using genetic programming for rotating machinery fault diagnosis[J]. IEEE transactions on Cybernetics, 2020.
* **Energy-efficient EEG Classification**：Lu J, Jia H, Verma N, et al. Genetic programming for energy-efficient and energy-scalable approximate feature computation in embedded inference systems[J]. IEEE Transactions on Computers, 2017.
* **Seismic Analysis**：Gandomi A H, Roke D. A Multi-Objective Evolutionary Framework for Formulation of Nonlinear Structural Systems[J]. IEEE Transactions on Industrial Informatics, 2021.
* **QoS Prediction**：FanJiang Y Y, Syu Y, Huang W L. Time series QoS forecasting for Web services using multi-predictor-based genetic programming[J]. IEEE Transactions on Services Computing, 2020.

##### **Symbolic Regression**

* **Domain Adaptation**：Chen Q, Xue B, Zhang M. Genetic programming for instance transfer learning in symbolic regression[J]. IEEE Transactions on Cybernetics, 2020.
* **Cross-domain Data Imputation**：Al-Helali B, Chen Q, Xue B, et al. Multitree Genetic Programming With New Operators for Transfer Learning in Symbolic Regression With Incomplete Data[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **Feature Selection**：Chen Q, Zhang M, Xue B. Feature selection to improve generalization of genetic programming for high-dimensional symbolic regression[J]. IEEE Transactions on Evolutionary Computation, 2017.
* **Multitask Learning**：Zhong J, Feng L, Cai W, et al. Multifactorial genetic programming for symbolic regression problems[J]. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2018.

Crossover/Mutation Operator

* **Linkage Learning**：Virgolin M, Alderliesten T, Witteveen C, et al. Improving model-based genetic programming for symbolic regression of small expressions[J]. Evolutionary Computation, 2021.
* **Bayesian Networks for Distributed Learning**：Wong P K, Wong M L, Leung K S. Probabilistic Contextual and Structural Dependencies Learning in Grammar-Based Genetic Programming[J]. Evolutionary Computation, 2021.
* **Semantic Crossover**：Chen Q, Xue B, Zhang M. Improving generalization of genetic programming for symbolic regression with angle-driven geometric semantic operators[J]. IEEE Transactions on Evolutionary Computation, 2018.
* **Semantic Crossover+MILP**：Huynh Q N, Chand S, Singh H K, et al. Genetic programming with mixed-integer linear programming-based library search[J]. IEEE Transactions on Evolutionary Computation, 2018.

Evaluation Operator

* **Adaptive Evaluation**：Drahosova M, Sekanina L, Wiglasz M. Adaptive fitness predictors in coevolutionary Cartesian genetic programming[J]. Evolutionary Computation, 2019.

Selection Operator

* **Diversity-based Selection Operator**：La Cava W, Helmuth T, Spector L, et al. A probabilistic and multi-objective analysis of lexicase selection and ε-lexicase selection[J]. Evolutionary Computation, 2019.
* **Diversity-based Selection Operator**：Chen Q, Xue B, Zhang M. Preserving Population Diversity Based on Transformed Semantics in Genetic Programming for Symbolic Regression[J]. IEEE Transactions on Evolutionary Computation, 2020.

##### **Image Classification**

* **Ensemble Learning**：Bi Y, Xue B, Zhang M. Genetic programming with a new representation to automatically learn features and evolve ensembles for image classification[J]. IEEE Transactions on Cybernetics, 2020.
* **Divide and Conquer**：Bi Y, Xue B, Zhang M. A divide-and-conquer genetic programming algorithm with ensembles for image classification[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **Low Quality Image**：Bi Y, Xue B, Zhang M. Genetic programming-based discriminative feature learning for low-quality image classification[J]. IEEE Transactions on Cybernetics, 2021.
* **Few-shot Learning**：Bi Y, Xue B, Zhang M. Dual-tree genetic programming for few-shot image classification[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **Multitask Learning**：Bi Y, Xue B, Zhang M. Learning and sharing: A multitask genetic programming approach to image feature learning[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **Surrogate Model**：Bi Y, Xue B, Zhang M. Instance Selection-Based Surrogate-Assisted Genetic Programming for Feature Learning in Image Classification[J]. IEEE Transactions on Cybernetics, 2021.
* **Skip-Connection**：Fan Q, Bi Y, Xue B, et al. Genetic Programming for Image Classification: A New Program Representation with Flexible Feature Reuse[J]. IEEE Transactions on Evolutionary Computation, 2022.
* **Texture Classification**：Al-Sahaf H, Al-Sahaf A, Xue B, et al. Automatically Evolving Texture Image Descriptors Using the Multitree Representation in Genetic Programming Using Few Instances[J]. Evolutionary Computation, 2021.
* **Evolutionary Neural Network**：Suganuma M, Kobayashi M, Shirakawa S, et al. Evolution of deep convolutional neural networks using cartesian genetic programming[J]. Evolutionary Computation, 2020.

##### **Unsupervised Learning**

* **Manifold Learning**：Lensen A, Xue B, Zhang M. Genetic Programming for Manifold Learning: Preserving Local Topology[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **Explainable Dimensionality Reduction**：Lensen A, Xue B, Zhang M. Genetic programming for evolving a front of interpretable models for data visualization[J]. IEEE Transactions on Cybernetics, 2020.
* **Clustering**：Lensen A, Xue B, Zhang M. Genetic programming for evolving similarity functions for clustering: Representations and analysis[J]. Evolutionary Computation, 2020.

##### **Operations Research**

The research of GP in the field of operations research (OR) mainly focuses on the automatic design of scheduling rules in the job-shop scheduling problem (JSP).

* **Multitask**：Zhang F, Mei Y, Nguyen S, et al. Multitask genetic programming-based generative hyperheuristics: A case study in dynamic scheduling[J]. IEEE Transactions on Cybernetics, 2021.
* **Multigene**：Nguyen S, Thiruvady D, Zhang M, et al. Automated design of multipass heuristics for resource-constrained job scheduling with self-competitive genetic programming[J]. IEEE transactions on Cybernetics, 2021.
* **Multifidelity**：Zhang F, Mei Y, Nguyen S, et al. Collaborative multifidelity-based surrogate models for genetic programming in dynamic flexible job shop scheduling[J]. IEEE Transactions on Cybernetics, 2021.
* **Feature Selection**：Zhang F, Mei Y, Nguyen S, et al. Evolving scheduling heuristics via genetic programming with feature selection in dynamic flexible job-shop scheduling[J]. IEEE Transactions on Cybernetics, 2020.
* **Surrogate Model**：Zhang F, Mei Y, Nguyen S, et al. Surrogate-assisted evolutionary multitask genetic programming for dynamic flexible job shop scheduling[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **Crossover Operator**：Zhang F, Mei Y, Nguyen S, et al. Correlation coefficient-based recombinative guidance for genetic programming hyperheuristics in dynamic flexible job shop scheduling[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **Improved Constraint Programming Solver**：Nguyen S, Thiruvady D, Zhang M, et al. A genetic programming approach for evolving variable selectors in constraint programming[J]. IEEE Transactions on Evolutionary Computation, 2021.

The uncertain capacitated arc routing problem is also a hot topic of GP in the domain of operational research (OR).

* **Multitask**：Ardeh M A, Mei Y, Zhang M, et al. Knowledge Transfer Genetic Programming with Auxiliary Population for Solving Uncertain Capacitated Arc Routing Problem[J]. IEEE Transactions on Evolutionary Computation, 2022.
* **Ensemble Learning**：Wang S, Mei Y, Zhang M, et al. Genetic programming with niching for uncertain capacitated arc routing problem[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **Knowledge Transfer**：Ardeh M A, Mei Y, Zhang M. Genetic Programming with Knowledge Transfer and Guided Search for Uncertain Capacitated Arc Routing Problem[J]. IEEE Transactions on Evolutionary Computation, 2021.
* **Co-evolution**：Liu Y, Mei Y, Zhang M, et al. A predictive-reactive approach with genetic programming and cooperative coevolution for the uncertain capacitated arc routing problem[J]. Evolutionary Computation, 2020.
* **Real-time Routing**：Xu B, Mei Y, Wang Y, et al. Genetic programming with delayed routing for multiobjective dynamic flexible job shop scheduling[J]. Evolutionary Computation, 2021.

In addition to job-shop scheduling and arc routing problems, GP has some applications in other operational research problems.

* **Container Scheduling**：Tan B, Ma H, Mei Y, et al. A cooperative coevolution genetic programming hyper-heuristic approach for on-line resource allocation in container-based clouds[J]. IEEE Transactions on Cloud Computing, 2020.
* **Multi-agent Scheduling**：Gao G, Mei Y, Xin B, et al. Automated Coordination Strategy Design Using Genetic Programming for Dynamic Multipoint Dynamic Aggregation[J]. IEEE Transactions on Cybernetics, 2021.
* **Pricing**：Kieffer E, Danoy G, Brust M R, et al. Tackling large-scale and combinatorial bi-level problems with a genetic programming hyper-heuristic[J]. IEEE Transactions on Evolutionary Computation.
* **Mobile Network**：Fenton M, Lynch D, Kucera S, et al. Multilayer optimization of heterogeneous networks using grammatical genetic programming[J]. IEEE Transactions on Cybernetics, 2017.
* **Constraint Generation**：Pawlak T P, Krawiec K. Synthesis of constraints for mathematical programming with one-class genetic programming[J]. IEEE Transactions on Evolutionary Computation, 2018.

##### **Reinforcement Learning**

* **Multitask Learning**：Kelly S, Heywood M I. Emergent solutions to high-dimensional multitask reinforcement learning[J]. Evolutionary Computation, 2018.
* **Transfer Learning**：Kelly S, Heywood M I. Discovering agent behaviors through code reuse: Examples from half-field offense and ms. pac-man[J]. IEEE Transactions on Games, 2017.
* **MCTS Enhancement**：Holmgård C, Green M C, Liapis A, et al. Automated playtesting with procedural personas through MCTS with evolved heuristics[J]. IEEE Transactions on Games, 2018.

##### **Program Synthesis**

* **Genetic Improvement**：Yuan Y, Banzhaf W. Arja: Automated repair of java programs via multi-objective genetic programming[J]. IEEE Transactions on Software Engineering, 2018.
* **Regular Expression Generation**：Bartoli A, De Lorenzo A, Medvet E, et al. Automatic search-and-replace from examples with coevolutionary genetic programming[J]. IEEE transactions on Cybernetics, 2019.
* **Entity Extraction**：Bartoli A, De Lorenzo A, Medvet E, et al. Active learning of regular expressions for entity extraction[J]. IEEE transactions on Cybernetics, 2017.
* **Formal Verification**：Błądek I, Krawiec K, Swan J. Counterexample-driven genetic programming: heuristic program synthesis from formal specifications[J]. Evolutionary Computation, 2018.
* **Game AI Synthesis**：Martinez-Arellano G, Cant R, Woods D. Creating AI characters for fighting games using genetic programming[J]. IEEE Transactions on Computational Intelligence and AI in Games, 2016.
* **Global Optimization Algorithm**：Fajfar I, Puhan J, Bűrmen Á. Evolving a Nelder–Mead algorithm for optimization with genetic programming[J]. Evolutionary Computation, 2017.
* **Linear-GP Redundancy Analysis**：Sotto L F D P, Rothlauf F, de Melo V V, et al. An Analysis of the Influence of Noneffective Instructions in Linear Genetic Programming[J]. Evolutionary Computation, 2022.


### Domain Experts

Reference Ranking: http://gpbib.cs.ucl.ac.uk/gp-html/index.html

* Mengjie Zhang
  * Victoria University of Wellington
  * Research Direction: Evolutionary Machine Learning
  * Representative Works: MOGP
* Michael O'Neill
  * University College Dublin
  * Research Direction: Procedural Synthesis
  * Representative Works: Grammatical Evolution
* Wolfgang Banzhaf
  * Michigan State University
  * Research Direction: Procedural Synthesis
  * Representative Works: Genetic Improvement
* Lee Spector 
  * University of Massachusetts Amherst
  * Research Direction: Procedural Synthesis
  * Representative Works: Lexicase Selection，PyshGP
* Krzysztof Krawiec
  * Poznan University of Technology
  * Research Direction: Symbolic Regression
  * Representative Works: GSGP(Geometric Semantic Genetic Programming)
* Una-May O'Reilly
  * Massachusetts Institute of Technology
  * Research Direction: Symbolic Regression
  * Representative Works: M4GP，EFS

### Open-source Framework

##### **AutoML**

* **TPOT:** https://github.com/EpistasisLab/tpot
  * 8000+ Star, one of the most popular AutoML frameworks
* **GAMA:** https://github.com/openml-labs/gama
  * AutoML framework that supports ensemble learning

##### **GP Research**

* **DEAP:** https://github.com/DEAP/deap
  * The most flexible GP framework, developed based on Python
* **ECJ:** https://github.com/GMUEClab/ecj
  * Java-based GP framework

##### **Symbolic Regression**

* **gplearn**: https://github.com/trevorstephens/gplearn
  * 1000+Star, the most mature symbolic regression framework in the GP field
* **PS-Tree**：https://github.com/hengzhe-zhang/PS-Tree/
  * Symbolic regression algorithm supporting piecewise symbolic regression
* **Other symbolic regression algorithms please refer to SRBench**：https://github.com/cavalab/srbench/

##### **Feature Engineering**

* **Evolutionary Forest**：https://github.com/hengzhe-zhang/EvolutionaryForest/
  * Feature engineering framework based on ensemble learning
* **FEW**：https://github.com/lacava/few
  * Feature engineering framework based on feature importance


##### **Operations Research**

* **GPJSS**：https://github.com/meiyi1986/GPJSS
  * Job-shop scheduling algorithm based on GP

### Test Suite

##### **Regression**

* SRBench
  * A machine learning benchmark suite, containing real-world machine learning problems and synthetic problems
  * https://github.com/cavalab/srbench/

##### **Classification**

* DIGEN
  * Classification problems synthesized by GP
  * https://github.com/EpistasisLab/digen/
* PMLB
  * Including a large number of real-world machine learning problems
  * https://github.com/EpistasisLab/pmlb

### Journal/Conference

##### **Journal**

* **《IEEE Transactions on Evolutionary Computation》（JCR Q1，CCF B）**
  * The best journal in the field of evolutionary computation, the first choice for groundbreaking achievements in the field of GP
* **《IEEE Transactions on Cybernetics》（JCR Q1，CCF B）**
  * There are many GP papers in the direction of machine learning/operation research
* **《Evolutionary Computation》（CCF B）**
  * The quality of papers is high, but due to the small amount of papers, the influence of the journal is not high
* **《Swarm and Evolutionary Computation》（JCR Q1）**
  * A comprehensive journal in the field of evolutionary computation, with high quality papers in the GP direction
* **《IEEE Transactions on Emerging Topics in Computational Intelligence》**
  * A comprehensive journal, with a few GP papers
* **《Genetic Programming and Evolvable Machines》**
  * The only professional journal in the field of GP with high quality papers

##### **Conference**

* **The Genetic and Evolutionary Computation Conference (*GECCO*, CCF C)**
  * The top annual conference in the field of evolutionary computation, with high quality GP papers
* **Parallel Problem Solving from Nature (*PPSN*, CCF B)**
  * Top bi-annual conference in evolutionary computation with high quality GP papers
* **European Conference on Genetic Programming (*EuroGP*)** 
  * The only professional conference in the GP field, mainly focusing on newcomers in the field
* **IEEE Congress on Evolutionary Computation (*IEEE CEC*)**
  * The conference in the field of evolutionary computation mainly focuses on newcomers in the field

### Learning Materials

* GP Development Documentation in DEAP
  * https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html
* Ranking of Researchers in The Field of GP
  * http://gpbib.cs.ucl.ac.uk/gp-html/index.html
* An Overview of GP 
  * https://geneticprogramming.com/

### Seven Tricks in Genetic Programming

Here we focus on seven tricks for GP-based machine learning algorithms. The following seven techniques have been validated in the open source framework [Evolutionary Forest](https://github.com/hengzhe-zhang/EvolutionaryForest/).

1. **Crossover Rate**: Please try to use a large crossover rate and a small mutation rate.

   Reason: Only a large crossover rate can fully combine building blocks, and building blocks are the driving force of evolutionary algorithms.

2. **Operator**: The division operator should use Analytical Quotient instead of Protected Division.

   Reason: Analytical Quotient can avoid the instability issues caused by Protected Division.

   Paper：Ni J, Drieberg R H, Rockett P I. The use of an analytic quotient operator in genetic programming[J]. IEEE Transactions on Evolutionary Computation, 2012.

3. **Selection operator**: Lexicase Selection can effectively enhance population diversity.

   Tip: Lexicase Selection needs to be used with a large population. In the original paper, the population size is 1000.

   Paper：La Cava W, Helmuth T, Spector L, et al. A probabilistic and multi-objective analysis of lexicase selection and ε-lexicase selection[J]. Evolutionary Computation, 2019.

4. **Preprocessing**: Feature standardization is an important way to improve the model fitting capability.

   Reason: GP is good at structural optimization, not good at optimizing scaling factor.

   Paper：C. A. Owen, G. Dick and P. A. Whigham, Standardisation and Data Augmentation in Genetic Programming[J]. IEEE Transactions on Evolutionary Computation, 2022.

5. **Post-processing**: Linear scaling is a great tool for improving fitting performance.

   Reason: GP is not good at coefficient fitting, and linear scaling can greatly alleviate the coefficient fitting dilemma.

   Paper：Keijzer M. Improving symbolic regression with interval arithmetic and linear scaling[C]//European Conference on Genetic Programming, 2003.

6. **Model characterization**: Multi-Gene genetic programming algorithms can often achieve better results.

   Reason: The multi-Gene GP system can make full use of the advantages of linear scaling and reduce the complexity of coefficient fitting.

7. **Ensemble Learning**: Ensemble learning can reduce the variance of genetic programming algorithms.

    Reason: There may be multiple GP individuals with good performance in the population, and integrating the prediction results of these individuals can effectively reduce the randomness of prediction.

    Paper：H. Zhang, A. Zhou and H. Zhang, An Evolutionary Forest for Regression[J]. IEEE Transactions on Evolutionary Computation, 2021.
