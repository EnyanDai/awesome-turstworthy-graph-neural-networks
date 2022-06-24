# Awesome Trustworthy Graph Neural Networks
This repository aims to provide links to works in trustworthy graph neural networks. If you find this repo useful, please cite our survey [A Comprehensive Survey on Trustworthy Graph Neural Networks: Privacy, Robustness, Fairness, and Explainability](https://arxiv.org/pdf/2204.08570.pdf) with:
```
@article{dai2022comprehensive,
  title={A Comprehensive Survey on Trustworthy Graph Neural Networks: Privacy, Robustness, Fairness, and Explainability},
  author={Dai, Enyan and Zhao, Tianxiang and Zhu, Huaisheng and Xu, Junjie and Guo, Zhimeng and Liu, Hui and Tang, Jiliang and Wang, Suhang},
  journal={arXiv preprint arXiv:2204.08570},
  year={2022}
}
```

## Content
* [1. Survey Papers](#1-survey-papers)
* [2. Datasets](#2-datasets)
    * [2.1 Fairness](#21-fairness)
    * [2.2 Privacy](#22-privacy)
    * [2.3 Explainability](#23-explainability)
* [3. Fairness](#3-fairness)
* [4. Privacy](#4-privacy)
    * [4.1 Privacy Attacks on Graphs](#41-privacy-attacks-on-graphs)
    * [4.2 Privacy-Preserving GNNs](#42-privacy-preserving-gnns)
* [5. Explainability](#5-explainability)
    * [5.1 Self-Explainable GNNs](#51-self-explainable-gnns)
    * [5.2 Posthoc Explainable GNNs](#52-posthoc-explainable-gnns)
* [6. Robustness](#6-robustness)
    * [6.1 Graph Adversarial Attacks](#61-graph-adversarial-attacks)
    * [6.2 Robust GNNs](#62-robust-gnns)
## 1. Survey Papers
1. **Adversarial Attacks and Defenses on Graphs: A Review and Empirical Study.**
    SIGKDD Explorations 2020. [[paper]](https://arxiv.org/abs/2003.00653) [[code]](https://github.com/DSE-MSU/DeepRobust/)
1. **A Survey of Adversarial Learning on Graphs.**
 arxiv, 2020. [[paper]](https://arxiv.org/abs/2003.05730)
1. **Adversarial Attacks and Defenses in Images, Graphs and Text: A Review.**
   arxiv, 2019. [[paper]](https://arxiv.org/pdf/1909.08072.pdf)
1. **Adversarial Attack and Defense on Graph Data: A Survey.**
    arxiv 2018. [[paper]](https://arxiv.org/pdf/1812.10528.pdf) 

## 2. Datasets
### 2.1 Fairness
|Dataset|  Task   | Labels | Sensitive Attributes | Link |
|------|---------|--------|----------------------|------|
| Pokec-n | Node classificaiton | Job | Region | [[code](https://github.com/EnyanDai/FairGNN/tree/main/dataset)]|
| Pokec-z | Node classificaiton | Job | Region | [[code](https://github.com/EnyanDai/FairGNN/tree/main/dataset)]|
| NBA | Node classificaiton | Salary | Nationality | [[code](https://github.com/EnyanDai/FairGNN/tree/main/dataset)]|
| German Credit | Node classificaiton | Credit Risk | Gender | [[code](https://github.com/chirag126/nifty/tree/main/dataset)]|
| Recidivism | Node classificaiton | Bail | Race | [[code](https://github.com/chirag126/nifty/tree/main/dataset)]|
| Credit Defaulter | Node classificaiton | Default | Age | [[code](https://github.com/chirag126/nifty/tree/main/dataset)]|
| MovieLens | Link Prediction | - | Multi-attribute | [[code](https://grouplens.org/datasets/movielens/)]|
| Reddit | Link Prediction | - | Multi-attribute | [[code](https://files.pushshift.io/reddit/submissions/)]|
| Polblog | Link Prediction | - | Community | [[code](http://konect.cc/networks/dimacs10-polblogs/)]|
| Twitter | Link Prediction | - | Politics | [[code](https://github.com/ahmadkhajehnejad/CrossWalk/tree/master/data)]|
| Facebook | Link Prediction | - | Gender | [[code](https://github.com/jiank2/inform/tree/main/data)]|
| Google+ | Link Prediction | - | Gender | [[code](http://konect.cc/networks/ego-gplus/)]|
| Dutch | Link Prediction | - | Gender | [[code](https://www.stats.ox.ac.uk/~snijders/siena/tutorial2010_data.htm)]|


### 2.2 Privacy
| Dataset | Task | #Graphs | #Nodes | Link |
|---------|------|--------|---------|------|
|To add|

### 2.3 Explainability
| Dataset | Task | #Graphs | #Nodes | Link |
|---------|------|--------|---------|------|
|To add|

## 3. Fairness
order by years
1. **EDITS: Modeling and Mitigating Data Bias for Graph Neural Networks** WWW 2022. [[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3512173)], [[code](https://github.com/yushundong/EDITS)]
1. **Unbiased graph embedding with biased graph observations** WWW 2022. [[paper](https://dl.acm.org/doi/pdf/10.1145/3485447.3512173)]
1. **CrossWalk: Fairness-enhanced Node Representation Learning** AAAI 2022. [[paper](https://www.aaai.org/AAAI22Papers/AISI-11833.KhajehnejadA.pdf)], [[code](https://github.com/ahmadkhajehnejad/CrossWalk)]
1. **Say No to the Discrimination: Learning Fair Graph Neural Networks with Limited Sensitive Attribute Information** WSDM 2021. [[paper](https://arxiv.org/pdf/2009.01454.pdf)], [[code](https://github.com/EnyanDai/FairGNN)]
1. **Towards a unified framework for fair and stable graph representation learning** UAI 2021. [[paper](https://proceedings.mlr.press/v161/agarwal21b/agarwal21b.pdf)], [[code](https://github.com/chirag126/nifty)]
1. **InFoRM: Individual Fairness on Graph Mining** KDD 2020. [[paper](https://dl.acm.org/doi/pdf/10.1145/3394486.3403080)], [[code](https://github.com/jiank2/inform)]
1. **FairDrop: Biased Edge Dropout for Enhancing Fairness in Graph Representation Learning** IEEE Transactions on Artificial Intelligence 2021. [[paper](https://arxiv.org/pdf/2104.14210.pdf)], [[code](https://github.com/ispamm/FairDrop)]
1. **On dyadic fairness: Exploring and mitigating bias in graph connections** ICLR 2021. [[paper](https://openreview.net/pdf?id=xgGS6PmzNq6)], [[code]([https://github.com/farzmas/FLIP](https://github.com/brandeis-machine-learning/FairAdj))]
1. **Individual fairness for graph neural networks: A ranking based approach** KDD 2021. [[paper](http://tonghanghang.org/pdfs/kdd21_redress.pdf)], [[code](https://github.com/yushundong/REDRESS)]
1. **Fairness-Aware Node Representation Learning** KDD 2021. [[paper](https://arxiv.org/pdf/2106.05391.pdf)]
1. **DeBayes: a Bayesian Method for Debiasing Network Embeddings** ICML 2020. [[paper](https://arxiv.org/pdf/2002.11442.pdf)], [[code](https://github.com/aida-ugent/DeBayes)]
1. **Bursting the filter bubble: Fairness-aware network link prediction** AAAI 2020. [[paper](https://arxiv.org/pdf/1905.10674.pdf)], [[code](https://github.com/farzmas/FLIP)]
1. **Compositional Fairness Constraints for Graph Embeddings** ICML 2019. [[paper](https://arxiv.org/pdf/1905.10674.pdf)], [[code](https://github.com/joeybose/Flexible-Fairness-Constraints)]
1. **Fairwalk: Towards fair graph embedding** IJCAI 2019. [[paper](https://www.ijcai.org/Proceedings/2019/0456.pdf], [[code](https://github.com/mridul2899/Fairwalk_Towards_Fair_Graph_Embedding)]
## 4. Privacy
### 4.1 Privacy Attacks on Graphs
1. 
1.

### 4.2 Privacy-Preserving GNNs
1. 


## 5. Explainability
### 5.1 Self-Explainable GNNs
1. **Towards Self-Explainable Graph Neural Network**. CIKM 2021. [[paper]](https://dl.acm.org/doi/abs/10.1145/3459637.3482306)
2. **ProtGNN: Towards Self-Explaining Graph Neural Networks**. AAAI 2022. [[paper]](https://arxiv.org/abs/2112.00911)
3. **Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism**. Arxiv 2022. [[paper]](https://arxiv.org/abs/2201.12987v1)
4. **KerGNNs: Interpretable Graph Neural Networks with Graph Kernels**. AAAI 2022. [[paper]](https://arxiv.org/pdf/2201.00491.pdf)

### 5.2 Posthoc Explainable GNNs

1. **Gnnexplainer: Generating explanations for graph neural networks**. *Ying Rex, Bourgeois Dylan, You Jiaxuan, Zitnik Marinka, Leskovec Jure*. NeurIPS 2019. [[paper]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7138248/) [[code]](https://github.com/RexYing/gnn-model-explainer)
2. **Explainability methods for graph convolutional neural networks**. *Pope Phillip E, Kolouri Soheil, Rostami Mohammad, Martin Charles E, Hoffmann Heiko*. CVPR 2019.[[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pope_Explainability_Methods_for_Graph_Convolutional_Neural_Networks_CVPR_2019_paper.pdf)
3. **Parameterized Explainer for Graph Neural Network**. *Luo Dongsheng, Cheng Wei, Xu Dongkuan, Yu Wenchao, Zong Bo, Chen Haifeng, Zhang Xiang*. NeurIPS 2020. [[paper]](https://arxiv.org/abs/2011.04573) [[code]](https://github.com/flyingdoog/PGExplainer)
4. **Xgnn: Towards model-level explanations of graph neural networks**. *Yuan Hao, Tang Jiliang, Hu Xia, Ji Shuiwang*. KDD 2020. [[paper]](https://dl.acm.org/doi/pdf/10.1145/3394486.3403085). 
5. **Evaluating Attribution for Graph Neural Networks**. *Sanchez-Lengeling Benjamin, Wei Jennifer, Lee Brian, Reif Emily, Wang Peter, Qian Wesley, McCloskey Kevin, Colwell  Lucy, Wiltschko Alexander*. NeurIPS  2020.[[paper]](https://proceedings.neurips.cc/paper/2020/file/417fbbf2e9d5a28a855a11894b2e795a-Paper.pdf)
6. **PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks**. *Vu Minh, Thai My T.*. NeurIPS  2020. [[paper]](https://arxiv.org/pdf/2010.05788.pdf)
7. **Causal Screening to Interpret Graph Neural Networks**. [[paper]](https://openreview.net/pdf?id=nzKv5vxZfge)
8.  **GraphSVX: Shapley Value Explanations for Graph Neural Networks**. ECML PKDD 2021. [[paper]](https://arxiv.org/abs/2104.10482)
9.  **GNES: Learning to Explain Graph Neural Networks**. ICDM 2021. [[paper]](https://www.researchgate.net/profile/Yuyang-Gao-4/publication/355259484_GNES_Learning_to_Explain_Graph_Neural_Networks/links/616986a6b90c512662459391/GNES-Learning-to-Explain-Graph-Neural-Networks.pdf)
10.  **Generative Causal Explanations for Graph Neural Networks**. ICML 2021. [[paper]](https://arxiv.org/abs/2104.06643)
11.  **On Explainability of Graph Neural Networks via Subgraph Explorations**. ICML 2021. [[paper]](https://arxiv.org/abs/2102.05152)
12.  **Zorro: Valid, Sparse, and Stable Explanations in Graph Neural Networks**.  [[paper]](https://arxiv.org/abs/2105.08621)
13.  **Robust Counterfactual Explanations on Graph Neural Networks**. Neurips 2021. [[paper]](https://arxiv.org/abs/2107.04086)
14.  **When Comparing to Ground Truth is Wrong: On Evaluating GNN Explanation Methods**. KDD 2021. [[paper]](https://dl.acm.org/doi/abs/10.1145/3447548.3467283)
15.  **Towards Multi-Grained Explainability for Graph Neural Networks**. Neurips 2021. [[paper]](http://staff.ustc.edu.cn/~hexn/papers/nips21-explain-gnn.pdf)
16.  **Reinforcement Learning Enhanced Explainer for Graph Neural Networks**. Neurips 2021. [[paper]](http://recmind.cn/papers/explainer_nips21.pdf)
17.  **Discovering Invariant Rationales for Graph Neural Networks**. ICLR 2022. [[paper]](https://arxiv.org/abs/2201.12872)
18.  **Zorro: Valid, Sparse, and Stable Explanations in Graph Neural Networks**. Arxiv 2021. [[paper]](https://arxiv.org/abs/2105.08621)
19.  **On Consistency in Graph Neural Network Interpretation**. Arxiv 2022. [[paper]](https://arxiv.org/abs/2205.13733)
20. **GRAPHSHAP: Motif-based Explanations for Black-box Graph Classifiers**. Arxiv 2022. [[paper]](https://arxiv.org/abs/2202.08815)
21.  **MotifExplainer: a Motif-based Graph Neural Network Explainer**. Arxiv 2022. [[paper]](https://arxiv.org/abs/2202.00519)
22.  **Reinforced Causal Explainer for Graph Neural Networks**. TPAMI 2022. [[paper]](https://arxiv.org/abs/2204.11028)
23.  **CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks**. AISTATS 2022  [[paper]](https://arxiv.org/abs/2102.03322)
24. **Prototype-Based Explanations for Graph Neural Networks**. AAAI 2022. [[paper]](https://www.aaai.org/AAAI22Papers/SA-00396-ShinY.pdf)
25. **FlowX: Towards Explainable Graph Neural Networks via Message Flows**. OpenReview 2021. [[paper]](https://openreview.net/pdf?id=mRF387I4Wl)


## 6. Robustness
### 6.1 Graph Adversarial Attacks
1. **Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective.** 
   *Kaidi Xu, Hongge Chen, Sijia Liu, Pin-Yu Chen, Tsui-Wei Weng, Mingyi Hong, Xue Lin.* IJCAI 2019. [[paper]](https://arxiv.org/pdf/1906.04214.pdf) [[code]](https://github.com/KaidiXu/GCN_ADV_Train)
1. **Fast Gradient Attack on Network Embedding.**
*Jinyin Chen, Yangyang Wu, Xuanheng Xu, Yixian Chen, Haibin Zheng, Qi Xuan.* arxiv 2018. [[paper]](https://arxiv.org/pdf/1809.02797.pdf) [[code]](https://github.com/DSE-MSU/DeepRobust)
1. **Adversarial Examples on Graph Data: Deep Insights into Attack and Defense.**
   *Huijun Wu, Chen Wang, Yuriy Tyshetskiy, Andrew Docherty, Kai Lu, Liming Zhu.* IJCAI 2019. [[paper]](https://arxiv.org/pdf/1903.01610.pdf) [[code]](https://github.com/DSE-MSU/DeepRobust)
1. **Robustness of Graph Neural Networks at Scale.** NeurIPS 2021. [[paper]](https://www.in.tum.de/daml/robustness-of-gnns-at-scale/) [[code]](https://github.com/sigeisler/robustness_of_gnns_at_scale)
1. **Adversarial Attack on Large Scale Graph.** TKDE 2021. [[paper]](https://arxiv.org/pdf/2009.03488.pdf)
1. **Scalable Attack on Graph Data by Injecting Vicious Nodes.** arxiv 2020. [[paper]](https://arxiv.org/pdf/2004.14734.pdf)
1. **Graph Backdoor.**
  *Zhaohan Xi, Ren Pang, Shouling Ji, Ting Wang.* USENIX 2021. [[paper]](https://arxiv.org/abs/2006.11890)
1. **Backdoor Attacks to Graph Neural Networks.**
  *Zaixi Zhang, Jinyuan Jia, Binghui Wang, Neil Zhenqiang Gong.* arxiv 2020. [paper](https://arxiv.org/abs/2006.11165)
1. **A Restricted Black-box Adversarial Framework Towards Attacking Graph Embedding Models.** 
   *Heng Chang, Yu Rong, Tingyang Xu, Wenbing Huang, Honglei Zhang, Peng Cui, Wenwu Zhu, Junzhou Huang.* AAAI 2020. [[paper]](https://arxiv.org/pdf/1908.01297.pdf) [[code]](https://github.com/SwiftieH/GFAttack)
1. **Adversarial Attacks on Node Embeddings via Graph Poisoning.** 
   *Aleksandar Bojchevski, Stephan Günnemann.* ICML 2019. [[paper]](https://arxiv.org/pdf/1809.01093.pdf) [[code]](https://github.com/abojchevski/node_embedding_attack)
1. **Adversarial Attack on Graph Structured Data.** [[paper]](https://arxiv.org/pdf/1806.02371.pdf) [[code]](https://github.com/Hanjun-Dai/graph_adversarial_attack)
1. **Adversarial Attacks on Neural Networks for Graph Data.**
   *Daniel Zügner, Amir Akbarnejad, Stephan Günnemann.*  KDD 2018. [[paper]](https://arxiv.org/pdf/1805.07984.pdf) [[code]](https://github.com/danielzuegner/nettack)
1. **Attacking Graph Neural Networks at Scale.** 
   *Simon Geisler, Daniel Zügner, Aleksandar Bojchevski, Stephan Günnemann.* AAAI workshop 2021. [[paper]](https://www.dropbox.com/s/ddrwoswpz3wwx40/Robust_GNNs_at_Scale__AAAI_Workshop_2020_CameraReady.pdf?dl=0)
1. **Attacking Graph-based Classification via Manipulating the Graph Structure.**
*Binghui Wang, Neil Zhenqiang Gong.* CCS 2019. [[paper]](https://arxiv.org/pdf/1903.00553.pdf)

1. **Adversarial Attacks on Graph Neural Networks via Meta Learning.**
   *Daniel Zugner, Stephan Gunnemann.* ICLR 2019. [[paper]](https://openreview.net/pdf?id=Bylnx209YX) [[code]](https://github.com/danielzuegner/gnn-meta-attack)
1. **Adversarial attacks on neural networks for graph data** KDD 2018. [[paper]()] [[code]()]
1. **Attacking Graph Convolutional Networks via Rewiring.**
   *Yao Ma, Suhang Wang, Lingfei Wu, Jiliang Tang.*  arxiv 2019. [[paper]](https://arxiv.org/pdf/1906.03750.pdf)
1. **Adversarial attacks on graph neural networks via node injections: A hierarchical reinforcement learning approach.** WWW 2020 [[paper]()]
1. **Towards More Practical Adversarial Attacks on Graph Neural Networks.**
   *Jiaqi Ma, Shuangrui Ding, Qiaozhu Mei.*  NeurIPS 2020. [[paper]](https://arxiv.org/abs/2006.05057) [[code]](https://github.com/Mark12Ding/GNN-Practical-Attack)
1. **Single Node Injection Attack against Graph Neural Networks** CIKM 2021. [[Paper]] [[code]]
1. **Towards Robust Graph Neural Networks for Noisy Graphs with Sparse Labels** arxiv 2022. [[Paper]] [[code]] 



## 6.2 Robust GNNs
1. **Adversarial training methods for network embedding.** WWW 2019. [[paper]] [[code]] 
1. **Robustness of Graph Neural Networks at Scale.** NeurIPS 2021. [[paper]](https://www.in.tum.de/daml/robustness-of-gnns-at-scale/) [[code]](https://github.com/sigeisler/robustness_of_gnns_at_scale)
1. **Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective.** 
   *Kaidi Xu, Hongge Chen, Sijia Liu, Pin-Yu Chen, Tsui-Wei Weng, Mingyi Hong, Xue Lin.* IJCAI 2019. [[paper]](https://arxiv.org/pdf/1906.04214.pdf) [[code]](https://github.com/KaidiXu/GCN_ADV_Train)
1. **Adversarial Attack on Graph Structured Data.** [[paper]](https://arxiv.org/pdf/1806.02371.pdf) [[code]](https://github.com/Hanjun-Dai/graph_adversarial_attack)
1. **Graph Adversarial Training: Dynamically Regularizing Based on Graph Structure**
   *Fuli Feng, Xiangnan He, Jie Tang, Tat-Seng Chua.*  TKDE 2019. [[paper]](https://arxiv.org/pdf/1902.08226.pdf)

1. **GraphDefense: Towards Robust Graph Convolutional Networks.**
   *Xiaoyun Wang, Xuanqing Liu, Cho-Jui Hsieh.*  arxiv 2019. [[paper]](https://arxiv.org/pdf/1911.04429v1.pdf)
1. **All You Need is Low (Rank): Defending Against Adversarial Attacks on Graphs.**
   *Negin Entezari, Saba Al-Sayouri, Amirali Darvishzadeh, and Evangelos E. Papalexakis.*  WSDM 2020. [[paper]](https://dl.acm.org/doi/pdf/10.1145/3336191.3371789?download=true) [[code]](https://github.com/DSE-MSU/DeepRobust/)
4. **GNNGuard: Defending Graph Neural Networks against Adversarial Attacks.** NeurIPS 2020. [[paper]](https://arxiv.org/abs/2006.08149)
3. **Node Similarity Preserving Graph Convolutional Networks.** WSDM 2021. [[paper]](https://arxiv.org/abs/2011.09643) [[code]](https://github.com/ChandlerBang/SimP-GCN)
1. **Robust Graph Neural Network Against Poisoning Attacks via Transfer Learning.**
   *Xianfeng Tang, Yandong Li, Yiwei Sun, Huaxiu Yao, Prasenjit Mitra, Suhang Wang.*  WSDM 2020. [[paper]](https://arxiv.org/pdf/1908.07558.pdf)
1. **Robust Graph Convolutional Networks Against Adversarial Attacks.**
   *Dingyuan Zhu, Ziwei Zhang, Peng Cui, Wenwu Zhu.*  KDD 2019. [[paper]](http://pengcui.thumedialab.com/papers/RGCN.pdf) 
1. **Adversarial Examples on Graph Data: Deep Insights into Attack and Defense.**
   *Huijun Wu, Chen Wang, Yuriy Tyshetskiy, Andrew Docherty, Kai Lu, Liming Zhu.*   IJCAI 2019. [[paper]](https://arxiv.org/pdf/1903.01610.pdf) [[code]](https://github.com/DSE-MSU/DeepRobust)
1. **Learning to drop: Robust graph neural network via topological denoising.** WSDM 2021. [[paper]] [[code]]
1. **Towards Robust Graph Neural Networks for Noisy Graphs with Sparse Labels.** WSDM 2022. [[paper]] [[code]] 
9. **Graph Structure Learning for Robust Graph Neural Networks.**
*Wei Jin, Yao Ma, Xiaorui Liu, Xianfeng Tang, Suhang Wang, Jiliang Tang*. KDD 2020. [[paper]](https://arxiv.org/abs/2005.10203) [[code]](https://github.com/ChandlerBang/Pro-GNN)
1. **Can Adversarial Network Attack be Defended?** arxiv 2019. [[paper]] [[code]]
1. **Learning robust representations with graph denoising policy network.** ICDM 2019. [[paper]] [[code]] 
1. **Batch Virtual Adversarial Training for Graph Convolutional Networks.**
   *Zhijie Deng, Yinpeng Dong, Jun Zhu.*  ICML 2019 Workshop. [[paper]](https://arxiv.org/pdf/1902.09192.pdf) 
1. **Understanding structural vulnerability in graph convolutional networks** arxiv 2021. [[paper]] [[code]] 
1. **Towards Self-Explainable Graph Neural Network.** CIKM 2021.  [[paper]] [[code]] 
1. **Graph Contrastive Learning with Augmentations.** NeurIPS 2020. [[paper]](https://arxiv.org/abs/2010.13902) [[code]](https://github.com/Shen-Lab/GraphCL)
1. **Robust Unsupervised Graph Representation Learning via Mutual Information Maximization** arxiv 2022. [[paper]]

1. **Certified Robustness of Graph Convolution Networks for Graph Classification under Topological Attacks.** NeurIPS 2020. [[paper]](https://arxiv.org/abs/2009.05872) [[code]](https://github.com/RobustGraph/RoboGraph)
1. **Adversarial Immunization for Improving Certifiable Robustness on Graphs.** Arxiv 2020. [[paper]](https://arxiv.org/abs/2007.09647)
1. **Certified Robustness of Graph Neural Networks against Adversarial Structural Perturbation.** Arxiv 2020. [[paper]](https://arxiv.org/abs/2008.10715)
1. **Efficient Robustness Certificates for Graph Neural Networks via Sparsity-Aware Randomized Smoothing.** ICML 2020. [[paper]](https://proceedings.icml.cc/static/paper_files/icml/2020/6890-Paper.pdf) [[code]](https://github.com/abojchevski/sparse_smoothing)
1. **Certifiable Robustness of Graph Convolutional Networks under Structure Perturbations.** KDD 2020. [[paper]](https://dl.acm.org/doi/abs/10.1145/3394486.3403217) [[code]](https://github.com/danielzuegner/robust-gcn-structure)
1. **Certified Robustness of Community Detection against Adversarial Structural Perturbation via Randomized Smoothing.**
*Jinyuan Jia, Binghui Wang, Xiaoyu Cao, Neil Zhenqiang Gong.* WWW 2020. [[paper]](https://arxiv.org/pdf/2002.03421.pdf)
1. **Certifiable Robustness to Graph Perturbations.**
   *Aleksandar Bojchevski, Stephan Günnemann.*  NeurIPS 2019. [[paper]](https://arxiv.org/pdf/1910.14356.pdf)[[code]](https://github.com/abojchevski/graph_cert)
1. **Certifiable Robustness and Robust Training for Graph Convolutional Networks.**
   *Daniel Zügner Stephan Günnemann.*  KDD 2019. [[paper]](https://arxiv.org/pdf/1906.12269.pdf) [[code]](https://github.com/danielzuegner/robust-gcn)
