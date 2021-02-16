# awesome-GNN
关于图神经网络的一些工作总结

## 图神经网络的更新方式除了一下全部aggregate以外，还可以按照分阶段来处理。
比如知识图谱我可以把每个三元组的进行bilinear。然后使用attention来搞。

# 1.Hypergraph Neural Networks@AAAI2020
high-order data correlation(更加flexible)
## Motivation:
1.主要就是high-order data correlation@

## 我们的技术核心创新
1.传统的hypergraph learning过程可以通过本文提出的hyperedge convolution进行优化～

## introduction介绍方式
就是说传统的gcn就是pairwise connections among data are employed，但是事实上data structure in real practice could be beyond pairwise connections and even far more complicated。

尤其是针对多模态数据，整个情形就会变得更佳复杂。

1.data correlation比起pairwise relaionship而言是很复杂的。

2.而且对于异质信息网络而言，传统的GCN就有它的limitation to formulate the data correlation；

3.咱们的HGNN比起传统的edge degree@mandatory 2，我们的可以encode high-order data correlation。

4.hypergraph在cv中很常见，但是由于其很高的计算量，所以wide application就limited了。

GCN可以看作是HGNN的一个special case。


### TODO 这里的公式太spectral了

# 2.Hypergraph Attention Networks for Multimodal Learning
## 1.Motivation:
1.在这个传统的过程中，我们表示align the information level of 异质模态是一个基础的任务@多模态学习

2。我们这里就是创建hypergraph来学习alignment～

![](HGNN.png)

# 3.Event Detection with Multi-order Graph Convolution and Aggregated Attention
这里的核心就是找到一个event trigger（main word to the 对应的event，依靠这个词就可以分类出这个话到底是什么类型的event，比如fired就是attack）
## Motivation:
1.这里就是前人工作使用的dependency tree的话，就是trigger words和related entities的syntactic relations可能是first-order的，也可能是high-order的。而且，据统计，51%的是高order的。

2.虽然目前的使用high-order都是stack more GCN layers。但是事实上oversmooth就有问题。那么我们搞几个～

3.我们就是同时使用first-order和high-order的graph来encoding，其中使用GAT来自适应决定邻居words的weight～

## 工作内容
### 1.Word Encoding
word embedding+entity type embedding+POStagging embedding

BiLSTM
### 2.Multi-order graph attention network
首先关于A有三个sub matrixs，他们的shape都是nxn的，Aalong,Arev,Aloop. along呢，就是如果这儿在dependency tree中有dependency arc的话，就有1@Aalong。Arev就是Aalong的转置矩阵～
Aloop就是identity matrix。

然后咱们这里额外搞了几个k-th order syntactic graph。就是edge仍然是三种，不过along的就是变成了之前的k-order。


计算的时候就是GAT自适应学习，然后element-addition就行～那么我们这里设置的K=3.

![](Multiorder.png)

# 3. Learning Multi-Granular Hypergraphs for Video-Based Person Re-id
### introduction介绍方式
this work aims to .In this sense.

对于person re-id而言，multi-granular的spatial relation和multi-granular 的temporal relation都是十分重要的～

# 4. Sentence Specified Dynamic Video Thumbnail Generation
Video Thumbnail 就是和视频摘要差不多～

Video Thumbnail Generation就是产生video content preview预演，这个对于影响users' online searching experiences~

这里就是提出一个新任务。不仅是生成Video Summarization还要针对用户的个人兴趣来自动生成～（多模态）
## 1.Motivation:
1.传统的video thumbnail仅仅只是利用了视觉特征@video，而没有user的搜索意图不能提供一个有意义的snapshot简介 of the video contents that users concern.

2.我们提出的模型GTP就是利用sentence specified video graph convolutional network@建模both 句子-视频 语义interaction以及内部的视频关系@结合sentence information。（基于temporal conditioned pointer network）（就是一种graph之前做了一个匹配的操作～）

# 5. Temporal Dynamic Graph LSTM for Action-driven Video Object Detection@TD-GraphLSTM

# 6. Object-Aware Multi-Branch Relation Networks for Spatio-Temporal Video Grounding
## 1.Motivation
1.许多现有的Grounding work都是局限于well-aligned segment-sentence pairs～那么我们的工作就是unaligned data and multi-form sentences.

2.这个任务就是需要capture重要的object relations去identify the queried target.

3.但是现有的办法无法区分出notable objects以及关系建模还不是那么有效@针对不重要的对象～

## 2.我们的Contributions@我们提出一个创新性的object-aware multi-branch relation network@object-aware relation discovery.
1.我们就是设计了multiple branches去发展object-aware region modeling（每个branch都关注一个crucial object in 句子）

2.然后我们就是提出一个multi-branch relation reasoning去捕捉关键object 关系@main branch以及附属branches～

3.除此以外，我们使用了一个diversity loss去确保每个branch只关注自己应该关注的corresponding objects。

# 7.Zero-Shot Video Object Segmentation via Attentive Graph Neural Networks
这里就是一个全连接图，然后relations between 绝对frame pairs作为边～

这个潜在的对关系就是使用注意力机制来进行迭代更新

实验结果证明我们的网络可以发现到常见的对象objects










