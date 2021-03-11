# 1.两种方式去抽取特征来表示graphs
1.feature engineering。
这个就是手工特征比较耗时间而且不是optimal。


2.representation learning。
这个就是minimal human efforts+可适应到下游任务。

# 2.三代的graph表达学习
1.traditional graph embedding

这个就是使用一些经典的**降维技术**，比如IsoMap/LLE/eigenmap。

2.modern graph embedding

这个就是受到Word2Vec在NLP的成功的启发，我们这里就是将其迁移到图结构里来。

3.深度学习

这里也来自于图像和文本上的成功～
**尤其是GNN促进了计算任务，比如node-focused和graph-focused。**

这里就是比如经典的图表示学习的domain，比如推荐系统和社交网络分析，GNN就已经获得了SOTA，bring them into new frontiers。

与此同时一些新的domain也比较时髦，比如组合优化/物理/医疗。GNN使得本身无关的discipline变得更加interdisciplinary了。

