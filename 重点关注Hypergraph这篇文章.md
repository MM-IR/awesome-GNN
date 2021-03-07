# 1.Learning Multi-Granular Hypergraphs for Video-Based Person Re-Identification

## 关于temporal multi-granular的graph学习
这个多粒度就是使用多个不同的邻接矩阵来学习。然后就是使用topk作为hyperedge@然后就是限制邻居范围。

本文中使用的邻居包括不同粒度的比如说temporal adjacent还有固定范围的。1/3/5，一直实验到饱和为止。

K主要包括
