# 我们可以检查节点和边缘
4个基本图形属性，包括:

1)G.nodes

2)G.edges

3)G.adj

4)G.degree

这里给图形结构提供了一个不断更新的只读视图。它们也是类似dict的，，因为您可以通过视图查找节点和边缘数据属性，并使用方法迭代数据属性。 .items() ， .data('span') 

## 如果您想要一个特定的容器类型而不是视图。list

尽管集合/dict/元组和其他容器在其他上下文中可能更好。

```
1.list(G.nodes)

2.list(G.edges)

3.list(G.adj[1])  # or list(G.neighbors(1))
[2, 3]
4.G.degree[1] # the number of edges incident to 1

5.重点，对于非1st order的edge而言也可以显示。
G.edges([2,'m'])


6.我们可以同时查看多个节点对应的度。
G.degree([2,3])

```


