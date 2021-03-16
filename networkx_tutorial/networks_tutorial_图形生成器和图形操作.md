# 1.应用经典图形操作，比如笛卡尔集/并集/交集等等.

![](standardgraph.jpg)

# 2.调用一个经典的小图。

![](SmallGraph.jpg)

# 3.对经典图形使用/构造生成器。

![](SPGraph.jpg)

这里就是一些方便的application，比如complete_graph等等。

```
>>> K_5 = nx.complete_graph(5)
>>> K_3_5 = nx.complete_bipartite_graph(3, 5)
>>> barbell = nx.barbell_graph(10, 10)
>>> lollipop = nx.lollipop_graph(10, 20)
```

# 4.使用随机图形生成器。
![](randomgraph.jpg)

# 5.使用常用的图形格式读取存储在文件中的图形，比如列表、邻接列表、gml、graphml、pickle、leda等。

```
>>> nx.write_gml(red, "path.to.file")
>>> mygraph = nx.read_gml("path.to.file")
```

# 6.分析图形+一系列算法

![](AnalysisGraph.jpg)

# 7.图形绘制

networkx主要不是一个图形绘制包，而是一个带有matplotlib的基本绘图，以及一个使用开源graphviz软件包的接口，这些是networkx.drawing模块，如果可能，将导入。

```
>>> import matplotlib.pyplot as plt
>>> G = nx.petersen_graph()
>>> plt.subplot(121)
<matplotlib.axes._subplots.AxesSubplot object at ...>
>>> nx.draw(G, with_labels=True, font_weight='bold')
>>> plt.subplot(122)
<matplotlib.axes._subplots.AxesSubplot object at ...>
>>> nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
```
