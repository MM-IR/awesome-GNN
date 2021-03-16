# 1.从图形中删除元素
```
G.remove_node(2)
G.remove_nodes_from("spam")
list(G.nodes)
G.remove_edge(1, 3)
```

# 2.使用图形构造函数
我们不一定就要按照增量式来不断添加边和节点来创建一个图。

```
G.add_edge(1, 2)

H = nx.DiGraph(G)   # create a DiGraph using the connections from G

list(H.edges()) # 这个就是根据图实例化一个图。
[(1, 2), (2, 1)]

edgelist = [(0, 1), (1, 2), (2, 3)]
H = nx.Graph(edgelist) # 这个就是按照边来创建图
```

# 3.用作节点和边的内容
虽然节点最常用的选择是:

1.数字

2.字符串

但是节点可以是任何可哈希对象(除了None), 并且edge可以与任何对象关联x。

```
G.add_edge(n1,n2,object=x) #这个就是边和x产生关联了。
```

# 4.访问边缘和邻居
我们除了使用视图Graph.edges+Graph.adj,我们可以使用下标表示法访问边和邻居。

```
>>> G = nx.Graph([(1, 2, {"color": "yellow"})])
>>> G[1]  # same as G.adj[1]
AtlasView({2: {'color': 'yellow'}})
>>> G[1][2]
{'color': 'yellow'}
>>> G.edges[1, 2]
{'color': 'yellow'}
```

## 如果边已经存在，我们还可以直接使用下标表示法**获取/设置** 边的属性。

```
>>> G.add_edge(1, 3)
>>> G[1][3]['color'] = "blue" # 这个就是针对连接1和3的那条边。
>>> G.edges[1, 2]['color'] = "red"
>>> G.edges[1, 2]
{'color': 'red'}
```

## 我们针对有权图的边的权重，还可以这么操作。注意的是，*对于无向图，邻接迭代可以看到每个边两次。*

```
>>> FG = nx.Graph()
>>> FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
>>> for n, nbrs in FG.adj.items():
...    for nbr, eattr in nbrs.items():
...        wt = eattr['weight'] # 属性，默认是weight
...        if wt < 0.5: print(f"({n}, {nbr}, {wt:.3})")
(1, 2, 0.125)
(2, 1, 0.125)
(3, 4, 0.375)
(4, 3, 0.375)
```

## 使用边属性可以方便地访问所有边缘。
```
>>> for (u, v, wt) in FG.edges.data('weight'):
...     if wt < 0.5:
...         print(f"({u}, {v}, {wt:.3})")
(1, 2, 0.125)
(3, 4, 0.375)
```

# 5.向图形/节点和边添加属性。
诸如**权重、标签、颜色或任何您喜欢的python对象等属性**都可以附加到图形、节点或边上。

每个图/节点和边都可以在关联的属性字典中保存“键/值属性对”。**键值必须是可哈希的**。

默认情况下，这些属性为空，但可以使用 add_edge ， add_node 或直接操作命名的属性字典 G.graph ， G.nodes 和 G.edges 对于图 G .

# 6.图形属性。

创建新图形时分配图形属性。

```
G = nx.Graph(day='Friday')
G.graph #这个就是显示:

{'day': 'Friday'}
```

或者您可以稍后修改属性。

```
>>> G.graph['day'] = "Monday"
>>> G.graph
{'day': 'Monday'}
```

## 7.节点属性
使用添加节点属性add_node()/add_nodes_from()/G.nodes

```
>>> G.add_node(1, time='5pm')
>>> G.add_nodes_from([3], time='2pm')
>>> G.nodes[1]
{'time': '5pm'}
>>> G.nodes[1]['room'] = 714
>>> G.nodes.data()
NodeDataView({1: {'time': '5pm', 'room': 714}, 3: {'time': '2pm'}})
```

这里不仅将节点添加到G.nodes不将其添加到图表中，使用G.add_nodes()添加新节点，这一点同样适用于边缘。

## 8.边缘属性
使用添加/更改边缘属性add_edge(), add_edges_from()或者下标符号。

```
>>> G.add_edge(1, 2, weight=4.7 )
>>> G.add_edges_from([(3, 4), (4, 5)], color='red')
>>> G.add_edges_from([(1, 2, {'color': 'blue'}), (2, 3, {'weight': 8})])
>>> G[1][2]['weight'] = 4.7
>>> G.edges[3, 4]['weight'] = 4.2
```

特殊属性weight应该是数字，因为它被需要加权边缘的算法使用。


