# 有向图directed graph@DiGraph
这个就是除了一般的graph的属性之外，还有一些别的属性，比如:
DiGraph.out_edges ， DiGraph.in_degree ， 
DiGraph.predecessors() ， DiGraph.successors() 等.

为了使得算法能够轻松地与两个类一起工作，有向版本的neighbors()等于
successors(),虽然degree和in_degree/out_degree尽管有时会觉得不一致。

```
>>> DG = nx.DiGraph()
>>> DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
>>> DG.out_degree(1, weight='weight')
0.5
>>> DG.degree(1, weight='weight') #这个就是显示入度和出度之和。
1.25
>>> list(DG.successors(1))
[2]
>>> list(DG.neighbors(1))
[2]
```

有些算法只适用于有向图，而另一些算法不适用于有向图。

实际上，如果将两者集合在一起的趋势是十分危险的。但是我们也可以将有向图转换成无向图，不过这里当然也要注意啦。

```
1.Graph.to_undirected()

2.H = nx.Graph(G) # create an undirected graph H from a directed graph G
```

## 2.多重图
这个就是multi-dimensional graph啦。

```
>>> MG = nx.MultiGraph()
>>> MG.add_weighted_edges_from([(1, 2, 0.5), (1, 2, 0.75), (2, 3, 0.5)])
>>> dict(MG.degree(weight='weight'))
{1: 1.25, 2: 1.75, 3: 0.5}
>>> GG = nx.Graph()
>>> for n, nbrs in MG.adjacency():
...    for nbr, edict in nbrs.items():
...        minvalue = min([d['weight'] for d in edict.values()])
...        GG.add_edge(n, nbr, weight = minvalue)
...
>>> nx.shortest_path(GG, 1, 3)
[1, 2, 3]
```


