# 找到最理想的分类器对应的mapping function～
a maps an input to a target category y. In this case, a feedforward network is supposed to find 
a mapping f (x|Θ) such that it can approximate the ideal classifier f∗(x)well.

信息x就是flow from the input，通过一些中间计算，然后最终获得output y。

**这个就是一系列的compositions of several functions**

但是只有最后的layer才会有supervision signals。中间的就没有direct 监督，所以这就叫hidden layers。

## FFN每层都可以看作是vector-valued function。
elements in the layer可以被看作是nodes/units。

这个网络之所以叫神经网络是因为这个是神经科学所启发的。

## 激活函数就是决定what extent the information can pass through to the next layer。
非线性就是改进approximation capability。
