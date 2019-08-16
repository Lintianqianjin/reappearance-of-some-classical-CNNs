[TOC]

---

####任务描述


本关任务：编写一个能不断输出一个固定个数的数据的类。


####相关知识


为了完成本关任务，你需要掌握：1.打乱列表顺序；2.返回列表对应元素的索引；3.简单的数组的切片。

#####打乱列表顺序

`numpy.random.shuffle()`可以打乱一个数组本来的顺序。

```python
import numpy as np
l = [0,1,2,3,4,5]
print(f'初始：{l}')
np.random.shuffle(l)
print(f'打乱后：{l}')
```

输出：

```
初始：[0, 1, 2, 3, 4, 5]
打乱后：[2, 1, 5, 4, 0, 3]
```

#####返回列表对应元素的索引
根据元素的值返回其在列表中的索引。列表自带的方法`index()`就可以完成这个任务。

示例如下：

```python
labels = ['bus','family sedan','fire engine','racing car']
i = labels.index('bus')
print(f'bus 的 index 是 {i}')
```

输出：`bus 的 index 是 0`

#####简单的数组的切片

即选择一个数组的某一部分。本例只需要用到一维数组的切片，这里仅用一维数组举例，使用列表索引时用`:`隔开起始和结尾索引就行，所得的元素包含起始，不包含结尾。如果索引是负数，代表从后往前数。如果需要最后的元素，可以使用`None`。

示例如下：

```python
labels = ['bus','family sedan','fire engine','racing car']
start,end = [1,-1]
print(labels[start:end])
print(labels[start:None])
```

输出

`['family sedan', 'fire engine']`
`['family sedan', 'fire engine', 'racing car']`

####编程要求

根据提示，在右侧编辑器补充代码，完成生成器。

####测试说明

平台会调用你写的生成器，检验是否正确地迭代返回了数据。


---
开始你的任务吧，祝你成功！
