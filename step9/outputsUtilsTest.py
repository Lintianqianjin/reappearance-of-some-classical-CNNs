from outputsUtilsForUsers import softmax as Softmax
from outputsUtilsForUsers import computeAccuracy as ComputeAccuracy
from outputsUtilsForUsers import returnOneHot as ReturnOneHot
#
# from outputsUtilsForUsers import softmax as Softmax
# from outputsUtilsForUsers import computeAccuracy as ComputeAccuracy
# from outputsUtilsForUsers import returnOneHot as ReturnOneHot


'''
测试矩阵:
[[-1  2 25  7]
 [20 15 10  5]
 [12  4 19  5]
 [ 9 13 61  8]]
对应标签:
[[0 0 1 0]
 [1 0 0 0]
 [0 0 1 0]
 [0 0 1 0]]
你softmax的值:
[[0.    0.    1.    0.   ]
 [0.993 0.007 0.    0.   ]
 [0.001 0.    0.999 0.   ]
 [0.    0.    1.    0.   ]]
你onehot编码后的值:
[[0 0 1 0]
 [1 0 0 0]
 [0 0 1 0]
 [0 0 1 0]]
你计算的准确率:
1.0

'''


import numpy as np

test_matrix = np.array([[-1,2,25,7],
                        [20,15,10,5],
                        [12,4,19,5],
                        [9,13,61,8],])

test_label = np.array([[0,0,1,0],
                       [1,0,0,0],
                       [0,0,1,0],
                       [0,0,1,0]])


print('测试矩阵:')
print(test_matrix)
print('对应标签:')
print(test_label)

softmaxed = Softmax(test_matrix)
print('你softmax的值:')
print(np.around(softmaxed,decimals=3))

onehoted = ReturnOneHot(softmaxed)
print('你onehot编码后的值:')
print(onehoted)

accuracy = ComputeAccuracy(onehoted,test_label)
print('你计算的准确率:')
print(accuracy)
