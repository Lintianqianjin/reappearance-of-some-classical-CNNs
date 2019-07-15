
from outputsUtilsCompleted import softmax as correctSoftmax
from outputsUtilsCompleted import computeAccuracy as correctComputeAccuracy
from outputsUtilsCompleted import returnOneHot as correctReturnOneHot

from outputsUtilsForUsers import softmax as UserSoftmax
from outputsUtilsForUsers import computeAccuracy as UserComputeAccuracy
from outputsUtilsForUsers import returnOneHot as UserReturnOneHot


import numpy as np
test_matrix = np.array([[-1,2,25,7],
                        [20,15,10,5],
                        [12,4,19,5],
                        [9,13,61,8],])

test_label = np.array([[0,0,1,0],
                       [1,0,0,0],
                       [0,0,1,0],
                       [0,0,1,0]])
try:
    if (correctSoftmax(test_matrix) == UserSoftmax(test_matrix)).all():
        softmaxed = correctSoftmax(test_matrix)
        if (correctReturnOneHot(softmaxed) == UserReturnOneHot(softmaxed)).all():
            onehoted = correctReturnOneHot(softmaxed)
            if correctComputeAccuracy(onehoted,test_label) == UserComputeAccuracy(onehoted,test_label):
                print('Right')
            else:
                print('Wrong')
        else:
            print('Wrong')
    else:
        print('Wrong')

except :
    print('Wrong')