import sys

import tensorflow as tf

sys.path.append('..\\step7')
sys.path.append('..\\step3')
sys.path.append('..\\step9')
sys.path.append('..\\step8')

from generatorCompleted import batchGenerator
from outputsUtilsCompleted import softmax, returnOneHot, computeAccuracy
from InceptionCompleted import Conv, maxpool
from ResNetCompleted import multiChannelWeightLayer,ResNetBlock


if __name__ == '__main__':
    pass
    #********** Begin **********#


    # saver.save(sess, "Model/FinalNet")

    #********** End **********#