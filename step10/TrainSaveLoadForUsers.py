import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

import sys
sys.path.append('step3')
sys.path.append('step9')

from generatorCompleted import batchGenerator
from outputsUtilsCompleted import softmax, returnOneHot, computeAccuracy

from prevModules import (Inception_traditional, Inception_parallelAsymmetricConv,
                               Inception_AsymmetricConv,InitialPart,reduction,ResNetBlock)


#********** Begin **********#
# 任意发挥，完成模型并训练，保存模型PATH为: 'step10/Model/FinalNet'
# 可选placeholder以及要求name参数值:
# 输入X: batchImgInput / 输入Y: Labels / dropout保存概率: dropout_keep_prob
# batchNorm层training: BNTraining / 前文所述ResNet实现中批数据数量: InputBatchSize
#********** End **********#