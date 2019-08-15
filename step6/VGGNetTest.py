import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import warnings
warnings.filterwarnings('ignore')

import VGGNetCompleted

import VGGNetForUsers


rightModelPath = 'step6/modelInfo/VGGNet'
userModelPath = 'step6/userModelInfo/VGGNet'

# print(os.path.exists(rightModelPath))
# print(os.path.exists(userModelPath))
# print(os.path.getsize(rightModelPath))
# print(os.path.getsize(userModelPath))


isRight = os.path.getsize(rightModelPath)==os.path.getsize(userModelPath)
# print(isRight)
if isRight:
    print('恭喜你通过本关测试!VGGNet的结构你已经掌握!',end='')
else:
    print('未能通过本关测试,模型结构有误!')
