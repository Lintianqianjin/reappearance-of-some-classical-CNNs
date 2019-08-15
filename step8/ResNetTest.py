import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

import warnings
warnings.filterwarnings('ignore')

import ResNetCompleted
import ResNetForUsers

rightModelPath = 'step8/modelInfo/ResNet'
userModelPath = 'step8/userModelInfo/ResNet'

# print(os.path.exists(rightModelPath))
# print(os.path.exists(userModelPath))
# print(os.path.getsize(rightModelPath))
# print(os.path.getsize(userModelPath))


try:

    isRight = os.path.getsize(rightModelPath)==os.path.getsize(userModelPath)
    # print(isRight)

    if isRight:
        print('恭喜你通过本关测试!你已掌握ResNet结构!',end='')
    else:
        print('未能通过本关测试!模块结构有误!')
except:
    print('未能通过本关测试!模型结构文件没能正确保存!')