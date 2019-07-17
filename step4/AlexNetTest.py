import filecmp

rightModelPath = 'modelInfo/AlexNet.ckpt.meta'
userModelPath = 'userModelInfo/AlexNet.ckpt.meta'

try:

    isRight = filecmp.cmp(rightModelPath, userModelPath)

    if isRight:
        print('Right')
    else:
        print('Wrong')
except:
    print('Wrong')