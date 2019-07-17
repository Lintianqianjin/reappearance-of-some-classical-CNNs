import filecmp

rightModelPath = 'modelInfo/ResNet.meta'
userModelPath = 'userModelInfo/ResNet.meta'

try:

    isRight = filecmp.cmp(rightModelPath, userModelPath)

    if isRight:
        print('Right')
    else:
        print('Wrong')
except:
    print('Wrong')