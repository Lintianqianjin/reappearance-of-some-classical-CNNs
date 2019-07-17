import filecmp

rightModelPath = 'modelInfo/VGGNet.meta'
userModelPath = 'userModelInfo/VGGNet.meta'

try:
    isRight = filecmp.cmp(rightModelPath, userModelPath)

    if isRight:
        print('Right')
    else:
        print('Wrong')
except:
    print('Wrong')