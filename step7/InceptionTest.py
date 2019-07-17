import filecmp

rightModelPath = 'modelInfo/InceptionNet.meta'
userModelPath = 'userModelInfo/InceptionNet.meta'

try:

    isRight = filecmp.cmp(rightModelPath, userModelPath)

    if isRight:
        print('Right')
    else:
        print('Wrong')
except:
    print('Wrong')