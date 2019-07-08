import filecmp

rightModelPath = 'modelInfo/InceptionNet.ckpt.meta'
userModelPath = 'userModelInfo/InceptionNet.ckpt.meta'

isRight = filecmp.cmp(rightModelPath, userModelPath)

if isRight:
    print('Right')
else:
    print('Wrong')
