import filecmp

rightModelPath = 'modelInfo/VGGNet.ckpt.meta'
userModelPath = 'userModelInfo/VGGNet.ckpt.meta'

isRight = filecmp.cmp(rightModelPath, userModelPath)

if isRight:
    print('Right')
else:
    print('Wrong')
