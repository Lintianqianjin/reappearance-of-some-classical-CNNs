import filecmp

rightModelPath = 'modelInfo/AlexNet.ckpt.meta'
userModelPath = 'userModelInfo/AlexNet.ckpt.meta'

isRight = filecmp.cmp(rightModelPath, userModelPath)

if isRight:
    print('Right')
else:
    print('Wrong')
