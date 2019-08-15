import numpy as np
from VGGPreprocessForUsers import VGGPreprocessingBatch as userV

if __name__ == '__main__':
    #     预期输出

    '''
    [[[ -1   0   0]
      [  1   2   3]
      [ -1  -1  -1]]
    
     [[  0   1   2]
      [  3   4   5]
      [ -5  -5  -5]]
    
     [[  1   2   3]
      [  4   5   6]
      [-11  -8  -5]]]'''

# ----------------------example---------------------------
    a = np.array([[[1, 2, 3], [4, 5, 6], [1, 1, 1]],
                  [[7, 8, 9], [10, 11, 12], [2, 2, 2]],
                  [[13, 14, 15], [16, 17, 18], [0, 3, 6]]])

    print('---原始矩阵---')
    print(a)
    print('---处理后矩阵---')
    c = userV(a)
    print(c,end='')