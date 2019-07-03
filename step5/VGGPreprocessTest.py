import numpy as np
from VGGPreprocessForUsers import VGGPreprocessingBatch as userV
from VGGPreprocessCompleted import VGGPreprocessingBatch as rightV

if __name__ == '__main__':
# ----------------------example---------------------------
    a = np.array([[[1, 2, 3], [4, 5, 6], [1, 1, 1]],
                  [[7, 8, 9], [10, 11, 12], [2, 2, 2]],
                  [[13, 14, 15], [16, 17, 18], [0, 3, 6]]])
    b = userV(a)
    c = rightV(a)
    bool = (b==c).all()
    if bool:
        print('Right')
    else:
        print('Wrong')