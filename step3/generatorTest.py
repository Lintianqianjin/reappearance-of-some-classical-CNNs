from generatorCompleted import batchGenerator as correctG
from generatorForUsers import batchGenerator as userG
import numpy as np

correct_g = correctG(basePath= '..\\step1\\processed\\valid_224' ,batchSize=80)
user_g = userG(basePath= '..\\step1\\processed\\valid_224' ,batchSize=80)

try:
    X_c,Y_c = correct_g.getBatch()
    X_u,Y_u = user_g.getBatch()
except:
    print('Wrong')
    exit()

if not (np.sum(X_c) == np.sum(X_u)):
    print('Wrong')
    exit()

if not (np.sum(Y_c,axis=0) == np.sum(Y_u,axis=0)).all():
    print('Wrong')
    exit()

print('Right')
