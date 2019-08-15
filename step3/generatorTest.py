from generatorCompleted import batchGenerator as correctG
from generatorForUsers import batchGenerator as userG
import numpy as np

# 首先至少能够正常构造一个对象
try:
    correct_g = correctG(basePath= 'data/processed/valid_224' ,batchSize=80)
    user_g = userG(basePath= 'data/processed/valid_224' ,batchSize=80)
except:
    print('未能通过本关测试,无法正确新建对象!')
    exit()

# 一次读完全部样本，检查用户产生的X,Y和标答产生的X,Y各个维度的总和是否相等，如果各个维度的和相等，则没问题
try:
    X_c,Y_c = correct_g.getBatch()
    X_u,Y_u = user_g.getBatch()
except:
    print('未能通过本关测试,无法正确调用 getBatch() !')
    exit()

if not(len(X_c)==len(X_u) and len(Y_c)==len(Y_u)):
    print('未能通过本关测试,返回数据长度不正确!')

if not (np.sum(X_c) == np.sum(X_u)):
    print('未能通过本关测试,返回的图片数据有误!')
    exit()

if not (np.sum(Y_c,axis=0) == np.sum(Y_u,axis=0)).all():
    print('未能通过本关测试,返回的标签数据有误!')
    exit()

# 检查是否可以做到最后一次读的数量是剩余的全部样本

correct_g_2 = correctG(basePath= 'data/processed/valid_224' ,batchSize=70)
user_g_2 = userG(basePath= 'data/processed/valid_224' ,batchSize=70)

# 取了70个样本
X_u,Y_u = user_g_2.getBatch()

# 这次应该只取了10个样本
X_u_rest,Y_u_rest = user_g_2.getBatch()

if not(len(X_u_rest)==10) and len(Y_u_rest)==10:
    print('未能通过本关测试,样本数不足batchSize时没能正确返回剩余样本!')
    exit()

# 再次读取时，如果索引更新正确，应该重新读70个样本了
X_u_new,Y_u_new = user_g_2.getBatch()
if not(len(X_u_new)==70) and len(Y_u_new)==70:
    print('未能通过本关测试,数据集全部读完后,不能正确开始第二次循环读取!')
    exit()
# 如果以上全部通过，几乎可以认定正确
print('恭喜你通过本关测试!',end='')




