# import numpy as np
# l = [0,1,2,3,4,5]
# print(f'初始：{l}')
# np.random.shuffle(l)
# print(f'打乱后：{l}')

# i = labels.index('bus')
# print(f'bus 的 index 是 {i}')

# labels = ['bus','family sedan','fire engine','racing car']
# start,end = [1,-1]
# print(labels[start:end])
# print(labels[start:None])


# import tensorflow as tf

# input = tf.placeholder(dtype=tf.float32,shape=(1))
#
# output = tf.multiply(input, tf.constant([2.]))
#
# with tf.Session() as sess:
#     a = sess.run(output, feed_dict={input: [3.]})
#     print(a)
import tensorflow as tf
import numpy as np
# X = np.reshape((np.random.ranf(size=50)*2*np.pi),newshape=(50,1))
# Y_np = np.sin(X)
# Y = tf.constant(Y_np,dtype=tf.float32)
# X = tf.constant(X,dtype=tf.float32)
#
# dense1 = tf.keras.layers.Dense(32)(X)
# out = tf.keras.layers.Dense(1)(X)
#
# loss = tf.reduce_mean(tf.losses.mean_squared_error(Y,out))
# train = tf.train.AdamOptimizer().minimize(loss)
#
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#
#     for i in range(64):
#         sess.run(train)
#         if i%8 ==0:
#             cur_out,cur_loss = sess.run([out,loss])
#             print(cur_loss,end=',')

# x = tf.constant(np.array([[1,2,3],
#               [2,4,5],
#               [5,8,7]]).flatten(),dtype=tf.float32)
# y = tf.nn.dropout(x,keep_prob=0.2)
#
# with tf.Session() as sess:
#     # print(x[0])
#     xx = sess.run(x)
#     print(xx)
#     a = sess.run(y)
#     print(a)
# def softmax(x):
#     np.seterr(divide='ignore', invalid='ignore')
#     return (np.exp(x).T / np.sum(np.exp(x), axis=1)).T

# def returnOneHot(NNOutput):
#     out = np.zeros(NNOutput.shape)
#     idx = NNOutput.argmax(axis=1)
#     out[np.arange(NNOutput.shape[0]), idx] = 1
#     return out
#
# a = np.array([[1,2,3],
#      [-2,-3,2]])
# print(returnOneHot(a))
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
x=[1,2,3,4]
y = np.square(x)
plt.plot(x, y,label = 'x**2')
plt.legend()
plt.show()