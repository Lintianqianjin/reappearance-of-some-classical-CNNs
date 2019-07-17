import tensorflow as tf
a = [[[1., 2., 3.], [3., 4., 5.]],
     [[3., 5., 4.], [2., 1., 4.]]]

x = tf.constant([[1.,3.],
                  [3.,2.]]
)

x_ = tf.constant([[10.,20.],
                  [30.,40.],
                  [50.,60.]
                  ])

b = [[[10., 20.],
      [30., 40.]],

     [[-10., -20.],
      [-30., -40.]],

     [[-5., -15.],
      [-3., -4.]]]


c = tf.transpose(a, [2,0,1])

sess = tf.Session()
sess.run(tf.initialize_all_variables())
x = tf.constant(3)
zz = tf.concat([tf.stack([x,1],axis=0),[1,1]],axis=0)
# x = tf.constant(3)
# y = tf.constant(1)
# yy = tf.constant([1,1])
# z = tf.stack([x,y],axis=0)
# zz = tf.concat([z,yy],axis=0)
# print(tf.constant([1]).shape)
# print(tf.constant([1,1,1]).shape)
print(sess.run(zz))
exit()
# d = sess.run(c)
# e = sess.run(tf.matmul(c,b))
# f = sess.run(tf.transpose(e,[1,2,0]))
# print(d)
# print(e)
# print(f)
# print(sess.run(tf.add(x_,x)))
g = tf.expand_dims(x_,0)
# g_ = tf.expand_dims(g,0)
# g_x = tf.concat([g_,x],axis=0)
# g__ = tf.expand_dims(g_x,1)
# print('x_')
# print(sess.run(x_))
# print(x_.shape)
print('g')
print(sess.run(g))
print(g.shape)
# print('g_')
# print(sess.run(g_))
# print(g_.shape)
# print('g_x')
# print(sess.run(g_x))
# print('g__')
# print(sess.run(g__))
i = tf.tile(g,tf.constant([1,2,1]))
print(sess.run(i))
print(i.shape)

# h = tf.expand_dims(i,0)
# print(sess.run(h))
# j = tf.tile(h,tf.constant([2,1,1,1]))
# print(sess.run(j))
# print(sess.run(i))
# print(sess.run(tf.add(i,b)))