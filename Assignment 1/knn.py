import tensorflow as tf

def euclidean_dist(x, z):
  n1, d = x.shape
  n2 = z.shape[0]
  x_ = tf.expand_dims(x,len(x.shape))
  x_ = tf.tile(x_, [1]*len(x.shape)+[int(n2)])
  z_ = tf.expand_dims(tf.transpose(z),0)
  res = tf.square(x_- z_)
  res = tf.reduce_sum(res, 1)
  return res


if __name__ == "__main__":
  x = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
  y = tf.constant([[10,11,12],[14,15,16]])
  sess = tf.Session()
  print(sess.run(euclidean_dist(x,y)))
