import tensorflow as tf

def euclidean_dist(x, z):
  n2 = tf.shape(z)[0]
  x_ = tf.expand_dims(x,len(x.shape))
  # x_ = tf.tile(x_, [1,1, n2]) #optional, will be broadcasted
  z_ = tf.expand_dims(tf.transpose(z),0)
  res = tf.square(x_- z_)
  res = tf.reduce_sum(res, 1)
  return res

import numpy as np
def split_data():
  np.random.seed(521)
  Data = np.linspace(1.0, 10.0, num=100)[:, np.newaxis]
  Target = np.sin(Data) + 0.1*np.power(Data,2) + 0.5*np.random.randn(100,1)
  randIdx = np.arange(100)
  np.random.shuffle(randIdx)
  trainData, trainTarget = Data[randIdx[:80]], Target[randIdx[:80]]
  validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
  testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]
  return trainData, trainTarget, validData, validTarget, testData, testTarget

def data_segmentation(data_path, target_path, task):
  # task = 0 >> select the name ID targets for face recognition task
  # task = 1 >> select the gender ID targets for gender recognition task
  data = np.load(data_path)/255
  data = np.reshape(data, [-1, 32*32])
  target = np.load(target_path)
  np.random.seed(45689)
  rnd_idx = np.arange(np.shape(data)[0])
  np.random.shuffle(rnd_idx)
  trBatch = int(0.8*len(rnd_idx))
  validBatch = int(0.1*len(rnd_idx))
  trainData, validData, testData = data[rnd_idx[1:trBatch],:], \
  data[rnd_idx[trBatch+1:trBatch + validBatch],:],\
  data[rnd_idx[trBatch + validBatch+1:-1],:]
  trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
  target[rnd_idx[trBatch+1:trBatch + validBatch], task],\
  target[rnd_idx[trBatch + validBatch + 1:-1], task]
  return trainData, validData, testData, trainTarget, validTarget, testTarget

class KNNBuilder:
  def __init__(self, k):
    self.k = k
  def setData(self, trainData, trainTarget, validData, validTarget, testData, testTarget):
    self.trainData = trainData
    self.trainTarget = trainTarget
    self.validData = validData
    self.validTarget = validTarget
    self.testData = testData
    self.testTarget = testTarget
  def build(self):
    trainData = tf.placeholder(tf.float32, shape=(None, 1), name="trainData")
    trainTarget = tf.placeholder(tf.float32, shape=(None, 1), name="trainTarget")
    
    X = tf.placeholder(tf.float32, shape=(None,1), name="X")
    distances = -euclidean_dist(X,trainData)
    k_neighbors, k_indices = tf.nn.top_k(distances, k=self.k, name="k_neighbors") # size is n*k

    k_indices = tf.expand_dims(k_indices, 2)
    predictions = tf.gather_nd(trainTarget,k_indices)
    print predictions.shape
    predictions = tf.reduce_mean(predictions, 1)
    return predictions


if __name__ == "__main__":
  x = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
  y = tf.constant([[10,11,12],[14,15,16]])
  sess = tf.Session()
  print(sess.run(euclidean_dist(x,y)))
  
  trainData, trainTarget, validData, validTarget, testData, testTarget = split_data()
  data = [trainData, trainTarget, validData, validTarget, testData, testTarget]
  knnBuilder = KNNBuilder(10)
  knnBuilder.setData(*data)
  knn = knnBuilder.build()
  print sess.run(knn, feed_dict={'X:0':validData, 'trainData:0':trainData, 'trainTarget:0':trainTarget })
  print validTarget
  

