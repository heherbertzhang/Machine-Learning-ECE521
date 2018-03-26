import tensorflow as tf
import numpy as np
#load data

def load_data1():
    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData,trainTarget,validData, validTarget,testData, testTarget


def fully_connected_layer(input,num_hidden_unit,name="fully_connected"):
    # initializer
    weight_init = tf.contrib.layers.xavier_initializer(uniform=True)
    shape=tf.shape(input)
    with tf.variable_scope(name):
        # weight bias #name ?
        W = tf.get_variable("W", shape=[shape[-1], num_hidden_unit], initializer=weight_init, dtype=tf.float32)
        b = tf.get_variable("b", shape=[1,num_hidden_unit], initializer=weight_init, dtype=tf.float32)  # broadcast
        tf.add_to_collection("weights",W)
    y_=tf.add(tf.matmul(input, W), b)
    return y_


def Q1_1(saver_path):

    saver = tf.train.Saver()
    if saver_path==None:
        pass


    class One_hidden_net:
        def build(self, num_of_features=1000, learnning_rate=0.005, weight_decay_scale=3*np.exp(-4)):
            #         with self.graph.as_default():

            # Input data
            self.X = tf.placeholder(
                dtype=tf.float32,
                shape=[None, num_of_features])
            self.Y = tf.placeholder(
                dtype=tf.float32,
                shape=[None, 1])
            X = self.X
            Y = self.Y


            # hidden layer
            self.y_ = tf.nn.relu(fully_connected_layer(X,1000))# apply relu
            y_ = self.y_
            # loss
            weights=tf.get_collection("weights") #tf.GraphKeys.TRAINABLE_VARIABLES or tf.GraphKeys.GLOBAL_VARIABLES
            weight_decay = (weight_decay_scale / 2.0) * tf.reduce_sum(tf.norm(weights)** 2)
            loss = tf.nn.l2_loss(y_ - self.Y) / tf.cast(tf.shape(X)[0], tf.float32) + weight_decay
            self.loss = tf.squeeze(loss)
            self.predict = tf.cast(y_ > 0.5, tf.float32)
            self.optimizer = tf.train.GradientDescentOptimizer(learnning_rate)
            self.train_op = self.optimizer.minimize(loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.Y), tf.float32))

            # self.input_batch=None
            # self.label_batch=None