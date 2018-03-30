import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


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

        trainData = trainData.reshape([trainData.shape[0], -1])
        validData = validData.reshape([validData.shape[0], -1])
        testData = testData.reshape([testData.shape[0], -1])
    return trainData,trainTarget,validData, validTarget,testData, testTarget


def fully_connected_layer(input,num_hidden_unit,name="fully_connected"):
    # initializer
    weight_init = tf.contrib.layers.xavier_initializer(uniform=True)
    bias_init = tf.constant_initializer(0.0)

    shape=input.get_shape()
    with tf.variable_scope(name):
        W = tf.get_variable("W", shape=[shape[-1], num_hidden_unit], initializer=weight_init, dtype=tf.float32)
        b = tf.get_variable("b", shape=[1,num_hidden_unit], initializer=bias_init, dtype=tf.float32)  # broadcast
        tf.add_to_collection("weights",W)
    y_=tf.add(tf.matmul(input, W), b)
    return y_

class Neural_Network:
    def build(self, num_of_features, learnning_rate=0.005, weight_decay_scale=3*np.exp(-4)):
        # Input data
        self.X = tf.placeholder(
            dtype=tf.float32,
            shape=[None, num_of_features])
        self.Y = tf.placeholder(
            dtype=tf.float32,
            shape=[None])
        X = self.X
        Y = tf.one_hot(self.Y, self.num_classes)

        # hidden layer
        hidden1 = tf.nn.relu(fully_connected_layer(X, 1000,"hidden1"))  # apply relu
        self.y_= fully_connected_layer(hidden1,self.num_classes,"output")
        y_ = self.y_
        # loss
        weights = tf.get_collection("weights")  # tf.GraphKeys.TRAINABLE_VARIABLES or tf.GraphKeys.GLOBAL_VARIABLES
        weight_decay = (weight_decay_scale / 2.0) * tf.reduce_sum(tf.norm(weights) ** 2)

        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y))
        loss = cross_entropy_loss + weight_decay
        self.loss = tf.squeeze(loss)
        self.predict = tf.argmax(tf.nn.softmax(y_), axis=1,output_type=tf.int32)

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.Y), tf.float32))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learnning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def add_training_data(self, inputs, labels, batch_size):

        self.epoch_size = labels.shape[0]
        self.trainData = inputs
        self.trainTarget = labels
        self.batch_size = batch_size

        def batch_generator(self):
            while True:
                end = 0
                for batch_i in range(self.epoch_size // self.batch_size):
                    start = batch_i * self.batch_size
                    end = start + self.batch_size
                    batch_xs = self.trainData[start:end, :]
                    batch_ys = self.trainTarget[start:end]
                    yield batch_xs, batch_ys

                remaining = self.epoch_size - end
                if remaining:
                    batch_xs = np.concatenate(
                        (self.trainData[end:, :], self.trainData[:self.batch_size - remaining, :]), axis=0)
                    batch_ys = np.concatenate((self.trainTarget[end:], self.trainTarget[:self.batch_size - remaining]),
                                              axis=0)
                    yield batch_xs, batch_ys

                ##shuffle
                randIndx = np.arange(len(self.trainData))
                np.random.shuffle(randIndx)
                self.trainData, self.trainTarget = self.trainData[randIndx], self.trainTarget[randIndx]

        self.generator = batch_generator(self)

        # method 2 use tensorflow queue runner, slow but memory efficient
        # self.trainning_data = tf.constant(inputs, dtype=tf.float32)
        # self.trainning_label = tf.constant(labels, dtype=tf.float32)
        # # create a runner queue, shuffle every epoch
        # x, y = tf.train.slice_input_producer([self.trainning_data, self.trainning_label], num_epochs=None, \
        #                                      capacity= batch_size*10,shuffle=True)
        # # build mini-batch
        # self.next_batch_op = tf.train.batch([x, y], batch_size=batch_size,capacity=batch_size*10,num_threads=6)

    def get_next_batch(self):
        return next(self.generator)

    def init(self, sess=None):
        sess = (sess or tf.get_default_session())
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(self.init_op)
        # self.coord = tf.train.Coordinator()
        # self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)
        return sess

    def get_loss(self, inputs, labels, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.loss, {self.X: inputs, self.Y: labels})

    def update(self, sess=None):
        sess = sess or tf.get_default_session()
        # input_batch, label_batch = sess.run(self.next_batch_op)
        input_batch, label_batch = self.get_next_batch()
        _, loss = sess.run([self.train_op, self.loss], {self.X: input_batch, self.Y: label_batch})
        return loss

    def get_prediction(self, data, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.predict, {self.X: data})

    def get_accuracy(self, inputs, labels, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.accuracy, {self.X: inputs, self.Y: labels})
    def get_status(self,trainData, trainTarget,validData, validTarget,testData, testTarget):
        summary = {}

        summary['train_loss']=(self.get_loss(trainData, trainTarget,))
        summary['valid_loss']=(self.get_loss(validData, validTarget))
        summary['test_loss']=(self.get_loss(testData, testTarget))
        summary['train_accuracy']=(self.get_accuracy(trainData, trainTarget))
        summary['valid_accuracy']=(self.get_accuracy(validData, validTarget))
        summary['test_accuracy']=(self.get_accuracy(testData, testTarget))

        return summary

    def train(self, trainData, trainTarget,validData, validTarget,testData, testTarget, learnning_rate, weight_decay_scale=3*np.exp(-4), batch_size=500, steps=20000,
              sess=None, optimizer=None):
        sess = sess or tf.get_default_session()
        epoch_size = trainData.shape[0]

        summary = {'train_loss': [], 'valid_loss': [], 'train_accuracy': [], 'valid_accuracy': [],'test_loss': [], 'test_accuracy': []}

        self.add_training_data(trainData, trainTarget, batch_size)
        self.build(trainData.shape[1], learnning_rate, weight_decay_scale)
        if optimizer:
            self.train_op = optimizer.minimize(self.loss)
        self.init()

        for step in range(steps):
            self.update(sess)
            if ( ((step * batch_size) % epoch_size) < batch_size):
                loss_val=self.get_loss(trainData, trainTarget)
                print(str(step)+": "+str(loss_val))
                summary['train_loss'].append(loss_val)
                summary['valid_loss'].append(self.get_loss(validData, validTarget))
                summary['test_loss'].append(self.get_loss(testData, testTarget))
                summary['train_accuracy'].append(self.get_accuracy(trainData, trainTarget))
                summary['valid_accuracy'].append(self.get_accuracy(validData, validTarget))
                summary['test_accuracy'].append(self.get_accuracy(testData, testTarget))
        return summary

def plot(summarys,labels,xlabel,ylabel,title):
    plt.figure()
    for i, summary in enumerate(summarys):
        plt.plot(summary, label=labels[i])
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
    plt.legend()
    plt.show()

def Q1_1(checkpoint_dir="./checkpoint"):




    class simple_net(Neural_Network):
        def __init__(self, num_classes):
            Neural_Network.__init__(self)
            self.num_classes = num_classes

        def build(self, num_of_features=1000, learnning_rate=0.005, weight_decay_scale=0):
            # Input data
            self.X = tf.placeholder(
                dtype=tf.float32,
                shape=[None, num_of_features])
            self.Y = tf.placeholder(
                dtype=tf.int32,
                shape=[None])
            X = self.X
            Y = tf.one_hot(self.Y, self.num_classes)

            # hidden layer
            hidden1 = tf.nn.relu(fully_connected_layer(X, 1000,"hidden1"))  # apply relu
            self.y_ = fully_connected_layer(hidden1, self.num_classes,"output")
            y_ = self.y_
            # loss
            weights = tf.get_collection("weights")  # tf.GraphKeys.TRAINABLE_VARIABLES or tf.GraphKeys.GLOBAL_VARIABLES
            weight_decay=0
            for weight in weights:
                weight_decay += (weight_decay_scale / 2.0) * tf.reduce_sum(tf.norm(weight) ** 2)

            cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=Y))
            loss = cross_entropy_loss + weight_decay
            self.loss = tf.squeeze(loss)
            self.predict = tf.squeeze(tf.argmax(tf.nn.softmax(y_), axis=1,output_type=tf.int32))


            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.Y), tf.float32))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learnning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

    trainData, trainTarget, validData, validTarget, testData, testTarget=load_data1()

    net = simple_net(10)

    # LEARNING_RATES = (0.005, 0.001, 0.0001)
    # summarys=[]
    # #find best learning rate
    # for LEARNING_RATE in LEARNING_RATES:
    #     tf.reset_default_graph()#reset must after sess close
    #     sess = tf.Session()
    #
    #     with sess.as_default():
    #         summary=net.train(trainData, trainTarget,validData, validTarget, testData, testTarget, LEARNING_RATE, batch_size=500, steps=100)
    #         summarys.append(summary['train_loss'])
    #     sess.close()
    #
    #
    # plot(summarys,["0.005","0.001","0.0001"],"a","a","a")

    #best learning rate 0.005


    # if checkpoint_dir!=None:
    #     saver = tf.train.Saver()
    sess = tf.Session()
    # ckpt = tf.train.get_checkpoint_state(checkpoint_dir, 'Q1.1_full')
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(sess, os.path.join(checkpoint_dir, 'Q1.1_full'))
    # else:
    with sess.as_default():
        summary=net.train(trainData, trainTarget,validData, validTarget, testData, testTarget, 0.005, weight_decay_scale=0.001,batch_size=500, steps=1000)
    # saver.save(sess,os.path.join(checkpoint_dir, 'Q1.1_full'))
    sess.close()

    loss_data = [summary['train_loss'],summary['valid_loss'],summary['test_loss']]
    error_data = [np.ones([1])-summary['train_accuracy'],np.ones([1])-summary['valid_accuracy'],np.ones([1])-summary['test_accuracy']]

    plot(loss_data,['train_loss','valid_loss','test_loss'],"num of epochs","loss","loss vs. epochs")
    plot(error_data, ['train', 'valid', 'test'], "num of epochs", "classification error", "classification error vs. epochs")

if __name__ == "__main__":
    input_ = input("what question to run?")
    {'1.1':Q1_1}[input_]()
