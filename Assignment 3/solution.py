import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

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
    with tf.variable_scope(name) as scope:
        try:
            W = tf.get_variable("W", shape=[shape[-1], num_hidden_unit], initializer=weight_init, dtype=tf.float32)
            b = tf.get_variable("b", shape=[1,num_hidden_unit], initializer=bias_init, dtype=tf.float32)  # broadcast
            tf.add_to_collection("weights",W)
        except Exception as e:
            scope.reuse_variables()
            W = tf.get_variable("W")#, shape=[shape[-1], num_hidden_unit], initializer=weight_init, dtype=tf.float32)
            b = tf.get_variable("b")#, shape=[1, num_hidden_unit], initializer=bias_init, dtype=tf.float32)  # broadcast

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
        self.predict = tf.cast(tf.argmax(tf.nn.softmax(y_), axis=1),tf.int32)

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

    def build_and_train(self, trainData, trainTarget,validData, validTarget,testData, \
                        testTarget, learnning_rate, weight_decay_scale=3*np.exp(-4), \
                        batch_size=500, steps=20000,
                        sess=None, optimizer=None):
        self.build(trainData.shape[1], learnning_rate, weight_decay_scale)
        return self.train(trainData, trainTarget,validData, validTarget,testData, \
                        testTarget, learnning_rate, weight_decay_scale=weight_decay_scale, \
                        batch_size=batch_size, steps=steps,
                        sess=sess, optimizer=optimizer)

    def train(self, trainData, trainTarget,validData, validTarget,testData, testTarget, learnning_rate, weight_decay_scale=3*np.exp(-4), batch_size=500, steps=20000,
              sess=None, optimizer=None):
        sess = sess or tf.get_default_session()
        epoch_size = trainData.shape[0]

        summary = {'train_loss': [], 'valid_loss': [], 'train_accuracy': [], 'valid_accuracy': [],'test_loss': [], 'test_accuracy': []}

        self.add_training_data(trainData, trainTarget, batch_size)

        if optimizer:
            self.train_op = optimizer.minimize(self.loss)
        self.init()
        saver = tf.train.Saver()
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
            if step%(steps//4) == 0 or step == steps-1:
                saver.save(sess, "./checkpoint/"+self.__class__.__name__+"_model", global_step=step)
                picklefile = open("./checkpoint/"+self.__class__.__name__+str(step)+"_summary", 'wb')
                pickle.dump(summary, picklefile)
                picklefile.close()
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


class simple_net(Neural_Network):
    def __init__(self, num_classes):
        Neural_Network.__init__(self)
        self.num_classes = num_classes

    def build(self, num_of_features=1000, learning_rate=0.005, weight_decay_scale=0, hidden_size=[1000]):
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
        hiddenl = tf.nn.relu(fully_connected_layer(X, hidden_size[0], "hidden_start"))  # apply relu
        for i,size in enumerate(hidden_size[1:]):
            hiddenl = tf.nn.relu(fully_connected_layer(hiddenl, size, "hidden_" + str(i+1)))  # apply relu

        self.y_ = fully_connected_layer(hiddenl, self.num_classes,"output")
        y_ = self.y_
        # loss
        weights = tf.get_collection("weights")  # tf.GraphKeys.TRAINABLE_VARIABLES or tf.GraphKeys.GLOBAL_VARIABLES
        weight_decay=0
        for weight in weights:
            weight_decay += (weight_decay_scale / 2.0) * tf.reduce_sum(tf.norm(weight) ** 2)

        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y))
        loss_train = cross_entropy_loss + weight_decay
        self.loss = tf.squeeze(cross_entropy_loss)
        self.predict = tf.squeeze(tf.cast(tf.argmax(tf.nn.softmax(y_), axis=1),tf.int32))


        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.Y), tf.float32))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(loss_train)
trainData, trainTarget, validData, validTarget, testData, testTarget=load_data1()
tf.set_random_seed(3475)
def Q1_1(checkpoint_dir="./checkpoint"):
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

    sess = tf.Session()
    # with sess.as_default():
    #     net.build(trainData.shape[1], 0.005, 0.001)
    #     saver = tf.train.Saver()
    #     saver.restore(sess, os.path.join("./checkpoint", 'simple_net_model-750'))
    #
    #     print(net.get_accuracy(testData, testTarget, sess))
    # return


    with sess.as_default():
        summary=net.build_and_train(trainData, trainTarget,validData, validTarget, testData, \
                                    testTarget, 0.001, weight_decay_scale=0.000001,\
                                    batch_size=1500, steps=300)
    # saver.save(sess,os.path.join(checkpoint_dir, 'Q1.1_full'))
    sess.close()

    loss_data = [summary['train_loss'],summary['valid_loss'],summary['test_loss']]
    # broadcast ones
    error_data = [np.ones([1])-summary['train_accuracy'],np.ones([1])-summary['valid_accuracy'],np.ones([1])-summary['test_accuracy']]

    plot(loss_data,['train_loss','valid_loss','test_loss'],"num of epochs","loss","loss vs. epochs")
    plot(error_data, ['train', 'valid', 'test'], "num of epochs", "classification error", "classification error vs. epochs")

def Q1_2_1():
    hiddens = [100, 500, 1000]
    loss = []
    classification_error = []
    for hidden in hiddens:
        net = simple_net(10)
        tf.reset_default_graph()
        sess = tf.Session()

        with sess.as_default():
            net.build(trainData.shape[1], learning_rate=0.001, weight_decay_scale=0.000001, hidden_size=[hidden])
            summary = net.train(trainData, trainTarget, validData, validTarget, testData, \
                                          testTarget, 0.001, weight_decay_scale=0.000001, \
                                          batch_size=1500, steps=300)
            loss.append(net.get_loss(validData, validTarget))
            classification_error.append(1-net.get_accuracy(testData, testTarget))
        sess.close()
    print(loss,classification_error)
    #[0.27146724, 0.30881488, 0.3118587] [0.08700442314147949, 0.08039647340774536, 0.07892805337905884] # batch 500 1000 iter
    #[0.41982716, 0.4271598, 0.38878557] [0.08406752347946167, 0.08039647340774536, 0.07378852367401123] # batch 1500 3000 iter
    #[0.29187372, 0.25225523, 0.24494112] [0.09471362829208374, 0.08259910345077515, 0.07782673835754395] # batch 1500 300 iter 0.000001 weight decay
    #[0.284581, 0.24820998, 0.24310654] [0.0917767882347107, 0.08406752347946167, 0.07856094837188721] #fix tf random seed
def Q1_2_2():
    net = simple_net(10)
    sess = tf.Session()
    with sess.as_default():
        net.build(trainData.shape[1], learning_rate=0.001, weight_decay_scale=0.000001, hidden_size=[500, 500])
        summary = net.train(trainData, trainTarget, validData, validTarget, testData, \
                            testTarget, 0.001, weight_decay_scale=0.000001, \
                            batch_size=1500, steps=300)
        print(net.get_loss(validData, validTarget))
        print(1 - net.get_accuracy(validData, validTarget))
        print(net.get_loss(testData, testTarget))
        print(1 - net.get_accuracy(testData, testTarget))
    sess.close()
    loss_data = [summary['train_loss'], summary['valid_loss']]
    plot(loss_data, ['train_error', 'valid_error'], "num of epochs", "error", "error vs. epochs")
    # 0.2715209
    # 0.06699997186660767
    # 0.3475434
    # 0.0734214186668396
    # 0.2618259
    # 0.06599998474121094
    # 0.3647777
    # 0.07892805337905884 more accu

class Dropout_Net(Neural_Network):
    def __init__(self, num_classes):
        Neural_Network.__init__(self)
        self.num_classes = num_classes
    def build(self, num_of_features=1000, learning_rate=0.005, weight_decay_scale=0, hidden_size=[1000]):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, num_of_features])
        self.Y = tf.placeholder(dtype=tf.int32, shape=[None])
        Y = tf.one_hot(self.Y, self.num_classes)
        X = self.X

        # hidden layer
        hiddenl = tf.nn.relu(fully_connected_layer(X, hidden_size[0], "hidden_start"))  # apply relu
        hidden_dol = tf.nn.dropout(hiddenl, 0.5)
        for i,size in enumerate(hidden_size[1:]):
            hidden_dol = tf.nn.dropout( \
                tf.nn.relu(fully_connected_layer(hidden_dol, size, "hidden_" + str(i + 1))), 0.5)
            hiddenl = tf.nn.relu(fully_connected_layer(hiddenl, size, "hidden_" + str(i+1)))  # apply relu


        #dropout1 = tf.nn.dropout(hidden, 0.5)
        y_ = (fully_connected_layer(hidden_dol, self.num_classes, "output"))
        y_no_dropout = fully_connected_layer(hiddenl, self.num_classes, "output")
        weights = tf.get_collection("weights")
        weight_decay=0
        for weight in weights:
            weight_decay += (weight_decay_scale / 2.0) * tf.reduce_sum(tf.norm(weight) ** 2)
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=Y))
        self.loss_train = tf.squeeze(cross_entropy_loss+weight_decay)
        self.loss = tf.squeeze(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_no_dropout, labels=Y)))
        self.predict = tf.squeeze(tf.cast(tf.argmax(tf.nn.softmax(y_no_dropout), axis=1),tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.Y), tf.float32))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_train)

def Q1_3_1():
    net = Dropout_Net(10)
    sess = tf.Session()
    with sess.as_default():
        net.build(trainData.shape[1], learning_rate=0.001, weight_decay_scale=0.0000, hidden_size=[1000])
        summary=net.train(trainData, trainTarget,validData, validTarget, testData, testTarget, \
                          0.001, weight_decay_scale=0, batch_size=1500, steps=500)
        print(net.get_accuracy(testData,testTarget))
    sess.close()
    error_data = [np.ones([1])-summary['train_accuracy'],np.ones([1])-summary['valid_accuracy']]
    loss_data = [summary['train_loss'], summary['valid_loss']]
    plot(loss_data, ['train_error', 'valid_error'], "num of epochs", "error", "error vs. epochs")

def Q1_3_2():

    stoppings=["75","150","225","299"]
    for net_name in ['simple_net_model', 'Dropout_Net_model']:
        sess = tf.Session()
        if net_name == 'simple_net_model':
            net = simple_net(10)
        else:
            net = Dropout_Net(10)
        net.build(trainData.shape[1], learning_rate=0.001, weight_decay_scale=0)
        with sess.as_default():
            net.train(trainData, trainTarget, validData, validTarget, testData, testTarget, \
                  0.001, weight_decay_scale=0.000001, batch_size=1500, steps=300)
        saver = tf.train.Saver()
        sample= np.random.random_sample()
        for progress in stoppings:
            print("./checkpoint/" + net_name + "-" + progress)
            saver.restore(sess,os.path.join("./checkpoint", net_name+"-"+progress))

            with tf.variable_scope("hidden_start",reuse=True):
                weight=sess.run(tf.get_variable("W"))
                for i in range(weight.shape[1]):
                    w=weight[:,i]
                    w=np.reshape(w,[28,28])
                    plt.imshow(w)
                    plt.subplot(25,40,i+1)
            plt.show()

def Q1_4_1():
    np.random.seed(3475)
    tf.set_random_seed(3475)
    result=[]
    for i in range(5):
        tf.reset_default_graph()
        todropout = np.random.randint(0,2)

        learning_rate = np.exp(np.random.rand() * 3 - 7.5)
        layers = np.random.randint(1, 6)
        hiddens = []
        for layer in range(layers):
            hiddens.append(np.random.randint(100, 501))
        weight_decay = np.exp(np.random.rand() * 3 - 9)

        sess = tf.Session()
        if todropout:
            net = Dropout_Net(10)
            net.build(trainData.shape[1], learning_rate=learning_rate, \
                      weight_decay_scale=weight_decay, hidden_size=hiddens)
        else:
            net = simple_net(10)
            net.build(trainData.shape[1], learning_rate=learning_rate, \
                      weight_decay_scale=weight_decay, hidden_size=hiddens)


        with sess.as_default():
            summary=net.train(trainData, trainTarget,validData, validTarget, testData, testTarget, \
                             learning_rate, weight_decay_scale=weight_decay, batch_size=1500, steps=300)
            valid_class_error=(1-net.get_accuracy(validData, validTarget))
            test_class_error=(1-net.get_accuracy(testData, testTarget))
        sess.close()
        result.append((todropout, layers, hiddens, learning_rate, weight_decay , \
                       valid_class_error, test_class_error))

    print(result)


    # [(1, 3, [410, 376, 284], 0.0005951274124940402, 0.001539334562242609, 0.8449999988079071, 0.8432452231645584),
    #  (1, 4, [371, 152, 187, 129], 0.008200252705786006, 0.00019158781933987086, 0.8929999992251396, 0.9041850194334984),
    #  (1, 1, [420], 0.0010812407954069393, 0.00018822095645644772, 0.9280000030994415, 0.9313509538769722),
    #  (0, 1, [272], 0.004250395461417586, 0.000365927163481687, 0.06999999284744263, 0.08406752347946167),
    #  (1, 5, [423, 282, 194, 348, 260], 0.0026714114821269337, 0.001063733012067351, 0.8929999992251396,0.9041850194334984)]


def Q1_4_2():
    learning_rate = 0
    hiddens = []
    weight_decay =0
    net = Dropout_Net(10)

    sess = tf.Session()
    net.build(trainData.shape[1], learning_rate= learning_rate, \
              weight_decay_scale=weight_decay, hidden_size=hiddens)
    with sess.as_default():
        summary=net.train(trainData, trainTarget,validData, validTarget, testData, testTarget, \
                          0.001, weight_decay_scale=0, batch_size=500, steps=1050)
        print(1-net.get_accuracy(validData, validTarget))
        print(1-net.get_accuracy(testData, testTarget))


if __name__ == "__main__":
    input_ = input("what question to run?")
    {'1.1':Q1_1,'1.2.1':Q1_2_1,'1.2.2':Q1_2_2,'1.3.1':Q1_3_1,'1.3.2':Q1_3_2,'1.4.1':Q1_4_1}[input_]()
