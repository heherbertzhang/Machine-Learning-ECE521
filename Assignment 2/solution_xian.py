import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#traindata1
import time
trainData, trainTarget, validData, validTarget, testData, testTarget=[None]*6
def get_data1():
    global trainData, trainTarget, validData, validTarget, testData, testTarget
    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    #reshape the training data
    trainData = trainData.reshape([trainData.shape[0], -1])
    validData = validData.reshape([validData.shape[0], -1])
    testData=testData.reshape([testData.shape[0], -1])



class Linear_Regression_Net:
    def build(self, num_of_features, learnning_rate, weight_decay_scale):

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

        # initializer
        weight_init = tf.constant_initializer(0.0001)#tf.random_normal_initializer(0,0.1) #tf.contrib.layers.xavier_initializer(uniform=True)  #
        const_init = tf.constant_initializer(0.01)

        # weight bias
        W1 = tf.get_variable("W1", shape=[num_of_features, 1], initializer=weight_init, dtype=tf.float32)
        b1 = tf.get_variable("b1", shape=[1, 1], initializer=const_init, dtype=tf.float32)  # broadcast
        # hidden layer
        self.y_ = tf.add(tf.matmul(X, W1), b1)
        y_=self.y_
        # loss
        weight_decay = (weight_decay_scale / 2.0) * tf.norm(W1)**2
        loss = tf.nn.l2_loss(y_ - self.Y) / tf.cast(tf.shape(X)[0], tf.float32) + weight_decay
        self.loss = tf.squeeze(loss)
        self.predict = tf.cast(y_ > 0.5, tf.float32)
        self.optimizer = tf.train.GradientDescentOptimizer(learnning_rate)
        self.train_op = self.optimizer.minimize(loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict, self.Y), tf.float32))

        # self.input_batch=None
        # self.label_batch=None


    def add_training_data(self, inputs, labels, batch_size):

        self.epoch_size = labels.shape[0]
        self.trainData=inputs
        self.trainTarget=labels
        self.batch_size=batch_size

        def batch_generator(self):
            while True:
                end=0
                for batch_i in range(self.epoch_size // self.batch_size):
                    start = batch_i * self.batch_size
                    end = start + self.batch_size
                    batch_xs = self.trainData[start:end, :]
                    batch_ys = self.trainTarget[start:end]
                    yield batch_xs, batch_ys

                remaining = self.epoch_size - end
                if remaining:
                    batch_xs = np.concatenate((self.trainData[end:,:], self.trainData[:self.batch_size - remaining,:]), axis=0)
                    batch_ys = np.concatenate((self.trainTarget[end:] , self.trainTarget[:self.batch_size - remaining]),axis=0)
                    yield batch_xs, batch_ys

                ##shuffle
                randIndx = np.arange(len(self.trainData))
                np.random.shuffle(randIndx)
                self.trainData, self.trainTarget = self.trainData[randIndx], self.trainTarget[randIndx]

        self.generator=batch_generator(self)

        #method 2 use tensorflow queue runner, slow but memory efficient
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
        input_batch,label_batch=self.get_next_batch()
        _, loss = sess.run([self.train_op, self.loss], {self.X: input_batch, self.Y: label_batch})
        return loss

    def get_prediction(self, data, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.predict, {self.X: data})

    def get_accuracy(self, inputs, labels, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.accuracy, {self.X: inputs, self.Y: labels})

    def train(self, trainData_rs, trainTarget, learnning_rate, weight_decay_scale, batch_size=500, steps=20000,
              sess=None, optimizer=None, accuList=None):
        sess = sess or tf.get_default_session()
        epoch_size = trainData_rs.shape[0]

        summary = []

        self.add_training_data(trainData_rs, trainTarget, batch_size)
        self.build(trainData_rs.shape[1], learnning_rate, weight_decay_scale)
        if optimizer:
            self.train_op = optimizer.minimize(self.loss)
        self.init()
        loss_val = 0
        for step in range(steps):
            loss_val = self.update(sess)
            if ((step * batch_size) % epoch_size == 0):
                loss_val=self.get_loss(self.trainData,self.trainTarget)
                summary.append(loss_val)
                if accuList != None:
                    accuList.append(self.get_accuracy(self.trainData, self.trainTarget))
                print(loss_val)
        # self.coord.request_stop()
        # self.coord.join(self.threads)
        return summary



def Q1_1():
    get_data1()
    # Q1
    BATCH_SIZE = 500
    LEARNING_RATES = (0.005, 0.001, 0.0001)
    weight_decay_scale = 0

    summarys = []

    net = Linear_Regression_Net()

    for LEARNING_RATE in LEARNING_RATES:
        tf.reset_default_graph()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        with sess.as_default():
            summary = net.train(trainData, trainTarget, LEARNING_RATE, weight_decay_scale, BATCH_SIZE, 20000)
            summarys.append(summary)
        sess.close()



    plt.figure()
    for i, summary in enumerate(summarys):
        plt.plot(summary, label='LearningRate= ' + str(LEARNING_RATES[i]))
        plt.ylabel("training loss")
        plt.xlabel("number of epochs")
        plt.title("Q1.1")
    plt.legend()
    plt.show()

def Q1_2():
    # Q2
    get_data1()
    import time
    B = [500, 1500, 3500]
    learning_rate = 0.005
    weight_decay_scale = 0
    losses = []
    times = []
    summarys = []

    net = Linear_Regression_Net()

    for batch_size in B:
        tf.reset_default_graph()
        sess = tf.Session()
        start = time.time()
        with sess.as_default():
            summary = net.train(trainData, trainTarget, learning_rate, weight_decay_scale, batch_size, steps=20000)
            summarys.append(summary)
            losses.append(net.get_loss(trainData,trainTarget))

        sess.close()
        end = time.time()
        times.append((end - start))


        # Report the final training MSE for each mini-batch value
        # What is the best mini-batch size in terms of training time?
        # Comment on your observation.
    for t, loss in zip(times, losses):
        print(str(t) + "," + str(loss))

    plt.figure()
    for i, summary in enumerate(summarys):
        plt.plot(summary, label='batch_size= ' + str(B[i]))
        plt.ylabel("training loss")
        plt.xlabel("number of epochs")
        plt.title("Q1.2")
    plt.legend()
    plt.show()
    #result
    # 58.574955224990845, 0.0126219
    # 169.57756733894348, 0.0127338
    # 385.44294452667236, 0.0128752

def Q1_3():
    get_data1()
    # Q3
    weight_decay_ratios = [0, 0.001, 0.1, 1]
    batch_size = 500
    learning_rate = 0.005
    accuracys = []
    net = Linear_Regression_Net()

    for weight_decay_scale in weight_decay_ratios:
        tf.reset_default_graph()
        sess = tf.Session()
        with sess.as_default():
            summary = net.train(trainData, trainTarget, learning_rate, weight_decay_scale, batch_size, steps=20000)
            validation_accuracy = net.get_accuracy(validData, validTarget)
            test_accuracy = net.get_accuracy(testData, testTarget)
            train_accuracy = net.get_accuracy(trainData, trainTarget)
            accuracys.append((validation_accuracy, test_accuracy,train_accuracy))
    for accuracy in accuracys:
        print(accuracy)

    # (0.99000001, 0.96551722)
    # (0.98000002, 0.96551722)
    # (0.98000002, 0.97241378)
    # (0.98000002, 0.96551722)

        # (0.94, 0.95862067) with random random_normal_initializer
        # (0.97000003, 0.95862067)
        # (0.98000002, 0.97241378)
        # (0.98000002, 0.96551722)

class Normal_Equation_Method:
    def optimal_weight(self, inputs, labels, weight_decay):
        X0 = tf.ones([inputs.shape[0], 1],dtype=tf.float64)
        X = tf.concat(axis=1, values=[tf.constant(inputs, tf.float64), X0])
        Y = tf.constant(labels, tf.float64)
        W_ = tf.matmul(tf.transpose(X), X) + weight_decay * tf.constant(np.identity(X.get_shape()[1]), tf.float64)
        W_ = tf.matmul(tf.matrix_inverse(W_), tf.transpose(X))
        W_ = tf.matmul(W_, Y)
        return W_

    def build(self, inputs, labels, weight_decay):
        self.X_input = tf.placeholder(tf.float64, [None, inputs.shape[1]])
        X0 = tf.ones([tf.shape(self.X_input)[0], 1],dtype=tf.float64)
        self.X = tf.concat(axis=1, values=[self.X_input, X0])
        self.Y = tf.placeholder(tf.float64, [None, 1])
        self.W_ = self.optimal_weight(inputs, labels, weight_decay)
        self.Y_ = tf.matmul(self.X, self.W_)
        self.predict = tf.cast(self.Y_ > 0.5, tf.float64)
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.predict, self.Y), tf.float64)) / tf.cast(tf.shape(self.Y)[0], tf.float64)
        self.loss = tf.nn.l2_loss(self.Y-self.Y_) / tf.cast(tf.shape(self.X)[0], tf.float64)

    def get_accuracy(self, inputs, labels, sess=None):
        sess = sess or tf.get_default_session()
        accu = sess.run(self.accuracy, {self.X_input: inputs, self.Y: labels})
        return accu
    def get_loss(self, inputs, labels, sess=None):
        sess = sess or tf.get_default_session()
        loss = sess.run(self.loss, {self.X_input: inputs, self.Y: labels})
        return loss

def Q1_4():
    net = Linear_Regression_Net()
    tf.reset_default_graph()
    sess = tf.Session()
    with sess.as_default():
        start=time.time()
        summary = net.train(trainData, trainTarget, 0.005, 0, 500, steps=20000)
        end = time.time()
        print(net.get_accuracy(validData, validTarget))
        print(net.get_accuracy(testData, testTarget))
        print(net.get_accuracy(trainData, trainTarget))
        print(net.get_loss(trainData, trainTarget))
        print(end-start)


    normal_eqn = Normal_Equation_Method()
    tf.reset_default_graph()
    sess = tf.Session()
    with sess.as_default():
        start = time.time()
        normal_eqn.build(trainData, trainTarget, 0)
        end = time.time()
        print(str(normal_eqn.get_accuracy(validData, validTarget)))
        print(str(normal_eqn.get_accuracy(testData, testTarget)))
        print(str(normal_eqn.get_accuracy(trainData, trainTarget)))
        print(str(normal_eqn.get_loss(trainData, trainTarget)))
        print(end - start)


class Logistic_Net:
    def build(self,num_of_features, learnning_rate, weight_decay_scale):
        # Input data
        self.X = tf.placeholder(
            dtype=tf.float32,
            shape=[None, num_of_features])
        self.Y = tf.placeholder(
            dtype=tf.float32,
            shape=[None, 1])
        X = self.X
        Y = self.Y
        weight_init = tf.contrib.layers.xavier_initializer(uniform=True)
        const_init = tf.constant_initializer(0.01)

        # weight bias
        W1 = tf.get_variable("W1", shape=[num_of_features, 1], initializer=weight_init, dtype=tf.float32)
        b1 = tf.get_variable("b1", shape=[1, 1], initializer=const_init, dtype=tf.float32)  # broadcast
        # hidden layer
        sample_numble = tf.cast(tf.shape(Y)[0],tf.float32)
        logits = tf.add(tf.matmul(X, W1), b1)
        cross_entropy_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y)) / sample_numble
        self.prob=tf.sigmoid(logits)
        self.predict = tf.cast(tf.sigmoid(logits) > 0.5, tf.float32)
        # loss
        weight_decay = weight_decay_scale / 2 * tf.matmul(tf.transpose(W1), (W1))
        loss = cross_entropy_loss + weight_decay
        self.loss = tf.squeeze(loss)
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(Y, self.predict), tf.float32)) / sample_numble

        self.optimizer = tf.train.GradientDescentOptimizer(learnning_rate)
        self.train_op = self.optimizer.minimize(loss)



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


    def get_next_batch(self):
        return next(self.generator)

    def init(self, sess=None):
        sess = (sess or tf.get_default_session())
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(self.init_op)
        return sess

    def get_loss(self, inputs, labels, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.loss, {self.X: inputs, self.Y: labels})

    def update(self, sess=None):
        sess = sess or tf.get_default_session()
        input_batch, label_batch = self.get_next_batch()
        _, loss = sess.run([self.train_op, self.loss], {self.X: input_batch, self.Y: label_batch})
        return loss

    def get_prediction(self, data, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.predict, {self.X: data})

    def get_accuracy(self, inputs, labels, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.accuracy, {self.X: inputs, self.Y: labels})

    def train(self, trainData, trainTarget, learnning_rate, weight_decay_scale, batch_size=500, steps=20000,
              sess=None, optimizer=None):
        sess = sess or tf.get_default_session()
        epoch_size = trainData.shape[0]

        summary = []

        self.add_training_data(trainData, trainTarget, batch_size)
        self.build(trainData.shape[1], learnning_rate, weight_decay_scale)
        if optimizer:
            self.train_op = optimizer.minimize(self.loss)

        self.init()
        loss_val = 0
        summary = {'loss_train':[], 'loss_valid':[], 'accuracy_train':[], 'accuracy_valid':[]}
        for step in range(steps):
            self.update(sess)
            if ((step * batch_size) % epoch_size == 0):
                loss_val_train = self.get_loss(trainData, trainTarget)
                loss_val_valid = self.get_loss(validData, validTarget)
                accu_train = self.get_accuracy(trainData, trainTarget)
                accu_valid = self.get_accuracy(validData, validTarget)
                summary['loss_train'].append(loss_val_train)
                summary['loss_valid'].append(loss_val_valid)
                summary['accuracy_train'].append(accu_train)
                summary['accuracy_valid'].append(accu_valid)
                print(loss_val_train)
                print(accu_train)

        return summary

def Q2_prep():
    get_data1()
    BATCH_SIZE = 500
    LEARNING_RATES = (0.005, 0.001, 0.0001)
    weight_decay_scale = 0.01
    tf.reset_default_graph()

    summarys=[]
    for LEARNING_RATE in LEARNING_RATES:
        tf.reset_default_graph()
        net = Logistic_Net()
        sess=tf.Session()
        with sess.as_default():
            summary = net.train(trainData, trainTarget, LEARNING_RATE, weight_decay_scale, BATCH_SIZE, 5000)
        summarys.append(summary)
    for i,summary in enumerate(summarys):
        plt.plot(summary['loss_train'], label=str(LEARNING_RATES[i])+" loss")
        plt.plot(summary['accuracy_train'], label=str(LEARNING_RATES[i])+" accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.show()


def Q2_1():
    get_data1()
    BATCH_SIZE = 500
    LEARNING_RATE = 0.005
    weight_decay_scale = 0.01
    tf.reset_default_graph()

    net = Logistic_Net()
    sess=tf.Session()
    with sess.as_default():
        summary = net.train(trainData, trainTarget, LEARNING_RATE, weight_decay_scale, BATCH_SIZE, 5000)
        print(net.get_accuracy(testData, testTarget))

    plt.figure(figsize=(10, 4))
    for key, points in summary.items():
        s = np.array(summary)
        plt.plot(points, label=key)

    plt.legend()
    plt.title("Loss & Accuracy vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.show()

def Q2_2():
    get_data1()
    BATCH_SIZE = 500
    LEARNING_RATE = 0.001
    weight_decay_scale = 0.01
    tf.reset_default_graph()

    net = Logistic_Net()
    sess=tf.Session()

    with sess.as_default():
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        summary = net.train(trainData, trainTarget, LEARNING_RATE, weight_decay_scale, BATCH_SIZE, 5000,\
            optimizer = optimizer)

    # without adam optimizer
    tf.reset_default_graph()
    net = Logistic_Net()
    sess=tf.Session()
    with sess.as_default():
        summary2 = net.train(trainData, trainTarget, LEARNING_RATE, weight_decay_scale, BATCH_SIZE, 5000)

    plt.figure(figsize=(10, 4))
    plt.plot(summary['loss_train'], label='trainning_loss_Adam')
    plt.plot(summary2['loss_train'], label='trainning_loss_SGD')

    plt.legend()
    plt.title("Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.show()

def Q2_3():
    get_data1()
    tf.reset_default_graph()
    normal_eqn = Normal_Equation_Method()
    sess = tf.Session()
    with sess.as_default():
        normal_eqn.build(trainData, trainTarget, 0)
        print(str(normal_eqn.get_accuracy(trainData, trainTarget)))
        print(str(normal_eqn.get_accuracy(validData, validTarget)))
        print(str(normal_eqn.get_accuracy(testData, testTarget)))

    # compare logistic
    BATCH_SIZE = 500
    LEARNING_RATE = 0.001
    weight_decay_scale = 0.0
    tf.reset_default_graph()

    net = Logistic_Net()
    sess=tf.Session()


    linear_graph = []
    logistic_graph = []

    with sess.as_default():
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        summary2 = net.train(trainData, trainTarget, LEARNING_RATE, weight_decay_scale, BATCH_SIZE, 500,\
            optimizer = optimizer)
        print(str(net.get_accuracy(trainData, trainTarget)))
        print(str(net.get_accuracy(validData, validTarget)))
        print(str(net.get_accuracy(testData, testTarget)))


        #run prob
        y2 = []
        for input,label in zip(testData,testTarget):

            input=input.reshape(1,-1)
            label=label.reshape(1,-1)
            logistic_prob = sess.run(tf.squeeze(net.prob), {net.X: input.reshape(1,-1), net.Y: label.reshape(1,-1)})
            logistic_loss = net.get_loss(input, label)
            logistic_graph.append(logistic_prob)
            y2.append(logistic_loss)




    plt.figure(figsize=(10, 4))
    plt.plot(summary2['loss_train'], label='trainning_loss_logistic')
    plt.plot(summary2['accuracy_train'], label='trainning_accuracy_logistic')


    tf.reset_default_graph()
    net = Linear_Regression_Net()
    sess = tf.Session()
    y = []
    with sess.as_default():
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        accuList = list()
        summary = net.train(trainData, trainTarget, LEARNING_RATE, weight_decay_scale, BATCH_SIZE, steps=500, \
            optimizer=optimizer, accuList=accuList)


        #run prob
        for input,label in zip(testData,testTarget):
            input = input.reshape(1, -1)
            label = label.reshape(1, -1)
            linear_prob=sess.run(tf.squeeze(net.y_),{net.X:input.reshape(1,-1),net.Y:label.reshape(1,-1)})
            linear_loss=net.get_loss(input,label)
            linear_graph.append(linear_prob)
            y.append(linear_loss)
    plt.plot(summary, label='trainning_loss_linear_reg')
    plt.plot(accuList, label='trainning_accuracy_linear_reg')

    plt.legend()
    plt.title("Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.scatter(linear_graph,y, label="linear")
    plt.scatter(logistic_graph, y2, label="logistic")

    plt.legend()
    plt.xlabel("prob")
    plt.ylabel("loss")
    plt.show()

class Multiclass_Logistic_Net(Logistic_Net):
    def __init__(self, num_classes):
        Logistic_Net.__init__(self)
        self.num_classes = num_classes

    def build(self,num_of_features, learnning_rate, weight_decay_scale):
        num_classes = self.num_classes
        # Input data
        self.X = tf.placeholder(
            dtype=tf.float32,
            shape=[None, num_of_features])
        self.Y = tf.placeholder(
            dtype=tf.int32,
            shape=[None, 1])
        X = tf.concat((tf.ones([tf.shape(self.X)[0],1]), self.X), axis=1)
        Y = tf.one_hot(self.Y, num_classes)
        sess=tf.get_default_session()
        weight_init = tf.contrib.layers.xavier_initializer(uniform=True)
        const_init = tf.constant_initializer(0.01)

        # weight bias
        W1 = tf.get_variable("W1",shape=[num_of_features+1,num_classes],initializer=weight_init, dtype=tf.float32)
        logits = tf.matmul(X, W1)
        # hidden layer
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
        self.predict = tf.argmax(tf.nn.softmax(logits,dim=1), axis=1,output_type=tf.int32)
        # loss
        weight_decay = weight_decay_scale/2 * tf.norm(W1)**2
        self.loss = cross_entropy_loss + weight_decay
        self.loss = tf.squeeze(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(self.Y),tf.squeez(self.predict)), tf.float32))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learnning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

def get_data3():
    global trainData, trainTarget, validData, validTarget, testData, testTarget
    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Target = Target[randIndx].reshape([-1,1])
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    trainData = trainData.reshape([trainData.shape[0], -1])
    validData = validData.reshape([validData.shape[0], -1])
    testData = testData.reshape([testData.shape[0], -1])
    return trainData, trainTarget, validData, validTarget, testData, testTarget
def get_faceData():
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
              data[rnd_idx[trBatch+1:trBatch+validBatch],:],\
              data[rnd_idx[trBatch+validBatch+1:-1],:]
      trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
              target[rnd_idx[trBatch+1:trBatch+validBatch], task],\
              target[rnd_idx[trBatch+validBatch+1:-1], task]
      return trainData, validData, testData, trainTarget, validTarget, testTarget

    trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation('data.npy', 'target.npy', 0)
    data = [trainData, trainTarget, validData, validTarget, testData, testTarget]
    return data

def Q3_1():
    get_data3()
    BATCH_SIZE = 500
    LEARNING_RATE = 0.005
    weight_decay_scale = 0.01
    tf.reset_default_graph()

    net = Multiclass_Logistic_Net(10)
    sess=tf.Session()

    with sess.as_default():
        summary = net.train(trainData, trainTarget, LEARNING_RATE, weight_decay_scale, BATCH_SIZE, 5000)

    plt.figure(figsize=(10,4))
    for key, points in summary.items():
        s = np.array(summary)
        plt.plot(points, label=key)
    plt.legend()
    plt.title("Values vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.show()
def Q3_2():
    trainData, trainTarget, validData, validTarget, testData, testTarget = get_faceData()
    BATCH_SIZE = 300
    LEARNING_RATE = 0.005
    weight_decay_scale = 0.01
    tf.reset_default_graph()

    net = Multiclass_Logistic_Net(10)
    sess=tf.Session()

    with sess.as_default():
        summary = net.train(trainData, trainTarget, LEARNING_RATE, weight_decay_scale, BATCH_SIZE, 5000)

    plt.figure(figsize=(10,4))
    for key, points in summary.items():
        s = np.array(summary)
        plt.plot(points, label=key)
    plt.legend()
    plt.title("Values vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.show()

if __name__ == "__main__":
    input_ = input("what question to run?")
    {'1.1':Q1_1, '1.2':Q1_2, '1.3':Q1_3, '1.4':Q1_4, '2.0':Q2_prep, '2.1':Q2_1, '2.2':Q2_2,
    '2.3':Q2_3
    ,'3.1':Q3_1
    ,'3.2':Q3_2}[input_]()
