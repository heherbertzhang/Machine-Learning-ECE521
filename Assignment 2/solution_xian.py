import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
        weight_init = tf.contrib.layers.xavier_initializer(uniform=True)
        const_init = tf.constant_initializer(0.01)

        # weight bias
        W1 = tf.get_variable("W1", shape=[num_of_features, 1], initializer=weight_init, dtype=tf.float32)
        b1 = tf.get_variable("b1", shape=[1, 1], initializer=const_init, dtype=tf.float32)  # broadcast
        # hidden layer
        y_ = tf.add(tf.matmul(X, W1), b1)

        # loss
        weight_decay = weight_decay_scale / 2 * tf.norm(W1)**2
        loss = tf.nn.l2_loss(y_ - self.Y) / tf.cast(tf.shape(X)[0], tf.float32) + weight_decay
        self.loss = tf.squeeze(loss)
        self.predict = tf.cast(y_ > 0.5, tf.float32)
        self.optimizer = tf.train.GradientDescentOptimizer(learnning_rate)
        self.train_op = self.optimizer.minimize(loss)
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.predict, self.Y), tf.float32)) / tf.cast(
            tf.shape(self.Y)[0], tf.float32)

    def add_training_data(self, inputs, labels, batch_size):
        trainning_data = tf.constant(inputs, dtype=tf.float32)
        trainning_label = tf.constant(labels, dtype=tf.float32)
        # create a runner queue, shuffle every epoch
        x, y = tf.train.slice_input_producer([trainning_data, trainning_label], num_epochs=None, shuffle=True)
        # build mini-batch
        epoch = labels.shape[0]
        self.next_batch_op = tf.train.batch([x, y], batch_size=batch_size)

    def init(self, sess=None):
        sess = (sess or tf.get_default_session())
        sess.run(self.init_op)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)
        return sess

    def get_loss(self, inputs, labels, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.loss, {self.X: inputs, self.Y: labels})

    def update(self, sess=None):
        sess = sess or tf.get_default_session()
        input_batch, label_batch = sess.run(self.next_batch_op)
        _, loss = sess.run([self.train_op, self.loss], {self.X: input_batch, self.Y: label_batch})
        return loss

    def get_prediction(self, data, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.predict, {self.X: data})

    def get_accuracy(self, inputs, labels, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.accuracy, {self.X: inputs, self.Y: labels})

    def train(self, trainData_rs, trainTarget, learnning_rate, weight_decay_scale, batch_size=500, steps=20000,
              sess=None):
        sess = sess or tf.get_default_session()
        epoch_size = trainData_rs.shape[0]

        summary = []

        self.add_training_data(trainData_rs, trainTarget, batch_size)
        self.build(trainData_rs.shape[1], learnning_rate, weight_decay_scale)

        self.init()
        loss_val = 0
        for step in range(steps):
            loss_val = self.update(sess)
            if ((step * batch_size) % epoch_size == 0):
                summary.append(loss_val)
                print(loss_val)
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.training_loss = loss_val
        return summary



def Q1_1():

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
            summary = net.train(trainData, trainTarget, learning_rate, weight_decay_scale, batch_size, steps=400)
            summarys.append(summary)
            losses.append(net.training_loss)

        sess.close()
        end = time.time()
        times.append((end - start))


        # Report the final training MSE for each mini-batch value
        # What is the best mini-batch size in terms of training time?
        # Comment on your observation.
        for time, loss in zip(times, losses):
            print(str(time) + "," + str(loss))

        plt.figure()
        for i, summary in enumerate(summarys):
            plt.plot(summary, label='batch_size= ' + str(B[i]))
            plt.ylabel("training loss")
            plt.xlabel("number of epochs")
            plt.title("Q1.2")
        plt.legend()
        plt.show()

def Q1_3():
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
            summary = net.train(validData, validTarget, learning_rate, weight_decay_scale, batch_size, steps=1000)
            validation_accuracy = net.get_accuracy(validData, validTarget)
            test_accuracy = net.get_accuracy(testData, testTarget)
            accuracys.append((validation_accuracy, test_accuracy))
    for accuracy in accuracys:
        print(accuracy)


def Q1_4():
    # Q1.4
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
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.predict, self.Y), tf.float64)) / tf.cast(
                tf.shape(self.Y)[0], tf.float64)
            self.loss = tf.nn.l2_loss(self.Y-self.Y_) / tf.cast(tf.shape(self.X)[0], tf.float64)

        def get_accuracy(self, inputs, labels, sess=None):
            sess = sess or tf.get_default_session()
            accu = sess.run(self.accuracy, {self.X_input: inputs, self.Y: labels})
            return accu
        def get_loss(self, inputs, labels, sess=None):
            sess = sess or tf.get_default_session()
            loss = sess.run(self.loss, {self.X_input: inputs, self.Y: labels})
            return loss

    normal_eqn = Normal_Equation_Method()
    tf.reset_default_graph()
    sess = tf.Session()
    with sess.as_default():
        normal_eqn.build(trainData, trainTarget, 0)
        print(str(normal_eqn.get_accuracy(validData, validTarget)))
        print(str(normal_eqn.get_loss(trainData, trainTarget)))

if __name__ == "__main__":
    Q1_1()