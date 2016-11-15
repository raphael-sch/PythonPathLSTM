import tensorflow as tf
from utils import find_divider
import numpy as np


class LSTMModel:

    def __init__(self, name, output_path):
        self.name = name + 'LSTM'
        self.output_path = output_path
        self.trained = False

    def train(self, feature_set, config):
        print('start training LSTM Model')
        num_classes = feature_set.num_classes
        row_width = feature_set.lstm_feature_row_width
        binary_feature_width = feature_set.binary_feature_width
        batch_size = config.get_value('batch_size', int)
        lstm_size = config.get_value('lstm_size', int)
        hidden_layer_size = config.get_value('hidden_layer', int)
        learning_rate = config.get_value('learning_rate', float)
        drop_out_rate = config.get_value('drop_out_rate', float)
        class_weights = np.asarray([feature_set.class_weights], dtype=np.float32)
        upsample = config.get_value('class_weights', lambda s: s == 'True')
        if not upsample:
            class_weights = np.ones(shape=class_weights.shape, dtype=np.float32)
        print('class weights: ', class_weights)

        iteration_factor = config.get_value('iteration_factor', int)
        iterations = int(int((feature_set.number_of_instances / batch_size)+1) * iteration_factor)

        graph = tf.Graph()
        with graph.as_default():

            sequence_lengths = tf.placeholder(tf.int32, shape=[batch_size])
            # path sequence one-hot vectors for lstm
            tf_train_dataset = tf.placeholder(tf.float32, shape=[None, batch_size, row_width])
            # binary features one-hot vectors
            tf_train_binary_dataset = tf.placeholder(tf.float32, shape=[batch_size, binary_feature_width])
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_classes))

            weights_e_hidden = tf.Variable(tf.random_normal([lstm_size, hidden_layer_size]), dtype=tf.float32)
            weights_B_hidden = tf.Variable(tf.random_normal([binary_feature_width, hidden_layer_size]), dtype=tf.float32)
            biases_hidden = tf.Variable(tf.random_normal([hidden_layer_size]), dtype=tf.float32)

            weights_e_softmax = tf.Variable(tf.random_normal([lstm_size, num_classes]), dtype=tf.float32)
            weights_h_softmax = tf.Variable(tf.random_normal([hidden_layer_size, num_classes]), dtype=tf.float32)
            biases_softmax = tf.Variable(tf.random_normal([num_classes]), dtype=tf.float32)

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=drop_out_rate)
            outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, tf_train_dataset, sequence_length=sequence_lengths,
                                                     dtype=tf.float32, time_major=True)
            e = final_state[1]
            h = tf.nn.dropout(tf.nn.relu(tf.matmul(tf_train_binary_dataset, weights_B_hidden) + tf.matmul(e, weights_e_hidden) + biases_hidden), drop_out_rate)
            s = tf.matmul(e, weights_e_softmax) + tf.matmul(h, weights_h_softmax) + biases_softmax
            loss = tf.nn.softmax_cross_entropy_with_logits(s, tf_train_labels)

            # create a tensor for the class weights
            weight_per_instance = tf.transpose(tf.matmul(tf_train_labels, tf.transpose(class_weights)))

            # multiply each loss with the class weight
            cost = tf.reduce_mean(tf.mul(weight_per_instance, loss))

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

            correct_pred = tf.equal(tf.argmax(s, 1), tf.argmax(tf_train_labels, 1))
            # accuracy handles class weights
            accuracy = tf.reduce_sum(tf.truediv(tf.mul(weight_per_instance,
                       tf.cast(correct_pred, tf.float32)), tf.reduce_sum(weight_per_instance)))
            saver = tf.train.Saver()
        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            print('iterations: ' + str(iterations))
            for epoch in range(iterations):
                # actual load data
                x_B_batch, x_batch, sequence_len, y_batch = feature_set.get_training_batch(batch_size, epoch)

                feed_dict = {tf_train_dataset: x_batch,}
                feed_dict.update({tf_train_binary_dataset: x_B_batch})
                feed_dict.update({tf_train_labels: y_batch})
                feed_dict.update({sequence_lengths: sequence_len})
                session.run(optimizer, feed_dict=feed_dict)

                if epoch % 10 == 0:
                    acc = session.run(accuracy, feed_dict=feed_dict)
                    loss = session.run(cost, feed_dict=feed_dict)
                    print('Iter ' + str(epoch * batch_size) + '/' + str(iterations * batch_size) + ', Minibatch Loss= ' + \
                          '{:.6f}'.format(loss) + ', Training Accuracy= ' + '{:.5f}'.format(acc))
            save_path = saver.save(session, self.output_path + self.name + '.model')
        self.trained = True
        print('finished training LSTM Model')

    def pred(self, feature_set, config):
        num_classes = feature_set.num_classes
        row_width = feature_set.lstm_feature_row_width
        binary_feature_width = feature_set.binary_feature_width
        lstm_size = config.get_value('lstm_size', int)
        hidden_layer_size = config.get_value('hidden_layer', int)
        num_instances = feature_set.number_of_instances
        if num_instances == 0:
            return []
        # batches can be of prior unknown length, but must then be all of same length
        batch_size = find_divider(num_instances)
        batches = int(num_instances / batch_size)
        batch_start_stop = [(i, min((i + batch_size), num_instances)) for i in range(0, num_instances, batch_size)]

        graph = tf.Graph()
        with graph.as_default():
            sequence_lengths = tf.placeholder(tf.int32, shape=[None])
            tf_pred_dataset = tf.placeholder(tf.float32, shape=[None, None, row_width])
            tf_train_binary_dataset = tf.placeholder(tf.float32, shape=[None, binary_feature_width])

            weights_e_hidden = tf.Variable(tf.random_normal([lstm_size, hidden_layer_size]), dtype=tf.float32)
            weights_B_hidden = tf.Variable(tf.random_normal([binary_feature_width, hidden_layer_size]), dtype=tf.float32)
            biases_hidden = tf.Variable(tf.random_normal([hidden_layer_size]), dtype=tf.float32)

            weights_e_softmax = tf.Variable(tf.random_normal([lstm_size, num_classes]), dtype=tf.float32)
            weights_h_softmax = tf.Variable(tf.random_normal([hidden_layer_size, num_classes]), dtype=tf.float32)
            biases_softmax = tf.Variable(tf.random_normal([num_classes]), dtype=tf.float32)

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)

            outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, tf_pred_dataset, sequence_length=sequence_lengths,
                                                     dtype=tf.float32, time_major=True)

            e = final_state[1]

            h = tf.nn.relu(tf.matmul(tf_train_binary_dataset, weights_B_hidden) + tf.matmul(e, weights_e_hidden) + biases_hidden)
            s = tf.matmul(e, weights_e_softmax) + tf.matmul(h, weights_h_softmax) + biases_softmax
            prediction = tf.nn.softmax(s)

            saver = tf.train.Saver()
        with tf.Session(graph=graph) as session:
            model_path = (config.get_value('lstm_model_path') + self.name + '.model')
            saver.restore(session, model_path)

            y_preds = list()
            print('prediction batch size: {}, batches: {}'.format(str(batch_size), batches))
            # much more memory efficient than complete test instances
            for i, start_stop in enumerate(batch_start_stop):
                start, stop = start_stop
                x_B_batch, x_batch, sequence_len = feature_set.get_prediction_instances(start, stop)

                feed_dict = {tf_pred_dataset: x_batch}
                feed_dict.update({tf_train_binary_dataset: x_B_batch})
                feed_dict.update({sequence_lengths: sequence_len})
                y_pred = session.run(prediction, feed_dict=feed_dict)
                y_preds.extend(y_pred.tolist())
                if i % int(batches/10+1) == 0:
                    print(str(i) + '/' + str(batches))
            print(str(batches) + '/' + str(batches))

        pred_labels = list()
        for probabilities in y_preds:
            index, _ = max(enumerate(probabilities), key=lambda x: x[1])
            pred_labels.append(feature_set.class_names[index])
        print([(c, pred_labels.count(c)) for c in set(pred_labels)])
        return pred_labels
