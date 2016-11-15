from features import available_features, LSTMFeature
from scipy.sparse import csr_matrix
import numpy as np
from random import randint


class FeatureSet(object):

    def __init__(self, iterator, step_name, pos_type, config, freezed, label_func=None, vocabs=None, class_names=None):
        self.config = config
        self.name = step_name + pos_type
        self.iterator = iterator
        self.step_name = step_name
        self.pos_type = pos_type
        self.freezed = freezed
        self.vocabs = vocabs
        self.label_func = label_func
        self.class_names = class_names
        feature_names = self.config.get_value('features', lambda s: s.split(' '))
        self.binary_features = [available_features[f_name]() for f_name in feature_names]
        self.lstm_feature = LSTMFeature()
        if self.vocabs:
            self._set_vocabs()

        self.binary_feature_matrix = None
        self.binary_feature_width = None
        self.lstm_feature_vectors = None
        self.lstm_feature_row_width = None
        self.label_array = None
        self.number_of_instances = None
        self.num_classes = None if self.class_names is None else len(self.class_names)
        self.class_indices = None

        self.print('get binary feature matrix')
        binary_features_vectors = list()
        for feature in self.binary_features:
            feature_vectors = feature.get_vector_batch(self.iterator, self.freezed)
            binary_features_vectors.append(feature_vectors)
        features_length = [len(f) for f in self.binary_features]
        self.print('binary features length ' + str(features_length))
        self.binary_feature_matrix = dicts_to_sparse_matrix(binary_features_vectors, features_length, add_bias=True)
        self.binary_feature_width = self.binary_feature_matrix[0].shape[1]
        self.print('finished binary feature matrix')
        self.print('get lstm feature matrix')
        self.lstm_feature_vectors = self.lstm_feature.get_vector_batch(self.iterator, self.freezed)
        self.lstm_feature_row_width = len(self.lstm_feature)
        self.print('finished lstm feature matrix')

        # get the labels
        if self.label_func is not None:
            label_array_raw = self.label_func(self.iterator)
            self.class_names = list(np.unique(label_array_raw).tolist())
            self.num_classes = len(self.class_names)
            label_array = list()
            # calculate the class weights by frequency
            class_weights = [1 - (list(label_array_raw).count(c) / float(len(list(label_array_raw))))
                                  for c in self.class_names]
            self.class_weights = [w / min(class_weights) for w in class_weights]
            print('classes: ', self.class_names)

            for i, label_raw in enumerate(label_array_raw):
                label = [0 for _ in range(self.num_classes)]
                label[self.class_names.index(label_raw)] = 1
                label_array.append(label)

            self.label_array = np.asarray(label_array)

        # check if number of lstm feature instances, binary feature instance and labels are identical
        if (self.binary_feature_matrix.shape[0] != len(self.lstm_feature_vectors)) or \
                (self.label_func and (self.binary_feature_matrix.shape[0] != len(self.label_array))):
            raise ValueError('No equal number of instances')
        self.number_of_instances = self.binary_feature_matrix.shape[0]

    def get_binary_feature_matrix(self):
        return self.binary_feature_matrix

    def get_lstm_features(self):
        return self.lstm_feature, self.lstm_feature_row_width

    def get_training_batch(self, batch_size, epoch):
        random_int = randint(0, self.number_of_instances)
        indices = [(i + random_int) % self.number_of_instances for i in range(batch_size)]
        batch_lstm_instances = list()
        batch_binary_instances = self.binary_feature_matrix[indices].toarray()
        labels = list()
        for i in indices:
            batch_lstm_instances.append(self.lstm_feature_vectors[i])
            labels.append(self.label_array[i])
        lstm_instance_time_major, sequence_lengths = self._lstm_time_major(batch_lstm_instances)
        labels = np.asarray(labels)
        batch_binary_instances = np.asarray(batch_binary_instances)
        return batch_binary_instances, lstm_instance_time_major, sequence_lengths, labels

    def get_prediction_instances(self, start, stop):
        lstm_instance_time_major, sequence_lengths = self._lstm_time_major(self.lstm_feature_vectors[start:stop])
        binary_instances = self.binary_feature_matrix[start:stop].toarray()
        return binary_instances, lstm_instance_time_major, sequence_lengths

    def get_prediction_instance(self, i):
        feature_vector = self.lstm_feature_vectors[i]
        sequence_lengths = [len(feature_vector)]
        lstm_instance_time_major = list()
        for row_index in range(len(feature_vector)):
            row = [1 if r in feature_vector[row_index] else 0 for r in range(self.lstm_feature_row_width)]
            lstm_instance_time_major.append([row])

        lstm_instance_time_major = np.asarray(lstm_instance_time_major, dtype=np.float32)
        sequence_lengths = np.asarray(sequence_lengths, dtype=np.int32)
        return lstm_instance_time_major, sequence_lengths

    def _lstm_time_major(self, lstm_feature_instances):
        # Tensorflow needs this format for sequences of different length
        sequence_lengths = [len(sequence) for sequence in lstm_feature_instances]
        max_sequence_length = max(sequence_lengths)

        instance_time_major = np.zeros(shape=(max_sequence_length, len(lstm_feature_instances),
                                              self.lstm_feature_row_width), dtype=np.float32)
        for s_id, sequence in enumerate(lstm_feature_instances):
            for r_id , row in enumerate(sequence):
                for key in row.keys():
                    instance_time_major[r_id][s_id][key] = 1

        sequence_lengths = np.asarray(sequence_lengths, dtype=np.int32)
        return instance_time_major, sequence_lengths

    def get_vocabs(self):
        # get the vocabularies from the features of this step to be saved in the model object
        vocabs = {}
        for feature in self.binary_features:
            if hasattr(feature, 'get_vocab'):
                vocabs.update(feature.get_vocab())
        return vocabs

    def _set_vocabs(self):
        # load the vocabularies into the features to be able to reproduce one-hot-vectors of the trained model
        for feature in self.binary_features:
            if hasattr(feature, 'set_vocab'):
                feature.set_vocab(self.vocabs)

    def get_label_array(self):
        if self.label_func is None:
            raise ValueError('Need label function to generate labels')
        return self.label_array

    def print(self, s):
        print(self.step_name + ' - ' + self.pos_type + ': ' + s)


def dicts_to_sparse_matrix(features_vectors, features_length, add_bias=False):
    if any([len(features_vectors[0]) != len(f) for f in features_vectors]):
        raise ValueError('Every Feature needs values for every instance')

    shape = (len(features_vectors[0]), sum(features_length))
    row = list()
    col = list()
    data = list()

    for feature_idx, feature_vectors in enumerate(features_vectors):
        col_offset = sum(features_length[0:feature_idx])
        for row_idx, feature_vector in enumerate(feature_vectors):
            for col_idx, v in feature_vector.items():
                col_idx += col_offset
                row.append(row_idx)
                col.append(col_idx)
                data.append(v)
    if add_bias:
        for i in range(shape[0]):
            row.append(i)
            col.append(shape[1])
            data.append(1)
        shape = (shape[0], shape[1]+1)
    return csr_matrix((data, (row, col)), shape=shape)