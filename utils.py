from scipy.sparse import csr_matrix
import os

def chunkify(lst,n):
    return [ lst[i::n] for i in range(n) ]


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


def find_divider(number):
    for i in reversed(range(500)):
        if number % i == 0:
            return i


def check_path(path):
    if path[-1] != '/':
        path += '/'
    if not os.access(path, os.W_OK):
        raise AttributeError('Can\'t write to path: ' + str(path))
    return path


class Config:

    config_file = './config.cfg'

    def __init__(self, config_name):
        self.config_name = config_name
        self.values = dict()
        self.read_config()

    def read_config(self):
        with open(Config.config_file, 'r') as f:
            config_name = None
            for line in f:
                line = line.rstrip()
                if line.startswith('>'):
                    config_name = line[1:]
                    self.values[config_name] = dict()
                elif line == '' or line == '#':
                    continue
                else:
                    key, value = line.split(' = ')
                    self.values[config_name][key] = value

    def get_value(self, key, cast=None):
        if cast == bool:
            cast = lambda v: v == 'True'
        elif cast is None:
            cast = lambda x: x
        if key in self.values[self.config_name]:
            return cast(self.values[self.config_name][key])
        elif key in self.values['default']:
            return cast(self.values['default'][key])
        else:
            raise AttributeError('key not in config')

    def set_value(self, key, value):
        self.values[self.config_name][key] = value

    def __str__(self):
        values = {k: v for k, v in self.values['default'].items()}
        values.update({k: v for k, v in self.values[self.config_name].items()})
        return '[CONFIG] ' + ''.join(['{}: "{}", '.format(k, v) for k, v in values.items()])[:-2]

