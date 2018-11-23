import sys
import math
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.io as sio
import tensorflow as tf
import scipy.sparse as sp

from includes.preprocessing import mask_test_edges, preprocess_graph, sparse_to_tuple


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_gumbel(shape, eps=1e-20):
    U = np.random.uniform(0, 1, shape)
    return - np.log(eps - np.log(U + eps))


def sample_bernoulli(probs):
    shape = tf.shape(probs)
    return tf.where(
        tf.random_uniform(shape) - probs < 0,
        tf.ones(shape), tf.zeros(shape)
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_data(datagroup, dataset, **args):
    def spiral(dataset="normal", N_tr=5000, N_ts=1000):
        D = 2
        K = 5
        X_tr = np.zeros((N_tr * K, D))
        X_ts = np.zeros((N_ts * K, D))

        for j in range(K):
            ix = range(N_tr * j, N_tr * (j + 1))
            r = np.linspace(2.5, 10.0, N_tr)
            t = np.linspace(j * 1.25, (j + 1) * 1.25, N_tr) + \
                np.random.randn(N_tr) * 0.05
            X_tr[ix] = np.c_[r * np.sin(t), r * np.cos(t)]

        for j in range(K):
            ix = range(N_ts * j, N_ts * (j + 1))
            r = np.linspace(2.5, 10.0, N_ts)
            t = np.linspace(j * 1.25, (j + 1) * 1.25, N_ts) + \
                np.random.randn(N_ts) * 0.05
            X_ts[ix] = np.c_[r * np.sin(t), r * np.cos(t)]

        return X_tr, X_ts

    def mnist(dataset="static", dir="data/mnist"):
        from tensorflow.examples.tutorials.mnist import input_data

        mnist = input_data.read_data_sets("data/mnist/", one_hot=True)

        test_data = mnist.test.images
        train_data = mnist.train.images

        return train_data, test_data

    def graph(dataset="citeseer", dir="data/graphs/"):
        if dataset == "yeast":
            mat_data = sio.loadmat(dir + "yeast.mat")

            adj = mat_data['B']
            adj = sp.csr_matrix(adj)

            features = sp.identity((adj.shape[0]))

        else:
            names = ['x', 'tx', 'allx', 'graph']
            objects = []
            for i in range(len(names)):
                with open(dir + "ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                    objects.append(pkl.load(f, encoding="latin1"))

            x, tx, allx, graph = tuple(objects)

            test_idx_reorder = parse_index_file(
                dir + "ind.{}.test.index".format(dataset)
            )
            test_idx_range = np.sort(test_idx_reorder)

            if dataset == 'citeseer':
                test_idx_range_full = range(
                    min(test_idx_reorder), max(test_idx_reorder)+1)
                tx_extended = sp.lil_matrix(
                    (len(test_idx_range_full), x.shape[1]))
                tx_extended[test_idx_range-min(test_idx_range), :] = tx
                tx = tx_extended

            features = sp.vstack((allx, tx)).tolil()
            features[test_idx_reorder, :] = features[test_idx_range, :]
            features = features.todense()

            adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        adj_orig = adj - sp.dia_matrix(
            (adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape
        )
        adj_orig.eliminate_zeros()

        adj_train, _, test_edges, test_edges_false = mask_test_edges(
            adj_orig
        )

        adj_orig = adj_orig.todense()

        adj_norm = preprocess_graph(adj_train)
        adj_norm = adj_norm.todense()

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = adj_label.todense()

        return (adj_norm, adj_label, adj_orig, features), (test_edges, test_edges_false)

    if datagroup == "spiral":
        return spiral(dataset=dataset, **args)
    elif datagroup == "mnist":
        return mnist(dataset=dataset, **args)
    elif datagroup == "graph":
        return graph(dataset=dataset, **args)
    else:
        assert(False)


class Dataset:
    def __init__(self, data, datagroup, type="train", batch_size=100, shuffle=True):
        self.type = type
        self.datagroup = datagroup
        self.batch_size = batch_size

        if datagroup == "graph":
            if type == "train":
                adj_norm, adj_label, adj_orig, features = data

                self.adj_orig = adj_orig
                self.adj_norm = adj_norm
                self.adj_label = adj_label
                self.features = features

                self.input_dim = features.shape[1]

            elif type == "test":
                edges_pos, edges_neg = data

                self.edges_pos = edges_pos
                self.edges_neg = edges_neg
            else:
                raise ValueError

            self.epoch_len = 1

        else:
            if datagroup == "mnist":
                self.input_type = "binary"
            elif datagroup == "spiral":
                self.input_type = "real"
            else:
                raise NotImplementedError

            self.data = np.copy(data)
            self.input_dim = data.shape[1]

            self.epoch_len = int(math.ceil(len(data) / batch_size))

            if shuffle:
                np.random.shuffle(self.data)

    def get_batches(self, shuffle=True):
        if self.datagroup == "graph":
            raise NotImplementedError

        if shuffle:
            np.random.shuffle(self.data)

        batch = []
        for row in self.data:
            batch.append(row)
            if len(batch) == self.batch_size:
                yield np.array(batch)
                batch = []
        if len(batch) > 0:
            yield np.array(batch)

    def __len__(self):
        return self.epoch_len
