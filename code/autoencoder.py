import numpy as np
import tensorflow as tf

from network import FeedForwardNetwork
from matplotlib import pyplot as plt

from includes.utils import sample_gumbel


class Wrapper:
    def __init__(self, network, activation):
        self.network = network
        self.activation = activation

    def build(self, output_dim, layer_sizes, input_var):
        return self.activation(
            self.network.build(output_dim, layer_sizes, input_var)
        )


class Encoder:
    def __init__(self, name, latent_type, activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer, weight_decay_coeff=1.0):
        self.name = name
        self.weight_decay_coeff = weight_decay_coeff

        self.activation = activation
        self.initializer = initializer

        self.latent_type = latent_type

        assert(self.latent_type in ["binary", "normal", "split_bn"])

        if self.latent_type == "normal":
            self.networks = {
                "mean": FeedForwardNetwork(
                    "mean/network", self.activation, self.initializer, weight_decay_coeff=0.3
                ),
                "log_var": FeedForwardNetwork(
                    "log_var/network", self.activation, self.initializer, weight_decay_coeff=0.3
                )
            }
        if self.latent_type == "binary":
            self.networks = {
                "log_weights": FeedForwardNetwork(
                    "mean/network", self.activation, self.initializer, weight_decay_coeff=0.3
                )
            }
        if self.latent_type == "split_bn":
            raise NotImplementedError

    def build(self, latent_dim, layer_sizes, input_var):
        with tf.variable_scope(self.name) as _:
            self.outputs = dict([
                (name, network.build(latent_dim, layer_sizes, input_var))
                for name, network in self.networks.iteritems()
            ])


class Decoder:
    def __init__(self, name, latent_type, activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer, weight_decay_coeff=1.0):
        self.name = name
        self.weight_decay_coeff = weight_decay_coeff

        self.activation = activation
        self.initializer = initializer

        self.network = FeedForwardNetwork(
            "network", self.activation, self.initializer, weight_decay_coeff=0.3
        )

    def build(self, input_dim, layer_sizes, latent_var):
        with tf.variable_scope(self.name) as _:
            self.output = self.network.build(
                input_dim, layer_sizes, latent_var
            )


class LatentVariable:
    def __init__(self, name, dim, type):
        self.name = name
        self.dim = dim

        self.type = type
        assert(self.type in ["binary", "normal", "split_bn"])

    def sample_reparametrization_variable(self, n, is_training, **kwargs):
        if self.type == "normal":
            return np.random.randn(n, self.dim)

        if self.type == "binary":
            if is_training:
                return sample_gumbel((n, self.dim, 2))
            else:
                raise NotImplementedError

        if self.type == "split_bn":
            raise NotImplementedError

            # assert("split" in kwargs)

            # split = kwargs["split"]
            # assert(0 < split < self.dim)

            # return np.concatenate([
            #     np.random.uniform(0, 1, (n, split)),
            #     np.random.randn(n, self.dim - split)
            # ], axis=1)

    def inverse_reparametrize(self, epsilon, **kwargs):
        if self.type == "normal":
            assert("mean" in kwargs and "log_var" in kwargs)

            return kwargs["mean"] + tf.exp(kwargs["log_var"] / 2) * epsilon

        if self.type == "binary":
            assert("log_weights" in kwargs and "temperature" in kwargs)

            q_z = tf.reshape(kwargs["log_weights"], (-1, self.dim, 1))
            q_z = tf.concat(
                [q_z, tf.zeros(dtype=tf.float32, shape=tf.shape(q_z))], axis=-1
            )
            q_z = q_z + epsilon
            q_z = tf.reshape(q_z, (-1, 2))
            q_z = tf.nn.softmax(q_z / kwargs["temperature"])
            q_z = tf.reshape(q_z, (-1, self.dim, 2))
            # eps = 1e-10
            # res = (
            #     kwargs["log_weights"] +
            #     tf.log(epsilon + eps) - tf.log(1 - epsilon + eps)
            # ) / kwargs["temperature"]
            # res = tf.minimum(100.0, tf.maximum(res, -100.0))
            # res = tf.exp(res)
            return q_z[:, 1]

        if self.type == "split_bn":
            raise NotImplementedError
