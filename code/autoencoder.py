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
    def __init__(self, name, networks, activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer, weight_decay_coeff=1.0):
        self.name = name
        self.weight_decay_coeff = weight_decay_coeff

        self.activation = activation
        self.initializer = initializer

        self.networks = dict([
            (network, FeedForwardNetwork(
                network + "/network", self.activation, self.initializer, weight_decay_coeff=0.3
            )) for network in networks
        ])

    def build(self, latent_dims, layer_sizes, latent_vars, input_parameters, extra_parameters, input_var):
        with tf.variable_scope(self.name) as _:
            self.latent_variables = list()
            self.outputs = dict()

            for i in range(len(latent_dims)):
                for parameter in input_parameters[i]:
                    self.outputs[parameter] = self.networks[parameter].build(
                        latent_dims[i], layer_sizes[i], input_var
                    )

                latent_var, epsilon = latent_vars[i]
                parameters = dict([
                    (parameter, value) for parameter, value in self.outputs.iteritems()
                ])
                parameters.update(extra_parameters[i])

                input_var = latent_var.inverse_reparametrize(
                    epsilon,
                    parameters=parameters
                )

                self.latent_variables.append((latent_var, input_var))

                input_var = tf.layers.flatten(input_var)

            return input_var

    def kl_from_priors(self):
        return sum([
            latent_var.kl_from_prior(self.outputs)
            for latent_var, _ in self.latent_variables
        ])

    def sample_reparametrization_variables(self, n):
        return [
            latent_var.sample_reparametrization_variable(n)
            for latent_var, _ in self.latent_variables
        ]


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
