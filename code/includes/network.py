import tensorflow as tf

from includes.layers import GraphConvolutionLayer


class FeedForwardNetwork:
    def __init__(self, name, activation=None, initializer=None, weight_decay_coeff=0.5):
        self.name = name
        self.weight_decay_coeff = weight_decay_coeff

        self.activation = activation
        self.initializer = initializer

    def build(self, output_dims, layer_sizes, input_var, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse) as _:
            input_var = tf.layers.flatten(input_var)
            layers = [input_var]

            for index, layer_size in enumerate(layer_sizes):
                layers.append(
                    tf.layers.dense(
                        layers[index],
                        layer_size,
                        activation=self.activation,
                        kernel_initializer=self.initializer,
                        name="network_layer_" + str(index + 1)
                    )
                )

            self.outputs = []
            for name, output_dim in output_dims:
                self.outputs.append(
                    tf.layers.dense(
                        layers[-1],
                        output_dim,
                        activation=None,
                        kernel_initializer=self.initializer,
                        name="network_output/" + name
                    )
                )

        self.layers = layers[1:]

        if len(self.outputs) == 1:
            return self.outputs[0]

        return self.outputs

    # def get_weight_decay_loss(self):
    #     params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    #     r1 = self.name + "\/.*\/kernel"
    #     r2 = self.name + "\/.*\/gamma"

    #     l2_norm_loss = 0
    #     for p in params:
    #         if re.search(r1, p.name) or re.search(r2, p.name):
    #             l2_norm_loss += tf.nn.l2_loss(p)

    #     return self.weight_decay_coeff * l2_norm_loss


class GraphConvolutionalNetwork:
    def __init__(self, name, activation=None, initializer=None, dropout=0.0):
        self.name = name

        self.dropout = dropout

        self.activation = activation
        self.initializer = initializer

    def build(self, input_dim, output_dims, layer_sizes, adjacency_matrix, input_var, reuse=False):
        layers = [input_var]
        layer_sizes = [input_dim] + list(layer_sizes)

        with tf.variable_scope(self.name, reuse=reuse) as _:
            for index, layer_size in enumerate(layer_sizes[:-1]):
                layers.append(
                    GraphConvolutionLayer(
                        layers[index],
                        layer_size,
                        layer_sizes[index + 1],
                        adjacency_matrix,
                        dropout=self.dropout,
                        activation=self.activation,
                        initializer=self.initializer,
                        name="network_layer_" + str(index + 1)
                    )
                )

            self.outputs = []
            for name, output_dim in output_dims:
                self.outputs.append(
                    GraphConvolutionLayer(
                        layers[-1],
                        layer_sizes[-1],
                        output_dim,
                        adjacency_matrix,
                        dropout=self.dropout,
                        initializer=self.initializer,
                        name="network_output/" + name
                    )
                )

        self.layers = layers[1:]

        if len(self.outputs) == 1:
            return self.outputs[0]

        return self.outputs
