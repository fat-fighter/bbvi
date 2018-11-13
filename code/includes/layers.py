import tensorflow as tf


def GraphConvolutionLayer(input_var, input_dim, output_dim, adjacency_matrix, dropout=0.0,
                          activation=None, initializer=None, name="graph_convolution_layer"
                          ):
    with tf.variable_scope(name):
        weights = tf.get_variable(
            "weights", shape=(input_dim, output_dim),
            initializer=initializer
        )

        x = tf.nn.dropout(input_var, 1 - dropout)
        x = tf.matmul(x, weights)
        x = tf.matmul(adjacency_matrix, x)

        if activation is not None:
            x = activation(x)

        return x
