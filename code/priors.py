
class DiscreteFactorial:
    def __init__(self, dim, n_classes):
        self.dim = dim
        self.n_classes = n_classes

    def sample_reparametrization_variable(self, n, is_training, **kwargs):
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
