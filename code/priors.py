import qupa
import qupa.pcd as pcd
import numpy as np
import tensorflow as tf

from includes.utils import sample_gumbel


class LatentVariable:
    def kl_from_prior(self, **kwargs):
        raise NotImplementedError

    def sample_reparametrize(self, **kwargs):
        raise NotImplementedError

    def inverse_reparametrize(self, **kwargs):
        raise NotImplementedError


class NormalFactorial(LatentVariable):
    def __init__(self, dim):
        self.dim = dim

    def sample_reparametrization_variable(self, n, is_training=True):
        return np.random.randn(n, self.dim)

    def inverse_reparametrize(self, epsilon, parameters):
        assert("mean" in parameters and "log_var" in parameters)

        return parameters["mean"] + tf.exp(parameters["log_var"] / 2) * epsilon

    def kl_from_prior(self, parameters, eps=1e-20):
        assert("mean" in parameters and "log_var" in parameters)

        mean = parameters["mean"]
        log_var = parameters["log_var"]

        res = tf.exp(log_var) + tf.square(mean) - 1. - log_var
        res = tf.reduce_mean(0.5 * tf.reduce_sum(res, axis=1))

        return res


class DiscreteFactorial(LatentVariable):
    def __init__(self, dim, n_classes):
        self.dim = dim
        self.n_classes = n_classes

    def sample_reparametrization_variable(self, n, is_training=True):
        samples = sample_gumbel((n, self.dim, self.n_classes))
        if not is_training:
            samples = np.reshape(samples, (-1, self.n_classes))
            samples = np.asarray(np.equal(
                samples, np.max(samples, 1, keepdims=True)
            ), dtype=samples.dtype)
            samples = np.reshape(samples, (-1, self.dim, self.n_classes))

        return samples

    def inverse_reparametrize(self, epsilon, parameters):
        assert("logits" in parameters and "temperature" in parameters)

        logits = parameters["logits"]
        logits = tf.reshape(logits, (-1, self.n_classes))

        res = tf.reshape(epsilon, (-1, self.n_classes))
        res = (logits + res) / parameters["temperature"]
        res = tf.nn.softmax(res)
        res = tf.reshape(res, (-1, self.dim, self.n_classes))

        return res

    def kl_from_prior(self, parameters, eps=1e-20):
        assert("logits" in parameters)

        logits = tf.reshape(parameters["logits"], (-1, self.n_classes))
        q_z = tf.nn.softmax(logits)

        res = tf.reshape(
            q_z * (tf.log(q_z + eps) - tf.log(1.0 / self.n_classes)),
            (-1, self.dim * self.n_classes)
        )
        res = tf.reduce_mean(tf.reduce_sum(res, axis=1))

        return res
