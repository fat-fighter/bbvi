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


class DiscreteFactorial(LatentVariable):
    def __init__(self, dim, n_classes):
        self.dim = dim
        self.n_classes = n_classes

    def sample_reparametrization_variable(self, n, is_training=True):
        samples = sample_gumbel((n, self.dim, self.n_classes))
        if not is_training:
            samples[samples >= 0] = 1
            samples[samples < 0] = 0
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
        res = tf.reduce_mean(tf.reduce_sum(res, axis=-1))

        return res


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


class RBM(LatentVariable):
    def __init__(self, name, num_var1, num_var2, weight_decay,
                 num_samples=100, num_gibbs_iter=40, use_qupa=False):
        self.num_var1 = num_var1
        self.num_var2 = num_var2
        self.num_var = num_var1 + num_var2
        self.weight_decay = weight_decay
        self.name = name

        # bias on the left side
        self.b1 = tf.Variable(
            tf.zeros(shape=[self.num_var1, 1], dtype=tf.float32, name='bias1'))
        # bias on the right side
        self.b2 = tf.Variable(
            tf.zeros(shape=[self.num_var2, 1], dtype=tf.float32, name='bias2'))
        # pairwise weight
        self.w = tf.Variable(tf.zeros(
            shape=[self.num_var1, self.num_var2], dtype=tf.float32, name='pairwise'))

        # sampling options
        self.num_samples = num_samples
        self.use_qupa = use_qupa

        # concat b
        b = tf.concat(values=[tf.squeeze(self.b1),
                              tf.squeeze(self.b2)], axis=0)

        if not self.use_qupa:
            # init pcd class implemented in QuPA
            self.sampler = pcd.PCD(left_size=self.num_var1, right_size=self.num_var2,
                                   num_samples=self.num_samples, dtype=tf.float32)
        else:
            # init population annealing class in QuPA
            self.sampler = qupa.PopulationAnnealer(left_size=self.num_var1, right_size=self.num_var2,
                                                   num_samples=self.num_samples, dtype=tf.float32)

        # This returns a scalar tensor with the gradient of log z. Don't trust its value.
        self.log_z_train = self.sampler.training_log_z(
            b, self.w, num_mcmc_sweeps=num_gibbs_iter)

        # This returns the internal log z variable in QuPA sampler. We wil use this variable in evaluation.
        self.log_z_value = self.sampler.log_z_var

        # get always the samples after updating train log z
        with tf.control_dependencies([self.log_z_train]):
            self.samples = self.sampler.samples()

        # Define inverse temperatures used for AIS. Increasing the # of betas improves the precision of log z estimates.
        betas = tf.linspace(tf.constant(0.), tf.constant(1.), num=1000)
        # Define log_z estimation for evaluation.
        eval_logz = qupa.ais.evaluation_log_z(
            b, self.w, init_biases=None, betas=betas, num_samples=1024)

        # Update QuPA internal log z variable with the eval_logz
        self.log_z_update = self.log_z_value.assign(eval_logz)

    def energy_tf(self, samples):
        samples1 = tf.slice(samples, [0, 0], [-1, self.num_var1])
        samples2 = tf.slice(samples, [0, self.num_var1], [-1, -1])

        energy = tf.matmul(samples1, self.b1) + tf.matmul(samples2, self.b2) + tf.reduce_sum(
            tf.matmul(samples1, self.w) * samples2, 1, keepdims=True)
        energy = - tf.squeeze(energy, axis=1)
        return energy

    def log_prob(self, samples, is_training=True):
        return - self.energy_tf(samples) - self.log_z_value
