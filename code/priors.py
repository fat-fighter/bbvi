import qupa
import qupa.pcd as pcd
import numpy as np
import tensorflow as tf

from includes.utils import sample_gumbel, sample_bernoulli


class LatentVariable:
    def kl_from_prior(self, **kwargs):
        raise NotImplementedError

    def sample_reparametrization_variable(self, **kwargs):
        raise NotImplementedError

    def sample_generative_feed(self, **kwargs):
        raise NotImplementedError

    def inverse_reparametrize(self, **kwargs):
        raise NotImplementedError


class NormalFactorial(LatentVariable):
    def __init__(self, name, dim):
        self.name = name

        self.dim = dim

    def sample_reparametrization_variable(self, n):
        return np.random.randn(n, self.dim)

    def sample_generative_feed(self, n, **kwargs):
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
    def __init__(self, name, dim, n_classes):
        self.name = name

        self.dim = dim
        self.n_classes = n_classes

    def sample_reparametrization_variable(self, n):
        return sample_gumbel((n, self.dim, self.n_classes))

    def sample_generative_feed(self, n, **kwargs):
        samples = sample_gumbel((n, self.dim, self.n_classes))
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


class RBM(DiscreteFactorial):
    def __init__(self, name, visible_dim, hidden_dim, trainable=False, init_gibbs_iters=5000):
        DiscreteFactorial.__init__(self, name, visible_dim + hidden_dim, 2)

        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        self.init_gibbs_iters = init_gibbs_iters

        with tf.variable_scope(self.name) as _:
            self.bv = tf.get_variable(
                "vbias", dtype=tf.float32, shape=(1, self.visible_dim),
                initializer=tf.initializers.zeros, trainable=trainable
            )
            self.bh = tf.get_variable(
                "hbias", dtype=tf.float32, shape=(1, self.hidden_dim),
                initializer=tf.initializers.zeros, trainable=trainable
            )

            self.w = tf.get_variable(
                "weights", dtype=tf.float32, shape=(self.visible_dim, self.hidden_dim),
                initializer=tf.initializers.random_normal, trainable=trainable
            )

        self.b = tf.expand_dims(
            tf.concat([tf.squeeze(self.bv), tf.squeeze(self.bh)], axis=0), 1
        )

        self.samples_visible = None

    def free_energy(self, samples):
        samples_visible = tf.slice(samples, [0, 0], [-1, self.visible_dim])
        samples_hidden = tf.slice(samples, [0, self.visible_dim], [-1, -1])

        energy = tf.matmul(samples, self.b) + tf.reduce_sum(
            tf.matmul(samples_visible, self.w) * samples_hidden, 1, keepdims=True
        )
        energy = - tf.squeeze(energy, axis=1)

        return energy

    def _propup(self, samples_visible):
        pre_sigmoid = self.bh + tf.matmul(
            samples_visible, self.w
        )
        return pre_sigmoid, tf.nn.sigmoid(pre_sigmoid)

    def _sample_hgv(self, samples_visible):
        _, probs = self._propup(samples_visible)
        return sample_bernoulli(probs)

    def _propdown(self, samples_hidden):
        pre_sigmoid = self.bv + tf.matmul(
            samples_hidden, tf.transpose(self.w)
        )
        return pre_sigmoid, tf.nn.sigmoid(pre_sigmoid)

    def _sample_vgh(self, samples_hidden):
        _, probs = self._propdown(samples_hidden)
        return sample_bernoulli(probs)

    def _gibbs_vhv(self, samples_visible):
        samples_hidden = self._sample_hgv(samples_visible)
        samples_visible = self._sample_vgh(samples_hidden)

        return samples_visible, samples_hidden

    def _gibbs_vhv_k(self, samples_visible, k):
        for _ in range(k - 1):
            samples_visible, _ = self._gibbs_vhv(samples_visible)

        return self._gibbs_vhv(samples_visible)

    def generate_gibbs_samples(self, session, k, t=1):
        clip = 0
        if self.samples_visible is None:
            clip = self.init_gibbs_iters / t
            k += clip

            probs = tf.ones((1, self.visible_dim)) / 2.0
            self.samples_visible = sample_bernoulli(probs)

        def sample(samples, _):
            samples_visible = tf.slice(
                samples, [0, 0], [-1, self.visible_dim]
            )

            samples = tf.concat(self._gibbs_vhv_k(
                samples_visible, t
            ), axis=1)

            return samples

        initializer = tf.concat(
            [self.samples_visible, tf.zeros((1, self.hidden_dim))], axis=1
        )
        samples = tf.layers.flatten(tf.scan(
            sample, tf.zeros(k),
            initializer=initializer
        ))
        samples = session.run(samples)[clip:, :]

        self.samples_visible = tf.convert_to_tensor(
            samples[-1:, :self.visible_dim], dtype=tf.float32
        )

        return samples

    def sample_reparametrization_variable(self, n):
        U = np.random.uniform(0, 1, (n, self.dim))
        return np.log(U / (1 - U))

    def sample_generative_feed(self, n, **kwargs):
        assert("session" in kwargs)
        return self.generate_gibbs_samples(kwargs["session"], n, t=100)

    def log_partition(self, samples):
        return - tf.reduce_mean(self.free_energy(samples))

    def inverse_reparametrize(self, epsilon, parameters):
        assert("log_ratios" in parameters and "temperature" in parameters)

        log_ratios = parameters["log_ratios"]

        res = (log_ratios + epsilon) / parameters["temperature"]
        res = tf.nn.sigmoid(res)

        return res

    def kl_from_prior(self, parameters, eps=1e-20):
        assert(
            "log_ratios" in parameters and
            "samples" in parameters and
            "zeta" in parameters
        )

        probs = tf.nn.sigmoid(parameters["log_ratios"])

        zeta = parameters["zeta"]
        samples = parameters["samples"]

        log_posterior = tf.reduce_sum(
            tf.log(zeta * probs + (1 - zeta) * (1 - probs)),
            axis=-1
        )
        log_prior_un = - self.free_energy(zeta)

        kl = tf.reduce_mean(
            log_posterior - log_prior_un
        )
        kl += self.log_partition(samples)

        return kl
