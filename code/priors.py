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
        if "mean" in kwargs:
            mean = kwargs["mean"]

            shape = list(mean.shape)
            shape.insert(0, n)

            samples = np.random.standard_normal(shape)
            samples += mean + samples

        else:
            samples = np.random.randn(n, self.dim)

        return samples

    def inverse_reparametrize(self, epsilon, parameters):
        assert("mean" in parameters and "log_var" in parameters)

        return parameters["mean"] + tf.exp(parameters["log_var"] / 2) * epsilon

    def kl_from_prior(self, parameters, eps=1e-20):
        assert("mean" in parameters and "log_var" in parameters)

        mean = parameters["mean"]
        log_var = parameters["log_var"]

        kl = tf.exp(log_var) + tf.square(mean) - 1. - log_var
        kl = tf.reduce_mean(0.5 * tf.reduce_sum(kl, axis=1))

        return kl


class DiscreteFactorial(LatentVariable):
    def __init__(self, name, dim, n_classes):
        self.name = name

        self.dim = dim
        self.n_classes = n_classes

    def sample_reparametrization_variable(self, n):
        return sample_gumbel((n, self.dim, self.n_classes))

    def sample_generative_feed(self, n, **kwargs):

        if "logits" in kwargs:
            logits = kwargs["logits"]

            shape = list(logits.shape)
            shape.insert(0, n)

            samples = sample_gumbel(shape)

            logits = np.reshape(logits, (1, -1, self.n_classes))
            samples = np.reshape(samples, (n, -1, self.n_classes))

            samples = samples + logits

            samples = np.reshape(samples, (-1, self.n_classes))

        else:
            shape = (n, self.dim, self.n_classes)

            samples = sample_gumbel(shape)
            samples = np.reshape(samples, (-1, self.n_classes))

        samples = np.asarray(np.equal(
            samples, np.max(samples, 1, keepdims=True)
        ), dtype=samples.dtype)
        samples = np.reshape(samples, shape)

        return samples

    def inverse_reparametrize(self, epsilon, parameters):
        assert("logits" in parameters and "temperature" in parameters)

        logits = parameters["logits"]
        logits = tf.reshape(logits, (-1, self.n_classes))

        res = tf.reshape(epsilon, (-1, self.n_classes))
        res = (logits + res) / parameters["temperature"]
        res = tf.nn.softmax(res)

        if self.n_classes == 2:
            res = res[:, 0]
            res = tf.reshape(res, (-1, self.dim))
        else:
            res = tf.reshape(res, (-1, self.dim, self.n_classes))

        return res

    def kl_from_prior(self, parameters, eps=1e-20):
        assert("logits" in parameters)

        logits = tf.reshape(parameters["logits"], (-1, self.n_classes))
        q_z = tf.nn.softmax(logits)

        kl = tf.reshape(
            q_z * (tf.log(q_z + eps) - tf.log(1.0 / self.n_classes)),
            (-1, self.dim * self.n_classes)
        )
        kl = tf.reduce_mean(tf.reduce_sum(kl, axis=1))

        return kl


class RBMPrior(DiscreteFactorial):
    def __init__(self, name, visible_dim, hidden_dim, beta=1.0, trainable=False, init_gibbs_iters=1000):
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

        self.beta = beta

        self.samples_visible = None

    def free_energy(self, samples):
        samples_visible = samples[:, :self.visible_dim]
        samples_hidden = samples[:, self.visible_dim:]

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

    def generate_gibbs_samples(self, session, k, g=1):
        clip = 0
        if self.samples_visible is None:
            clip = self.init_gibbs_iters // g
            k += clip

            probs = tf.ones((1, self.visible_dim)) / 2.0

            samples_visible = sample_bernoulli(probs)
            self.samples_visible = self._gibbs_vhv_k(
                samples_visible, self.init_gibbs_iters
            )[0]

        def sample(samples, _):
            samples_visible = samples[:, :self.visible_dim]

            samples = tf.concat(self._gibbs_vhv_k(
                samples_visible, g
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
        return self.generate_gibbs_samples(kwargs["session"], n, g=100)

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
            # zeta * tf.log(probs + eps) + (1 - zeta) * tf.log(1 - probs + eps),
            tf.log(zeta * probs + (1 - zeta) * (1 - probs)),
            axis=-1
        )
        log_prior_un = - self.free_energy(zeta)

        kl = tf.reduce_mean(
            log_posterior - log_prior_un
        )
        kl += self.log_partition(samples)

        return self.beta * kl
