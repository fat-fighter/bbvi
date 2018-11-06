class RBM(LatentVariable):
    def __init__(self, name, visible_dim, hidden_dim, weight_decay=0.3,
                 num_samples=100, num_gibbs_iter=40, use_qupa=True):
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.dim = self.num_var = visible_dim + hidden_dim
        self.weight_decay = weight_decay
        self.name = name

        with tf.variable_scope(self.name) as _:
            # bias on the left side
            self.b1 = tf.Variable(
                tf.zeros(shape=[self.visible_dim, 1], dtype=tf.float32, name='bias1'))
            # bias on the right side
            self.b2 = tf.Variable(
                tf.zeros(shape=[self.hidden_dim, 1], dtype=tf.float32, name='bias2'))
            # pairwise weight
            self.w = tf.Variable(tf.zeros(
                shape=[self.visible_dim, self.hidden_dim], dtype=tf.float32, name='pairwise'))

        # sampling options
        self.num_samples = num_samples
        self.use_qupa = use_qupa

        # concat b
        self.b = tf.concat(values=[tf.squeeze(self.b1),
                                   tf.squeeze(self.b2)], axis=0)

        if not self.use_qupa:
            # init pcd class implemented in QuPA
            self.sampler = pcd.PCD(left_size=self.visible_dim, right_size=self.hidden_dim,
                                   num_samples=self.num_samples, dtype=tf.float32)
        else:
            # init population annealing class in QuPA
            self.sampler = qupa.PopulationAnnealer(left_size=self.visible_dim, right_size=self.hidden_dim,
                                                   num_samples=self.num_samples, dtype=tf.float32)

        # This returns a scalar tensor with the gradient of log z. Don't trust its value.
        self.log_z_train = self.sampler.training_log_z(
            self.b, self.w, num_mcmc_sweeps=num_gibbs_iter
        )

        # This returns the internal log z variable in QuPA sampler. We wil use this variable in evaluation.
        self.log_z_value = self.sampler.log_z_var

        # get always the samples after updating train log z
        with tf.control_dependencies([self.log_z_train]):
            self.samples = self.sampler.samples()

        # Define inverse temperatures used for AIS. Increasing the # of betas improves the precision of log z estimates.
        betas = tf.linspace(tf.constant(0.), tf.constant(1.), num=1000)
        # Define log_z estimation for evaluation.
        eval_logz = qupa.ais.evaluation_log_z(
            self.b, self.w, init_biases=None, betas=betas, num_samples=1024
        )

        # Update QuPA internal log z variable with the eval_logz
        self.log_z_update = self.log_z_value.assign(eval_logz)

    def energy_tf(self, samples):
        samples_visible = tf.slice(samples, [0, 0], [-1, self.visible_dim])
        samples_hidden = tf.slice(samples, [0, self.visible_dim], [-1, -1])

        energy = tf.matmul(samples, tf.expand_dims(self.b, 1)) + tf.reduce_sum(
            tf.matmul(samples_visible, self.w) * samples_hidden, 1, keepdims=True
        )
        energy = - tf.squeeze(energy, axis=1)

        return energy

    def log_prob(self, samples, is_training=True):
        return - self.energy_tf(samples) - self.log_z_value

    def sample_reparametrization_variable(self, n, is_training=True):
        samples = sample_gumbel((n, self.dim, 2))
        if not is_training:
            samples = np.reshape(samples, (-1, 2))
            samples = np.asarray(np.equal(
                samples, np.max(samples, 1, keepdims=True)
            ), dtype=samples.dtype)
            samples = np.reshape(samples, (-1, self.dim, 2))

        return samples

    def inverse_reparametrize(self, epsilon, parameters):
        assert("logits" in parameters and "temperature" in parameters)

        logits = parameters["logits"]
        logits = tf.reshape(logits, (-1, 2))

        res = tf.reshape(epsilon, (-1, 2))
        res = (logits + res) / parameters["temperature"]
        res = tf.nn.softmax(res)
        res = tf.reshape(res, (-1, self.dim, 2))

        return res

    def kl_from_prior(self, parameters, eps=1e-20):
        assert("logits" in parameters and "zeta" in parameters)

        logits = parameters["logits"]
        logits = tf.reshape(logits, (-1, 2))
        logits = tf.nn.softmax(logits)
        logits = tf.reshape(logits, (-1, self.dim, 2))

        zeta = parameters["zeta"]

        log_posterior = tf.reduce_sum(
            tf.log(tf.reduce_sum(zeta * logits, axis=-1)),
            axis=-1
        )
        log_prior_un = - self.energy_tf(zeta[:, :, 0])

        kl = tf.reduce_mean(log_prior_un - log_posterior) - self.log_z_train

        return kl
