import priors
import tensorflow as tf

from includes.network import FeedForwardNetwork, GraphConvolutionalNetwork


class VAE:
    def __init__(self, name, input_dim, latent_dim, activation=None, initializer=None):
        self.name = name

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.activation = activation
        self.initializer = initializer

        self.train_step = None

    def build_graph(self, encoder_layer_sizes, decoder_layer_sizes):
        with tf.variable_scope(self.name) as _:
            self.X = tf.placeholder(tf.float32, shape=(
                None, self.input_dim), name="X")
            self.epsilon = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim), name="reparametrization_variable"
            )

            self.encoder_network = FeedForwardNetwork(
                name="encoder_network",
                activation=self.activation,
                initializer=self.initializer
            )

            self.mean, self.log_var = self.encoder_network.build(
                [("mean", self.latent_dim), ("log_var", 10)],
                encoder_layer_sizes, self.X
            )

            self.latent_variables = {
                "Z": (
                    priors.NormalFactorial(
                        "latent_representation", self.latent_dim
                    ), self.epsilon,
                    {"mean": self.mean, "log_var": self.log_var}
                )
            }

            lv, eps, params = self.latent_variables["Z"]
            self.Z = lv.inverse_reparametrize(eps, params)

            self.decoder_network = FeedForwardNetwork(
                name="decoder_network",
                activation=self.activation,
                initializer=self.initializer
            )
            self.decoded_X = self.decoder_network.build(
                [("decoded_X", self.input_dim)], decoder_layer_sizes, self.Z
            )
            self.reconstructed_X = tf.nn.sigmoid(self.decoded_X)

    def sample_reparametrization_variables(self, n, feed=False):
        samples = dict()
        if not feed:
            for lv, eps, _ in self.latent_variables.itervalues():
                samples[eps] = lv.sample_reparametrization_variable(n)
        else:
            for name, (lv, _, _) in self.latent_variables.iteritems():
                samples[name] = lv.sample_reparametrization_variable(n)

        return samples

    def sample_generative_feed(self, n, **kwargs):
        samples = dict()
        for name, (lv, _, _) in self.latent_variables.iteritems():
            kwargs_ = dict() if name not in kwargs else kwargs[name]
            samples[name] = lv.sample_generative_feed(n, **kwargs_)

        return samples

    def define_latent_loss(self):
        self.latent_loss = tf.add_n(
            [lv.kl_from_prior(params)
             for lv, _, params in self.latent_variables.itervalues()]
        )

    def define_recon_loss(self):
        self.recon_loss = tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.X,
                logits=self.decoded_X
            ), axis=1
        ))

    def define_train_loss(self):
        self.define_latent_loss()
        self.define_recon_loss()

        self.loss = tf.reduce_mean(self.recon_loss + self.latent_loss)

    def define_train_step(self, init_lr, decay_steps, decay_rate=0.9):
        learning_rate = tf.train.exponential_decay(
            learning_rate=init_lr,
            global_step=0,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

        self.define_train_loss()
        self.train_step = tf.train.AdamOptimizer(
            learning_rate=learning_rate
        ).minimize(self.loss)

    def train_op(self, session, data):
        assert(self.train_step is not None)

        loss = 0.0
        for batch in data.get_batches():
            feed = {
                self.X: batch
            }
            feed.update(
                self.sample_reparametrization_variables(len(batch))
            )

            batch_loss, _ = session.run(
                [self.loss, self.train_step],
                feed_dict=feed
            )
            loss += batch_loss / data.epoch_len

        return loss


class DiscreteVAE(VAE):
    def __init__(self, name, input_dim, latent_dim, n_classes, activation=None, initializer=None):
        VAE.__init__(self, name, input_dim, latent_dim,
                     activation=activation, initializer=initializer)

        self.n_classes = n_classes

    def build_graph(self, encoder_layer_sizes, decoder_layer_sizes):
        with tf.variable_scope(self.name) as _:
            self.X = tf.placeholder(
                tf.float32, shape=(None, self.input_dim), name="X"
            )
            self.epsilon = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim, self.n_classes), name="epsilon_Z"
            )
            self.temperature = tf.placeholder_with_default(
                [0.1], shape=(1,), name="temperature"
            )

            self.encoder_network = FeedForwardNetwork(
                name="encoder_network",
                activation=self.activation,
                initializer=self.initializer
            )

            logits = self.encoder_network.build(
                [("logits", self.latent_dim * self.n_classes)],
                encoder_layer_sizes, self.X
            )
            self.logits = tf.reshape(
                logits, (-1, self.latent_dim, self.n_classes)
            )

            self.latent_variables = {
                "Z": (
                    priors.DiscreteFactorial(
                        "discrete-prior", self.latent_dim, self.n_classes
                    ), self.epsilon,
                    {"logits": self.logits, "temperature": self.temperature}
                )
            }

            lv, eps, params = self.latent_variables["Z"]
            self.Z = lv.inverse_reparametrize(eps, params)

            self.decoder_network = FeedForwardNetwork(
                name="decoder_network",
                activation=self.activation,
                initializer=self.initializer
            )
            self.decoded_X = self.decoder_network.build(
                [("decoded_X", self.input_dim)], decoder_layer_sizes, self.Z
            )
            self.reconstructed_X = tf.nn.sigmoid(self.decoded_X)


class GumboltVAE(VAE):
    def __init__(self, name, input_dim, visible_dim, hidden_dim, num_gibbs_samples=200,
                 gibbs_sampling_gap=10, activation=None, initializer=None):
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        VAE.__init__(self, name, input_dim, visible_dim + hidden_dim,
                     activation=activation, initializer=initializer)

        self.num_gibbs_samples = num_gibbs_samples
        self.gibbs_sampling_gap = gibbs_sampling_gap

    def build_graph(self, encoder_layer_sizes, decoder_layer_sizes):
        with tf.variable_scope(self.name) as _:
            self.X = tf.placeholder(
                tf.float32, shape=(None, self.input_dim), name="X"
            )
            self.epsilon = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim), name="epsilon_Z"
            )
            self.rbm_prior_samples = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim), name="rbm_prior_samples"
            )
            self.temperature = tf.placeholder_with_default(
                [0.5], shape=(1,), name="temperature"
            )

            self.encoder_network = FeedForwardNetwork(
                name="encoder_network",
                activation=self.activation,
                initializer=self.initializer
            )
            self.log_ratios = self.encoder_network.build(
                [("log_ratios", self.latent_dim)],
                encoder_layer_sizes, self.X
            )

            self.latent_variables = {
                "Z": (
                    priors.RBM(
                        "rbm_prior", self.visible_dim, self.hidden_dim, trainable=True
                    ), self.epsilon,
                    {"log_ratios": self.log_ratios, "temperature": self.temperature,
                        "samples": self.rbm_prior_samples}
                )
            }

            lv, eps, params = self.latent_variables["Z"]
            self.Z = lv.inverse_reparametrize(eps, params)
            self.latent_variables["Z"][2]["zeta"] = self.Z

            self.decoder_network = FeedForwardNetwork(
                name="decoder_network",
                activation=self.activation,
                initializer=self.initializer
            )
            self.decoded_X = self.decoder_network.build(
                [("decoded_X", self.input_dim)], decoder_layer_sizes, self.Z
            )
            self.reconstructed_X = tf.nn.sigmoid(self.decoded_X)

    def generate_prior_samples(self, session):
        return self.latent_variables["Z"][0].generate_gibbs_samples(
            session, self.num_gibbs_samples, self.gibbs_sampling_gap
        )

    def train_op(self, session, data):
        assert(self.train_step is not None)

        rbm_prior_samples = self.generate_prior_samples(session)

        loss = 0.0
        for batch in data.get_batches():
            feed = {
                self.X: batch
            }
            feed.update(
                self.sample_reparametrization_variables(len(batch))
            )
            feed[self.rbm_prior_samples] = rbm_prior_samples

            batch_loss, _ = session.run(
                [self.loss, self.train_step],
                feed_dict=feed
            )
            loss += batch_loss / data.epoch_len

        return loss


class GVAE(VAE):
    def __init__(self, name, input_dim, latent_dim, activation=None, initializer=None):
        VAE.__init__(self, name, input_dim, latent_dim,
                     activation=activation, initializer=initializer)

    def build_graph(self, encoder_layer_sizes, decoder_layer_sizes):
        with tf.variable_scope(self.name) as _:
            self.X = tf.placeholder(tf.float32, name="X")

            self.A = tf.placeholder(
                tf.float32, shape=(None, None),
                name="adjacency_matrix"
            )
            self.A_orig = tf.placeholder(
                tf.float32, shape=(None, None),
                name="adjacency_matrix_orig"
            )

            self.epsilon = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim),
                name="reparametrization_variable"
            )

            self.encoder_network = GraphConvolutionalNetwork(
                name="encoder_network",
                activation=self.activation,
                initializer=self.initializer
            )

            self.mean, self.log_var = self.encoder_network.build(
                self.input_dim,
                [("mean", self.latent_dim), ("log_var", 10)],
                encoder_layer_sizes, self.A, self.X
            )

            self.latent_variables = {
                "Z": (
                    priors.NormalFactorial(
                        "latent_representation", self.latent_dim
                    ), self.epsilon,
                    {"mean": self.mean, "log_var": self.log_var}
                )
            }

            lv, eps, params = self.latent_variables["Z"]
            self.Z = lv.inverse_reparametrize(eps, params)

            self.decoder_network = FeedForwardNetwork(
                name="decoder_network",
                activation=self.activation,
                initializer=self.initializer
            )
            self.node_features = self.decoder_network.build(
                [("node_features", self.input_dim)], decoder_layer_sizes, self.Z
            )

            self.link_weights = tf.matmul(
                self.node_features, self.node_features, transpose_b=True
            )

    def define_recon_loss(self):
        shape = tf.cast(tf.shape(self.A)[0], dtype=tf.float32)
        num_edges = tf.reduce_sum(self.A)

        pos_weight = (shape ** 2 - num_edges) / num_edges
        norm = shape ** 2 / (shape ** 2 - num_edges) / 2

        self.recon_loss = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                targets=tf.reshape(self.A_orig, (-1,)),
                logits=tf.reshape(self.link_weights, (-1,)),
                pos_weight=pos_weight
            )
        )

    def train_op(self, session, data):
        assert(self.train_step is not None)

        loss = 0.0
        feed = {
            self.A: data["adj_norm"],
            self.A_orig: data["adj_label"],
            self.X: data["features"],
        }
        feed.update(
            self.sample_reparametrization_variables(len(data["features"]))
        )

        loss, _ = session.run(
            [self.loss, self.train_step],
            feed_dict=feed
        )
        return loss
