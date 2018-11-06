import priors
import tensorflow as tf

from includes.network import FeedForwardNetwork


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
            self.X = tf.placeholder(tf.float32, shape=(None, self.input_dim))
            self.epsilon = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim))

            self.encoder_network = FeedForwardNetwork(name="encoder_network")

            self.mean, self.log_var = self.encoder_network.build(
                [("mean", self.latent_dim), ("log_var", 10)],
                encoder_layer_sizes, self.X
            )

            self.latent_variables = {
                "Z": (
                    priors.NormalFactorial(self.latent_dim), self.epsilon,
                    {"mean": self.mean, "log_var": self.log_var}
                )
            }

            lv, eps, params = self.latent_variables["Z"]
            self.Z = lv.inverse_reparametrize(eps, params)

            self.decoder_network = FeedForwardNetwork(name="decoder_network")
            self.decoded_X = self.decoder_network.build(
                [("decoded_X", self.input_dim)], decoder_layer_sizes, self.Z
            )
            self.reconstructed_X = tf.nn.sigmoid(self.decoded_X)

    def sample_reparametrization_variables(self, n, feed=False, is_training=True):
        samples = dict()
        if feed:
            for lv, eps, _ in self.latent_variables.itervalues():
                samples[eps] = lv.sample_reparametrization_variable(
                    n, is_training=is_training
                )
        else:
            for name, (lv, _, _) in self.latent_variables.iteritems():
                samples[name] = lv.sample_reparametrization_variable(
                    n, is_training=is_training
                )

        return samples

    def define_train_loss(self):
        self.latent_loss = tf.add_n(
            [lv.kl_from_prior(params)
             for lv, _, params in self.latent_variables.itervalues()]
        )
        self.recon_loss = tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.X,
                logits=self.decoded_X
            ), axis=1
        ))

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
                self.sample_reparametrization_variables(
                    len(batch), feed=True, is_training=True
                )
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
                1.0, shape=None, name="temperature"
            )

            self.encoder_network = FeedForwardNetwork(name="encoder_network")

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
                        self.latent_dim, self.n_classes
                    ), self.epsilon,
                    {"logits": self.logits, "temperature": self.temperature}
                )
            }

            lv, eps, params = self.latent_variables["Z"]
            self.Z = lv.inverse_reparametrize(eps, params)
            self.latent_variables["Z"][2]["Xi"] = self.Z

            self.decoder_network = FeedForwardNetwork(name="decoder_network")
            self.decoded_X = self.decoder_network.build(
                [("decoded_X", self.input_dim)], decoder_layer_sizes, self.Z
            )
            self.reconstructed_X = tf.nn.sigmoid(self.decoded_X)


class GumboltVAE(VAE):
    def __init__(self, name, input_dim, visible_dim, hidden_dim, activation=None, initializer=None):
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        VAE.__init__(self, name, input_dim, visible_dim + hidden_dim,
                     activation=activation, initializer=initializer)

        self.n_classes = 2

    def build_graph(self, encoder_layer_sizes, decoder_layer_sizes):
        with tf.variable_scope(self.name) as _:
            self.X = tf.placeholder(
                tf.float32, shape=(None, self.input_dim), name="X"
            )
            self.epsilon = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim, self.n_classes), name="epsilon_Z"
            )
            self.temperature = tf.placeholder_with_default(
                1.0, shape=None, name="temperature"
            )

            self.encoder_network = FeedForwardNetwork(name="encoder_network")

            logits = self.encoder_network.build(
                [("logits", self.latent_dim * self.n_classes)],
                encoder_layer_sizes, self.X
            )
            self.logits = tf.reshape(
                logits, (-1, self.latent_dim, self.n_classes)
            )

            self.latent_variables = {
                "Z": (
                    priors.RBM(
                        "rbm_prior", self.visible_dim, self.hidden_dim, self.n_classes
                    ), self.epsilon,
                    {"logits": self.logits, "temperature": self.temperature}
                )
            }

            lv, eps, params = self.latent_variables["Z"]
            self.Z = lv.inverse_reparametrize(eps, params)
            self.latent_variables["Z"][2]["zeta"] = self.Z

            self.decoder_network = FeedForwardNetwork(name="decoder_network")
            self.decoded_X = self.decoder_network.build(
                [("decoded_X", self.input_dim)], decoder_layer_sizes, self.Z
            )
            self.reconstructed_X = tf.nn.sigmoid(self.decoded_X)
