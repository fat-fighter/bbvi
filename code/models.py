import priors
import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from includes.utils import sigmoid
from includes.network import FeedForwardNetwork, GraphConvolutionalNetwork


class VAE:
    def __init__(self, name, input_type, input_dim, latent_dim,
                 activation=None, initializer=None):
        self.name = name

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.input_type = input_type

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

            encoder_network = FeedForwardNetwork(
                name="encoder_network",
                activation=self.activation,
                initializer=self.initializer
            )

            self.mean, self.log_var = encoder_network.build(
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
                [("decoded_X", self.input_dim)
                 ], decoder_layer_sizes, self.Z
            )

            if self.input_type is None:
                self.reconstructed_X = self.decoded_X
            elif self.input_type == "real":
                self.reconstructed_X = self.decoded_X
            elif self.input_type == "binary":
                self.reconstructed_X = tf.nn.sigmoid(self.decoded_X)

        return self

    def sample_reparametrization_variables(self, n, feed=False):
        samples = dict()
        if not feed:
            for lv, eps, _ in self.latent_variables.values():
                samples[eps] = lv.sample_reparametrization_variable(n)
        else:
            for name, (lv, _, _) in self.latent_variables.items():
                samples[name] = lv.sample_reparametrization_variable(n)

        return samples

    def sample_generative_feed(self, n, name_index=False, **kwargs):
        samples = dict()
        if name_index:
            for name, (lv, eps, _) in self.latent_variables.items():
                kwargs_ = dict() if name not in kwargs else kwargs[name]
                samples[eps] = lv.sample_generative_feed(n, **kwargs_)
        else:
            for name, (lv, _, _) in self.latent_variables.items():
                kwargs_ = dict() if name not in kwargs else kwargs[name]
                samples[name] = lv.sample_generative_feed(n, **kwargs_)

        return samples

    def define_latent_loss(self):
        self.latent_loss = tf.add_n(
            [lv.kl_from_prior(params)
             for lv, _, params in self.latent_variables.values()]
        )

    def define_recon_loss(self):
        if self.input_type is None:
            pass
        elif self.input_type == "binary":
            self.recon_loss = tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.X,
                    logits=self.decoded_X
                ), axis=1
            ))
        elif self.input_type == "real":
            self.recon_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(
                tf.square(self.X - self.decoded_X), axis=1
            ))
        else:
            raise NotImplementedError

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
    def __init__(self, name, input_type, input_dim, latent_dim, n_classes,
                 activation=None, initializer=None):
        VAE.__init__(self, name, input_type, input_dim, latent_dim,
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

            encoder_network = FeedForwardNetwork(
                name="encoder_network",
                activation=self.activation,
                initializer=self.initializer
            )

            logits = encoder_network.build(
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

        return self


class GumboltVAE(VAE):
    def __init__(self, name, input_type, input_dim, visible_dim, hidden_dim,
                 num_gibbs_samples=200, gibbs_sampling_gap=10,
                 activation=None, initializer=None):
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        VAE.__init__(self, name, input_type, input_dim, visible_dim + hidden_dim,
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
                0.2, shape=(), name="temperature"
            )

            encoder_network = FeedForwardNetwork(
                name="encoder_network",
                activation=self.activation,
                initializer=self.initializer
            )
            self.log_ratios = encoder_network.build(
                [("log_ratios", self.latent_dim)],
                encoder_layer_sizes, self.X
            )

            self.latent_variables = {
                "Z": (
                    priors.RBMPrior(
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

        return self

    def generate_prior_samples(self, session, n=None, g=None):
        if n is None:
            n = self.num_gibbs_samples
        if g is None:
            g = self.gibbs_sampling_gap

        return self.latent_variables["Z"][0].generate_gibbs_samples(
            session, n, g
        )

    def train_op(self, session, rbm_prior_samples, data):
        assert(self.train_step is not None)

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
    def __init__(self, name, input_dim, latent_dim,
                 activation=None, initializer=None):
        VAE.__init__(self, name, None, input_dim, latent_dim,
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

            self.bias = tf.get_variable(
                "bias", shape=(1,), dtype=tf.float32,
                initializer=tf.initializers.zeros
            )

            self.epsilon = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim),
                name="reparametrization_variable"
            )

            self.dropout = tf.placeholder_with_default(
                0.0, shape=(), name="dropout"
            )

            encoder_network = GraphConvolutionalNetwork(
                name="encoder_network",
                dropout=self.dropout,
                activation=self.activation,
                initializer=self.initializer
            )
            self.mean, self.log_var = encoder_network.build(
                self.input_dim,
                [("mean", self.latent_dim), ("log_var", self.latent_dim)],
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

            features_dim = decoder_layer_sizes[-1]
            decoder_layer_sizes = decoder_layer_sizes[:-1]

            self.decoder_network = FeedForwardNetwork(
                name="decoder_network",
                activation=self.activation,
                initializer=self.initializer
            )
            self.node_features = self.decoder_network.build(
                [("node_features", features_dim)], decoder_layer_sizes, self.Z
            )

            self.link_weights = tf.matmul(
                self.node_features, self.node_features, transpose_b=True
            ) + self.bias

            self.preds = tf.reshape(self.link_weights, (-1,))
            self.labels = tf.reshape(self.A_orig, (-1,))

            correct_prediction = tf.equal(
                tf.cast(tf.greater_equal(
                    tf.sigmoid(self.preds), 0.5), tf.int32),
                tf.cast(self.labels, tf.int32)
            )
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

        return self

    def define_recon_loss(self):
        num_nodes = tf.cast(tf.shape(self.A)[0], dtype=tf.float32)
        num_edges = tf.reduce_sum(self.A)

        pos_weight = (num_nodes ** 2 - num_edges) / num_edges
        norm = num_nodes ** 2 / (num_nodes ** 2 - num_edges) / 2

        self.recon_loss = num_nodes * norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                targets=self.labels,
                logits=self.preds,
                pos_weight=pos_weight
            )
        )

    def train_op(self, session, data):
        assert(self.train_step is not None)

        loss = 0.0
        feed = {
            self.A: data.adj_norm,
            self.A_orig: data.adj_label,
            self.X: data.features,
            self.dropout: 0.2
        }
        feed.update(
            self.sample_reparametrization_variables(len(data.features))
        )

        loss, acc, _ = session.run(
            [self.loss, self.accuracy, self.train_step],
            feed_dict=feed
        )
        return loss, acc

    def get_roc_score(self, session, train_data, test_data):
        feed = {
            self.A: train_data.adj_norm,
            self.A_orig: train_data.adj_label,
            self.X: train_data.features,
            self.dropout: 0.0
        }

        Z = session.run(self.mean, feed_dict=feed)

        adj_rec = session.run(self.link_weights, feed_dict={self.Z: Z})
        adj_rec = sigmoid(adj_rec)

        adj_orig = train_data.adj_orig

        preds = []
        pos = []
        for e in test_data.edges_pos:
            preds.append(adj_rec[e[0], e[1]])
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in test_data.edges_neg:
            preds_neg.append(adj_rec[e[0], e[1]])
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score


class DGLFRM(GVAE):
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

            self.epsilon_real = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim),
                name="real_reparametrization_variable"
            )
            self.epsilon_binary = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim, 2),
                name="binary_reparametrization_variable"
            )

            self.temperature = tf.placeholder_with_default(
                0.1, shape=(), name="temperature"
            )
            self.dropout = tf.placeholder_with_default(
                0.0, shape=(), name="dropout"
            )

            self.bias = tf.get_variable(
                "bias", shape=(1,), dtype=tf.float32,
                initializer=tf.initializers.zeros
            )

            real_encoder_network = GraphConvolutionalNetwork(
                name="real_encoder_network",
                dropout=self.dropout,
                activation=self.activation,
                initializer=self.initializer
            )
            self.mean, self.log_var = real_encoder_network.build(
                self.input_dim,
                [("mean", self.latent_dim), ("log_var", self.latent_dim)],
                encoder_layer_sizes, self.A, self.X
            )

            binary_encoder_network = GraphConvolutionalNetwork(
                name="binary_encoder_network",
                dropout=self.dropout,
                activation=self.activation,
                initializer=self.initializer
            )
            logits = binary_encoder_network.build(
                self.input_dim, [("logits", self.latent_dim * 2)],
                encoder_layer_sizes, self.A, self.X
            )
            self.logits = tf.reshape(
                logits, (-1, self.latent_dim, 2)
            )

            self.latent_variables = {
                "Z_real": (
                    priors.NormalFactorial(
                        "latent_representation", self.latent_dim
                    ), self.epsilon_real,
                    {"mean": self.mean, "log_var": self.log_var}
                ),
                "Z_binary": (
                    priors.DiscreteFactorial(
                        "latent_representation", self.latent_dim, 2
                    ), self.epsilon_binary,
                    {"logits": self.logits, "temperature": self.temperature}
                )
            }

            lv, eps, params = self.latent_variables["Z_real"]
            self.Z_real = lv.inverse_reparametrize(eps, params)

            lv, eps, params = self.latent_variables["Z_binary"]
            self.Z_binary = lv.inverse_reparametrize(eps, params)

            self.Z = self.Z_binary * self.Z_real

            features_dim = decoder_layer_sizes[-1]
            decoder_layer_sizes = decoder_layer_sizes[:-1]

            self.decoder_network = FeedForwardNetwork(
                name="decoder_network",
                activation=self.activation,
                initializer=self.initializer
            )
            self.node_features = self.decoder_network.build(
                [("node_features", features_dim)], decoder_layer_sizes, self.Z
            )

            self.link_weights = tf.matmul(
                self.node_features, self.node_features, transpose_b=True
            ) + self.bias

            self.preds = tf.reshape(self.link_weights, (-1,))
            self.labels = tf.reshape(self.A_orig, (-1,))

            correct_prediction = tf.equal(
                tf.cast(tf.greater_equal(
                    tf.sigmoid(self.preds), 0.5), tf.int32),
                tf.cast(self.labels, tf.int32)
            )
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

        return self

    def get_roc_score(self, session, train_data, test_data):
        feed = {
            self.A: train_data.adj_norm,
            self.A_orig: train_data.adj_label,
            self.X: train_data.features,
            self.dropout: 0.0
        }

        Z_real = session.run(self.mean, feed_dict=feed)

        Z_binary = session.run(self.logits, feed_dict=feed)

        shape = list(Z_binary.shape)

        Z_binary = np.reshape(Z_binary, (-1, 2))
        Z_binary = np.asarray(np.equal(
            Z_binary, np.max(Z_binary, 1, keepdims=True)
        ), dtype=Z_binary.dtype)
        Z_binary = Z_binary[:, 0]
        Z_binary = np.reshape(Z_binary, shape[:-1])

        adj_rec = session.run(
            self.link_weights, feed_dict={self.Z: Z_real * Z_binary}
        )
        adj_rec = sigmoid(adj_rec)

        adj_orig = train_data.adj_orig

        preds = []
        pos = []
        for e in test_data.edges_pos:
            preds.append(adj_rec[e[0], e[1]])
            pos.append(adj_orig[e[0], e[1]])

        preds_neg = []
        neg = []
        for e in test_data.edges_neg:
            preds_neg.append(adj_rec[e[0], e[1]])
            neg.append(adj_orig[e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score


class GumboltVGAE(GVAE):
    def __init__(self, name, input_dim, visible_dim, hidden_dim,
                 num_gibbs_samples=200, gibbs_sampling_gap=10,
                 activation=None, initializer=None):
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim

        GVAE.__init__(self, name, input_dim, visible_dim + hidden_dim,
                      activation=activation, initializer=initializer)

        self.num_gibbs_samples = num_gibbs_samples
        self.gibbs_sampling_gap = gibbs_sampling_gap

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

            self.epsilon_real = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim),
                name="real_reparametrization_variable"
            )
            self.epsilon_binary = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim),
                name="binary_reparametrization_variable"
            )
            self.rbm_prior_samples = tf.placeholder(
                tf.float32, shape=(None, self.latent_dim), name="rbm_prior_samples"
            )

            self.temperature = tf.placeholder_with_default(
                0.2, shape=(), name="temperature"
            )
            self.dropout = tf.placeholder_with_default(
                0.0, shape=(), name="dropout"
            )

            self.bias = tf.get_variable(
                "bias", shape=(1,), dtype=tf.float32,
                initializer=tf.initializers.zeros
            )

            real_encoder_network = GraphConvolutionalNetwork(
                name="real_encoder_network",
                dropout=self.dropout,
                activation=self.activation,
                initializer=self.initializer
            )
            self.mean, self.log_var = real_encoder_network.build(
                self.input_dim,
                [("mean", self.latent_dim), ("log_var", self.latent_dim)],
                encoder_layer_sizes, self.A, self.X
            )

            binary_encoder_network = GraphConvolutionalNetwork(
                name="binary_encoder_network",
                dropout=self.dropout,
                activation=self.activation,
                initializer=self.initializer
            )
            self.log_ratios = binary_encoder_network.build(
                self.input_dim,
                [("log_ratios", self.latent_dim)],
                encoder_layer_sizes, self.A, self.X
            )

            self.latent_variables = {
                "Z_real": (
                    priors.NormalFactorial(
                        "latent_representation", self.latent_dim
                    ), self.epsilon_real,
                    {"mean": self.mean, "log_var": self.log_var}
                ),
                "Z_binary": (
                    priors.RBMPrior(
                        "rbm_prior", self.visible_dim, self.hidden_dim, beta=10.0, trainable=True
                    ), self.epsilon_binary,
                    {"log_ratios": self.log_ratios, "temperature": self.temperature,
                        "samples": self.rbm_prior_samples}
                )
            }

            lv, eps, params = self.latent_variables["Z_real"]
            self.Z_real = lv.inverse_reparametrize(eps, params)

            lv, eps, params = self.latent_variables["Z_binary"]
            self.Z_binary = lv.inverse_reparametrize(eps, params)
            self.latent_variables["Z_binary"][2]["zeta"] = self.Z_binary

            self.Z = self.Z_binary * self.Z_real

            # features_dim = decoder_layer_sizes[-1]
            # decoder_layer_sizes = decoder_layer_sizes[:-1]

            # self.decoder_network = FeedForwardNetwork(
            #     name="decoder_network",
            #     activation=self.activation,
            #     initializer=self.initializer
            # )
            # self.node_features = self.decoder_network.build(
            #     [("node_features", features_dim)], decoder_layer_sizes, self.Z
            # )

            # self.link_weights = tf.matmul(
            #     self.node_features, self.node_features, transpose_b=True
            # ) + self.bias

            self.link_weights = tf.matmul(
                self.Z, self.Z, transpose_b=True
            ) + self.bias

            self.preds = tf.reshape(self.link_weights, (-1,))
            self.labels = tf.reshape(self.A_orig, (-1,))

            correct_prediction = tf.equal(
                tf.cast(tf.greater_equal(
                    tf.sigmoid(self.preds), 0.5), tf.int32),
                tf.cast(self.labels, tf.int32)
            )
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

        return self

    def get_roc_score(self, session, train_data, test_data):
        feed = {
            self.A: train_data.adj_norm,
            self.A_orig: train_data.adj_label,
            self.X: train_data.features,
            self.dropout: 0.0
        }

        Z_real = session.run(self.mean, feed_dict=feed)

        Z_binary = session.run(self.log_ratios, feed_dict=feed)
        Z_binary = np.round(sigmoid(Z_binary))

        adj_rec = session.run(
            self.link_weights, feed_dict={self.Z: Z_real * Z_binary}
        )
        adj_rec = sigmoid(adj_rec)

        edges_u, edges_v = zip(*test_data.edges_pos)
        preds_pos = adj_rec[edges_u, edges_v]

        edges_u, edges_v = zip(*test_data.edges_neg)
        preds_neg = adj_rec[edges_u, edges_v]

        preds_all = np.hstack([preds_pos, preds_neg])
        labels_all = np.hstack([
            np.ones(len(test_data.edges_pos)),
            np.zeros(len(test_data.edges_neg))
        ])

        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score

    def generate_prior_samples(self, session):
        return self.latent_variables["Z_binary"][0].generate_gibbs_samples(
            session, self.num_gibbs_samples, self.gibbs_sampling_gap
        )

    def train_op(self, session, rbm_prior_samples, data):
        assert(self.train_step is not None)

        feed = {
            self.A: data.adj_norm,
            self.A_orig: data.adj_label,
            self.X: data.features,
            self.dropout: 0.2
        }
        feed.update(
            self.sample_reparametrization_variables(len(data.features))
        )
        feed[self.rbm_prior_samples] = rbm_prior_samples

        loss, acc, _ = session.run(
            [self.loss, self.accuracy, self.train_step],
            feed_dict=feed
        )
        return loss, acc
