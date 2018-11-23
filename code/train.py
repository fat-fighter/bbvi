import os
import models
import numpy as np
import tensorflow as tf
import matplotlib as mpl

from tqdm import tqdm

from absl import app
from absl import flags

from matplotlib import pyplot as plt
from matplotlib import gridspec as grid

import includes.visualization as visualization
from includes.utils import load_data, Dataset

mpl.rc_file_defaults()

tf.logging.set_verbosity(tf.logging.ERROR)


FLAGS = flags.FLAGS

flags.DEFINE_string("model", "gumbolt-vae",
                    "Model to use [vae, dvae, gumbolt-vae, gvae]")
flags.DEFINE_string("datagroup", "mnist",
                    "Datagroup to use [mnist, spiral, graph]")
flags.DEFINE_string("dataset", "static",
                    "Dataset to use {mnist:[static], spiral:[normal], graph:[citeseer]}")

flags.DEFINE_integer("latent_dim", 10,
                     "Number of dimensions for latent variable Z")
flags.DEFINE_integer("hidden_dim", 50,
                     "Number of hidden dimensions for latent variable Z")
flags.DEFINE_integer("visible_dim", 50,
                     "Number of visible dimensions for latent variable Z")

flags.DEFINE_integer("n_clusters", 10,
                     "Number of clusters for Discrete Latent Variables")

flags.DEFINE_integer("n_epochs", 500, "Number of epochs for training a model")

flags.DEFINE_integer("gibbs_sampling_gap", 1,
                     "Gap between consecutive gibbs samples")
flags.DEFINE_integer("num_gibbs_samples", 100,
                     "Number of gibbs samples to use")
flags.DEFINE_integer("rbm_sampling_gap", 1,
                     "Number of epochs sharing same rbm samples")

flags.DEFINE_list("encoder_size", [256, 256, 512],
                  "Layer sizes for the encoder network")
flags.DEFINE_list("decoder_size", [512, 256, 256],
                  "Layer sizes for the decoder network")

flags.DEFINE_boolean("plotting", True,
                     "Whether to generate sampling and regeneration plots")
flags.DEFINE_integer("plot_epochs", 100,
                     "Nummber of epochs before generating plots")

flags.DEFINE_float("init_lr", 0.002, "Intial Learning Rate")
flags.DEFINE_float("decay_rate", 0.9, "Learning Rate Decay Rate")
flags.DEFINE_float("decay_steps", 10.0, "Learning Rate Decay Steps")


def sample_gumbolt_plot(model, session):
    figure = []
    for _ in range(20):
        model.generate_prior_samples(session, 1, 100)
        rbm_prior_samples = model.generate_prior_samples(session, 5, 1)

        X = session.run(
            model.reconstructed_X, feed_dict={model.Z: rbm_prior_samples}
        )
        X = np.concatenate(np.reshape(X, (5, 28, 28)), axis=-1)

        figure.append(X)

    figure_left = np.concatenate(figure[:10], axis=0)
    figure_right = np.concatenate(figure[10:], axis=0)

    figure = np.concatenate([figure_left, figure_right], axis=-1)

    plt.imshow(figure, cmap="Greys_r")

    ax = plt.axes()

    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig("plots/%s/mnist/rbm_sampled.png" % model.name)
    plt.clf()


def main(argv):
    datagroup = FLAGS.datagroup
    dataset = FLAGS.dataset

    model_str = FLAGS.model
    latent_dim = FLAGS.latent_dim

    n_clusters = FLAGS.n_clusters

    hidden_dim = FLAGS.hidden_dim
    visible_dim = FLAGS.visible_dim

    plotting = FLAGS.plotting
    plot_epochs = FLAGS.plot_epochs

    init_lr = FLAGS.init_lr
    decay_rate = FLAGS.decay_rate
    decay_steps = FLAGS.decay_steps

    encoder_layer_sizes = FLAGS.encoder_size
    decoder_layer_sizes = FLAGS.decoder_size

    rbm_sampling_gap = FLAGS.rbm_sampling_gap
    num_gibbs_samples = FLAGS.num_gibbs_samples
    gibbs_sampling_gap = FLAGS.gibbs_sampling_gap

    n_epochs = FLAGS.n_epochs

    train_data, test_data = load_data(datagroup, dataset)

    train_data = Dataset(train_data, datagroup, type="train", batch_size=200)
    test_data = Dataset(test_data, datagroup, type="test", batch_size=200)

    if datagroup == "mnist":
        sample_plot = visualization.mnist_sample_plot
        regeneration_plot = visualization.mnist_regeneration_plot

    elif datagroup == "spiral":
        sample_plot = visualization.spiral_sample_plot
        regeneration_plot = visualization.spiral_regeneration_plot

    elif datagroup == "graph":
        plotting = False

    else:
        raise NotImplementedError

    if model_str == "vae":
        model = models.VAE(
            model_str, train_data.input_type, train_data.input_dim, latent_dim,
            activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer()
        )

    elif model_str == "dvae":
        model = models.DiscreteVAE(
            model_str, train_data.input_type, train_data.input_dim, latent_dim, n_clusters,
            activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer()
        )

    elif model_str == "gumbolt-vae":
        model = models.GumboltVAE(
            model_str, train_data.input_type, train_data.input_dim, visible_dim, hidden_dim,
            num_gibbs_samples=num_gibbs_samples, gibbs_sampling_gap=gibbs_sampling_gap,
            activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer()
        )

    elif model_str == "gvae":
        model = models.GVAE(
            model_str, train_data.input_dim, latent_dim,
            activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer()
        )

    elif model_str == "dglfrm":
        model = models.DGLFRM(
            model_str, train_data.input_dim, latent_dim,
            activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer()
        )

    elif model_str == "gumbolt-vgae":
        model = models.GumboltVGAE(
            model_str, train_data.input_dim, visible_dim, hidden_dim,
            num_gibbs_samples=num_gibbs_samples, gibbs_sampling_gap=gibbs_sampling_gap,
            activation=tf.nn.relu, initializer=tf.contrib.layers.xavier_initializer()
        )

    else:
        raise NotImplementedError

    model.build_graph(
        encoder_layer_sizes, decoder_layer_sizes
    ).define_train_step(
        init_lr, decay_steps, decay_rate
    )

    sess = tf.Session()
    tf.global_variables_initializer().run(session=sess)

    with tqdm(range(n_epochs), postfix={"loss": "inf"}) as bar:
        for epoch in bar:
            if plotting and epoch % plot_epochs == 0 and epoch != 0:
                sample_plot(model, sess)
                regeneration_plot(model, test_data, sess)

                if epoch % 200 == 0:
                    sample_gumbolt_plot(model, sess)

            if model_str in ["gumbolt-vgae", "gumbolt-vae"]:
                if (epoch > 200 or epoch % rbm_sampling_gap == 0):
                    rbm_prior_samples = model.generate_prior_samples(sess)

                loss = model.train_op(sess, rbm_prior_samples, train_data)

            else:
                loss = model.train_op(sess, train_data)

            if datagroup == "graph":
                roc, ap = model.get_roc_score(sess, train_data, test_data)

                bar.set_postfix(
                    {
                        "loss": "%.4f" % loss[0],
                        "acc": "%.4f" % (loss[1] * 100),
                        "roc": "%2.2f" % (roc * 100),
                        "ap": "%2.2f" % (ap * 100)
                    }
                )

            else:
                bar.set_postfix(
                    {"loss": "%.4f" % loss}
                )

    if plotting:
        sample_plot(model, sess)
        regeneration_plot(model, test_data, sess)

        if model_str in ["gumbolt-vae"]:
            sample_gumbolt_plot(model, sess)


if __name__ == "__main__":
    app.run(main)
