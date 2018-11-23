import os
import numpy as np
import tensorflow as tf
import matplotlib as mpl

from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
from matplotlib import gridspec as grid


mpl.rc_file_defaults()


def mnist_regeneration_plot(model, data, sess):
    if not os.path.exists("plots/%s/mnist" % model.name):
        os.makedirs("plots/%s/mnist" % model.name)

    gs = grid.GridSpec(1, 2)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    def reshape(images):
        images = images.reshape((10, 10, 28, 28))
        images = np.concatenate(np.split(images, 10, axis=0), axis=3)
        images = np.concatenate(np.split(images, 10, axis=1), axis=2)
        images = np.squeeze(images)

        return images

    orig_X = data.data[:100]

    feed = model.sample_reparametrization_variables(len(orig_X))
    feed.update({
        model.X: orig_X,
    })

    recn_X = sess.run(model.reconstructed_X, feed_dict=feed)

    ax1.imshow(reshape(orig_X), cmap="Greys_r")
    ax2.imshow(reshape(recn_X), cmap="Greys_r")

    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig("plots/%s/mnist/regenerated.png" %
                model.name, transparent=True)
    plt.close()


def mnist_sample_plot(model, sess):
    if not os.path.exists("plots/%s/mnist" % model.name):
        os.makedirs("plots/%s/mnist" % model.name)

    kwargs = {"Z": {"session": sess}}
    sample_Z = model.sample_generative_feed(100, **kwargs)["Z"]

    figure = sess.run(model.reconstructed_X, feed_dict={model.Z: sample_Z})
    figure = np.concatenate(
        np.concatenate(
            np.reshape(figure, (10, 10, 28, 28)),
            axis=-1
        ), axis=0
    )
    plt.imshow(figure, cmap="Greys_r")

    ax = plt.axes()

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    plt.savefig("plots/%s/mnist/sampled.png" % model.name, transparent=True)
    plt.close()


def spiral_regeneration_plot(model, data, sess):
    if not os.path.exists("plots/%s/spiral" % model.name):
        os.makedirs("plots/%s/spiral" % model.name)

    gs = grid.GridSpec(1, 2)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    orig_X = data.data

    feed = model.sample_reparametrization_variables(len(orig_X))
    feed.update({
        model.X: orig_X,
    })

    recn_X = sess.run(model.reconstructed_X, feed_dict=feed)

    ax1.scatter(orig_X[:, 0], orig_X[:, 1], s=0.5)
    ax2.scatter(recn_X[:, 0], recn_X[:, 1], s=0.5)

    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig("plots/%s/spiral/regenerated.png" %
                model.name, transparent=True)
    plt.close()


def spiral_sample_plot(model, sess):
    if not os.path.exists("plots/%s/spiral" % model.name):
        os.makedirs("plots/%s/spiral" % model.name)

    kwargs = {"Z": {"session": sess}}
    sample_Z = model.sample_generative_feed(100, **kwargs)["Z"]

    out = sess.run(model.reconstructed_X, feed_dict={model.Z: sample_Z})
    plt.scatter(out[:, 0], out[:, 1], s=0.5)

    ax = plt.axes()

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout()
    plt.savefig("plots/%s/spiral/sampled.png" % model.name)
    plt.close()
