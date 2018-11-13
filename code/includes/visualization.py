import numpy as np
import tensorflow as tf
import matplotlib as mpl

from matplotlib import pyplot as plt
from matplotlib import gridspec as grid


mpl.rc_file_defaults()


def mnist_regeneration_plot(model, sess, data):
    gs = grid.GridSpec(1, 2)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    def reshape(images):
        images = images.reshape((10, 10, 28, 28))
        images = np.concatenate(np.split(images, 10, axis=0), axis=3)
        images = np.concatenate(np.split(images, 10, axis=1), axis=2)
        images = np.squeeze(images)

        return images

    eps = model.sample_reparametrization_variables(100, feed=True)["Z"]

    orig_X = data.data[:100]
    recn_X = sess.run(
        model.reconstructed_X, feed_dict={
            model.X: orig_X,
            model.epsilon: eps
        }
    )

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
    plt.savefig("plots/regenerated.png", transparent=True)


def mnist_sample_plot(model, sess):
    figure = np.zeros((280, 280))

    kwargs = {
        "Z": {"session": sess}
    }

    eps = model.sample_generative_feed(100, **kwargs)["Z"]
    out = sess.run(model.reconstructed_X, feed_dict={vae.Z: eps})

    for i in range(10):
        for j in range(10):
            figure[i * 28: (i + 1) * 28, j * 28: (j + 1) *
                   28] = out[10 * i + j].reshape((28, 28)) * 255

    ax = plt.axes()
    ax.imshow(figure, cmap="Greys_r")

    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.savefig("plots/sampled.png", transparent=True)

    plt.close()
