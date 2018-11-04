import os
import models
import numpy as np
import tensorflow as tf
import matplotlib as mpl

from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import gridspec as grid

from includes.config import Config
from includes.utils import load_data, Dataset

mpl.rc_file_defaults()

tf.logging.set_verbosity(tf.logging.ERROR)

datagroup = "mnist"
dataset = "binary"

config = Config(datagroup)

train_data, test_data = load_data(datagroup, dataset=dataset)

train_data = Dataset(train_data, batch_size=config.batch_size)
test_data = Dataset(test_data, batch_size=config.batch_size)


def regeneration_plot():
    gs = grid.GridSpec(1, 2)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    def reshape(images):
        images = images.reshape((100, 28, 28))
        images = np.concatenate(images, axis=1)
        images = np.array([images[:, x:x+280]
                           for x in range(0, 2800, 280)])
        images = np.concatenate(images, axis=0)
        images = np.concatenate(
            [np.zeros((280, 10)), images, np.zeros((280, 10))], axis=1
        )
        images = np.concatenate(
            [np.zeros((10, 300)), images, np.zeros((10, 300))], axis=0
        )

        return images

    eps = vae.sample_reparametrization_variables(100, is_training=False)["Z"]

    orig_X = test_data.data[:100]
    recn_X = sess.run(
        vae.reconstructed_X, feed_dict={
            vae.X: orig_X,
            vae.epsilon: eps
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


def sample_plot():
    figure = np.zeros((280, 280))

    for i in range(10):
        for j in range(10):
            eps = vae.sample_reparametrization_variables(
                1, is_training=False)["Z"]
            print sess.run(vae.reconstructed_X, feed_dict={vae.Z: eps})
            out = sess.run(vae.reconstructed_X, feed_dict={vae.Z: eps})

            figure[i * 28: (i + 1) * 28, j * 28: (j + 1) *
                   28] = out.reshape((28, 28)) * 255

    ax = plt.axes()
    ax.imshow(figure, cmap="Greys_r")

    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.savefig("plots/sampled.png", transparent=True)

    plt.close()


vae = models.DiscreteVAE("discrete_vae", 784, 50, 10, activation=tf.nn.relu,
                         initializer=tf.contrib.layers.xavier_initializer)
vae.build_graph([500, 500, 2000], [2000, 500, 500])
vae.define_train_step(0.002, train_data.epoch_len * 10)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

with tqdm(range(100), postfix={"loss": "inf"}) as bar:
    for epoch in bar:
        if epoch % 5 == 0:
            sample_plot()
            regeneration_plot()

        bar.set_postfix({"loss": "%.4f" % vae.train_op(sess, train_data)})

sample_plot()
regeneration_plot()
