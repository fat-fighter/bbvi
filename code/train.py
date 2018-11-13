import os
import models
import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from tqdm import tqdm

from includes.config import Config
from includes.utils import load_data, Dataset
from includes.visualization import mnist_regeneration_plot, mnist_sample_plot

tf.logging.set_verbosity(tf.logging.ERROR)

datagroup = "graph"
dataset = "citeseer"

config = Config(datagroup)

train_data, test_data = load_data(datagroup, dataset=dataset)
(adj, adj_norm, adj_label, adj_orig, features, adj_train, train_edges) = train_data

# train_data = Dataset(train_data, batch_size=config.batch_size)
# test_data = Dataset(test_data, batch_size=config.batch_size)

folders = ["models", "plots"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)


# vae = models.VAE("standard_vae", 784, 10, activation=tf.nn.relu,
#                  initializer=tf.contrib.layers.xavier_initializer)

# vae = models.DiscreteVAE("discrete_vae", 784, 30, 10, activation=tf.nn.relu,
#                          initializer=tf.contrib.layers.xavier_initializer)

# vae = models.GumboltVAE("gumbolt_vae", 784, 60, 140, activation=tf.nn.relu,
#                         initializer=tf.contrib.layers.xavier_initializer)

# vae.define_train_step(0.0001, train_data.epoch_len * 10, decay_rate=0.9)


vae = models.GVAE("gvae", features.shape[1], 10, activation=tf.nn.relu,
                  initializer=tf.contrib.layers.xavier_initializer())

vae.build_graph([500, 500, 2000], [2000, 500, 500])
vae.define_train_step(0.0001, 100, decay_rate=0.9)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

saver = tf.train.Saver()

with tqdm(range(1000), postfix={"loss": "inf"}) as bar:
    for epoch in bar:
        # if epoch % 5 == 0:
        #     mnist_sample_plot(vae, sess)
        #     mnist_regeneration_plot(vae, sess, test_data)

        if epoch % 50 == 0:
            save_path = saver.save(sess, "models/%s.ckpt" % vae.name)

        bar.set_postfix({"loss": "%.4f" % vae.train_op(
            sess, {"features": features, "adj_label": adj_label, "adj_norm": adj_norm})})

# mnist_sample_plot(vae, sess)
# mnist_regeneration_plot(vae, sess)
