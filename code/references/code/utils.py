import tensorflow as tf
import numpy as np

SMALL = 1e-16
EULER_GAMMA = 0.5772156649015329

"""
Sample z_real, z_discrete, v from posterior params
"""


def sample(z_mean, z_log_std, pi_logit, a, b, temp, calc_v=True, calc_real=True):

    if calc_real:
        # mu + standard_samples * stand_deviation
        z_real = z_mean + \
            tf.random_normal(tf.shape(z_mean)) * tf.exp(z_log_std)
    else:
        z_real = None

    # Concrete instead of Bernoulli
    y_sample = reparametrize_discrete(pi_logit, temp)
    z_discrete = tf.sigmoid(y_sample)

    if calc_v:
        # draw v from kumarswamy instead of Beta
        v = kumaraswamy_sample(a, b)
    else:
        v = None

    return z_discrete, z_real, v, y_sample


def tf_monte_carlo_sample(z_mean, z_log_std, pi_logit, a, b, temp, calc_v=True, calc_real=True, S=5):

    # S x N x K

    z_mean = tf.expand_dims(z_mean, 0)
    z_mean = tf.tile(z_mean, [S, 1, 1])

    print z_mean.get_shape()
    z_real = z_mean + tf.random_normal(tf.shape(z_mean)) * tf.exp(z_log_std)

    pi_logit = tf.expand_dims(pi_logit, 0)
    pi_logit = tf.tile(pi_logit, [S, 1, 1])

    y_sample = reparametrize_discrete(pi_logit, temp)
    z_discrete = tf.round(tf.sigmoid(y_sample))

    emb = tf.multiply(z_real, z_discrete)
    emb_t = tf.transpose(emb, (0, 2, 1))

    adj_rec = tf.matmul(emb, emb_t)
    adj_rec = tf.reduce_mean(adj_rec, 0)

    return adj_rec, z_activated


def monte_carlo_sample(z_mean, z_log_std, pi_logit, temp, S, sigmoid_fn):

    shape = list(np.shape(z_mean))
    shape.insert(0, S)

    # mu + standard_samples * stand_deviation
    z_real = z_mean + \
        np.multiply(np.random.normal(0, 1, shape), np.exp(z_log_std))

    # Concrete instead of Bernoulli => equivalent to reparametrize_discrete in tensorflow
    uniform = np.random.uniform(1e-4, 1. - 1e-4, shape)
    logistic = np.log(uniform) - np.log(1 - uniform)
    y_sample = (pi_logit + logistic) / temp

    z_discrete = sigmoid_fn(y_sample)
    z_discrete = np.round(z_discrete)

    z_activated = np.sum(z_discrete) / (shape[0] * shape[1])

    emb = np.multiply(z_real, z_discrete)
    emb_t = np.transpose(emb, (0, 2, 1))
    adj_rec = np.matmul(emb, emb_t)
    adj_rec = np.mean(adj_rec, axis=0)

    return adj_rec, z_activated


def monte_carlo_sample_for_weighted_links(z_mean, z_log_std, pi_logit, W, temp, S, sigmoid_fn, c=0, e=None):

    shape = list(np.shape(z_mean))
    shape.insert(0, S)

    # mu + standard_samples * stand_deviation
    z_real = z_mean + \
        np.multiply(np.random.normal(0, 1, shape), np.exp(z_log_std))

    # Concrete instead of Bernoulli => equivalent to reparametrize_discrete in tensorflow
    uniform = np.random.uniform(1e-4, 1. - 1e-4, shape)
    logistic = np.log(uniform) - np.log(1 - uniform)
    y_sample = (pi_logit + logistic) / temp

    z_discrete = sigmoid_fn(y_sample)
    z_discrete = np.round(z_discrete)

    z_activated = np.sum(z_discrete) / (shape[0] * shape[1])

    emb = np.multiply(z_real, z_discrete)
    emb_t = np.transpose(emb, (0, 2, 1))

    adj_rec = np.matmul(np.matmul(emb, W), emb_t)
    #adj_rec = np.matmul(emb + np.matmul(emb, W), emb_t)
    adj_rec = np.mean(adj_rec, axis=0)

    adj_rec += np.tile(c, np.shape(adj_rec))

    if e is not None:
        e_bias = np.tile(e, (1, np.shape(adj_rec)[0]))
        e_bias = e_bias + np.transpose(e_bias)
        adj_rec += e_bias

    return adj_rec, z_activated


def monte_carlo_sample(pi_logit, W, temp, S, sigmoid_fn, c=0, e=None):

    shape = list(np.shape(pi_logit))
    shape.insert(0, S)

    # Concrete instead of Bernoulli => equivalent to reparametrize_discrete in tensorflow
    uniform = np.random.uniform(1e-4, 1. - 1e-4, shape)
    logistic = np.log(uniform) - np.log(1 - uniform)
    y_sample = (pi_logit + logistic) / temp

    z_discrete = sigmoid_fn(y_sample)
    z_discrete = np.round(z_discrete)

    z_activated = np.sum(z_discrete) / (shape[0] * shape[1])

    emb_t = np.transpose(z_discrete, (0, 2, 1))
    #adj_rec = np.matmul(z_discrete + np.matmul(z_discrete, W), emb_t)
    adj_rec = np.matmul(np.matmul(z_discrete, W), emb_t)
    adj_rec = np.mean(adj_rec, axis=0)

    adj_rec += np.tile(c, np.shape(adj_rec))

    if e is not None:
        e_bias = np.tile(e, (1, np.shape(adj_rec)[0]))
        e_bias = e_bias + np.transpose(e_bias)
        adj_rec += e_bias

    return adj_rec, z_activated

# See concrete distribution paper


def reparametrize_discrete(logalphas, temp):
    """
    input: logit, output: logit
    """
    uniform = tf.random_uniform(tf.shape(logalphas), 1e-4, 1. - 1e-4)
    logistic = tf.log(uniform) - tf.log(1. - uniform)
    ysample = (logalphas + logistic) / temp
    return ysample


def kumaraswamy_sample(a, b):
    u = tf.random_uniform(tf.shape(a), 1e-4, 1. - 1e-4)
    # return (1. - u.pow(1./b)).pow(1./a)
    return tf.exp(tf.log(1. - tf.exp(tf.log(u) / (b+SMALL)) + SMALL) / (a+SMALL))


def random_draw_logpi(a, b, n_samples, hidden):

    v = np.random.beta(a, b, hidden)
    v_term = np.log(v+1e-16)
    log_prior = np.cumsum(v_term)
    logit_pi = log_prior - np.log(1. - np.exp(log_prior) + 1e-16)
    logit_pi = logit_pi.reshape((1, logit_pi.shape[0]))
    print logit_pi.shape

    logit_pi = np.tile(logit_pi, (n_samples, 1))

    print logit_pi.shape

    return logit_pi


def logit(x):
    return tf.log(x + SMALL) - tf.log(1. - x + SMALL)


def log_density_logistic(logalphas, y_sample, temp):
    """
    log-density of the Logistic distribution, from 
    Maddison et. al. (2017) (right after equation 26)
    Input logalpha is a logit (alpha is a probability ratio)
    """
    exp_term = logalphas + y_sample * -temp
    log_prob = exp_term + np.log(temp) - 2. * tf.nn.softplus(exp_term)
    return log_prob


def Beta_fn(a, b):
    beta_ab = tf.exp(tf.lgamma(a) + tf.lgamma(b) - tf.lgamma(a + b))
    return beta_ab

# the prior is default Beta(alpha_0, 1)


def kl_kumar_beta(a, b, prior_alpha=10., log_beta_prior=np.log(1./10.)):
    """
    KL divergence between Kumaraswamy(a, b) and Beta(prior_alpha, prior_beta)
    as in Nalisnick & Smyth (2017) (12)
    - we require you to calculate the log of beta function, since that's a fixed quantity
    """
    prior_beta = 1.

    # digamma = b.log() - 1/(2. * b) - 1./(12 * b.pow(2)) # this doesn't seem to work
    first_term = ((a - prior_alpha)/(a+SMALL)) * \
        (-1 * EULER_GAMMA - tf.digamma(b) - 1./(b+SMALL))
    second_term = tf.log(a+SMALL) + tf.log(b+SMALL) + log_beta_prior
    third_term = -(b - 1)/(b+SMALL)

    ab = a*b + SMALL
    kl = 1./(1+ab) * Beta_fn(1./(a+SMALL), b)
    kl += 1./(2+ab) * Beta_fn(2./(a+SMALL), b)
    kl += 1./(3+ab) * Beta_fn(3./(a+SMALL), b)
    kl += 1./(4+ab) * Beta_fn(4./(a+SMALL), b)
    kl += 1./(5+ab) * Beta_fn(5./(a+SMALL), b)
    kl += 1./(6+ab) * Beta_fn(6./(a+SMALL), b)
    kl += 1./(7+ab) * Beta_fn(7./(a+SMALL), b)
    kl += 1./(8+ab) * Beta_fn(8./(a+SMALL), b)
    kl += 1./(9+ab) * Beta_fn(9./(a+SMALL), b)
    kl += 1./(10+ab) * Beta_fn(10./(a+SMALL), b)
    kl *= (prior_beta-1)*b

    kl += first_term + second_term + third_term
    return tf.reduce_mean(tf.reduce_sum(kl, 1))


def kl_discrete(logit_posterior, logit_prior, y_sample, temp, temp_prior):
    """
    KL divergence between the prior and posterior
    inputs are in logit-space
    """
    logprior = log_density_logistic(logit_prior, y_sample, temp_prior)
    logposterior = log_density_logistic(logit_posterior, y_sample, temp)
    kl = logposterior - logprior
    return tf.reduce_mean(tf.reduce_sum(kl, 1))


def kl_real(z_log_std, z_mean):

    kl = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + 2 * z_log_std -
                                             tf.square(z_mean) - tf.square(tf.exp(z_log_std)), 1))
    return kl


def nz(n=100, z=10):

    nz = np.zeros((n, z))

    nz[0:30, 0] = 1
    nz[30:50, 1] = 1
    nz[50:60, 2] = 1
    nz[60:70, 3] = 1
    nz[70:75, 4] = 1
    nz[75:80, 5] = 1
    nz[80:85, 6] = 1
    nz[85:90, 7] = 1
    nz[90:95, 8] = 1
    nz[95:100, 9] = 1

    # overlap
    nz[85:90, 0] = 1
    nz[50:60, 9] = 1
    nz[30:50, 5] = 1
    nz[0:20, 7] = 1

    return nz
