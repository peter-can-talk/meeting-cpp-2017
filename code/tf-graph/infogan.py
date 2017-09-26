import keras.backend as K
import matplotlib.pyplot as plot
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          Dense, Flatten, Input, LeakyReLU, Reshape, Lambda,
                          UpSampling2D)
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.examples.tutorials.mnist import input_data

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(session)

# Supress warnings about wrong compilation of TensorFlow.
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(42)

noise_size = 100

latent_discrete = 10
latent_continuous = 2
latent_size = latent_discrete + latent_continuous

## G

z = Input(shape=[noise_size], name='z')
c = Input(shape=[latent_size], name='c')
G = Concatenate()([z, c])

G = Dense(7 * 7 * 256)(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)
G = Reshape((7, 7, 256))(G)

G = UpSampling2D()(G)
G = Conv2D(128, (5, 5), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)

G = UpSampling2D()(G)
G = Conv2D(64, (5, 5), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)

G = Conv2D(32, (5, 5), padding='same')(G)
G = BatchNormalization(momentum=0.9)(G)
G = LeakyReLU(alpha=0.2)(G)

G = Conv2D(1, (5, 5), padding='same')(G)
G = Activation('tanh', name='G_final')(G)

## D

x = Input(shape=(28, 28, 1))
D = Conv2D(32, (5, 5), strides=(2, 2), padding='same')(x)
D = LeakyReLU(alpha=0.2)(D)

D = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(D)
D = LeakyReLU(alpha=0.2)(D)

D = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(D)
D = LeakyReLU(alpha=0.2)(D)

D = Conv2D(256, (5, 5), padding='same')(D)
D = LeakyReLU(alpha=0.2)(D)
D = Flatten(name='D_final')(D)


def latent_activations(Q):
    Q_discrete = Activation('softmax')(Q[:, :latent_discrete])
    Q_continuous = Activation('sigmoid')(Q[:, -latent_continuous:])
    return Concatenate(axis=1)([Q_discrete, Q_continuous])


Q = Dense(latent_discrete + 2 * latent_continuous)(D)
Q = Lambda(latent_activations)(Q)

P = Dense(1, activation='sigmoid')(D)


def mutual_information(prior_c, c_given_x):
    h_c = K.categorical_crossentropy(prior_c, prior_c)
    h_c_given_x = K.categorical_crossentropy(prior_c, c_given_x)
    return K.mean(h_c_given_x - h_c)


def joint_mutual_information(prior_c, c_given_x):
    discrete = mutual_information(prior_c[:, :latent_discrete],
                                  c_given_x[:, :latent_discrete],
                                  K.categorical_crossentropy)
    continuous_1 = mutual_information(prior_c[:, -2], c_given_x[:, -2])
    continuous_2 = mutual_information(prior_c[:, -1], c_given_x[:, -1])
    return discrete + continuous_1 + continuous_2


generator = Model([z, c], G, name='G')

discriminator = Model(x, P, name='D')
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=5e-4, beta_1=0.5, decay=2e-7))

# x = G(z, c)
q = Model(x, Q, name='Q')
q.compile(
    loss=joint_mutual_information,
    optimizer=Adam(lr=2e-4, beta_1=0.5, decay=2e-7))

discriminator.trainable = False
q.trainable = False
infogan = Model([z, c], [discriminator(G), q(G)], name='InfoGAN')
infogan.compile(
    loss=['binary_crossentropy', joint_mutual_information],
    optimizer=Adam(lr=2e-4, beta_1=0.5, decay=1e-7))

generator.summary()
discriminator.summary()

data = input_data.read_data_sets('MNIST_data').train.images
data = data.reshape(-1, 28, 28, 1) * 2 - 1

number_of_epochs = 30
batch_size = 256

print(generator.outputs[0])


def sample_noise(size):
    return np.random.randn(size, noise_size)


def sample_prior(size):
    discrete = np.random.multinomial(1, [0.1] * 10, size=size)
    continuous_1 = np.random.uniform(-1, +1, size).reshape(-1, 1)
    continuous_2 = np.random.uniform(-1, +1, size).reshape(-1, 1)
    return np.concatenate([discrete, continuous_1, continuous_2], axis=1)


def smooth_labels(size):
    return np.random.uniform(low=0.8, high=1.0, size=size)


saver = tf.train.Saver(max_to_keep=1)
saver_def = saver.as_saver_def()

print(saver_def.filename_tensor_name)
print(saver_def.restore_op_name)

try:
    for epoch in range(number_of_epochs):
        print('Epoch: {0}/{1}'.format(epoch + 1, number_of_epochs))
        for batch_start in range(0, len(data) - batch_size + 1, batch_size):
            noise = sample_noise(batch_size)
            latent_code = sample_prior(batch_size)
            generated_images = generator.predict([noise, latent_code])

            real_images = data[batch_start:batch_start + batch_size]
            assert len(generated_images) == len(real_images)
            all_images = np.concatenate(
                [generated_images, real_images], axis=0)
            all_images += np.random.normal(0, 0.1, all_images.shape)

            labels = np.zeros(len(all_images))
            labels[batch_size:] = smooth_labels(batch_size)
            d_loss = discriminator.train_on_batch(all_images, labels)

            q_loss = q.train_on_batch(generated_images, latent_code)

            labels = np.ones(batch_size)
            noise = sample_noise(batch_size)
            latent_code = sample_prior(batch_size)
            g_loss, _, _ = infogan.train_on_batch([noise, latent_code],
                                                  [labels, latent_code])

            batch_index = batch_start // batch_size + 1
            message = '\rBatch: {0} | D: {1:.10f} | G: {2:.10f} | Q: {3:.10f}'
            print(message.format(batch_index, d_loss, g_loss, q_loss), end='')
        print()
        np.random.shuffle(data)
        tf.train.write_graph(
            session.graph_def, 'graphs', 'graph.pb', as_text=False)
        saver.save(session, 'checkpoints/chkp')

except KeyboardInterrupt:
    print()

print('Training complete!')
