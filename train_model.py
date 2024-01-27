import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, Input, Flatten, BatchNormalization, Multiply, Add, LeakyReLU, \
    Reshape
from keras.activations import selu
from keras.optimizers import Adam
from keras import backend as K

# defining input and output directory
image_dir = "./image"
output_dir = './results_while_training'

# defining size
image_size = 64
batch_size = 128
latent_dim = 512
learning_rate = 0.0001
beta_1 = 0.5
epochs = 100
filters = [64, 128, 256, 512]
x_plot = 5
y_plot = 5
mse_loss = []
kl_losses = []
kernel_size_decoder = (5, 5)
kernel_size_encoder = 5


with tf.device('/device:GPU:0'):
    # defining some useful functions
    def preprocess(image):
        image = tf.io.read_file(image)
        image = tf.io.decode_jpeg(image)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (image_size, image_size))
        image = image / 255.0
        image = tf.reshape(image, shape=(image_size, image_size, 3))
        return image


    def reconstruction_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))


    def kl_loss(mu, log_var):
        loss = 0.5 * tf.reduce_mean(tf.square(mu) + tf.exp(log_var) - log_var - 1)
        return loss


    def vae_loss(y_true, y_pred, mu, log_var):
        return reconstruction_loss(y_true, y_pred) + kl_loss(mu, log_var)


    def save_images(model, epoch, step, input_):
        prediction = model.predict(input_)
        print(len(prediction))
        fig, axes = plt.subplots(x_plot, y_plot, figsize=(14, 14))
        idx = 0
        for row in range(x_plot):
            for col in range(y_plot):
                img = prediction[idx] * 255.0
                img = img.astype("int32")
                axes[row, col].imshow(img)
                idx += 1
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        plt.savefig(output_dir + "/Epoch_{:04d}_step_{:04d}.jpg".format(epoch, step))
        plt.close()


    # taking input
    images = [os.path.join(image_dir, image) for image in os.listdir(image_dir)]

    # make training dataset
    training_dataset = tf.data.Dataset.from_tensor_slices((images))
    training_dataset = training_dataset.map(preprocess)
    training_dataset = training_dataset.shuffle(1000).batch(batch_size)

    K.clear_session()

    # create encoder

    encoder_input = Input(shape=(image_size, image_size, 3))
    x = Conv2D(32, kernel_size=kernel_size_encoder, activation=LeakyReLU(0.02), strides=1, padding="same")(
        encoder_input)
    for i in filters:
        x = Conv2D(i, kernel_size=kernel_size_encoder, activation=LeakyReLU(0.02), strides=2, padding="same")(x)
        x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(1024, activation=selu)(x)
    encoder_output = BatchNormalization()(x)

    mu = Dense(latent_dim)(encoder_output)
    log_var = Dense(latent_dim)(encoder_output)
    sigma = tf.exp(0.5 * log_var)
    epsilon = K.random_normal(shape=(tf.shape(mu)[0], tf.shape(mu)[1]))
    z_eps = Multiply()([sigma, epsilon])
    z = Add()([mu, z_eps])

    encoder = Model(encoder_input, outputs=[mu, log_var, z], name="encoder")

    # create decoder
    decoder = Sequential()

    decoder.add(Dense(1024, activation=selu, input_shape=(latent_dim,)))
    decoder.add(BatchNormalization())
    decoder.add(Dense(8192, activation=selu))
    decoder.add(Reshape((4, 4, filters[3])))
    decoder.add(Conv2DTranspose(filters[2], kernel_size_decoder, activation=LeakyReLU(0.02), strides=2, padding="same"))
    decoder.add(BatchNormalization())
    decoder.add(Conv2DTranspose(filters[1], kernel_size_decoder, activation=LeakyReLU(0.02), strides=2, padding="same"))
    decoder.add(BatchNormalization())
    decoder.add(Conv2DTranspose(filters[0], kernel_size_decoder, activation=LeakyReLU(0.02), strides=2, padding="same"))
    decoder.add(BatchNormalization())
    decoder.add(Conv2DTranspose(32, kernel_size_decoder, activation=LeakyReLU(0.02), strides=2, padding="same"))
    decoder.add(BatchNormalization())
    decoder.add(Conv2DTranspose(3, kernel_size_decoder, activation="sigmoid", strides=1, padding="same"))
    decoder.add(BatchNormalization())

    # create model
    mu, log_var, z = encoder(encoder_input)
    reconstructed = decoder(z)

    VAE = Model(encoder_input, reconstructed, name='vae')
    loss = kl_loss(mu, log_var)
    VAE.add_loss(loss)

    # run model
    optimizer = Adam(learning_rate, beta_1)
    random_vector = tf.random.normal(shape=(x_plot * y_plot, latent_dim,))

    for epoch in range(1, epochs + 1):
        print("Epoch: ", epoch)
        for step, training_batch in enumerate(training_dataset):
            with tf.GradientTape() as tape:
                reconstructed = VAE(training_batch)
                y_true = tf.reshape(training_batch, shape=[-1])
                y_pred = tf.reshape(reconstructed, shape=[-1])

                mse_l = reconstruction_loss(y_true, y_pred)
                mse_loss.append(mse_l.numpy())

                kl = sum(VAE.losses)
                kl_losses.append(kl.numpy())

                train_loss = 0.01 * kl + mse_l
                grads = tape.gradient(train_loss, VAE.trainable_variables)
                optimizer.apply_gradients(zip(grads, VAE.trainable_variables))

                if step % 10 == 0:
                    save_images(decoder, epoch, step, random_vector)
                print("Epoch: %s - Step: %s - MSE Loss: %s - KL Loss: %s" % (epoch, step, mse_l.numpy(), kl.numpy()))

    VAE.save("./trained_model.keras")
