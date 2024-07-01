import numpy
import os
import cv2
from tensorflow import keras
import random
from keras.initializers import RandomNormal
from keras.optimizers import Adam

DATA_DIRECTORY = '/kaggle/input/dataset-00'
images = []

for SUB_DIRECTORY in os.listdir(DATA_DIRECTORY):
    for count, imgs in enumerate(os.listdir(os.path.join(DATA_DIRECTORY, SUB_DIRECTORY))):
        img = cv2.imread(os.path.join(DATA_DIRECTORY, SUB_DIRECTORY, imgs))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        img = cv2.resize(img, (128, 128))
        images.append(img)
        if count == 30000:
            break
        
images = numpy.asarray(images)
images = images.astype('float32')
images = (images - 127.5)/127.5


def generator(latent_dim):

    model = keras.Sequential()
    init = RandomNormal(stddev=0.02)
    model.add(keras.layers.Dense(8192, input_dim=latent_dim))
    model.add(keras.layers.Reshape((8, 8, 128)))
    model.add(keras.layers.Conv2DTranspose(128, (4, 4), (2, 2), padding='same', kernel_initializer=init))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.Conv2DTranspose(64, (4, 4), (2, 2), padding='same', kernel_initializer=init))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.Conv2DTranspose(32, (4, 4), (2, 2), padding='same', kernel_initializer=init))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv2DTranspose(3, (4, 4), (2, 2), padding='same', activation='tanh', kernel_initializer=init))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    return model


def discriminator():

    init = RandomNormal(stddev=0.02)
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, input_shape=(128, 128, 3)))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def gan(generator_model, discriminator_model):

    discriminator_model.trainable = False
    model = keras.Sequential()
    model.add(generator_model)
    model.add(discriminator_model)
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    return model


def generate_real_samples(dataset, n_samples):

    imgs = numpy.random.randint(0, dataset.shape[0], n_samples)
    x = dataset[imgs]
    y = numpy.ones((n_samples, 1))
    return x, y

def generate_latent_points(latent_dim, n_samples):

    x = numpy.random.randn(latent_dim*n_samples)
    x = x.reshape(n_samples, latent_dim)
    return x

def generate_fake_samples(generator_model, latent_dim, n_samples):

    x = generate_latent_points(latent_dim, n_samples)
    x = generator_model.predict(x)
    y = numpy.zeros((n_samples, 1))
    return x, y

def train(generator_model, discriminator_model, gan_model, latent_dim, n_epochs, dataset, n_batches=128):

    batch_per_epoch = int(dataset.shape[0]/n_batches)
    half_batch = int(n_batches/2)

    for epoch in range(n_epochs):
        for batch in range(batch_per_epoch):
            real_images, real_labels = generate_real_samples(dataset, half_batch)
            discriminator_loss_real, disc_acc_real = discriminator_model.train_on_batch(real_images, real_labels)

            fake_images, fake_labels = generate_fake_samples(generator_model, latent_dim, half_batch)
            discriminator_loss_fake, disc_acc_fake = discriminator_model.train_on_batch(fake_images, fake_labels)

            gan_fake_images = generate_latent_points(latent_dim, n_batches)
            gan_fake_labels = numpy.ones((n_batches, 1))

            gan_loss = gan_model.train_on_batch(gan_fake_images, gan_fake_labels)

            print(f'epoch: {epoch}, gan loss: {gan_loss:4f}, discriminator loss fake: {discriminator_loss_fake:4f}, discriminator loss real: {discriminator_loss_real:4f}')

            gan_model.save('cifar_generator.keras')
            
latent_dim = 128
generator = generator(latent_dim)
discriminator = discriminator()
gan = gan(generator, discriminator)
dataset = images

train(generator, discriminator, gan, latent_dim, 300, dataset)
