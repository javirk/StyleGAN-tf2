import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from utils.nns import generator_network, discriminator_network
from datetime import datetime
from time import time
import os

class StyleGAN:
    def __init__(self, input_dim, mapping_size, noise_shape, latent_dim, output_dim, output_shape, LR=0.0001,
                 checkpoint_dir='checkpoints/', log_interval=2, save_interval=50):
        self.lr = LR
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.noise_shape = noise_shape
        self.output_dim = output_dim
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.generator = generator_network(self.latent_dim, self.mapping_size, self.noise_shape, self.output_dim)
        self.discriminator = discriminator_network(output_shape)

        self.generator_optimizer, self.discriminator_optimizer = self.compiler()

        self.checkpoint, self.checkpoint_prefix, self.manager = self.create_checkpoints(checkpoint_dir)
        self.train_summary_writer = self.writers_tensorboard()


    def compiler(self):
        gen_comp = Adam(lr=self.lr, beta_1=0, beta_2=0.9)
        dis_comp = Adam(lr=self.lr * 4, beta_1=0, beta_2=0.9)

        return gen_comp, dis_comp

    @staticmethod
    def generator_loss(x):
        return K.mean(x)


    def discriminator_loss(self, x_real, x_fake, images):
        divergence = K.mean(K.relu(1 + x_real) + K.relu(1 - x_fake))
        disc_loss = divergence + self.r1_gradient_penalty(x_real, images)

        return disc_loss

    @staticmethod
    def r1_gradient_penalty(y_pred, samples):
        gradients = K.gradients(y_pred, samples)[0]
        gradients_sqr = K.square(gradients)
        gradient_norm = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        return K.mean(gradient_norm)

    def train(self, images, batch_size, epochs=1000, verbose=False):
        latent_shape = (batch_size, self.latent_dim)
        noise_shape_batch = (batch_size, self.noise_shape[0], self.noise_shape[1], self.noise_shape[2])

        for epoch in range(epochs):
            start = time()
            np.random.shuffle(images)

            indices = np.random.randint(0, images.shape[0] - 1, [batch_size])
            real_images = images[indices]
            latent = np.random.normal(0.0, 1.0, size=latent_shape).astype('f')
            noise = np.random.uniform(0, 1, noise_shape_batch).astype('f')

            d_loss, g_loss = self._train_step(real_images, latent, noise)

            if verbose:
                print('Epoch ' + str(epoch) + ' took ' + str(time() - start))
                print(f'Epoch {epoch}')
                print(f'D: {d_loss}')
                print(f'G: {g_loss}')

            if (epoch + 1) % self.log_interval == 0:
                with self.train_summary_writer.as_default():
                    # print('Generator: ' + str(g_loss))
                    # print('Discriminator: ' + str(d_loss))
                    tf.summary.scalar('Generator loss', g_loss, step=epoch + 1)
                    tf.summary.scalar('Discriminator loss', d_loss, step=epoch + 1)

            if (epoch + 1) % self.save_interval == 0:
                # self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                self.manager.save()
                print('Checkpoint saved.')
                self.generate_images_tb(3, epoch + 1, channels=3)
                # self.generate_images(10, epoch + 1)
                print('Images saved.')

    @tf.function
    def _train_step(self, images, latent, noise):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            x_fake = self.generator([latent, noise])
            x_fake_from_discriminator = self.discriminator(x_fake, training=True)
            x_real_from_discriminator = self.discriminator(images, training=True)

            gen_loss = self.generator_loss(x_fake_from_discriminator)
            disc_loss = self.discriminator_loss(x_real_from_discriminator, x_fake_from_discriminator, images)

        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        return disc_loss, gen_loss

    @staticmethod
    def writers_tensorboard():
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        return train_summary_writer

    def create_checkpoints(self, checkpoint_dir):
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                         discriminator_optimizer=self.discriminator_optimizer,
                                         generator=self.generator,
                                         discriminator=self.discriminator)

        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

        return checkpoint, checkpoint_prefix, manager

    def generate_images_tb(self, n_images_to_present, epoch, channels=1):
        latent_shape = (n_images_to_present, self.latent_dim)
        noise_shape = (n_images_to_present, self.noise_shape[0], self.noise_shape[1], self.noise_shape[2])

        latent = np.random.normal(0.0, 1.0, size=latent_shape).astype('f')
        noise = np.random.uniform(0, 1, noise_shape).astype('f')

        x = self.generator([latent, noise])

        if channels == 1:
            t1 = np.reshape(x[..., 0] + 1 / 2, (-1, x.shape[1], x.shape[2], 1))
            flair = np.reshape(x[..., 1] + 1 / 2, (-1, x.shape[1], x.shape[2], 1))
            mask = np.reshape(x[..., 2] + 1 / 2, (-1, x.shape[1], x.shape[2], 1))

            with self.train_summary_writer.as_default():
                tf.summary.image('FLAIR', flair, max_outputs=n_images_to_present, step=epoch)
                tf.summary.image('T1', t1, max_outputs=n_images_to_present, step=epoch)
                tf.summary.image('Mask', mask, max_outputs=n_images_to_present, step=epoch)
        else:
            with self.train_summary_writer.as_default():
                tf.summary.image('Output', x, max_outputs=n_images_to_present, step=epoch)
