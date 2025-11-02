import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras import Model
from tensorflow.keras.metrics import Mean, MSE
from tensorflow.keras.losses import MeanSquaredError

mse = MeanSquaredError()

from utilities import MinMaxScaler, compute_mse_loss

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# tf.config.run_functions_eagerly(True)

# Decoders

class Autoencoder(Model):
    def __init__(self, input_dim:int, hidden_layers:list, hidden_dim:int, use_bias:bool):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers + [hidden_dim]
        self.use_bias = use_bias
        # Encoder
        self.encoder = self._get_encoder()
        
        # Decoder
        self.decoder = self._get_decoder()

        self.use_mask = False

        # MSE loss
        self.mse_loss_tracker = Mean(name="mse_loss")
        self.mse_test_loss_tracker = Mean(name="mse_loss_test")

    def _get_encoder(self):
        raise NotImplementedError

    def _get_decoder(self):
        raise NotImplementedError

    def enable_masked_loss_function(self):
        self.use_mask = True
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
    
    # Override the train_step method
    def train_step(self, data):
        x = data[0]
        # Create a boolean mask where True represents NaN positions
        m_mask = tf.math.is_nan(x)
        # Replace NaN values with zeros
        x = tf.where(m_mask, tf.zeros_like(x), x)
        with tf.GradientTape() as tape:

            reconstruction = self.call(x)

            if self.use_mask:

                mse_loss = compute_mse_loss(x, reconstruction, m_mask=m_mask)
            else:
                mse_loss = compute_mse_loss(x, reconstruction, m_mask=None)


        grads = tape.gradient(mse_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.mse_loss_tracker.update_state(mse_loss)
        return {
            "mse_loss": self.mse_loss_tracker.result()
        }


    def test_step(self, data):
        x = data[0]
        # Create a boolean mask where True represents NaN positions
        m_mask = tf.math.is_nan(x)
        # Replace NaN values with zeros
        x = tf.where(m_mask, tf.zeros_like(x), x)
        with tf.GradientTape() as tape:

            reconstruction = self.call(x)

            if self.use_mask:

                mse_loss = compute_mse_loss(x, reconstruction, m_mask=m_mask)
            else:
                mse_loss = compute_mse_loss(x, reconstruction, m_mask=None)


        grads = tape.gradient(mse_loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.mse_test_loss_tracker.update_state(mse_loss)
        return {
            "mse_loss_test": self.mse_test_loss_tracker.result()
        }



class LinearAutoencoder(Autoencoder):
    def __init__(self, input_dim: int, hidden_layers: list, hidden_dim:int, use_bias:bool=False):
        super().__init__(input_dim, hidden_layers,  hidden_dim, use_bias=use_bias)

    def _get_encoder(self):
        input_lay = Input(shape=(self.input_dim,))

        encoded = Dense(self.hidden_layers[0], activation=None, use_bias=self.use_bias)(input_lay)
        for i in range(1,len(self.hidden_layers)):
            encoded = Dense(self.hidden_layers[i], activation=None, use_bias=self.use_bias)(encoded)
        return Model(input_lay, encoded, name='encoder')

    def _get_decoder(self):
        input_lay = Input(shape=(self.hidden_layers[-1],))
        if len(self.hidden_layers)==1:
            decoded = Dense(self.input_dim, activation=None, use_bias=self.use_bias)(input_lay)
        else:
            decoded = Dense(self.hidden_layers[-2], activation=None, use_bias=self.use_bias)(input_lay)
            for i in range(3,len(self.hidden_layers)+1):
                decoded = Dense(self.hidden_layers[-i], activation=None, use_bias=self.use_bias)(decoded)
            decoded = Dense(self.input_dim, activation=None, use_bias=self.use_bias)(decoded)
        return Model(input_lay, decoded, name="decoder")



class Linear2DAutoencoder(Autoencoder):
    def __init__(self, input_dim: list, hidden_layers: list, hidden_dim: int, use_bias: bool = False):
        super().__init__(input_dim, hidden_layers, hidden_dim, use_bias=use_bias)

    def _get_encoder(self):
        input_lay = Input(shape=self.input_dim)
        encoded = Flatten()(input_lay)

        encoded = Dense(self.hidden_layers[0], activation=None, use_bias=self.use_bias)(encoded)
        for i in range(1, len(self.hidden_layers)):
            encoded = Dense(self.hidden_layers[i], activation=None, use_bias=self.use_bias)(encoded)
        return Model(input_lay, encoded, name='encoder')

    def _get_decoder(self):
        input_lay = Input(shape=(self.hidden_layers[-1],))
        if len(self.hidden_layers) == 1:
            decoded = Dense(self.input_dim[0]*self.input_dim[1], activation=None, use_bias=self.use_bias)(input_lay)
            decoded = Reshape(self.input_dim)(decoded)
        else:
            decoded = Dense(self.hidden_layers[-2], activation=None, use_bias=self.use_bias)(input_lay)
            for i in range(3, len(self.hidden_layers) + 1):
                decoded = Dense(self.hidden_layers[-i], activation=None, use_bias=self.use_bias)(decoded)
            decoded = Dense(self.input_dim[0] * self.input_dim[1], activation=None, use_bias=self.use_bias)(decoded)
            decoded = Reshape(self.input_dim)(decoded)
        return Model(input_lay, decoded, name="decoder")


class SimpleAutoencoder(Autoencoder):
    def __init__(self, input_dim: int, hidden_layers: list, hidden_dim:int,use_bias:bool):
        super().__init__(input_dim, hidden_layers,  hidden_dim, use_bias=use_bias)

    def _get_encoder(self):
        input_lay = Input(shape=(self.input_dim,))

        encoded = Dense(self.hidden_layers[0], activation='relu', use_bias=self.use_bias)(input_lay)
        for i in range(1,len(self.hidden_layers)-1):
            encoded = Dense(self.hidden_layers[i], activation='relu', use_bias=self.use_bias)(encoded)
        bottleneck = Dense(self.hidden_layers[-1], activation='relu', use_bias=self.use_bias)(encoded)
        return Model(input_lay, bottleneck, name='encoder')


    def _get_decoder(self):
        input_lay = Input(shape=(self.hidden_layers[-1],))
        if len(self.hidden_layers)==1:
            decoded = Dense(self.input_dim, activation=None, use_bias=self.use_bias)(input_lay)
        else:
            decoded = Dense(self.hidden_layers[-2], activation='relu',use_bias=self.use_bias)(input_lay)
            for i in range(3,len(self.hidden_layers)+1):
                decoded = Dense(self.hidden_layers[-i], activation='relu', use_bias=self.use_bias)(decoded)
            decoded = Dense(self.input_dim, activation=None, use_bias=self.use_bias)(decoded)
        return Model(input_lay, decoded, name="decoder")
    


if __name__ == "__main__":
    # Define input and hidden dimensions
    input_dim = 784  # For MNIST images (28x28)
    hidden_layers = [2*128,128]

    autoencoder = SimpleAutoencoder(input_dim, hidden_layers,hidden_dim=15)


    autoencoder.compile(optimizer=Adam())

    # Print the model architecture
    autoencoder.summary()

    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32').reshape(x_train.shape[0],-1) / 255.
    x_test = x_test.astype('float32').reshape(x_test.shape[0],-1) / 255.

    print (x_train.shape)
    print (x_test.shape)

    autoencoder.fit(x_train, x_train,
                epochs=10, batch_size=64, 
                shuffle=True,
                validation_data=(x_test, x_test))


    decoded_img = autoencoder.predict(x_test)

    decoded_img = decoded_img.reshape(decoded_img.shape[0], 28,28)
    x_test = x_test.reshape(x_test.shape[0], 28,28)

    from matplotlib import pyplot as plt

    plt.imshow(decoded_img[3,:,:], cmap="gray")
    plt.show()

    plt.imshow(x_test[3,:,:], cmap="gray")
    plt.show()

