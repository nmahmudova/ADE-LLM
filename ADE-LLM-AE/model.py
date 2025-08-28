import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from tensorflow import convert_to_tensor

def build_autoencoder(input_dim, output_activation):
    #print(f"[DEBUG] Building autoencoder with input_dim={input_dim}, output_activation={output_activation}")
    inp = Input(shape=(input_dim,), name='input')
    x = Dense(256, activation='relu', name='dense_1')(inp)
    x = Dropout(0.5, name='dropout_1')(x)
    x = Dense(128, activation='relu', name='dense_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)
    x = Dense(256, activation='relu', name='dense_3')(x)
    x = Dropout(0.5, name='dropout_3')(x)
    out = Dense(input_dim, activation=output_activation, name='output')(x)
    
    model = Model(inp, out)
    model.summary()  # Show architecture
    return model

def create_autoencoder(input_dim, encoding_type):
    output_activation = 'sigmoid' if encoding_type == 'onehot' else 'linear'
    #print(f"[DEBUG] Creating autoencoder for encoding_type={encoding_type}")
    return build_autoencoder(input_dim, output_activation)

def compute_anomaly_score(autoencoder, sequence):
    # Ensure batch dimension
    if len(sequence.shape) == 1:
        sequence = tf.expand_dims(sequence, axis=0)
    
    reconstructed = autoencoder(sequence, training=False)
    
    mse = np.mean(np.square(sequence - reconstructed))
    #print(f"[DEBUG] Anomaly score computed → sequence shape: {sequence.shape}, reconstructed shape: {reconstructed.shape}, MSE: {mse:.6f}")
    
    return mse

def add_noise(data, noise_factor=0.3, encoding_type='onehot'):
    #print(f"[DEBUG] Adding noise → data shape: {data.shape}, noise_factor: {noise_factor}, encoding_type: {encoding_type}")
    noisy = data + noise_factor * np.random.normal(0.0, 1.0, size=data.shape)
    
    if encoding_type == 'onehot':
        noisy = np.clip(noisy, 0.0, 1.0)
    
    # Debug sample values
    #print(f"[DEBUG] Original sample: {data[0][:5]}, Noisy sample: {noisy[0][:5]}")
    
    return noisy