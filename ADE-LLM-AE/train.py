import numpy as np
from tensorflow.keras.backend import clear_session
from tensorflow import convert_to_tensor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score

from model import create_autoencoder, add_noise, compute_anomaly_score

def train_autoencoders_per_prefix(prefix_data, prefix_input_dims, encoding_type, noise_factor=0.3, epochs=50, batch_size=32):
    models = {}
    thresholds = {}
    
    for k, v in prefix_data.items():
        labels = [label for _, label in v]
        #print(f"[DEBUG] Prefix {k} → labels set:", set(labels), "total samples:", len(labels))
    
    for prefix_len, data_list in prefix_data.items():
        input_dim = prefix_input_dims[prefix_len]

        # Prepare training data
        X_train = np.array([embedding for embedding, _ in data_list])
        #print(f"[DEBUG] Prefix {prefix_len} → X_train shape: {X_train.shape}, input_dim: {input_dim}")

        # Add noise
        adaptive_noise = min(noise_factor, 0.3 / max(1, prefix_len/3))
        X_train_noisy = add_noise(X_train, adaptive_noise, encoding_type)
        #print(f"[DEBUG] Prefix {prefix_len} → noise_factor applied: {adaptive_noise}")

        # Validation split
        val_split = max(1, int(0.1 * len(X_train)))
        X_val = X_train[-val_split:]
        X_val_noisy = add_noise(X_val, adaptive_noise, encoding_type)
        X_train_final = X_train[:-val_split]
        X_train_noisy_final = X_train_noisy[:-val_split]

        #print(f"[DEBUG] Prefix {prefix_len} → training samples: {len(X_train_final)}, validation samples: {len(X_val)}")

        clear_session()
        autoencoder = create_autoencoder(input_dim=input_dim, encoding_type=encoding_type)
        autoencoder.compile(optimizer=Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.99), loss='mse')

        #print(f"[DEBUG] Training autoencoder for prefix {prefix_len}...")
        autoencoder.fit(
            X_train_noisy_final,
            X_train_final,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_val_noisy, X_val),
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=2)]
        )
        
        models[prefix_len] = autoencoder

        # Compute reconstruction errors
        train_errors = []
        for i, x in enumerate(X_train_final):
            tensor = convert_to_tensor(np.expand_dims(x, axis=0), dtype='float32')
            score = compute_anomaly_score(autoencoder, tensor)
            train_errors.append(score)
            #if i < 5:  # debug first 5
                #print(f"[DEBUG] Prefix {prefix_len} train sample {i} → error: {score:.6f}")

        val_errors = []
        for i, x in enumerate(X_val):
            tensor = convert_to_tensor(np.expand_dims(x, axis=0), dtype='float32')
            score = compute_anomaly_score(autoencoder, tensor)
            val_errors.append(score)
            #if i < 5:
                #print(f"[DEBUG] Prefix {prefix_len} val sample {i} → error: {score:.6f}")

        # Validation labels
        y_true_val = [label for _, label in data_list[-val_split:]]
        #print(f"[DEBUG] Prefix {prefix_len} → validation labels: {y_true_val[:10]} ... (total {len(y_true_val)})")

        # Determine best threshold
        best_f1 = -1
        best_th = None
        candidate_percentiles = range(80, 100)

        for p in candidate_percentiles:
            th = np.percentile(train_errors, p)
            y_pred = [err > th for err in val_errors]
            f1 = f1_score(y_true_val, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
            #print(f"[DEBUG] Prefix {prefix_len} → percentile {p}, threshold {th:.6f}, F1 {f1:.6f}")

        thresholds[prefix_len] = best_th
        #print(f"[DEBUG] Prefix {prefix_len} → selected threshold: {best_th:.6f}, best F1: {best_f1:.6f}")

    return models, thresholds