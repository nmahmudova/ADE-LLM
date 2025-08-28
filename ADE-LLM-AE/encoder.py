import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import convert_to_tensor
from gensim.models import Word2Vec

def return_features_events(events):
    #print(f"[DEBUG] Extracting features for {len(events)} events")
    return events[['name']]

def init_word2vec_model(dataset, vector_size=32, window=5, min_count=1, workers=4):
    sequences = dataset.groupby("case_id")["name"].apply(list).tolist()
    #print(f"[DEBUG] Initializing Word2Vec with {len(sequences)} sequences")
    model = Word2Vec(sentences=sequences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    #print(f"[DEBUG] Word2Vec model trained. Vocab size: {len(model.wv)}")
    return model

def word2vec_encode_data(sequence, w2v_model):
    if isinstance(sequence, pd.DataFrame):
        names = sequence["name"].tolist()
    elif isinstance(sequence, list):
        names = [x["name"] if isinstance(x, dict) else x for x in sequence]
    else:
        raise ValueError("Unsupported input type for word2vec_encode_data")

    embeddings = []
    missing = 0
    for name in names:
        if name in w2v_model.wv:
            embeddings.append(w2v_model.wv[name])
        else:
            embeddings.append(np.zeros(w2v_model.vector_size))  # OOV handling
            missing += 1

    #if missing > 0:
        #print(f"⚠️ {missing}/{len(names)} events missing from Word2Vec vocab")

    if not embeddings:
        #print("⚠️ No valid embeddings found; returning zeros")
        return np.zeros(w2v_model.vector_size)

    #if len(names) != len(embeddings):
        #print(f"⚠️ Length mismatch. Original: {len(names)}, Embeddings: {len(embeddings)}")

    concatenated = np.concatenate(embeddings)
    #print(f"[DEBUG] Word2Vec encoding shape: {concatenated.shape}")
    return concatenated

def init_encoder(dataset, method="onehot"):
    dataset = return_features_events(dataset)

    if method == "onehot":
        df_values = dataset.drop_duplicates()
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(df_values)
        #print(f"[DEBUG] OneHotEncoder initialized with {len(encoder.categories_[0])} categories")
        return encoder
    elif method == "":
        w2v_model = init_word2vec_model(dataset)
        return w2v_model
    else:
        raise ValueError("Unsupported encoding method.")

def onehot_encode_data(sequence, encoder):
    df = pd.DataFrame(sequence, columns=['name'])
    transformed = encoder.transform(df).flatten()
    #print(f"[DEBUG] One-hot encoded shape: {transformed.shape}")
    return transformed
