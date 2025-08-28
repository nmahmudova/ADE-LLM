from encoder import return_features_events, onehot_encode_data, word2vec_encode_data

def prepare_prefix_data(train_df, encoder, encoding_type="onehot"):
    prefix_data = {}
    prefix_input_dims = {}
    train_df = train_df.reset_index(drop=True)
    train_cases = train_df.groupby('case_id', sort=False)

    for case_id, group in train_cases:
        #print(f"[DEBUG] Processing case_id={case_id} with {len(group)} events")
        max_len = len(group)
        
        for prefix_len in range(1, max_len + 1):
            prefix = group.iloc[:prefix_len]

            prefix_seq = return_features_events(prefix)
            #print(f"[DEBUG] prefix_len={prefix_len}, features shape: {prefix_seq.shape}")

            if encoding_type == "onehot":
                embedding = onehot_encode_data(prefix_seq.values, encoder)
                one_event_size = sum([len(c) for c in encoder.categories_])
                input_dim = one_event_size * prefix_len
                #print(f"[DEBUG] One-hot embedding shape: {embedding.shape}, input_dim: {input_dim}")

            elif encoding_type == "word2vec":
                #if prefix_seq.empty:
                    #print(f"⚠️ Empty prefix sequence at prefix_len={prefix_len}, case={case_id}")
                embedding = word2vec_encode_data(prefix_seq, encoder)
                if embedding.shape[0] == 0:
                    #print(f"⚠️ Skipping empty embedding at prefix_len={prefix_len}, case={case_id}")
                    continue
                input_dim = embedding.shape[0]
                #print(f"[DEBUG] Word2Vec embedding shape: {embedding.shape}, input_dim: {input_dim}")

            else:
                raise ValueError("Unsupported encoding type.")

            if prefix_len not in prefix_data:
                prefix_data[prefix_len] = []
                prefix_input_dims[prefix_len] = input_dim
                #print(f"[DEBUG] Initialized prefix_data[{prefix_len}] with input_dim={input_dim}")
            else:
                assert input_dim == prefix_input_dims[prefix_len], \
                    f"Dim mismatch for prefix={prefix_len}: expected {prefix_input_dims[prefix_len]}, got {input_dim}"

            label = int(prefix['isAnomaly'].iloc[-1])  # label of last event in the prefix
            prefix_data[prefix_len].append((embedding, label))
            #print(f"[DEBUG] Added prefix of length {prefix_len}, label={label}, total samples for this prefix: {len(prefix_data[prefix_len])}")

    #print(f"[DEBUG] Completed preparing prefix data. Prefix lengths: {list(prefix_data.keys())}")
    return prefix_data, prefix_input_dims