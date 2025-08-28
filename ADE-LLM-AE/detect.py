from encoder import return_features_events, onehot_encode_data, word2vec_encode_data
from collections import defaultdict
from tensorflow import convert_to_tensor
from model import compute_anomaly_score
import numpy as np
import time

def detect_anomalies(test_df, models, thresholds, prefix_input_dims, encoder, encoding_type="onehot"):
    predicted_labels = {}
    anomaly_scores = {}
    per_prefix_predictions = defaultdict(list)
    per_prefix_truth = defaultdict(list)
    per_prefix_counts = defaultdict(lambda: {'normal': 0, 'anomaly': 0})
    csv_records = []

    # Track encoding times
    encoding_times = defaultdict(list)  # {prefix_len: [times]}

    test_cases = test_df.groupby('case_id')
    #print(f"[DEBUG] Number of test cases: {len(test_cases)}")

    for case_id, group in test_cases:
        evaluated = False
        max_score = -np.inf
        true_label = group["isAnomaly"].iloc[0]
        anomaly_detected = False

        max_len = len(group)
        #print(f"[DEBUG] Processing case_id={case_id} with {max_len} events, true label={true_label}")

        for prefix_len in range(1, max_len + 1):
            if prefix_len not in models:
                #print(f"[DEBUG] No model for prefix length {prefix_len}, skipping")
                continue

            prefix = group.iloc[:prefix_len]
            prefix_seq = return_features_events(prefix)
            #print(f"[DEBUG] Prefix {prefix_len} names: {prefix_seq['name'].tolist()}")

            start_time = time.time()
            if encoding_type == "onehot":
                embedding = onehot_encode_data(prefix_seq.values, encoder)
            elif encoding_type == "word2vec":
                embedding = word2vec_encode_data(prefix_seq, encoder)
            else:
                raise ValueError("Unsupported encoding type.")
            elapsed = time.time() - start_time

            encoding_times[prefix_len].append((elapsed, len(embedding)))
            #print(f"[DEBUG] Encoding time: {elapsed:.4f}s, embedding dim: {len(embedding)}")

            input_dim = prefix_input_dims[prefix_len]
            if len(embedding) < input_dim:
                embedding = np.pad(embedding, (0, input_dim - len(embedding)), 'constant')
                #print(f"[DEBUG] Padded embedding from {len(embedding) - (input_dim - len(embedding))} to {input_dim}")
            if len(embedding) != input_dim:
                raise ValueError(f"Embedding dim {len(embedding)} != expected {input_dim} for prefix {prefix_len}")

            tensor = convert_to_tensor(np.expand_dims(embedding, axis=0), dtype='float32')
            score = compute_anomaly_score(models[prefix_len], tensor)
            evaluated = True

            max_score = max(max_score, score)
            predicted = 1 if score > thresholds[prefix_len] else 0
            if predicted == 1:
                anomaly_detected = True

            prefix_text = ", ".join(prefix["name"].tolist())

            csv_records.append({
                "case_id": case_id,
                "prefix_len": prefix_len,
                "trace": prefix_text,
                "score": score,
                "true_label": true_label,
                "predicted": predicted
            })

            per_prefix_predictions[prefix_len].append(predicted)
            per_prefix_truth[prefix_len].append(true_label)
            label_key = 'anomaly' if true_label == 1 else 'normal'
            per_prefix_counts[prefix_len][label_key] += 1

            #print(f"[DEBUG] Prefix {prefix_len} — Score: {score:.4f} | Threshold: {thresholds[prefix_len]:.4f} | Predicted: {predicted}")

        if not evaluated:
            predicted_labels[case_id] = 0
            anomaly_scores[case_id] = np.nan
            #print(f"[DEBUG] No evaluation for case_id={case_id}, assigning default 0")
        else:
            predicted_labels[case_id] = 1 if anomaly_detected else 0
            anomaly_scores[case_id] = max_score
            #print(f"[DEBUG] case_id={case_id} max score: {max_score:.4f} | Predicted label: {predicted_labels[case_id]}")

    # Calculate average encoding time per prefix length and embedding dimension
    avg_encoding_times = {}
    for prefix_len, times in encoding_times.items():
        avg_time = np.mean([t[0] for t in times])
        avg_dim = np.mean([t[1] for t in times])
        avg_encoding_times[prefix_len] = {"avg_time_sec": avg_time, "avg_embedding_dim": avg_dim}
        #print(f"[DEBUG] Avg encoding time for prefix {prefix_len}: {avg_time:.4f}s, avg dim: {avg_dim}")

    return predicted_labels, anomaly_scores, per_prefix_predictions, per_prefix_truth, per_prefix_counts, csv_records, avg_encoding_times
