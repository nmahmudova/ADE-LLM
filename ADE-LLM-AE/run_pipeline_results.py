from data_loader import load_data, split_data
from encoder import init_encoder, onehot_encode_data, word2vec_encode_data
from prepare_prefix_data import prepare_prefix_data
from train import train_autoencoders_per_prefix
from detect import detect_anomalies
import pandas as pd
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from encoder import init_word2vec_model


def main(dataset_name, results_dir, encoding_type="onehot"):
    # read only first 100 rows
    #df = pd.read_csv("huge-0.1-2.csv", nrows=1000)
    #results_dir="evaluation-results_run5"
    #os.makedirs(results_dir, exist_ok=True)
    df = load_data(dataset_name)
    train_df, test_df = split_data(df)
    
    start_total = time.time()
    
    if encoding_type == "onehot":
        encoder = init_encoder(train_df)
    elif encoding_type == "word2vec":
        encoder = init_word2vec_model(train_df, vector_size=32)
    else:
        raise ValueError("Unsupported encoding type.")

    prefix_data, prefix_input_dims = prepare_prefix_data(train_df, encoder, encoding_type)
    models, thresholds = train_autoencoders_per_prefix(prefix_data, prefix_input_dims, encoding_type)

    predicted_labels, anomaly_scores, per_prefix_predictions, per_prefix_truth, per_prefix_counts, csv_records, avg_encoding_times = detect_anomalies(
        test_df, models, thresholds, prefix_input_dims, encoder, encoding_type
    )
    
    total_time_sec = time.time() - start_total
    
    # Execution time
    time_df = pd.DataFrame([{
        "dataset": dataset_name,
        "encoding": encoding_type,
        "total_execution_time_sec": total_time_sec
    }])
    time_csv = os.path.join(results_dir, f"execution_time_{dataset_name.split('.csv')[0]}_{encoding_type}_run2.csv")
    time_df.to_csv(time_csv, index=False)

    # Average encoding times
    encoding_times_df = pd.DataFrame([
        {"prefix_len": k, "avg_time_sec": v["avg_time_sec"], "avg_embedding_dim": v["avg_embedding_dim"]}
        for k, v in avg_encoding_times.items()
    ])
    encoding_times_csv = os.path.join(
        results_dir, f"avg_encoding_times_{dataset_name.split('.csv')[0]}_{encoding_type}_run2.csv"
    )
    encoding_times_df.to_csv(encoding_times_csv, index=False)


    # Prefix counts
    prefix_counts_list = []
    for prefix_len, counts in per_prefix_counts.items():
        prefix_counts_list.append({
            "prefix_len": prefix_len,
            "normal_count": counts['normal'],
            "anomaly_count": counts['anomaly']
        })
    
    prefix_counts_df = pd.DataFrame(prefix_counts_list)
    prefix_counts_df = prefix_counts_df.sort_values("prefix_len")
    prefix_counts_csv = os.path.join(results_dir, f"counts_vs_prefix_{dataset_name.split('.csv')[0]}_{encoding_type}_run2.csv")
    prefix_counts_df.to_csv(prefix_counts_csv, index=False)
    
    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(data=prefix_counts_df, x="prefix_len", y="normal_count", marker="o", label="Normal", ax=ax)
    sns.lineplot(data=prefix_counts_df, x="prefix_len", y="anomaly_count", marker="o", label="Anomaly", ax=ax)
    ax.set_title(f"Normal vs Anomalous Instances vs Prefix Length ({dataset_name}, {encoding_type})")
    fig.savefig(os.path.join(results_dir, f"counts_vs_prefix_{dataset_name.split('.csv')[0]}_{encoding_type}_run2.png"), bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved normal/anomaly counts vs prefix length to {prefix_counts_csv}")

    # F1 vs prefix
    prefix_f1_list = []
    for prefix_len in per_prefix_predictions.keys():
        y_pred_prefix = per_prefix_predictions[prefix_len]
        y_true_prefix = per_prefix_truth[prefix_len]
        f1_prefix = f1_score(y_true_prefix, y_pred_prefix, average="weighted", zero_division=0)
        prefix_f1_list.append({"prefix_len": prefix_len, "f1_score": f1_prefix})
    
    prefix_f1_df = pd.DataFrame(prefix_f1_list)
    prefix_f1_df     = prefix_f1_df.sort_values("prefix_len")
    prefix_f1_csv = os.path.join(results_dir, f"f1_vs_prefix_{dataset_name.split('.csv')[0]}_{encoding_type}_run2.csv")
    prefix_f1_df.to_csv(prefix_f1_csv, index=False)
    
    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(data=prefix_f1_df, x="prefix_len", y="f1_score", marker="o", ax=ax)
    ax.set_title(f"F1-score vs Prefix Length ({dataset_name}, {encoding_type})")
    fig.savefig(os.path.join(results_dir, f"f1_vs_prefix_{dataset_name.split('.csv')[0]}_{encoding_type}_run2.png"), bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved F1 vs prefix length to {prefix_f1_csv}")

    case_ids = test_df["case_id"].drop_duplicates().tolist()  # stable order
    is_anom_map = test_df.groupby("case_id")["isAnomaly"].first().to_dict()
    
    y_true = [is_anom_map[cid] for cid in case_ids]
    y_pred = [predicted_labels[cid] for cid in case_ids]
    y_score = [anomaly_scores[cid] for cid in case_ids]

    
    # Metrics
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    auc = roc_auc_score(y_true, y_score)  # binary ROC-AUC
    
    metrics = {
        "dataset": dataset_name,
        "encoding": encoding_type,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    }

    
    metrics_df = pd.DataFrame([metrics])
    metrics_csv = os.path.join(results_dir, f"metrics_{dataset_name.split('.csv')[0]}_{encoding_type}_run2.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    
    metrics_df_melted = metrics_df.melt(id_vars=["dataset","encoding"], 
                                    value_vars=["precision","recall","f1_score","roc_auc"],
                                    var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=metrics_df_melted, x="dataset", y="value", hue="metric", ax=ax)
    ax.set_title(f"Metrics for {dataset_name} ({encoding_type})")
    fig.savefig(os.path.join(results_dir, f"metrics_{dataset_name.split('.csv')[0]}_{encoding_type}_run2.png"), bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved metrics to {metrics_csv}")

    # F1 by anomaly type
    anomaly_type_map = test_df.groupby("case_id")["anomaly"].first().to_dict()
    is_anom_map      = test_df.groupby("case_id")["isAnomaly"].first().to_dict()
    case_ids         = test_df["case_id"].drop_duplicates().tolist()
    
    # Normalize anomaly strings
    anomaly_type_map = {k: str(v).strip().lower() for k, v in anomaly_type_map.items() if pd.notna(v)}
    is_anom_map      = {k: v for k, v in is_anom_map.items()}
    
    f1_by_type = []
    # Only keep anomalies that are not 'normal'
    types = sorted({t for t in anomaly_type_map.values() if t != "normal"})
    for a_type in types:
        ids = [cid for cid in case_ids if anomaly_type_map.get(cid) == a_type]
        if not ids:
            continue
        y_true_sub = [is_anom_map[cid] for cid in ids]
        y_pred_sub = [predicted_labels[cid] for cid in ids]
        f1_sub = f1_score(y_true_sub, y_pred_sub, average="weighted", zero_division=0)
        f1_by_type.append({"anomaly": a_type, "f1_score": f1_sub})

    
    f1_type_df = pd.DataFrame(f1_by_type)
    f1_type_df = f1_type_df[f1_type_df["anomaly"] != "normal"]  # remove normal just in case
    f1_type_df["anomaly"] = f1_type_df["anomaly"].astype(str)
    f1_type_csv = os.path.join(results_dir, f"f1_by_anomaly_type_{dataset_name.split('.csv')[0]}_{encoding_type}_run2.csv")
    f1_type_df.to_csv(f1_type_csv, index=False)
    
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=f1_type_df, x="anomaly", y="f1_score", ax=ax, order=sorted(f1_type_df["anomaly"].unique()))
    ax.set_title(f"F1-score by Anomaly Type ({dataset_name}, {encoding_type})")
    fig.savefig(os.path.join(results_dir, f"f1_by_anomaly_type_{dataset_name.split('.csv')[0]}_{encoding_type}_run2.png"), bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved F1 by anomaly type to {f1_type_csv}")


datasets = [
        "huge-0.1-2.csv", "huge-0.2-2.csv", "huge-0.3-2.csv",
        "large-0.1-2.csv", "large-0.2-2.csv", "large-0.3-2.csv",
        "medium-0.1-2.csv", "medium-0.2-2.csv", "medium-0.3-2.csv",
        "small-0.1-2.csv", "small-0.3-2.csv",
        "wide-0.1-2.csv", "wide-0.2-2.csv", "wide-0.3-2.csv"
]

encoding_type = "onehot"  # or "onehot"
num_runs = 5

for dataset_name in datasets:
    for run in range(1, num_runs + 1):
        results_dir = f"evaluation-results/{dataset_name.split('.csv')[0]}/onehot/run_{run}"
        os.makedirs(results_dir, exist_ok=True)
        print(f"Running {dataset_name}, Run {run} -> Results: {results_dir}")

        # Pass the results_dir to your main function
        main(dataset_name, results_dir=results_dir, encoding_type=encoding_type)
        
