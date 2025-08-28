import pandas as pd
import numpy as np

def load_data(filepath):
    #print(f"[DEBUG] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    #print(f"[DEBUG] Data shape: {df.shape}")
    #print(f"[DEBUG] Columns: {df.columns.tolist()}")
    #print(f"[DEBUG] Sample rows:\n{df.head(3)}")

    # Optional: convert timestamp (uncomment if needed)
    #df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    #df = df.dropna(subset=['timestamp'])

    # Optional: sort by case_id or timestamp
    #df.sort_values(by=['case_id'], inplace=True)
    #df.sort_values(by=['case_id', 'timestamp'], inplace=True)
    
    return df

def split_data(df, train_ratio=0.8):
    np.random.seed(42)  # for reproducibility
    #print(f"[DEBUG] Splitting data: train_ratio={train_ratio}")
    
    # Get only normal traces (case_ids)
    normal_case_ids = df[df['isAnomaly'] == 0]['case_id'].unique()
    #print(f"[DEBUG] Total normal cases: {len(normal_case_ids)}")
    
    normal_case_ids = np.random.permutation(normal_case_ids)
    train_size = int(len(normal_case_ids) * train_ratio)
    train_cases = normal_case_ids[:train_size]
    #print(f"[DEBUG] Number of training normal cases: {len(train_cases)}")

    # Train df contains only normal traces from train_cases
    train_df = df[(df['case_id'].isin(train_cases)) & (df['isAnomaly'] == 0)]
    #print(f"[DEBUG] Train DF shape: {train_df.shape}")

    # Test df contains all cases NOT in train_cases (can be normal or anomalous)
    test_df = df[~df['case_id'].isin(train_cases)]
    #print(f"[DEBUG] Test DF shape: {test_df.shape}")
    
    return train_df, test_df


def load_training_and_testing_data(train_path="train_dataset.csv", test_path="test_dataset_1.csv"):
    #print(f"[DEBUG] Loading training data from: {train_path}")
    df = pd.read_csv(train_path)
    df.columns = ["case_id", "name"]
    #print(f"[DEBUG] Training DF loaded, shape: {df.shape}")

    #print(f"[DEBUG] Loading testing data from: {test_path}")
    test_df = pd.read_csv(test_path)
    test_df.columns = ["case_id", "name", "isAnomaly"]
    #print(f"[DEBUG] Testing DF loaded, shape: {test_df.shape}")

    # For demo purposes, limit rows
    train_df = df.head(800)
    test_df = test_df.head(200)
    #print(f"[DEBUG] Using first 800 rows for training, 200 rows for testing")

    # Sort training data by case_id
    train_df.sort_values(by=['case_id'], inplace=True)
    test_df.sort_values(by=['case_id'], inplace=True)
    
    #print(f"[DEBUG] Train DF sorted by case_id. Sample rows:\n{train_df.head(3)}")
    #print(f"[DEBUG] Test DF sorted by case_id. Sample rows:\n{test_df.head(3)}")

    return train_df, test_df
