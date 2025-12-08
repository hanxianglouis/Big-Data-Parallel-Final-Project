import pandas as pd
import numpy as np
import os

def preprocessing(raw_train_path: str, raw_val_path: str,export_train_path: str, export_val_path: str) :
    if (not raw_train_path.endswith('.csv')) or (not raw_val_path.endswith('.csv')) or (not export_train_path.endswith('.parquet')) or (not export_val_path.endswith('.parquet')) :
        raise ValueError("The paths do not math the type of the files. Raw files must be .csv, and export files must be .parquet")
    
    if os.path.exists(export_train_path) and os.path.exists(export_val_path) :
        print("Proceeded data exists. Skip preprocessing!")
        return
    print("Reading raw data...")
    df_train = pd.read_csv(raw_train_path)[:60000]
    df_val = pd.read_csv(raw_val_path)[:7000]

    feature_cols = [f"f{i}" for i in range(12)]

    train_mean = df_train[feature_cols].mean()
    train_std  = df_train[feature_cols].std()

    # normalize
    df_train[feature_cols] = (df_train[feature_cols] - train_mean) / (train_std + 1e-6)
    df_val[feature_cols]   = (df_val[feature_cols]   - train_mean) / (train_std + 1e-6)

    df_train.to_parquet(export_train_path,index=False)
    df_val.to_parquet(export_val_path,index=False)
    print("Data preprocessing finished!")