import torch
from torch.utils.data import Dataset
import pandas as pd

class CriteoDataset(Dataset):
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path)   # lazy，不会真的全部加载

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        features = torch.tensor(
            [row[f"f{i}"] for i in range(12)], dtype=torch.float32
        )
        treatment = torch.tensor(row["treatment"], dtype=torch.long)

        cost = torch.tensor(
            row["visit"],
            dtype=torch.float32
        )

        revenue = torch.tensor(
            row["conversion"],
            dtype=torch.float32
        )
        return features, treatment, cost, revenue