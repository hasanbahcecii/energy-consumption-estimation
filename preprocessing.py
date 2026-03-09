import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # normalization
import torch
from torch.utils.data import Dataset

# load clean data
df = pd.read_csv("cleaned_power.csv", index_col= "datetime", parse_dates= True)

# normalization 
scaler = MinMaxScaler()
df["values"] = scaler.fit_transform(df[["Global_active_power"]]) # scale only target value

print(df.head())

# sequence creation function
def create_sequence(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len]) # input: last 24 hours
        y.append(data[i + seq_len]) # target value.  25. hour

    return np.array(X), np.array(y)


SEQ_LEN = 24
series = df["values"].values
X, y = create_sequence(series, SEQ_LEN)


# pytorch dataset definition
class EnergyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1) # (num_samples, seq_len, 1)
        self.y = torch.tensor(y, dtype= torch.float32).unsqueeze(-1) # (num_samples, 1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# train test split
split = int(0.8 * len(X)) # training set: %80, test set: %20
train_dataset = EnergyDataset(X[: split], y[: split])
test_dataset = EnergyDataset(X[split :], y[split :])
