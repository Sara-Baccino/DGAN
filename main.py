import polars as pl
import torch
from processing import DataProcessor
from model import DGAN
import json

# Load config
cfg = json.load(open("config.json"))

# Load data
df = pl.read_excel("pbc_data.xlsx")

# Process
proc = DataProcessor("config.json")
df_proc = proc.fit_transform(df)

# Convert to tensor
X = torch.tensor(df_proc.select(pl.exclude("patient_id")).to_numpy(), dtype=torch.float)

# Train
model = DGAN(cfg, X.shape[-1])
model.train(X, cfg["model"]["epochs"])

# Generate
X_gen = model.generate(500)

# Post-process
df_gen = pl.DataFrame(X_gen.reshape(-1, X.shape[-1]).detach().numpy(), schema=df_proc.columns)
df_gen = proc.inverse_transform(df_gen)
df_gen = proc.drop_observation_cols(df_gen)

df_gen.write_excel("synthetic_pbc.xlsx")
