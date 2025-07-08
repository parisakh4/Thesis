#!/usr/bin/env python3
import sys
# Prepend the parent folder that contains the 'tabula' package
sys.path.insert(0, "/Users/parisakhosravi/Tabula")
#!/usr/bin/env python3
import os
# must come before any torch import!
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
# confirm it took effect
print("MPS high‐watermark ratio:", os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"))
print("MPS available:", torch.backends.mps.is_available())
print("Running on device:", torch.device("mps") if torch.backends.mps.is_available() else "cpu")


import time
import pandas as pd
import torch
from tabula import Tabula


def main():
    # ┌─────────────────────────────────────────────────────────┐
    # │ 1) Paths                                              │
    # └─────────────────────────────────────────────────────────┘
    root = os.path.expanduser("~/Thesis")
    csv_in = os.path.join(root, "Data/Real_Datasets/wine_processed.csv")
    synth_dir = os.path.join(root, "Data/Synthetic_Datasets/TabuLa")
    model_dir = os.path.join(root, "tabula_wine_training")

    os.makedirs(synth_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ┌─────────────────────────────────────────────────────────┐
    # │ 2) Load & preprocess                                  │
    # └─────────────────────────────────────────────────────────┘
    df = pd.read_csv(csv_in)
    # replace spaces with underscores exactly as in your notebook
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    print(f"Loaded data: {df.shape}")
    print(df.dtypes, "\n")

    # ┌─────────────────────────────────────────────────────────┐
    # │ 3) Device detection & TabuLa instantiation             │
    # └─────────────────────────────────────────────────────────┘
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print("→ Using device:", device)

    model = Tabula(
        llm="distilgpt2",
        experiment_dir=model_dir,
        batch_size=32,
        epochs=400,
        categorical_columns=["type"],  # your wine categorical column(s)
        no_cuda=False,                 # **enable** GPU use
        use_mps_device=True            # **allow** MPS
    )
    # ensure the PyTorch model is on the chosen device
    model.model.to(device)

    # ┌─────────────────────────────────────────────────────────┐
    # │ 4) Fine-tune                                         │
    # └─────────────────────────────────────────────────────────┘
    t0 = time.time()
    print("Starting TabuLa fine-tune...")
    model.fit(df)
    print(f"✔ Training done in {time.time() - t0:.1f}s")

    # save final weights
    ckpt = os.path.join(model_dir, "model_state.pt")
    torch.save(model.model.state_dict(), ckpt)
    print("Model weights saved to", ckpt, "\n")

    # ┌─────────────────────────────────────────────────────────┐
    # │ 5) Generate 5 synthetic datasets                     │
    # └─────────────────────────────────────────────────────────┘
    for i in range(1, 6):
        t1 = time.time()
        syn = model.sample(n_samples=len(df), device=device)
        out_csv = os.path.join(synth_dir, f"wine_tabula_{i}.csv")
        syn.to_csv(out_csv, index=False)
        print(f"• Sample {i} saved ({time.time() - t1:.1f}s) → {out_csv}")

    print("\nAll done! Five datasets in", synth_dir)

if __name__ == "__main__":
    main()
