#!/usr/bin/env python3
import pandas as pd
import time
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator  import DataGenerator

def main():
    # 1) Paths
    csv_path = "Data/Real_Datasets/wine_processed.csv"

    # 2) Load and inspect
    df = pd.read_csv(csv_path)
    print("Loaded real data:", df.shape)
    print(df.dtypes)
    print("Missing vals:\n", df.isna().sum())

    # 3) (Optional) Downsample to speed up—but you can skip this if you want the full set
    # df = df.sample(3000, random_state=42).reset_index(drop=True)
    # print("Downsampled to:", df.shape)

    # 4) Describe in correlated‐attribute mode (no binning)
    describer = DataDescriber(category_threshold=3)
    start = time.time()
    describer.describe_dataset_in_correlated_attribute_mode(
        dataset_file=csv_path,
        epsilon=1.0
    )
    print(f"Description done in {time.time() - start:.1f}s")

    # 5) Generate synthetic data
    
    generator = DataGenerator()

    for i in range(1, 6):           # ← 5 samples
        t0 = time.time()
        syn = generator.generate_synthetic_data_from_privbayes(
            describer,
            n=df.shape[0]
        )
        elapsed = time.time() - t0
        fname = f"Data/Synthetic_Datasets/PrivBayes/privbayes_wine_{i}.csv"
        syn.to_csv(fname, index=False)
        print(f"Sample {i} done in {elapsed:.1f}s → saved to {fname}")

if __name__ == "__main__":
    main()

