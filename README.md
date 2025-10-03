# Thesis: Preservation of Pairwise Correlation in Synthetic Tabular Data

This repository contains the analysis, experiments, and datasets for my Master’s thesis on **synthetic tabular data generation**, with a particular focus on **correlation preservation in generative models **.  
The goal of this work is to compare different deep learning–based and statistical models for their ability to preserve inter-feature dependencies in continuous data.

---

##  Repository Structure

###  Analysis
- **`analysis/`** – Jupyter notebooks for each model/tool evaluation.
  - `CTABGANPlus analysis.ipynb` – Experiments with CTAB-GAN+.  
  - `TVAE analysis.ipynb` – Experiments with TVAE.  
  - `TabSyn-analysis.ipynb` – Experiments with TabSyn.  
  - `TabDIff analysis.ipynb` – Experiments with TabDDPM.  
  - `GaussianCopula analysis.ipynb` – Statistical baseline experiments.  
  - *(and the rest of the models)*

### Model Comparison
- **`model comparison/`** – Cross-model comparison notebooks.  
  - Includes  correlation evaluation between different models on their best performance as well as a common setup

### Data Analysis
- **`Data Processing/`** – Exploratory analysis of the original dataset.  
  - Data profiling, preprocessing steps, and feature distribution checks.  

### Data
- **`data/`** – All datasets used in the thesis.  
  - `real/` – Original datasets (raw, preprocessed, deduplicated).  
  - `synthetic/` – Synthetic datasets organized by generator:  
    - `CTABGANPlus/`  
    - `TVAE/`  
    - `TabDDPM/`  
    - `TabSyn/`  
    - *(etc.)*  

---

##  Requirements

- Python 3.9+  
- Jupyter Notebook / JupyterLab  
- Key packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `torch`  

Install dependencies via:  
```bash
pip install -r requirements.txt
