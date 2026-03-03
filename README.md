# Py-cSKL: Scalable Molecular Similarity for Biological Datasets

Py-cSKL is a highly scalable, Python-based implementation of the Curated Symmetric Kullback-Leibler (c-SKL) divergence method. It is designed to compute statistical similarities between high-dimensional, low-sample omics datasets (e.g., transcriptomics, methylomics) based purely on their molecular data distributions.

This library is a faithful, modernized Python port of the original methodology and R implementation introduced in:
> *A data driven approach reveals disease similarity on a molecular level* (Lakiotaki et al., 2019, npj Systems Biology and Applications).

## Features

* **Fast PCA Signatures:** Compute low-rank PCA signatures for datasets with automatic handling of zero-variance features and standardization.
* **c-SKL Computation:** Efficiently calculate the distributional similarity between datasets using Eq. 1 of the c-SKL methodology.
* **Statistical Significance:** Perform semi-parametric bootstrap testing with advanced in-memory caching to quickly generate $p$-values and Benjamini-Hochberg FDR-adjusted $q$-values.
* **Feature Explanations ($B(k)$ & $W(k)$):** Identify the top $k$ molecular features (e.g., genes) that best explain why two datasets are similar or dissimilar using alternating optimization.
* **Interactive Network Visualization:** Automatically generate an interactive, physics-based HTML network graph to explore the biological data landscape.

## Repository Structure

* `cskl.py`: The core mathematical library containing the PCA signature generation, c-SKL metric, statistical pooling, and explainers.
* `replicate_oldway.py`: A complete end-to-end pipeline script to ingest GEO datasets, align features, remove shared profiles, and compute the similarity network.
* `generate_network_html.py`: A utility to parse the network outputs and generate an interactive `vis.js` HTML visualization.

## Quick Start & Demo

### 1. Requirements
Ensure you have Python 3.8+ installed along with the required scientific libraries:
```bash
pip install numpy scipy pandas

```

*(Note: If you are pulling raw `.tar` files from GEO requiring `SCAN.UPC` normalization, an active R installation with the required Bioconductor packages is also needed).*

### 2. Running the Pipeline

You can run the end-to-end pipeline on pre-processed dataset CSV matrices. The pipeline aligns features, filters duplicates, and computes the network.

```bash
python replicate_oldway.py \
    --csvs "data/GSE*.csv" \
    --outdir "./results" \
    --alpha 0.5 \
    --B 100 

```

### 3. Generating the Interactive Network

Once the pipeline finishes, generate the visual landscape of your datasets:

```bash
python generate_network_html.py

```

Open the resulting `interactive_cskl_network.html` in your browser to interactively explore the dataset communities and feature explanations!

## Methodological Faithfulness

This Python implementation was carefully built to reproduce the mathematical behaviors of the original `BioDataome` R repository. It maintains identical data standardization techniques (including deterministic noise injection for flat signals), accurate eigenvalue scaling, and faithful semi-parametric bootstrapping, while offering significant improvements in speed and memory management.

## Citation

If you use this library or the underlying c-SKL methodology, please cite:

> Lakiotaki, K., Georgakopoulos, G., Castanas, E. et al. A data driven approach reveals disease similarity on a molecular level. *npj Syst Biol Appl* 5, 39 (2019). https://doi.org/10.1038/s41540-019-0117-0
