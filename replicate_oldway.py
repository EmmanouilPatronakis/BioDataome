#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import tarfile
import inspect
from pathlib import Path

import glob
import numpy as np
import pandas as pd

import cskl

# ---------------------------------------------------------------------
# R script written per-dataset into outdir/work/<GSE>/scan_normalize_affy.R
# ---------------------------------------------------------------------
R_NORMALIZE_SCRIPT = r"""
suppressPackageStartupMessages({
  library(SCAN.UPC)
  library(Biobase)
})

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript scan_normalize_affy.R <cel_root_dir> <out_tsv_gz>")
}

raw_root <- normalizePath(args[1], winslash="/", mustWork=TRUE)
out_path <- args[2]

# Find CEL / CEL.gz recursively (GEO raw tars can have nested folders)
cel_files <- list.files(
  raw_root,
  pattern="\\.cel(\\.gz)?$",
  full.names=TRUE,
  recursive=TRUE,
  ignore.case=TRUE
)

if (length(cel_files) == 0) {
  stop(paste("No CEL/CEL.gz files found under:", raw_root))
}

# Flatten into a temp dir so we can pass ONE wildcard to SCAN(), BioDataome-style.
flat_dir <- file.path(tempdir(), paste0("scan_upc_cels_", as.integer(Sys.time())))
dir.create(flat_dir, recursive=TRUE, showWarnings=FALSE)
flat_dir <- normalizePath(flat_dir, winslash="/", mustWork=TRUE)

# Copy files (avoid touching the extracted raw tree)
ok <- file.copy(cel_files, flat_dir, overwrite=FALSE)
if (!all(ok)) {
  # If name collisions happen (rare), fall back to overwrite=TRUE
  file.copy(cel_files[!ok], flat_dir, overwrite=TRUE)
}

pattern <- paste0(flat_dir, "/*.CEL*")  # ONE string; matches .CEL and .CEL.gz

message("SCAN.UPC::SCAN('", pattern, "') with ", length(cel_files), " files")
eset <- SCAN.UPC::SCAN(pattern, outFilePath=NA, verbose=TRUE)
mat <- Biobase::exprs(eset)  # probes x samples

con <- gzfile(out_path, "wt")
write.table(mat, file=con, sep="\t", quote=FALSE, col.names=NA)
close(con)

cat("Wrote:", out_path, "\n")
"""


def run(cmd, **kwargs):
    print("+", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, **kwargs)


def safe_name_from_file(file_path: Path) -> str:
    # e.g. GSE35570_RAW.tar -> GSE35570 or GSE10327.csv -> GSE10327
    m = re.match(r"(GSE\d+)", file_path.name, re.IGNORECASE)
    return m.group(1).upper() if m else file_path.stem.replace("_RAW", "").upper()


def extract_tar(tar_path: Path, dest_dir: Path) -> None:
    """
    Extract tar to dest_dir.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        sig = inspect.signature(tf.extractall)
        if "filter" in sig.parameters:
            tf.extractall(dest_dir, filter="data")
        else:
            tf.extractall(dest_dir)


def write_r_script(path: Path) -> None:
    path.write_text(R_NORMALIZE_SCRIPT, encoding="utf-8")


def normalize_affy_scan(tar_path: Path, work_root: Path, rscript: str) -> Path:
    gse = safe_name_from_file(tar_path)
    gse_dir = work_root / gse
    raw_dir = gse_dir / "raw_extracted"
    out_expr = gse_dir / "expr.tsv.gz"
    gse_dir.mkdir(parents=True, exist_ok=True)

    if out_expr.exists():
        print(f"[{gse}] Using existing normalized matrix: {out_expr}")
        return out_expr

    print(f"[{gse}] Extracting {tar_path} -> {raw_dir}")
    extract_tar(tar_path, raw_dir)

    rfile = gse_dir / "scan_normalize_affy.R"
    write_r_script(rfile)

    print(f"[{gse}] Normalizing with SCAN.UPC -> {out_expr}")
    run([rscript, str(rfile), str(raw_dir), str(out_expr)])
    return out_expr


def load_expr(expr_path: Path) -> pd.DataFrame:
    # Adapt reading parameters based on file extension
    if expr_path.name.endswith(".csv"):
        df = pd.read_csv(expr_path, sep=",", index_col=0)
    else:
        df = pd.read_csv(expr_path, sep="\t", compression="gzip", index_col=0)

    # Force numeric (SCAN and BioDataome CSVs should already be numeric)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def compute_global_good_features(expr: dict[str, pd.DataFrame], common: list[str]) -> list[str]:
    """
    Finds features that are finite (no NaNs/Infs) across ALL datasets.

    Note: Zero-variance features are intentionally kept here so cskl.py can inject deterministic noise later, matching the original R implementation.
    """
    good_all = None
    for name, df in expr.items():
        X = df.loc[common].to_numpy(dtype=np.float64)

        # Only check that the values are mathematically valid (finite).
        finite = np.isfinite(X).all(axis=1)

        good_all = finite if good_all is None else (good_all & finite)

    return [f for f, ok in zip(common, good_all) if ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars", nargs="+", default=[], help="Paths to GSE*_RAW.tar files")
    ap.add_argument("--csvs", nargs="+", default=[], help="Paths to pre-processed dataset CSV files")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument("--rscript", default="Rscript", help="Rscript executable")
    ap.add_argument("--alpha", type=float, default=0.5, help="Explained variance threshold (paper used 0.5).")
    ap.add_argument("--a", type=float, default=0.5,
                    help="(Ignored) Mixing parameter. cskl.py uses symmetric 0.5 natively.")
    ap.add_argument("--B", type=int, default=100, help="Bootstrap iterations for significance testing (StatSig.R).")
    ap.add_argument("--pool_matrix", type=str, default=None,
                    help="Path to a global background pool TSV/CSV (features x samples).")
    args = ap.parse_args()

    # Expand wildcards for CSVs
    expanded_csvs = []
    for c in args.csvs:
        # glob.glob expands the path. If it finds nothing, it keeps the original string to fail gracefully later.
        expanded_csvs.extend(glob.glob(c) if glob.glob(c) else [c])
    args.csvs = expanded_csvs

    # Expand wildcards for TARs
    expanded_tars = []
    for t in args.tars:
        expanded_tars.extend(glob.glob(t) if glob.glob(t) else [t])
    args.tars = expanded_tars

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    work_root = outdir / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    # 1) Load datasets (normalize TARs and/or read direct CSVs)
    expr = {}
    names = []

    if args.tars:
        for t in args.tars:
            tp = Path(t).resolve()
            gse = safe_name_from_file(tp)
            if gse not in names:
                names.append(gse)
                expr_path = normalize_affy_scan(tp, work_root, args.rscript)
                expr[gse] = load_expr(expr_path)

    if args.csvs:
        for c in args.csvs:
            cp = Path(c).resolve()
            gse = safe_name_from_file(cp)
            if gse not in names:
                names.append(gse)
                print(f"[{gse}] Loading pre-processed CSV directly: {cp}")
                expr[gse] = load_expr(cp)

    if not names:
        ap.error("Must provide at least one valid dataset via --tars or --csvs.")

    # 2) Load global background pool (if provided)
    pool_df = None
    if args.pool_matrix:
        print(f"Loading global background pool from {args.pool_matrix}...")
        pool_df = load_expr(Path(args.pool_matrix))
        # Temporarily add to expr to ensure perfect feature intersection
        expr["__GLOBAL_POOL__"] = pool_df

    # 3) Align features across all datasets (and pool, if provided)
    common = None

    for _, df in expr.items():
        idx = df.index.astype(str)
        common = set(idx) if common is None else (common & set(idx))
    common = sorted(common)

    if len(common) < 1000:
        raise SystemExit(
            f"Too few common features across datasets ({len(common)}). "
            f"Likely different platforms; you must map to a common gene set first."
        )

    # compute ONE shared "good feature" set (finite) across ALL datasets
    good_feats = compute_global_good_features(expr, common)
    if len(good_feats) < 1000:
        raise SystemExit(
            f"Too few usable common features after filtering ({len(good_feats)}). "
            f"Check for missing/constant rows or mismatched platforms."
        )

    print(f"Common features across all datasets: {len(common)}")
    print(f"Usable common features (finite in all datasets): {len(good_feats)}")

    # 3.5) Filter out datasets with shared molecular profiles (Faithful to the paper's methodology)
    # The paper explicitly states: "To account for shared profiles, in this work, we remove all
    # datasets with at least one shared profile".
    # In the original R code (compareDsets.R), this is done using a numeric matrix check.
    # We replicate this efficiently by checking the Pearson correlation of sample columns.

    datasets_to_drop = set()
    # Only compare the target datasets (skip the pool if it was temporarily added)
    dataset_names = [k for k in expr.keys() if k != "__GLOBAL_POOL__"]

    for i in range(len(dataset_names)):
        name_a = dataset_names[i]
        if name_a in datasets_to_drop:
            continue

        # Standardize samples (columns) to compute correlation matrix efficiently
        A = expr[name_a].loc[good_feats].to_numpy(dtype=np.float64)
        # Avoid division by zero warnings by using a safe standard deviation
        A_sd = A.std(axis=0, ddof=1)
        A_sd[A_sd == 0] = 1.0
        A_norm = (A - A.mean(axis=0)) / A_sd

        for j in range(i + 1, len(dataset_names)):
            name_b = dataset_names[j]
            if name_b in datasets_to_drop:
                continue

            B = expr[name_b].loc[good_feats].to_numpy(dtype=np.float64)
            B_sd = B.std(axis=0, ddof=1)
            B_sd[B_sd == 0] = 1.0
            B_norm = (B - B.mean(axis=0)) / B_sd

            # Compute correlation matrix between samples of A and B
            # Shape will be: (n_samples_A, n_samples_B)
            corr_matrix = (A_norm.T @ B_norm) / (len(good_feats) - 1)

            # If any correlation is extremely high (~1.0), they share a profile
            if np.any(corr_matrix > 0.999):
                print(f"[Filter] Flagged shared molecular profiles between {name_a} and {name_b}.")
                # Drop the smaller dataset to retain as much unique data as possible
                if A.shape[1] < B.shape[1]:
                    datasets_to_drop.add(name_a)
                else:
                    datasets_to_drop.add(name_b)

    # Apply the drops
    for drop_name in datasets_to_drop:
        print(f"[Filter] Dropping {drop_name} from analysis to prevent artificial similarity artifacts.")
        del expr[drop_name]
        if drop_name in names:
            names.remove(drop_name)

    if pool_df is not None:
        del expr["__GLOBAL_POOL__"]

    # 4) PCA decomposition per dataset using cskl.py
    signatures = {}
    metas = {}
    for name, df in expr.items():
        # Transpose: cskl.py expects shape (n_samples, n_features)
        X = df.loc[good_feats].to_numpy(dtype=np.float64).T

        # fit_pca_signature handles noise injection, standardization, and proper eigenvalue scaling!
        sig = cskl.fit_pca_signature(X, alpha=args.alpha, feature_names=good_feats)
        signatures[name] = sig

        metas[name] = {
            "n_features_common_before_filter": int(len(common)),
            "n_features_used": sig.n_features,
            "n_samples": sig.m_samples,
            "c_components": len(sig.lam),
            "alpha": float(sig.alpha),
            "lambda_sum": float(np.sum(sig.lam)),
        }
        print(
            f"[{name}] samples={sig.m_samples}  "
            f"features_used={sig.n_features}  "
            f"c={len(sig.lam)}  "
            f"lambda_sum={np.sum(sig.lam):.6f}"
        )

    (outdir / "pca_meta.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")

    # 5) Pairwise c-SKL matrix
    mat = np.zeros((len(names), len(names)), dtype=np.float64)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            # Using the mathematically correct cskl function
            mat[i, j] = cskl.cskl(signatures[ni], signatures[nj])

    df_cskl = pd.DataFrame(mat, index=names, columns=names)
    df_cskl.to_csv(outdir / "cskl_matrix.tsv", sep="\t")

    df_sim = 1.0 / (1.0 + df_cskl)
    df_sim.to_csv(outdir / "cskl_similarity_1_over_1_plus.tsv", sep="\t")

    print("\n=== c-SKL divergence matrix (lower = more similar) ===")
    print(df_cskl.round(8).to_string())

    # 6) Statistical Significance Testing Network (Matches StatSig.R)
    if len(names) > 1:
        print("\nBuilding statistical significance network...")

        if pool_df is not None:
            # Safely extract the perfectly aligned features for the global pool
            X_pool_all = pool_df.loc[good_feats].to_numpy(dtype=np.float64).T
        else:
            print(
                "\nWARNING: No --pool_matrix provided. Significance testing will use ONLY the queried datasets as the background.")
            print("This lacks statistical power. Please provide a global platform pool for accurate p-values.\n")
            # Fallback to the current logic
            X_pool_all = np.vstack([df.loc[good_feats].to_numpy(dtype=np.float64).T for df in expr.values()])

        pool = cskl.Pool(X_pool_all, alpha=args.alpha, feature_names=good_feats)

        all_pairs, kept_edges = cskl.build_dataset_network(signatures, pool, B=args.B, fdr_alpha=0.05)

        # Save network edges
        network_df = pd.DataFrame(all_pairs, columns=["Dataset_A", "Dataset_B", "cSKL", "p_value", "q_value"])
        network_df.to_csv(outdir / "cskl_network_edges.tsv", sep="\t", index=False)
        print(f"Discovered {len(kept_edges)} statistically significant similarities (FDR <= 0.05).")

    print("\nWrote:")
    print(" -", outdir / "cskl_matrix.tsv")
    print(" -", outdir / "cskl_similarity_1_over_1_plus.tsv")
    if len(names) > 1:
        print(" -", outdir / "cskl_network_edges.tsv")
    print(" -", outdir / "pca_meta.json")
    print(" - normalized matrices under:", outdir / "work" / "<GSE>" / "expr.tsv.gz")


if __name__ == "__main__":
    main()
