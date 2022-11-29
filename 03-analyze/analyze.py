#%% Import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.decomposition
import umap
import phate
import scprep

#%% Load data and metadata

data = pd.read_csv(
    "../02-clean/out/batch-corrected-data.csv", header=0, index_col=0
)

metadata = pd.read_csv(
    "../02-clean/out/filtered-metadata.csv", header=0, index_col=0
)

#%% Set healthy and TLE indexes

healthy_idx = metadata["Sample"] == 0
tle_idx = metadata["Sample"] > 0

#%% Normalize library size

normalized_data = scprep.transform.sqrt(
    scprep.normalize.library_size_normalize(data)
)

#%% Train PCA on healthy cells

pca_healthy = sklearn.decomposition.PCA(n_components=50).fit(
    normalized_data[healthy_idx]
)

print("Healthy PCA explained variance:", pca_healthy.explained_variance_ratio_)
print(
    "Healthy PCA explained variance (sum):",
    sum(pca_healthy.explained_variance_ratio_),
)

#%% Train PCA on TLE cells

pca_tle = sklearn.decomposition.PCA(n_components=50).fit(
    normalized_data[tle_idx]
)

print("TLE PCA explained variance:", pca_tle.explained_variance_ratio_)
print(
    "TLE PCA explained variance (sum):", sum(pca_tle.explained_variance_ratio_)
)

#%% Apply PCA

data_healthy_pca = pd.DataFrame(
    pca_healthy.transform(data[healthy_idx]), index=data[healthy_idx].index
)
data_tle_pca = pd.DataFrame(
    pca_tle.transform(data[tle_idx]), index=data[tle_idx].index
)

#%% Plot Healthy PCA

fig, ax = plt.subplots(1, 1)
ax.scatter(
    data_healthy_pca.iloc[:, 0],
    data_healthy_pca.iloc[:, 1],
    label="Healthy",
)
ax.legend()

#%% Plot TLE PCA

fig, ax = plt.subplots(1, 1)
ax.scatter(
    data_tle_pca.iloc[:, 0],
    data_tle_pca.iloc[:, 1],
    label="TLE",
)
ax.legend()

#%% Plot PCA

fig, ax = plt.subplots(1, 1)
ax.scatter(
    data_pca.iloc[:, 0][healthy_idx],
    data_pca.iloc[:, 1][healthy_idx],
    label="Healthy",
)
ax.scatter(
    data_pca.iloc[:, 0][tle_idx],
    data_pca.iloc[:, 1][tle_idx],
    label="TLE",
)
ax.legend()

fig, ax = plt.subplots(1, 1)
ax.scatter(
    data_pca.iloc[:, 2][healthy_idx],
    data_pca.iloc[:, 3][healthy_idx],
    label="Healthy",
)
ax.scatter(
    data_pca.iloc[:, 2][tle_idx],
    data_pca.iloc[:, 3][tle_idx],
    label="TLE",
)
ax.legend()

#%% Train UMAP

umap_fit = umap.UMAP().fit(data_pca)

#%% Apply UMAP

data_umap = pd.DataFrame(umap_fit.transform(data_pca), index=data.index)

#%% Plot UMAP

fig, ax = plt.subplots(1, 1)
ax.scatter(
    data_umap.iloc[:, 0][healthy_idx],
    data_umap.iloc[:, 1][healthy_idx],
    label="Healthy",
)
ax.scatter(
    data_umap.iloc[:, 0][tle_idx],
    data_umap.iloc[:, 1][tle_idx],
    label="TLE",
)
ax.legend()

#%% Set up PHATE

phate_op_healthy = phate.PHATE(n_jobs=-2)
phate_op_tle = phate.PHATE(n_jobs=-2)

#%% Train PHATE on healthy cells

phate_fit_healthy = phate_op_healthy.fit(data_pca[healthy_idx])

#%% Train PHATE on TLE cells

phate_fit_tle = phate_op_tle.fit(data_pca[tle_idx])

#%% Apply PHATE

data_phate_healthy = pd.DataFrame(
    phate_fit_healthy.transform(data_pca[healthy_idx]),
    index=data[healthy_idx].index,
)
data_phate_tle = pd.DataFrame(
    phate_fit_tle.transform(data_pca[tle_idx]),
    index=data[tle_idx].index,
)

#%% Plot healthy PHATE

fig, ax = plt.subplots(1, 1)
for key, group in data_phate_healthy.join(metadata).groupby("cell type"):
    ax.scatter(
        group.iloc[:, 0],
        group.iloc[:, 1],
        label=key,
        # alpha=0.25,
        s=5,
    )
ax.legend()

#%% Plot TLE PHATE

fig, ax = plt.subplots(1, 1)
for key, group in data_phate_tle.join(metadata).groupby("Sample"):
    ax.scatter(
        group.iloc[:, 0],
        group.iloc[:, 1],
        label=f"Sample {key}",
        alpha=0.25,
        s=5,
    )
ax.legend()

#%% Calcululate differential expression report

de_report = scprep.stats.differential_expression(
    data[healthy_idx], data[tle_idx], measure="ttest"
)

#%% Investigate differential expression report

top_genes = de_report.sort_values(by="rank")["ttest"].head(n=100)
print(top_genes)

#%% Print out top differentially expressed genes

with open("out/tle-downregulated.csv", "w") as f:
    for gene, ttest in top_genes.iteritems():
        if ttest < 0:
            f.write(gene + "\n")

with open("out/tle-upregulated.csv", "w") as f:
    for gene, ttest in top_genes.iteritems():
        if ttest > 0:
            f.write(gene + "\n")

# LIMITATION: no cell type markers

#%%
