#%% Import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats
import sklearn.decomposition
import sklearn.cluster
import umap
import phate
import scprep

#%% Load data and metadata

data = pd.read_csv(
    # "../02-clean/out/batch-corrected-data.csv"
    "../02-clean/out/filtered-data.csv",
    header=0,
    index_col=0,
)

metadata = pd.read_csv(
    "../02-clean/out/filtered-metadata.csv", header=0, index_col=0
)

#%% Set marker genes

# marker_genes = ["AQP4", "GFAP", "PDGFRA", "BCAS1", "AIF1"]
marker_genes = ["PAX6", "SOX2", "GAD1", "GAD2", "GFAP"]

#%% Set healthy and TLE indexes

healthy_idx = metadata["Sample"] == 0
tle_idx = metadata["Sample"] > 0

data_healthy = data[healthy_idx]
data_tle = data[tle_idx]

#%% Normalize library size

# log10(1 + x)

normalized_data_healthy = scprep.transform.log(
    scprep.normalize.library_size_normalize(data_healthy)
)

normalized_data_tle = scprep.transform.log(
    scprep.normalize.library_size_normalize(data_tle)
)

#%% Train PCA on healthy cells

pca_healthy = sklearn.decomposition.PCA(n_components=200).fit(
    normalized_data_healthy
)

# print("Healthy PCA explained variance:", pca_healthy.explained_variance_ratio_)
print(
    "Healthy PCA explained variance (sum):",
    sum(pca_healthy.explained_variance_ratio_),
)

#%% Train PCA on TLE cells

pca_tle = sklearn.decomposition.PCA(n_components=200).fit(normalized_data_tle)

# print("TLE PCA explained variance:", pca_tle.explained_variance_ratio_)
print(
    "TLE PCA explained variance (sum):", sum(pca_tle.explained_variance_ratio_)
)

#%% Apply PCA

data_healthy_pca = pd.DataFrame(
    pca_healthy.transform(data_healthy), index=data_healthy.index
)
data_tle_pca = pd.DataFrame(pca_tle.transform(data_tle), index=data_tle.index)

#%% Plot Healthy PCA

fig, ax = plt.subplots(1, 1)
ax.scatter(
    data_healthy_pca.iloc[:, 0],
    data_healthy_pca.iloc[:, 1],
    label="Healthy",
)
ax.legend()

#%% Plot TLE PCA

for g in marker_genes:
    fig, ax = plt.subplots(1, 1)
    sc = ax.scatter(
        data_tle_pca.iloc[:, 0],
        data_tle_pca.iloc[:, 1],
        label="TLE",
        c=normalized_data_tle[g],
        s=1,
        cmap="inferno",
    )
    fig.colorbar(sc)
    ax.set_title(g)
    ax.legend()

#%% Train UMAP on healthy cells

umap_healthy_fit = umap.UMAP().fit(data_healthy_pca)

#%% Train UMAP on TLE cells

umap_tle_fit = umap.UMAP().fit(data_tle_pca)

#%% Apply UMAP

data_healthy_umap = pd.DataFrame(
    umap_healthy_fit.transform(data_healthy_pca), index=data_healthy_pca.index
)
data_tle_umap = pd.DataFrame(
    umap_tle_fit.transform(data_tle_pca), index=data_tle_pca.index
)

#%% Plot TLE UMAP

for g in marker_genes:
    fig, ax = plt.subplots(1, 1)
    sc = ax.scatter(
        data_tle_umap.iloc[:, 0],
        data_tle_umap.iloc[:, 1],
        label="TLE",
        c=normalized_data_tle[g],
        s=1,
        cmap="inferno",
    )
    fig.colorbar(sc)
    ax.set_title(g)
    ax.legend()

#%% Prediction strength code from Homework 4

from sklearn.model_selection import train_test_split
from itertools import combinations
from scipy import sparse


def get_comemberships(dat, n_clusters, model, kwargs={}):
    ## helper function
    # returns comembership matrix and model
    model = model(n_clusters=n_clusters, **kwargs)
    model.fit(dat)

    labels = model.predict(dat)
    comember = labels_to_mat(labels, n_clusters, N=dat.shape[0])

    return comember, model


def labels_to_mat(labels, n_clusters, N):
    ## helper function
    # comembership matrix helper

    # create and fill an NxN co-membership matrix
    comember = np.zeros((N, N), dtype=bool)
    for i in range(n_clusters):
        match_idx = np.where(labels == i)[0]

        # slow, memory safe (we will symmetrize + add diagonal after)
        match_idx = iter(combinations(match_idx, 2))
        for (j, k) in match_idx:
            comember[j, k] = True

    # transpose here is memory bottleneck
    comember = np.maximum(comember, comember.transpose())
    comember[np.diag_indices_from(comember)] = True

    # sparsify output for storage space
    comember = sparse.csr_matrix(comember)
    return comember


def prediction_strength(
    full_dat,
    n_clusters,
    n_CV=3,
    verbose=False,
    model=sklearn.cluster.KMeans,
    kwargs={},
):
    ## USE THIS FUNCTION :)
    test_size = 1.0 / n_CV
    res = []
    for i in range(n_CV):
        train_dat, test_dat = train_test_split(full_dat, test_size=test_size)
        N_test = test_dat.shape[0]
        __, train_model = get_comemberships(
            train_dat, n_clusters, model=model, kwargs=kwargs
        )
        ground_truth, __ = get_comemberships(
            test_dat, n_clusters, model=model, kwargs=kwargs
        )

        pred_labels = labels_to_mat(
            train_model.predict(test_dat), n_clusters, N_test
        )

        # when match_count is N_test**2, there is no error
        match_count = ground_truth.minimum(pred_labels).sum()
        res.append(match_count / (N_test**2))

    if verbose:
        print(res)

    return np.mean(res)


#%% Choose k-means optimal TLE clusters

start = 2
end = 15

pred_strengths = []
for i in range(start, end):
    pred_strengths.append(prediction_strength(data_tle_umap, i))

x = np.arange(start, end, 1)
y = np.array(pred_strengths)

fig, ax = plt.subplots(1, 1)
ax.set_xticks(x)
ax.plot(x, y)

#%% Train and apply k-means on TLE cells

n_tle_clusters = 5
kmeans_tle = sklearn.cluster.KMeans(n_clusters=n_tle_clusters).fit_predict(
    data_tle_umap
)

#%% Update metadata to include k-means

metadata["Cluster"] = -1
metadata.loc[tle_idx, "Cluster"] = kmeans_tle

#%% Plot k-means

fig, ax = plt.subplots(1, 1)
sc = ax.scatter(
    data_tle_umap.iloc[:, 0],
    data_tle_umap.iloc[:, 1],
    label="TLE",
    c=metadata[tle_idx]["Cluster"],
    s=1,
    cmap="inferno",
)
# fig.colorbar(sc)
ax.legend()

#%% Look at representative genes

for g in marker_genes:
    scprep.plot.jitter(
        metadata[tle_idx]["Cluster"],
        normalized_data_tle[g],
        c=metadata[tle_idx]["Cluster"],
        figsize=(12, 5),
        legend_anchor=(1, 1),
        xlabel="Cluster",
        ylabel=f"{g} expression level",
        title=g,
    )
    # fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    # ax.violinplot(
    #     [
    #         normalized_data_tle[metadata[tle_idx]["Cluster"] == i][g].values
    #         for i in range(0, n_tle_clusters)
    #     ]
    # )
    # ax.set_xlabel("Cluster")
    # ax.set_ylabel(f"{g} expression level")
    # ax.set_title(g)

# Clusters:
#   1 and 4: astrocyte
#   1, 4, (6, 0): reactive astrocyte
#
#   0: premyelinating oligodendrocyte
#   6: microglia
#

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

#%% Compute astrocyte differential expression

astrocytes_healthy = normalized_data_healthy[
    metadata[healthy_idx]["cell type"] == "astrocytes"
]
astrocytes_tle = normalized_data_tle[data_tle["GFAP"] > 1]

de_report_data = {"statistic": [], "corrected_pvalue": [], "fold_change": []}
num_genes = len(astrocytes_healthy.columns)

# Limitation: t-test not appropriate for counts
for gene in astrocytes_healthy.columns:
    result = scipy.stats.ttest_ind(
        astrocytes_healthy[gene], astrocytes_tle[gene]
    )
    de_report_data["statistic"].append(result.statistic)
    de_report_data["corrected_pvalue"].append(result.pvalue * num_genes)
    de_report_data["fold_change"].append(
        astrocytes_tle[gene].mean() / (astrocytes_healthy[gene].mean() + 0.01)
    )

de_report = pd.DataFrame(
    de_report_data, index=astrocytes_healthy.columns.values
)

#%% Plot astrocyte differential expression

significant_idx = de_report["corrected_pvalue"] <= 0.5

x_sig = np.log2(de_report[significant_idx]["fold_change"])
y_sig = -np.log10(de_report[significant_idx]["corrected_pvalue"])

x_insig = np.log2(de_report[~significant_idx]["fold_change"])
y_insig = -np.log10(de_report[~significant_idx]["corrected_pvalue"])

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.scatter(x_sig, y_sig, s=1, c="red", label="Significant")
ax.scatter(x_insig, y_insig, s=1, c="black", label="Not significant")
ax.set_xlabel("log2(Fold change)")
ax.set_ylabel("-log10(Bonferroni corrected p-value)")
ax.legend()

# Annotate genes with low corrected p-value

for gene, row in de_report[significant_idx].iterrows():
    x = np.log2(row["fold_change"])
    y = -np.log10(row["corrected_pvalue"])
    if y > 22:
        adjusted_x = x + 0.1
        if gene == "PTPRS":
            adjusted_x -= 0.8
        if gene == "AASS":
            adjusted_x -= 0.7
        ax.annotate(gene, (adjusted_x, y + 0.1))

#%% Save significant astrocyte differential expression

de_report[significant_idx].sort_values(by="corrected_pvalue").to_csv(
    "out/significant-astrocyte-differential-expression.csv"
)

np.savetxt(
    "out/significant-astrocyte-overexpressed-genes.csv",
    de_report[significant_idx & (de_report["fold_change"] > 1)]
    .sort_values(by="corrected_pvalue")
    .index.values,
    delimiter="\n",
    fmt="%s",
)

np.savetxt(
    "out/significant-astrocyte-underexpressed-genes.csv",
    de_report[significant_idx & (de_report["fold_change"] < 1)]
    .sort_values(by="corrected_pvalue")
    .index.values,
    delimiter="\n",
    fmt="%s",
)

#%%

print(
    "Number overexpressed in TLE:",
    len(de_report[significant_idx & (de_report["fold_change"] > 1)]),
)
print(
    "Percent of significant overexpressed in TLE:",
    len(de_report[significant_idx & (de_report["fold_change"] > 1)])
    / len(de_report[significant_idx]),
)
print(
    "Number underexpressed in TLE:",
    len(de_report[significant_idx & (de_report["fold_change"] < 1)]),
)
print(
    "Percent of significant underexpressed in TLE:",
    len(de_report[significant_idx & (de_report["fold_change"] < 1)])
    / len(de_report[significant_idx]),
)

#%% Get astrocyte background genes

astrocyte_healthy_gene_expr = astrocytes_healthy.sum(axis=0)
astrocyte_tle_gene_expr = astrocytes_tle.sum(axis=0)

astrocyte_healthy_gene_low = 1
fig, ax = plt.subplots(1, 1)
ax.hist(astrocyte_healthy_gene_expr, bins=50)
ax.axvline(astrocyte_healthy_gene_low, color="red")
ax.set_title("Astrocyte Healthy Gene Expression")

astrocyte_tle_gene_low = 2
fig, ax = plt.subplots(1, 1)
ax.hist(astrocyte_tle_gene_expr, bins=50)
ax.axvline(astrocyte_tle_gene_low, color="red")
ax.set_title("Astrocyte TLE Gene Expression")

nonlow_astrocyte_expression_idx = (
    (astrocyte_healthy_gene_expr >= astrocyte_healthy_gene_low)
    | (astrocyte_tle_gene_expr >= astrocyte_tle_gene_low)
).values

nonlow_astrocyte_genes = astrocytes_healthy.columns[
    nonlow_astrocyte_expression_idx
].values

#%% Save astrocyte background genes

np.savetxt(
    "out/astrocyte-background-genes.csv",
    nonlow_astrocyte_genes,
    delimiter="\n",
    fmt="%s",
)
