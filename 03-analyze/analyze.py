#%% Import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats
import scprep

#%% Load data and metadata

data = pd.read_csv(
    "../02-clean/out/batch-corrected-data.csv",
    # "../02-clean/out/filtered-data.csv",
    header=0,
    index_col=0,
)

metadata = pd.read_csv(
    "../02-clean/out/filtered-metadata.csv", header=0, index_col=0
)

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

#%% Compute astrocyte differential expression

astrocytes_healthy = normalized_data_healthy[
    metadata[healthy_idx]["cell type"] == "astrocytes"
]
astrocytes_tle = normalized_data_tle[data_tle["GFAP"] > 1]

de_report_data = {"statistic": [], "corrected_pvalue": [], "fold_change": []}
num_genes = len(astrocytes_healthy.columns)

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

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.scatter(x_sig, y_sig, s=1, c="red", label="Significant")
ax.scatter(x_insig, y_insig, s=1, c="black", label="Not significant")
ax.set_xlabel("log2(Fold change)")
ax.set_ylabel("-log10(Bonferroni corrected p-value)")
ax.legend()

fig.savefig("out/BATCH-astrocyte-volcano.png")
# fig.savefig("out/astrocyte-volcano.png")

#%% Save significant astrocyte differential expression

de_report[significant_idx].sort_values(by="corrected_pvalue").to_csv(
    "out/BATCH-significant-astrocyte-differential-expression.csv"
    # "out/significant-astrocyte-differential-expression.csv"
)

np.savetxt(
    "out/BATCH-significant-astrocyte-overexpressed-genes.csv",
    # "out/significant-astrocyte-overexpressed-genes.csv",
    de_report[significant_idx & (de_report["fold_change"] > 1)]
    .sort_values(by="corrected_pvalue")
    .index.values,
    delimiter="\n",
    fmt="%s",
)

np.savetxt(
    "out/BATCH-significant-astrocyte-underexpressed-genes.csv",
    # "out/significant-astrocyte-underexpressed-genes.csv",
    de_report[significant_idx & (de_report["fold_change"] < 1)]
    .sort_values(by="corrected_pvalue")
    .index.values,
    delimiter="\n",
    fmt="%s",
)

#%% Print out some basic information about over/under-expression

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
    "out/BATCH-astrocyte-background-genes.csv",
    # "out/astrocyte-background-genes.csv",
    nonlow_astrocyte_genes,
    delimiter="\n",
    fmt="%s",
)
