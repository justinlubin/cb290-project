#%% Import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.io

#%% Load data and metadata

genes = pd.read_csv("../01-combine/out/combined-genes.csv", header=0)[
    "Gene"
].values

healthy_metadata = pd.read_csv(
    "../01-combine/out/combined-healthy-metadata.csv", header=0, index_col=0
)

tle_metadata = pd.read_csv(
    "../01-combine/out/combined-tle-metadata.csv", header=0, index_col=0
)

healthy_data = pd.read_csv(
    "../01-combine/out/combined-healthy-data.csv", header=0, index_col=0
)

tle_data_matrix = scipy.io.mmread("../01-combine/out/combined-tle-data.mtx")

#%% Compute healthy gene expression

healthy_gene_expression = healthy_data.sum(axis=0)

#%% Plot TLE gene expression and set lower cutoff

healthy_gene_low = 50
fig, ax = plt.subplots(1, 1)
ax.hist(
    np.log10(
        healthy_gene_expression[healthy_gene_expression >= healthy_gene_low]
    )
)

#%% Compute TLE gene expression

tle_csc = tle_data_matrix.tocsc()
tle_gene_expression = np.asarray(tle_csc.sum(axis=0))[0]

#%% Plot TLE gene expression and set lower cutoff

tle_gene_low = 50
fig, ax = plt.subplots(1, 1)
ax.hist(np.log10(tle_gene_expression[tle_gene_expression >= tle_gene_low]))

#%% Filter lowly-expressed genes

nonlow_expression = (
    (healthy_gene_expression >= healthy_gene_low)
    | (tle_gene_expression >= tle_gene_low)
).values

healthy_data_trimmed_genes = healthy_data.iloc[
    :, nonlow_expression.nonzero()[0]
]
tle_csc_trimmed_genes = tle_csc[:, nonlow_expression]

#%% Compute library sizes

healthy_metadata["TrimmedLibrarySize"] = healthy_data_trimmed_genes.sum(axis=1)
tle_metadata["TrimmedLibrarySize"] = tle_csc_trimmed_genes.sum(axis=1)

#%% Plot healthy library size and set cut offs

healthy_cell_low = 0.25e6
healthy_cell_high = 0.75e6
fig, ax = plt.subplots(1, 1)
ax.hist(healthy_metadata["TrimmedLibrarySize"], bins=50)
ax.axvline(healthy_cell_low, color="red")
ax.axvline(healthy_cell_high, color="red")

#%% Plot healthy library size and set cut offs

tle_cell_low = 200
tle_cell_high = 2000

fig, ax = plt.subplots(1, 1)
tle_idx = (
    (tle_metadata["TrimmedLibrarySize"] >= tle_cell_low)
    & (tle_metadata["TrimmedLibrarySize"] <= tle_cell_high)
).values
ax.hist(
    tle_metadata[tle_idx]["TrimmedLibrarySize"],
    bins=50,
)

#%% Filter low library size

healthy_idx = (
    (healthy_metadata["TrimmedLibrarySize"] > healthy_cell_low)
    & (healthy_metadata["TrimmedLibrarySize"] < healthy_cell_high)
).values

healthy_data_trimmed = healthy_data_trimmed_genes[healthy_idx]
healthy_metadata_trimmed = healthy_metadata[healthy_idx]

tle_data_trimmed = tle_csc_trimmed_genes.tocsr()[tle_idx, :]
tle_metadata_trimmed = tle_metadata[tle_idx]

#%% Combine all into dense arrays

dense_tle_data_trimmed = pd.DataFrame(
    tle_data_trimmed.toarray(),
    columns=healthy_data_trimmed.columns,
    index=tle_metadata_trimmed.index,
)

final_data = pd.concat([healthy_data_trimmed, dense_tle_data_trimmed])
final_metadata = pd.concat([healthy_metadata_trimmed, tle_metadata_trimmed])

#%% Save

final_data.to_csv("out/filtered-data.csv")
final_metadata.to_csv("out/filtered-metadata.csv")
