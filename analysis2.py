#%% Import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.io
import csv
import sklearn.decomposition
import umap

#%% Load healthy data

healthy = pd.read_csv("cached-data/healthy.csv", index_col=0)
healthy_metadata = pd.DataFrame(
    {"Sample": np.repeat(-1, len(healthy)), "Healthy": np.repeat(True, len(healthy))}, index=healthy.index
)

#%% Load rest of data

separated_data = [healthy]
separated_metadata = [healthy_metadata]

for i in range(1, 6):
    prefix = f"cached-data/tle{i}-"
    
    barcodes = []
    with open(prefix + "barcodes.tsv") as barcodes_file:
        for row in csv.reader(barcodes_file, delimiter="\t"):
            barcodes.append(row[0])
            
    features = []
    with open(prefix + "features.tsv") as features_file:
        for row in csv.reader(features_file, delimiter="\t"):
            features.append(row[1])
            
    counts = scipy.io.mmread(prefix + "matrix.mtx").transpose()
    
    df = pd.DataFrame.sparse.from_spmatrix(
        counts, index=barcodes, columns=features
    )
    
    mdf = pd.DataFrame(
        {"Sample": np.repeat(i, len(barcodes)), "Healthy": np.repeat(False, len(barcodes))},
        index=barcodes,
    )
    
    separated_data.append(df)
    separated_metadata.append(mdf)
    
    
#%% Merge

data = pd.concat(separated_data, join="inner")
metadata = pd.concat(separated_metadata)

#%% Train dimensionality reduction

raw_svd = sklearn.decomposition.TruncatedSVD(n_components=50).fit(data)

#%% Apply dimensionality reduction

data_dr = pd.DataFrame(
    raw_svd.transform(data), index=data.index
)

#%% Sample

sample = data_dr.sample(9000)

#%% Plot principal components

plt.scatter(np.log1p(sample.iloc[:,0]), np.log1p(sample.iloc[:,1]))
plt.xlim(-5, 7)
plt.ylim(-5, 7)
plt.figure()
plt.scatter(np.log1p(sample.iloc[:,2]), np.log1p(sample.iloc[:,3]))
plt.xlim(-5, 7)
plt.ylim(-5, 7)

#%% Train UMAP

raw_umap = umap.UMAP().fit(sample)

#%% Apply UMAP

data_umap = pd.DataFrame(
    raw_umap.transform(sample), index=sample.index
)

#%% Plot UMAP join

m = data_umap.join(metadata)

#%% Plot UMAP

plt.figure()
plt.scatter(m.iloc[:,0], m.iloc[:,1], c=m["Sample"], marker=".", s=1)