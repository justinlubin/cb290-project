#%% Import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.io

#%% Load data and metadata

metadata = pd.read_csv(
    "../01-combine/out/combined-metadata.csv", header=0, index_col=0
)

data = pd.DataFrame.sparse.from_spmatrix(
    scipy.io.mmread("../01-combine/out/combined-data.mtx"),
    index=metadata.index,
    columns=pd.read_csv("../01-combine/out/combined-genes.csv", header=0)[
        "Genes"
    ].values,
)

#%% Compute library sizes

metadata["LibrarySize"] = data.sum(axis=1)

# %%

include = metadata["Sample"] > 0

fig, ax = plt.subplots(1, 1)
ax.hist(metadata[include]["LibrarySize"], bins=40)


# %%
