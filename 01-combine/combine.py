#%% Import packages

import pandas as pd
import numpy as np

import scipy.io
import csv

#%% Load healthy data and create metadata

healthy_data = pd.read_csv("../00-download/out/healthy-data.csv", index_col=0)
healthy_metadata = pd.read_csv(
    "../00-download/out/healthy-metadata.csv", index_col=0
)

healthy_metadata["Sample"] = np.repeat(0, len(healthy_metadata))
healthy_metadata["Healthy"] = np.repeat(True, len(healthy_metadata))

#%% Load TLE data and metadata

separated_data = [healthy_data]
separated_metadata = [healthy_metadata]

for i in range(1, 6):
    prefix = f"../00-download/out/tle{i}-"

    barcodes = []
    with open(prefix + "barcodes.tsv") as barcodes_file:
        for row in csv.reader(barcodes_file, delimiter="\t"):
            barcodes.append(f"Sample{i}_{row[0]}")

    features = []
    with open(prefix + "features.tsv") as features_file:
        for row in csv.reader(features_file, delimiter="\t"):
            features.append(row[1])

    counts = scipy.io.mmread(prefix + "matrix.mtx").transpose()

    df = pd.DataFrame.sparse.from_spmatrix(
        counts, index=barcodes, columns=features
    ).drop(
        columns=[
            "RGS5",
            "TBCE",
            "PDE11A",
            "LINC01238",
            "PRSS50",
            "CYB561D2",
            "ATXN7",
            "TXNRD3NB",
            "CCDC39",
            "MATR3",
            "SOD2",
            "POLR2J3",
            "ABCF2",
            "TMSB15B",
            "PINX1",
            "LINC01505",
            "IGF2",
            "HSPA14",
            "EMG1",
            "DIABLO",
            "LINC02203",
            "COG8",
            "SCO2",
            "H2BFS",
        ]
    )

    mdf = pd.DataFrame(
        {
            "Sample": np.repeat(i, len(barcodes)),
            "Healthy": np.repeat(False, len(barcodes)),
        },
        index=barcodes,
    )

    separated_data.append(df)
    separated_metadata.append(mdf)


#%% Merge data and metadata

data = pd.concat(separated_data, join="inner")
metadata = pd.concat(separated_metadata)
genes = pd.DataFrame({"Genes": data.columns.values})

#%% Save data and metadata

scipy.io.mmwrite("out/combined-data.mtx", data.sparse.to_coo())
metadata.to_csv("out/combined-metadata.csv")
genes.to_csv("out/combined-genes.csv")

# %%
