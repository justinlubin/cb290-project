#%% Run all

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

for under_over, xticks in [
    # ("under", [1, 1.5, 2, 2.5]),
    # ("over", [4.0, 4.5, 5.0, 5.5, 6.0, 6.5]),
    ("under", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
]:
    data = pd.read_csv(
        f"panther-overexpression-tests/BATCH-astrocytes-{under_over}expressed.txt",
        # f"panther-overexpression-tests/astrocytes-{under_over}expressed.txt",
        header=6,
        delimiter="\t",
        usecols=[0, 5, 6, 7],
    )

    data.columns.values[0] = "GO Category"
    data.columns.values[1] = "Fold enrichment"
    data.columns.values[2] = "Raw p-value"
    data.columns.values[3] = "FDR"

    if data["Fold enrichment"].dtype != np.float64:
        data = data[~data["Fold enrichment"].str.startswith(" <")]
        data["Fold enrichment"] = data["Fold enrichment"].astype(np.float64)

    subdata = data.sort_values(by="Fold enrichment", ascending=False).head(
        # 10
        3
    )

    fig, ax = plt.subplots(1, 1)
    y_pos = np.arange(len(subdata))
    ax.set_yticks(y_pos, labels=subdata["GO Category"])
    ax.invert_yaxis()
    ax.scatter(np.log2(subdata["Fold enrichment"]), y_pos)
    ax.set_xlabel("log2(Fold enrichment)")
    ax.set_xticks(xticks)
    ax.grid(True)
    ax.set_axisbelow(True)

    fig.savefig(
        f"panther-overexpression-tests/BATCH-astrocytes-{under_over}expressed.png",
        # f"panther-overexpression-tests/astrocytes-{under_over}expressed.png",
        bbox_inches="tight",
    )

# %%
