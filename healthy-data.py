#%% Import packages

import pandas as pd

import urllib.request
import gzip

#%% Load SOFT Data

soft_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE67nnn/GSE67835/soft/GSE67835_family.soft.gz"
raw_soft_data = None

with urllib.request.urlopen(soft_url) as soft_gz_file:
    raw_soft_data = gzip.GzipFile(fileobj=soft_gz_file).read().decode("utf-8")

#%% Iterate through SOFT Data

samples = {}

for entry in raw_soft_data.split("^")[1:]:
    if not entry.startswith("SAMPLE"):
        continue
    attributes = entry.splitlines()
    _, sample_name = attributes[0].split(" = ", 2)
    samples[sample_name] = {}
    for line in attributes[1:]:
        key, val = line.split(" = ", 2)
        if key == "!Sample_characteristics_ch1":
            val_left, val_right = val.split(": ", 2)
            samples[sample_name][val_left] = val_right
        if key == "!Sample_supplementary_file_1":
            samples[sample_name]["url"] = val

#%% Download additional files

for _, sample in samples.items():
    with urllib.request.urlopen(sample["url"]) as data_gz_file:
        sample["raw_data"] = (
            gzip.GzipFile(fileobj=data_gz_file).read().decode("utf-8")
        )

#%% Make matrix columns

first_name = next(iter(samples))

genes = []
for line in samples[first_name]["raw_data"].splitlines():
    genes.append(line.split("\t", 2)[0].strip())

columns = {}

for sample_name, sample in samples.items():
    counts = []
    for line in sample["raw_data"].splitlines():
        counts.append(int(line.split("\t", 2)[1].strip()))
    columns[sample_name] = counts

#%% Make matrix

data = (
    pd.DataFrame(columns, index=genes)
    .transpose()
    .drop(columns=["no_feature", "ambiguous", "alignment_not_unique"])
)

#%% Save matrix

data.to_csv("cached-data/healthy.csv")
