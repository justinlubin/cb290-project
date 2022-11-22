#%% Import packages

import pandas as pd

import urllib.request
import gzip
import csv

#%% Download and save TLE data

with open("tle-sources.csv") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        with urllib.request.urlopen(row["url"]) as data_gz_file:
            f = gzip.GzipFile(fileobj=data_gz_file)
            with open("out/" + row["name"], "wb") as data_file:
                data_file.write(f.read())


#%% Load healthy GSE data

soft_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE67nnn/GSE67835/soft/GSE67835_family.soft.gz"
raw_soft_data = None

with urllib.request.urlopen(soft_url) as soft_gz_file:
    raw_soft_data = gzip.GzipFile(fileobj=soft_gz_file).read().decode("utf-8")

#%% Iterate through healthy GSE data

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

#%% Download healthy data files

for _, sample in samples.items():
    with urllib.request.urlopen(sample["url"]) as data_gz_file:
        sample["raw_data"] = (
            gzip.GzipFile(fileobj=data_gz_file).read().decode("utf-8")
        )

#%% Make healthy count matrix columns

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

#%% Make healthy count matrix

data = (
    pd.DataFrame(columns, index=genes)
    .transpose()
    .drop(columns=["no_feature", "ambiguous", "alignment_not_unique"])
)

#%% Make healthy metadata

metadata_columns = {}

for key in samples[first_name]:
    if key in ["raw_data", "url"]:
        continue
    metadata_columns[key] = []

for sample_name, sample in samples.items():
    for key in metadata_columns:
        metadata_columns[key].append(sample[key])

metadata = pd.DataFrame(metadata_columns, index=samples.keys())

#%% Save healthy data and metadata

data.to_csv("out/healthy-data.csv")
metadata.to_csv("out/healthy-metadata.csv")
