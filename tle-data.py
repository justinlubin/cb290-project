import urllib.request
import gzip
import csv

with open("data-sources.csv") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        with urllib.request.urlopen(row["url"]) as data_gz_file:
            f = gzip.GzipFile(fileobj=data_gz_file)
            with open("cached-data/" + row["name"], "wb") as data_file:
                data_file.write(f.read())