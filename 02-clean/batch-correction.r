# setwd("02-clean")

data <- read.csv("out/filtered-data.csv", header = TRUE, row.names = 1)
metadata <- read.csv("out/filtered-metadata.csv", header = TRUE, row.names = 1)

count_matrix = t(data)
batch <- metadata$Sample

adjusted <- sva::ComBat_seq(count_matrix, batch = batch, group = NULL)

adjusted_data = t(adjusted)

write.csv(adjusted_data, file = "out/batch-corrected-data.csv")
