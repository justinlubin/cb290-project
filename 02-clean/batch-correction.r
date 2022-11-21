# library(dplyr)

data <- read.csv("cached-data/combined-data.csv", header = TRUE, row.names = 1)
metadata <- read.csv("cached-data/combined-metadata.csv", header = TRUE, row.names = 1)

count_matrix = t(data)
batch <- metadata$Sample

adjusted <- sva::ComBat_seq(count_matrix, batch = batch, group = NULL)

# data_relevant_genes <- data %>% filter(if_any(everything(), ~ . != 0))

# count_matrix = t(data_relevant_genes)
# count_matrix_relevant <- count_matrix %>% filter(if_any(everything(), ~ . != 0))



# adjusted <- sva::ComBat_seq(count_matrix_relevant, batch = batch, group = NULL)
