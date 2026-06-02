library(xCell2)
library(tidyverse)
library(data.table)
library(arrow)
library(tibble)

# Set the URL of the pre-trained reference
#ref_url <- "https://github.com/dviraran/xCell2refs/raw/refs/heads/main/references/TabulaSapiensBlood.xCell2Ref.rds"
# Set the local filename to save the reference
local_filename <- "../../../data/references/xCell/TabulaSapiendBlood.xCell2Ref.rds"
# Download the file
#download.file(ref_url, local_filename, mode = "wb")
# Load the downloaded reference
TabulaSapiendBlood.xCell2Ref <- readRDS(local_filename)

expr <- fread("../../../data/processed/RNA/Norm_counts.csv", check.names = FALSE)
expr <- as.data.frame(expr)
rownames(expr) <- expr[[1]]
expr[[1]] <- NULL

# The analysis uses a bulk mixture of gene expression data (genes in rows, samples in columns). The input must use the same gene annotation system as the reference object.
xcell2_results <- xCell2::xCell2Analysis(
  mix = t(expr),
  xcell2object = TabulaSapiendBlood.xCell2Ref,
  minSharedGenes=0.5
)


write.csv(xcell2_results, "../../../data/processed/RNA/Deconvoluted_cell_types.csv")
