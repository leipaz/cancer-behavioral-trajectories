library(data.table)
library(maftools)
library(vcfR)
library(tidyr)

library(readxl)
library(data.table)

#setwd("/Volumes/Stage/Projects/CRP_BC_0002/Topics_wearables/DNA/updated")

# Load the excel file (replace with your actual filename)
mapping <- read_excel("../../../data/processed/DNA/Sample_names.xlsx") 
# Convert to data.table for easy merging
setDT(mapping)

# Ensure column names match your logic
# The 'sample_ID' column in Excel matches 'Tumor_Sample_Barcode' in your data
setnames(mapping, old = c("sample_ID", "patient"), new = c("Tumor_Sample_Barcode", "Patient_ID"))

# 1. Load the TSV
# Note: Ensure the column names match your query order
dt <- fread("../../../data/raw/DNA/putative_somatic_mutations.tsv", header = FALSE, sep = "\t")
setnames(dt, c("Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2", "CSQ", "Samples"))

# 2. Expand rows
dt_expanded <- dt[, list(Samples = unlist(strsplit(as.character(Samples), ","))), 
                  by = .(Chromosome, Start_Position, Reference_Allele, Tumor_Seq_Allele2, CSQ)]
dt_expanded <- dt_expanded[Samples != ""]

# 3. Split "Sample=GT=AF" into 3 columns
dt_expanded[, c("Tumor_Sample_Barcode", "GT", "VAF") := tstrsplit(Samples, "=", fixed=TRUE)]

# 4. Convert VAF to numeric (it was extracted as a string)
dt_expanded[, VAF := as.numeric(VAF)]

# 4. Filter for mutated genotypes only
# Keep only rows where a mutation exists (0/1, 1/1, 0|1, etc.)
final_maf_data <- dt_expanded[!GT %in% c("./.", "0/0", "0|0") & VAF > 0.05]

# 3. Parse VEP and Map to MAF terms
# (Using a helper to handle the common VEP term 'missense_variant')
# 1. Clean the CSQ string first by removing the "CSQ=" prefix if it exists
final_maf_data[, CSQ_clean := sub("^CSQ=", "", CSQ)]
final_maf_data[, Variant_Classification_Raw := tstrsplit(CSQ_clean, "\\|")[[2]]]
#final_maf_data[, VAF := as.numeric(sub(".*AF=([^;]+).*", "\\1", INFO))]
# 3. Handle cases where there are multiple consequences (e.g., missense&splice)
# We take only the first one before the '&'
final_maf_data[, Variant_Classification_Raw := sub("&.*", "", Variant_Classification_Raw)]

# 4. Apply your Map
# 6. Map VEP Consequence to MAF Variant_Classification
# maftools requires these specific terms to color the plot correctly
vc_map <- c(
  "missense_variant"          = "Missense_Mutation",
  "stop_gained"               = "Nonsense_Mutation",
  "frameshift_variant"        = "Frame_Shift_Del", # Or Ins, maftools treats both as truncating
  "splice_acceptor_variant"   = "Splice_Site",
  "splice_donor_variant"      = "Splice_Site",
  "inframe_deletion"          = "In_Frame_Del",
  "inframe_insertion"         = "In_Frame_Ins",
  "protein_altering_variant"  = "Missense_Mutation", 
  "start_lost"                = "Translation_Start_Site",
  "stop_lost"                 = "Nonstop_Mutation"
)

final_maf_data[, Variant_Classification := vc_map[Variant_Classification_Raw]]
final_maf_data[, Hugo_Symbol := tstrsplit(CSQ, "\\|")[[4]]] # Extract 4th field from VEP CSQ
final_maf_data[, End_Position := Start_Position]
# 2. Ensure positions are numeric/integer (maftools can be picky)
final_maf_data[, Start_Position := as.numeric(Start_Position)]
final_maf_data[, End_Position := as.numeric(End_Position)]

# 1. Add Variant_Type based on REF and ALT lengths
final_maf_data[, Variant_Type := ifelse(nchar(Reference_Allele) == nchar(Tumor_Seq_Allele2), "SNP",
                                        ifelse(nchar(Reference_Allele) < nchar(Tumor_Seq_Allele2), "INS", "DEL"))]

# 2. Safety Check: If any multi-nucleotide variants exist (e.g., AT -> CG), 
# some pipelines call them DNP/TNP, but 'SNP' or 'MNP' works for maftools.
final_maf_data[nchar(Reference_Allele) > 1 & nchar(Reference_Allele) == nchar(Tumor_Seq_Allele2), Variant_Type := "MNP"]


# 4. Final filter and Plot
final_dt_i <- final_maf_data[!is.na(Variant_Classification) & !is.na(Hugo_Symbol)]

# 1. Merge the mapping with your final_dt
final_dt <- merge(final_dt_i, mapping[, .(Tumor_Sample_Barcode, Patient_ID)], 
                  by = "Tumor_Sample_Barcode", all.x = TRUE)

# 2. Update the barcode column to use the Patient ID
# If a mapping is missing, we keep the original ID as a fallback
final_dt[, Tumor_Sample_Barcode := ifelse(is.na(Patient_ID), Tumor_Sample_Barcode, Patient_ID)]
# Remove everything from the first underscore to the end of the string
final_dt[, Tumor_Sample_Barcode := sub("_.*", "", Tumor_Sample_Barcode)]
final_dt[, Tumor_Sample_Barcode := sub("-", "", Tumor_Sample_Barcode)]
final_dt[, Tumor_Sample_Barcode := sub("HDM", "", Tumor_Sample_Barcode)]


###########################
## METADATA 
############################

# 1. Load the single-sheet metadata
metadata <- read_excel("../../../data/processed/clinical/Subjects_data.xlsx")
setDT(metadata)

library(dplyr)

metadata <- metadata %>%
  mutate(
    Cohort = substr(as.character(S_RECORD_id), 2, 2),
    Cohort = recode(
      Cohort,
      "1" = "Breast",
      "2" = "Colon",
      "3" = "Lung"
    )
  )

setnames(metadata, 
         old = c("Hyper_DCABP_general", "Hyper_DCABP_month_1"), 
         new = c("General", "M1")
)

# 2. Clean the VCF data barcodes to match S_RECORD_id
# If raw barcode is "HDM-73001_S216", this makes it "73001"
final_dt[, S_RECORD_id := sub("HDM-", "", Tumor_Sample_Barcode)] # Remove prefix
final_dt[, S_RECORD_id := sub("HDM", "", Tumor_Sample_Barcode)] # Remove prefix
final_dt[, S_RECORD_id := as.numeric(sub("_.*", "", S_RECORD_id))] # Remove suffix and make numeric

# 3. Rename Tumor_Sample_Barcode in final_dt to match the Patient ID for the plot
# First, let's get the mapping from the metadata
final_dt <- merge(final_dt, metadata, by = "S_RECORD_id", all.x = TRUE)





# 2. Tell read.maf exactly which terms are non-synonymous
# This tells maftools: "Don't ignore these categories!"
non_syn_list <- c(
  "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", 
  "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins"
)

setnames(metadata, old = "S_RECORD_id", new = "Tumor_Sample_Barcode")

my_maf <- read.maf(maf = final_dt, vc_nonSyn = non_syn_list,clinicalData = metadata)
samples_to_keep <- setdiff(
  unique(my_maf@clinical.data$Tumor_Sample_Barcode),
  c("11002","11012","11013","12008","31017","31018","31019","31020","31021","62007")
)
my_maf_filt <- subsetMaf(
  maf = my_maf,
  tsb = samples_to_keep
)

oncoplot(maf = my_maf_filt, top = 20, removeNonMutated = TRUE, clinicalFeatures = c("M1", "General",'Cohort'))


# Standard passenger blacklist
passengers <- c(
  "TTN", "MUC16", "OBSCN", "PCLO", "RYR2", "NEB", "SYNE1", "SYNE2", 
  "DNAH11", "HMCN1", "USH2A", "ADAMTSL1", "ZFHX3", "KMT2D", "MACF1"
)

oncoplot(
  maf = my_maf_filt, 
  top = 20,                  # Plot only the top 20 most frequent AFTER filtering
  genesToIgnore = passengers, 
  removeNonMutated = TRUE,    # Hide samples that have no mutations in the top genes
  fontSize = 0.8,
  showTumorSampleBarcodes = TRUE, 
  clinicalFeatures = c("M1", "General",'Cohort')
)

cosmic_genes<-c('RAD51C', 'UBR5', 'FLT3', 'FBXW7', 'CUX1', 'ERBB3', 'PBRM1', 'ATM',
                 'ERCC5', 'KRAS', 'GATA1', 'CSF1R', 'AXIN2', 'PIK3CA', 'RPL10',
                 'SMAD3', 'SUFU', 'GNA11', 'BAP1', 'NFE2L2', 'IDH1', 'BLM',
                 'ARHGAP35', 'KIT', 'TNFRSF14', 'MEN1', 'PIK3CB', 'CTCF', 'FES',
                 'GNAQ', 'SETD2', 'SMO', 'MED12', 'TET2', 'DNM2', 'PRDM1', 'CREBBP',
                 'CBL', 'MAP2K2', 'PTPRT', 'SIX1', 'NCOR2', 'EGFR', 'IL6ST', 'NF2',
                 'RET', 'RECQL4', 'TENT5C', 'H3F3A', 'SOCS1', 'CHEK2', 'NTRK2',
                 'CACNA1D', 'DDR2', 'IDH2', 'TSC2', 'SPOP', 'BARD1', 'POLD1',
                 'POLG', 'BRCA1', 'SMARCD1', 'QKI', 'HIF1A', 'ASPM', 'SMARCA4',
                 'DDX3X', 'KMT2C', 'EPAS1', 'NPM1', 'DDB2', 'ZFHX3', 'SOX2',
                 'LRP1B', 'ARID1B', 'RBM10', 'FGFR1', 'MUC6', 'CD79A', 'TGFBR2',
                 'XPO1', 'RAD50', 'PPP6C', 'NOTCH2', 'CARD11', 'CHD4', 'CRLF2',
                 'POLE', 'CDK12', 'NT5C2', 'BCL9L', 'NR2F2', 'PMS2', 'ATP1A1',
                 'TSC1', 'NTRK3', 'HNF1A', 'STAT3', 'GATA3', 'TERT', 'POT1',
                 'DROSHA', 'FBXO11', 'ERBB4', 'CBLC', 'ERCC2', 'FLT4', 'STAT5B',
                 'KLF4', 'DNMT3A', 'KDR', 'FAT4', 'NF1', 'TP63', 'BIRC3', 'IL7R',
                 'LATS2', 'JAK1', 'SPEN', 'SF3B1', 'CYLD', 'KCNJ5', 'MYOD1',
                 'NFKBIE', 'MET', 'RPL5', 'AKT1', 'CEBPA', 'PTPN13', 'GATA2',
                 'ATRX', 'RUNX1', 'TRRAP', 'STK11', 'KDM6A', 'POLR2A', 'ERBB2',
                 'MSH6', 'CSF3R', 'NRAS', 'PTPRB', 'PDGFRA', 'IKBKB', 'PTCH1',
                 'FOXA1', 'APC', 'DAXX', 'IKZF3', 'PPM1D', 'ALK', 'RAD21', 'FEN1',
                 'FAT1', 'RXRA', 'PLEC', 'SETBP1', 'YAP1', 'PRKACA', 'IRS4', 'MPL',
                 'ARID2', 'HIST1H3B', 'KEAP1', 'MYC', 'CDH1', 'ESR1', 'WT1', 'RHOA',
                 'DICER1', 'ATR', 'MAP3K1', 'PTPN11', 'FOXL2', 'RNF43', 'ZRSR2',
                 'IKZF1', 'ACVR1B', 'VHL', 'AMER1', 'TNFAIP3', 'MAP2K1', 'MLH1',
                 'ARID1A', 'ARHGAP26', 'SRSF2', 'FGFR3', 'FUBP1', 'MYCN', 'MYB',
                 'GSK3B', 'PTEN', 'PAX5', 'PPP2R1A', 'CASP8', 'PRKAR1A', 'ELF3',
                 'CTNNA1', 'PMS1', 'PIK3R2', 'SMARCB1', 'CDC73', 'CDKN2C', 'JAK3',
                 'LATS1', 'RB1', 'COL2A1', 'KDM5C', 'BRAF', 'PIK3R3', 'ACVR2A',
                 'BAX', 'PIK3R1', 'HRAS', 'DGCR8', 'SIX2', 'JAK2', 'MAP3K13',
                 'SMAD4', 'BCOR', 'U2AF1', 'CIC', 'SMAD2', 'AXIN1', 'BRCA2',
                 'PHOX2B', 'CALR', 'CBLB', 'MTOR', 'B2M', 'AR', 'MAX', 'ACVR1',
                 'ERCC3', 'POLQ', 'FGFR2', 'LEF1', 'TRAF7', 'NCOA2', 'BCORL1',
                 'KMT2A', 'PREX2', 'MAP2K4', 'TBL1XR1', 'EZH2', 'KMT2D', 'CXCR4',
                 'EP300', 'SETDB1', 'TBX3', 'CDKN1B', 'FAS', 'GNAS', 'NCOR1',
                 'SH2B3', 'SDHA', 'CD79B', 'H3F3B', 'ATP2B3', 'PRKD1', 'CTNNB1',
                 'PTK6', 'MSH2', 'PHF6', 'GRIN2A', 'PLCG1', 'CNOT3', 'ASXL1', 'SRC',
                 'MYD88', 'CDKN2A', 'SOS1', 'NOTCH1', 'TSHR', 'ETNK1', 'USP8',
                 'MAPK1', 'STAG2', 'RAC1', 'BTK', 'FGFR4', 'SALL4', 'LZTR1', 'ABL1',
                 'TP53')



# Subset the data to only include drivers
driver_dt <- final_dt[Hugo_Symbol %in% cosmic_genes]
# Initialize the driver-specific MAF
driver_maf <- read.maf(maf = driver_dt, clinicalData = metadata)
# Filter 11002 again
driver_maf <- subsetMaf(
  maf = driver_maf,
  tsb = samples_to_keep
)


## Plot
# Check for repeated mutations
maf_df <- driver_maf@data
# Count how many samples each variant appears in
variant_counts <- maf_df %>%
  group_by(Chromosome, Start_Position, End_Position, Reference_Allele, Tumor_Seq_Allele2) %>%
  summarise(n_samples = n_distinct(Tumor_Sample_Barcode), .groups = "drop")

library(ggplot2)
ggplot(variant_counts, aes(x = n_samples)) +
  geom_histogram(bins = 50) +
  theme_minimal() +
  coord_cartesian(ylim = c(0, 100)) +
  labs(title = "Variant recurrence",
       x = "Number of samples",
       y = "Count")

# Join back and filter
driver_maf_filt_df <- driver_maf@data %>%
  group_by(Chromosome, Start_Position, Reference_Allele, Tumor_Seq_Allele2) %>%
  mutate(n_samples = n_distinct(Tumor_Sample_Barcode)) %>%
  ungroup() %>%
  filter(n_samples <= 25)                      # remove recurrent artifacts
# Generate MAF object again
driver_maf_filt <- read.maf(maf = driver_maf_filt_df, clinicalData = metadata)

# Plot
# Save old margin
old_mar <- par()$mar
# Increase left margin (e.g., from 5 to 10)
par(mar = c(5, 20, 4, 2))  # c(bottom, left, top, right)
# Oncoplot
oncoplot(
  maf = driver_maf_filt, 
  top=25,
  showTumorSampleBarcodes = TRUE,
  fontSize = 0.7,
  clinicalFeatures = c("General","Cohort"),
  sortByAnnotation=TRUE,
  removeNonMutated = FALSE
)
# Reset margin
par(mar = old_mar)

###################
#### EXPORT SPECIFIC MUTATIONS
###################

library(maftools)
# Define the genes of interest
genes_of_interest <- c("TP53", "PLEC")
# 1️⃣ Get mutation counts per sample per gene
mut_table <- table(driver_maf_filt@data$Tumor_Sample_Barcode, driver_maf_filt@data$Hugo_Symbol)
# Subset only the genes we care about
mut_table <- mut_table[, genes_of_interest, drop = FALSE]
# Convert counts to 0/1
binary_table <- ifelse(mut_table > 0, 1, 0)
# 2️⃣ Convert to data.frame
binary_df <- data.frame(Sample = rownames(binary_table), binary_table, row.names = NULL)
# 3️⃣ Add "General" column from clinical data (metadata)
# Make sure the clinical data has Sample IDs
clinical_df <- driver_maf_filt@clinical.data[, c("Tumor_Sample_Barcode", "General")]
colnames(clinical_df)[1] <- "Sample"  # rename to match
# 4️⃣ Merge mutation table with clinical data
final_df <- merge(binary_df, clinical_df, by = "Sample", all.x = TRUE)
# View result
print(final_df)
write.csv(final_df, file = "../../../data/processed/DNA/mutation_table.csv", row.names = FALSE)


#############
## REMOVE SAMPLES WITHOUT TOPIC!!!
#############

maf_df <- driver_maf_filt@data
maf_df_clean <- maf_df[!is.na(maf_df$General) & maf_df$General != "", ]

driver_maf_filt<- read.maf(maf = maf_df_clean, clinicalData = metadata)

###########
## SIGNATURES
#############
library(MutationalPatterns)
library(BSgenome.Hsapiens.UCSC.hg38)
library(ggplot2)


#### 
# Check for repeated mutations
###
maf_df <- my_maf_filt@data
maf_df <- maf_df[!is.na(maf_df$General) & maf_df$General != "", ]
# Count how many samples each variant appears in
variant_counts <- maf_df %>%
  group_by(Chromosome, Start_Position, End_Position, Reference_Allele, Tumor_Seq_Allele2) %>%
  summarise(n_samples = n_distinct(Tumor_Sample_Barcode), .groups = "drop")

library(ggplot2)
ggplot(variant_counts, aes(x = n_samples)) +
  geom_histogram(bins = 50) +
  theme_minimal() +
  coord_cartesian(ylim = c(0, 100)) +
  labs(title = "Variant recurrence",
       x = "Number of samples",
       y = "Count")

# Join back and filter
my_maf_filt_df <- my_maf_filt@data %>%
  group_by(Chromosome, Start_Position, Reference_Allele, Tumor_Seq_Allele2) %>%
  mutate(n_samples = n_distinct(Tumor_Sample_Barcode)) %>%
  ungroup() %>%
  filter(n_samples <= 47)  %>%                      # remove recurrent artifacts
  filter(!grepl("Low_complexity", CSQ))
# Remove samples without topic
maf_df_clean <- my_maf_filt_df[!is.na(my_maf_filt_df$General) & my_maf_filt_df$General != "", ]

# Generate MAF object again
my_maf_filt_clean <- read.maf(maf = my_maf_filt_df, clinicalData = metadata)

maf_df <- my_maf_filt_clean@data
# Count how many samples each variant appears in
variant_counts <- maf_df %>%
  group_by(Chromosome, Start_Position, End_Position, Reference_Allele, Tumor_Seq_Allele2) %>%
  summarise(n_samples = n_distinct(Tumor_Sample_Barcode), .groups = "drop")

ggplot(variant_counts, aes(x = n_samples)) +
  geom_histogram(bins = 50) +
  theme_minimal() +
  coord_cartesian(ylim = c(0, 100)) +
  labs(title = "Variant recurrence",
       x = "Number of samples",
       y = "Count")

###
# Get TNM matrix
###
# First, get the trinucleotide matrix
maf.tnm <- trinucleotideMatrix(maf = my_maf_filt_clean, # Using ALL genes!
                               add = FALSE, 
                               ref_genome = "BSgenome.Hsapiens.UCSC.hg38")  # or hg19
# 3. CRITICAL STEP: Extract the matrix and drop zero-sum rows
# trinucleotideMatrix returns a list; the actual matrix is in $nmf_matrix
# Clean matrix extraction
mat <- maf.tnm$nmf_matrix
mat <- mat[rowSums(mat) > 0, ]   # Remove empty samples
mat <- mat[, colSums(mat) > 0]   # Remove empty mutation types
#mat <- as.matrix(mat)           # Ensure matrix format
maf.tnm$nmf_matrix <- mat


##
# Now fit the signatures
##
# Download the latest COSMIC signatures
signatures <- get_known_signatures(source = "COSMIC_v3.2")
# Fit your samples to the signatures
fit_res <- fit_to_signatures(t(maf.tnm$nmf_matrix), signatures)
# Filter to signatures contributing more than 5% in at least one sample
contri <- fit_res$contribution
relevant_sigs <- rowSums(contri) > (0.05 * sum(contri))
p<-plot_contribution(contri[relevant_sigs, ], mode = "relative")
# 2. Modify the theme to rotate labels
p + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
write.csv(contri[relevant_sigs, ], file = "../../../data/processed/DNA/signatures_per_sample.csv", row.names = TRUE)


##########
## EXPORT TMB
##########
tmb_df <- tmb(driver_maf_filt)  # returns a data.frame with Tumor_Sample_Barcode and tmb values
write.csv(tmb_df, file = "../../../data/processed/DNA/tmb_per_sample.csv", row.names = FALSE)

#tmb_df <- tmb(my_maf_filt)  # returns a data.frame with Tumor_Sample_Barcode and tmb values
#write.csv(tmb_df, file = "tmb_full_per_sample.csv", row.names = FALSE)



