#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(CovBat)
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

# -------------------------
# Parse command-line args
# -------------------------
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript Step2_covbat_harmonization.R <main|cognition|income>")
}

analysis_type <- tolower(args[1])
valid_analyses <- c("main", "cognition", "income")

if (!(analysis_type %in% valid_analyses)) {
  stop("Invalid analysis type. Use one of: main, cognition, income")
}

cat("Running analysis:", analysis_type, "\n")

# -------------------------
# Paths
# -------------------------
ROOT_DIR <- "/mnt/isilon/bgdlab_hbcd/projects/macedo_wm_exposome/macedo_wm_exposome"

OUTPUT_DIR <- file.path(ROOT_DIR, "output_data")
CLEANED_DIR <- file.path(OUTPUT_DIR, "cleaned")
HARMONIZED_DIR <- file.path(OUTPUT_DIR, "harmonized")

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(CLEANED_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(HARMONIZED_DIR, recursive = TRUE, showWarnings = FALSE)

# -------------------------
# Load data
# -------------------------
if (analysis_type == "main") {
  exposome_df <- read.csv(file.path(CLEANED_DIR, "exposome_FINAL_11_3.csv"))
}

if (analysis_type == "cognition") {
  exposome_df <- read.csv(file.path(CLEANED_DIR, "exposome_FINAL_cognition_11_10.csv"))
}

if (analysis_type == "income") {
  exposome_df <- read.csv(file.path(CLEANED_DIR, "parental_edu_income_sensitivity_df.csv"))
}

cat("Loaded rows:", nrow(exposome_df), "\n")
cat("Loaded cols:", ncol(exposome_df), "\n")

# -------------------------
# Train/test split flags
# -------------------------
exposome_group_vector_A_train <- exposome_df$matched_group == 1
exposome_group_vector_B_train <- exposome_df$matched_group == 2

n_A <- sum(exposome_group_vector_A_train, na.rm = TRUE)
n_B <- sum(exposome_group_vector_B_train, na.rm = TRUE)

cat("Group A:", n_A, "\n")
cat("Group B:", n_B, "\n")

# -------------------------
# Batch variable
# -------------------------
if (!("batch_device_software" %in% names(exposome_df))) {
  stop("Column 'batch_device_software' not found in exposome_df")
}

exposome_batch <- factor(as.character(exposome_df$batch_device_software))

cat("Batch counts (full sample):\n")
print(sort(table(exposome_batch, useNA = "ifany"), decreasing = FALSE))

if (any(is.na(exposome_batch))) {
  stop("Batch variable contains NA values")
}

# -------------------------
# Drop globally tiny batches
# Need at least 2 subjects per batch for CovBat/rowVars
# -------------------------
batch_counts <- table(exposome_batch)
valid_batches_global <- names(batch_counts[batch_counts >= 2])

keep_global <- as.character(exposome_batch) %in% valid_batches_global

cat("Subjects kept after global batch-size filter (>=2):", sum(keep_global), "\n")
cat("Subjects dropped after global batch-size filter (<2):", sum(!keep_global), "\n")

if (sum(!keep_global) > 0) {
  cat("Dropped globally tiny batches:\n")
  print(sort(table(droplevels(exposome_batch[!keep_global])), decreasing = TRUE))
}

if (sum(keep_global) == 0) {
  stop("No subjects remain after global batch-size filtering")
}

# Apply global filter to dataframe and train indicators
exposome_df_filt <- exposome_df[keep_global, , drop = FALSE]
exposome_batch_filt <- droplevels(exposome_batch[keep_global])
train_A_filt <- exposome_group_vector_A_train[keep_global]
train_B_filt <- exposome_group_vector_B_train[keep_global]

cat("Filtered rows:", nrow(exposome_df_filt), "\n")
cat("Filtered Group A:", sum(train_A_filt), "\n")
cat("Filtered Group B:", sum(train_B_filt), "\n")

# -------------------------
# Feature matrix for covbat
# covbat expects features x subjects
# -------------------------
msmt_cols <- grep("^bundle", names(exposome_df_filt), value = TRUE)

cat("Number of msmt columns:", length(msmt_cols), "\n")
print(head(msmt_cols, 20))

if (length(msmt_cols) == 0) {
  stop("No msmt columns found in exposome_df_filt")
}

data_exposome_clean <- exposome_df_filt[, msmt_cols, drop = FALSE]
data_exposome <- data.matrix(data_exposome_clean)
storage.mode(data_exposome) <- "double"
data_exposome <- t(data_exposome)

cat("data_exposome dim (features x subjects):", paste(dim(data_exposome), collapse = " x "), "\n")
print(object.size(data_exposome), units = "MB")

# -------------------------
# Covariate model matrix for covbat
# -------------------------
if (analysis_type == "main") {
  mod_exposome <- model.matrix(
    ~ age + sex +
      General_SES + School + Family_Values + Family_Turmoil +
      Dense_Urban_Poverty + Extracurriculars + Screen_Time,
    data = exposome_df_filt
  )
}

if (analysis_type == "cognition") {
  mod_exposome <- model.matrix(
    ~ age + sex +
      General_SES + School + Family_Values + Family_Turmoil +
      Dense_Urban_Poverty + Extracurriculars + Screen_Time +
      neurocog_pc1.bl + neurocog_pc2.bl + neurocog_pc3.bl,
    data = exposome_df_filt
  )
}

if (analysis_type == "income") {
  mod_exposome <- model.matrix(
    ~ age + sex + parental_education + income + le_l_adi__addr1__national_prcnt,
    data = exposome_df_filt
  )
}

cat("mod_exposome dim:", paste(dim(mod_exposome), collapse = " x "), "\n")
print(object.size(mod_exposome), units = "MB")

if (any(is.na(mod_exposome))) {
  stop("mod_exposome contains NA values")
}

if (any(is.na(data_exposome))) {
  stop("data_exposome contains NA values")
}

# Optional sanity check
mod_tr_A <- mod_exposome[train_A_filt, , drop = FALSE]
cat("rank(mod_tr_A):", qr(mod_tr_A)$rank, "\n")
cat("ncol(mod_tr_A):", ncol(mod_tr_A), "\n")

# -------------------------
# A-trained CovBat
# Keep only subjects whose batch appears in A training set
# -------------------------
batches_in_A <- unique(as.character(exposome_batch_filt[train_A_filt]))
keep_A_run <- as.character(exposome_batch_filt) %in% batches_in_A

cat("Subjects kept for A-trained CovBat:", sum(keep_A_run), "\n")
cat("Subjects dropped for A-trained CovBat:", sum(!keep_A_run), "\n")

if (sum(!keep_A_run) > 0) {
  cat("Dropped for A-trained CovBat (batch absent from A training set):\n")
  print(sort(table(droplevels(exposome_batch_filt[!keep_A_run])), decreasing = TRUE))
}

data_exposome_A <- data_exposome[, keep_A_run, drop = FALSE]
bat_A <- droplevels(exposome_batch_filt[keep_A_run])
mod_A <- mod_exposome[keep_A_run, , drop = FALSE]
train_A <- train_A_filt[keep_A_run]
df_A <- exposome_df_filt[keep_A_run, , drop = FALSE]

cat("Running covbat with A as training set...\n")
gc()
covbat_exposome_A <- covbat(
  dat   = data_exposome_A,
  bat   = bat_A,
  mod   = mod_A,
  train = train_A
)
gc()

# -------------------------
# B-trained CovBat
# Keep only subjects whose batch appears in B training set
# -------------------------
batches_in_B <- unique(as.character(exposome_batch_filt[train_B_filt]))
keep_B_run <- as.character(exposome_batch_filt) %in% batches_in_B

cat("Subjects kept for B-trained CovBat:", sum(keep_B_run), "\n")
cat("Subjects dropped for B-trained CovBat:", sum(!keep_B_run), "\n")

if (sum(!keep_B_run) > 0) {
  cat("Dropped for B-trained CovBat (batch absent from B training set):\n")
  print(sort(table(droplevels(exposome_batch_filt[!keep_B_run])), decreasing = TRUE))
}

data_exposome_B <- data_exposome[, keep_B_run, drop = FALSE]
bat_B <- droplevels(exposome_batch_filt[keep_B_run])
mod_B <- mod_exposome[keep_B_run, , drop = FALSE]
train_B <- train_B_filt[keep_B_run]
df_B <- exposome_df_filt[keep_B_run, , drop = FALSE]

cat("Running covbat with B as training set...\n")
gc()
covbat_exposome_B <- covbat(
  dat   = data_exposome_B,
  bat   = bat_B,
  mod   = mod_B,
  train = train_B
)
gc()

# -------------------------
# Extract harmonized data
# covbat output is features x subjects, transpose back
# -------------------------
harmonized_exposome_A <- t(covbat_exposome_A$dat.covbat)
harmonized_exposome_B <- t(covbat_exposome_B$dat.covbat)

cat("harmonized_exposome_A dim:", paste(dim(harmonized_exposome_A), collapse = " x "), "\n")
cat("harmonized_exposome_B dim:", paste(dim(harmonized_exposome_B), collapse = " x "), "\n")

# -------------------------
# Combine harmonized metrics back with original dataframe
# Use the filtered dataframes matching each run
# -------------------------
msmt_idx_A <- grep("^bundle", names(df_A))
msmt_idx_B <- grep("^bundle", names(df_B))

df_harmonized_exposome_A <- cbind(
  df_A[, -msmt_idx_A, drop = FALSE],
  as.data.frame(harmonized_exposome_A)
)

df_harmonized_exposome_B <- cbind(
  df_B[, -msmt_idx_B, drop = FALSE],
  as.data.frame(harmonized_exposome_B)
)

# preserve original metric names
colnames(df_harmonized_exposome_A)[
  (ncol(df_harmonized_exposome_A) - length(msmt_cols) + 1):ncol(df_harmonized_exposome_A)
] <- msmt_cols

colnames(df_harmonized_exposome_B)[
  (ncol(df_harmonized_exposome_B) - length(msmt_cols) + 1):ncol(df_harmonized_exposome_B)
] <- msmt_cols

cat("Final A output rows:", nrow(df_harmonized_exposome_A), "\n")
cat("Final B output rows:", nrow(df_harmonized_exposome_B), "\n")

# -------------------------
# Output filenames
# -------------------------
if (analysis_type == "main") {
  fileA <- "df_harmonized_exposome_A_12_10.csv"
  fileB <- "df_harmonized_exposome_B_12_10.csv"
}

if (analysis_type == "cognition") {
  fileA <- "df_harmonized_exposome_cognition_A_12_10.csv"
  fileB <- "df_harmonized_exposome_cognition_B_12_10.csv"
}

if (analysis_type == "income") {
  fileA <- "df_income_sens_A_11_30.csv"
  fileB <- "df_income_sens_B_11_30.csv"
}

outputfileA <- file.path(HARMONIZED_DIR, fileA)
outputfileB <- file.path(HARMONIZED_DIR, fileB)

# -------------------------
# Save
# -------------------------
write.csv(df_harmonized_exposome_A, file = outputfileA, row.names = FALSE)
write.csv(df_harmonized_exposome_B, file = outputfileB, row.names = FALSE)

cat("[SAVE] A ->", outputfileA, "\n")
cat("[SAVE] B ->", outputfileB, "\n")