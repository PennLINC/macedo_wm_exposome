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
# ROOT_DIR <- "/Users/bmacedo/Desktop/final_WM"
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
exposome_batch <- factor(as.character(exposome_df$site))

cat("Batch counts:\n")
print(table(exposome_batch, useNA = "ifany"))

# -------------------------
# Feature matrix for covbat
# covbat expects features x subjects
# -------------------------
msmt_cols <- grep("^bundle", names(exposome_df), value = TRUE)

cat("Number of msmt columns:", length(msmt_cols), "\n")
print(head(msmt_cols, 20))

if (length(msmt_cols) == 0) {
  stop("No msmt columns found in exposome_df")
}

data_exposome_clean <- exposome_df[, msmt_cols, drop = FALSE]
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
    data = exposome_df
  )
}

if (analysis_type == "cognition") {
  mod_exposome <- model.matrix(
    ~ age + sex +
      General_SES + School + Family_Values + Family_Turmoil +
      Dense_Urban_Poverty + Extracurriculars + Screen_Time +
      neurocog_pc1.bl + neurocog_pc2.bl + neurocog_pc3.bl,
    data = exposome_df
  )
}

if (analysis_type == "income") {
  mod_exposome <- model.matrix(
    ~ age + sex + parental_education + income + le_l_adi__addr1__national_prcnt,
    data = exposome_df
  )
}

cat("mod_exposome dim:", paste(dim(mod_exposome), collapse = " x "), "\n")
print(object.size(mod_exposome), units = "MB")

# -------------------------
# Run CovBat
# Older API with train indicator
# -------------------------
gc()

cat("Running covbat with A as training set...\n")
covbat_exposome_A <- covbat(
  dat   = data_exposome,
  bat   = exposome_batch,
  mod   = mod_exposome,
  train = exposome_group_vector_A_train
)

gc()

cat("Running covbat with B as training set...\n")
covbat_exposome_B <- covbat(
  dat   = data_exposome,
  bat   = exposome_batch,
  mod   = mod_exposome,
  train = exposome_group_vector_B_train
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
# -------------------------
msmt_idx <- grep("^bundle", names(exposome_df))

df_harmonized_exposome_A <- cbind(
  exposome_df[, -msmt_idx, drop = FALSE],
  as.data.frame(harmonized_exposome_A)
)

df_harmonized_exposome_B <- cbind(
  exposome_df[, -msmt_idx, drop = FALSE],
  as.data.frame(harmonized_exposome_B)
)

# preserve original metric names
colnames(df_harmonized_exposome_A)[(ncol(df_harmonized_exposome_A) - length(msmt_cols) + 1):ncol(df_harmonized_exposome_A)] <- msmt_cols
colnames(df_harmonized_exposome_B)[(ncol(df_harmonized_exposome_B) - length(msmt_cols) + 1):ncol(df_harmonized_exposome_B)] <- msmt_cols

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