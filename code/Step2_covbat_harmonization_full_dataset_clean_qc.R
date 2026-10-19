#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(CovBat)
  library(data.table)
  library(dplyr)
  library(tidyr)
})

# ============================================================================
# Step 2 — Full-dataset CovBat harmonization with explicit sample-flow QC
# ============================================================================
#
# Purpose:
#   Harmonize the full cleaned dataset in one CovBat fit, rather than fitting
#   CovBat parameters separately in split A or split B.
#
# IMPORTANT:
#   * This version is appropriate for pooled/full-sample downstream analyses.
#   * Do NOT use this version to claim a strictly held-out split-B replication,
#     because split-B participants contribute to the harmonization estimates.
#   * The biological covariates in `mod_formula` are preserved during
#     harmonization; this is not the same as adjusting for them in downstream
#     regression models.
#
# Usage:
#   Rscript Step2_covbat_harmonization_full_dataset_clean_qc.R main
#   Rscript Step2_covbat_harmonization_full_dataset_clean_qc.R cognition
#   Rscript Step2_covbat_harmonization_full_dataset_clean_qc.R income
# ============================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript Step2_covbat_harmonization_full_dataset_clean_qc.R <main|cognition|income>")
}

analysis_type <- tolower(args[1])
valid_analyses <- c("main", "cognition", "income")
if (!(analysis_type %in% valid_analyses)) {
  stop("Invalid analysis type. Use one of: main, cognition, income")
}

cat("Running analysis:", analysis_type, "\n")
cat("Harmonization mode: FULL DATASET (single CovBat fit)\n")

# -------------------------
# Paths
# -------------------------
ROOT_DIR <- "/mnt/isilon/bgdlab_hbcd/projects/macedo_wm_exposome/macedo_wm_exposome"
OUTPUT_DIR <- file.path(ROOT_DIR, "output_data")
CLEANED_DIR <- file.path(OUTPUT_DIR, "cleaned")
HARMONIZED_DIR <- file.path(OUTPUT_DIR, "harmonized")
QC_DIR <- file.path(HARMONIZED_DIR, "qc")

dir.create(HARMONIZED_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(QC_DIR, recursive = TRUE, showWarnings = FALSE)

input_files <- c(
  main = "exposome_FINAL_clean_qc.csv",
  cognition = "exposome_FINAL_cognition_clean_qc.csv",
  income = "parental_edu_income_sensitivity_clean_qc.csv"
)

output_files_full <- c(
  main = "df_harmonized_exposome_FULL_clean_qc.csv",
  cognition = "df_harmonized_exposome_cognition_FULL_clean_qc.csv",
  income = "df_income_sens_FULL_clean_qc.csv"
)

input_file <- file.path(CLEANED_DIR, input_files[[analysis_type]])
if (!file.exists(input_file)) {
  stop("Cleaned Step-1 input not found: ", input_file)
}

# -------------------------
# Load + structural QC
# -------------------------
exposome_df <- read.csv(input_file, check.names = FALSE)
cat("Loaded rows:", nrow(exposome_df), "\n")
cat("Loaded cols:", ncol(exposome_df), "\n")

ID_COL <- "subject_id_clean"
GROUP_COL <- "matched_group"
BATCH_COL <- "batch_device_software"

required_admin <- c(ID_COL, GROUP_COL, BATCH_COL)
missing_admin <- setdiff(required_admin, names(exposome_df))
if (length(missing_admin) > 0) {
  stop("Missing required administrative columns: ", paste(missing_admin, collapse = ", "))
}

if (anyDuplicated(exposome_df[[ID_COL]]) > 0) {
  dup <- exposome_df[
    duplicated(exposome_df[[ID_COL]]) |
      duplicated(exposome_df[[ID_COL]], fromLast = TRUE),
  ]
  dup_file <- file.path(
    QC_DIR,
    paste0(analysis_type, "_step2_FULL_duplicate_subject_rows.csv")
  )
  write.csv(dup, dup_file, row.names = FALSE)
  stop("Duplicate subject IDs in Step-2 input. Details saved to: ", dup_file)
}

if (any(is.na(exposome_df[[GROUP_COL]]))) {
  stop("matched_group contains NA in Step-2 input")
}
if (!all(exposome_df[[GROUP_COL]] %in% c(1, 2))) {
  stop("Step-2 input contains matched_group values outside {1,2}")
}

n_A_loaded <- sum(exposome_df[[GROUP_COL]] == 1)
n_B_loaded <- sum(exposome_df[[GROUP_COL]] == 2)
cat("Loaded Group A:", n_A_loaded, "\n")
cat("Loaded Group B:", n_B_loaded, "\n")

sample_flow <- data.frame(
  stage = "01_loaded_step1_cleaned",
  total_n = nrow(exposome_df),
  group1_n = n_A_loaded,
  group2_n = n_B_loaded,
  stringsAsFactors = FALSE
)

# -------------------------
# Batch variable + global singleton/tiny-batch filter
# -------------------------
if (any(is.na(exposome_df[[BATCH_COL]]))) {
  stop("Batch variable contains NA values; Step 1 should have removed these explicitly")
}

exposome_batch <- factor(as.character(exposome_df[[BATCH_COL]]))
batch_counts <- table(exposome_batch)
cat("Batch counts before global filter:\n")
print(sort(batch_counts, decreasing = FALSE))

# CovBat / rowVars require at least 2 subjects per batch globally.
valid_batches_global <- names(batch_counts[batch_counts >= 2])
keep_global <- as.character(exposome_batch) %in% valid_batches_global

global_drop <- exposome_df[
  !keep_global,
  c(ID_COL, GROUP_COL, BATCH_COL),
  drop = FALSE
]

if (nrow(global_drop) > 0) {
  global_drop$exclusion_reason <- "global_batch_size_lt_2"
  global_drop_file <- file.path(
    QC_DIR,
    paste0(analysis_type, "_step2_FULL_global_tiny_batch_exclusions.csv")
  )
  write.csv(global_drop, global_drop_file, row.names = FALSE)
  cat("Dropped globally tiny batches (<2):", nrow(global_drop), "subjects\n")
  print(global_drop)
} else {
  cat("No subjects dropped by global batch-size filter.\n")
}

exposome_df_filt <- exposome_df[keep_global, , drop = FALSE]
exposome_batch_filt <- droplevels(exposome_batch[keep_global])

n_A_global <- sum(exposome_df_filt[[GROUP_COL]] == 1)
n_B_global <- sum(exposome_df_filt[[GROUP_COL]] == 2)

cat(
  "After global batch filter — total:", nrow(exposome_df_filt),
  "A:", n_A_global,
  "B:", n_B_global, "\n"
)

sample_flow <- rbind(
  sample_flow,
  data.frame(
    stage = "02_after_global_batch_filter",
    total_n = nrow(exposome_df_filt),
    group1_n = n_A_global,
    group2_n = n_B_global,
    stringsAsFactors = FALSE
  )
)

# -------------------------
# WM feature matrix
# -------------------------
msmt_cols <- grep("^bundle", names(exposome_df_filt), value = TRUE)
if (length(msmt_cols) == 0) {
  stop("No WM measurement columns beginning with 'bundle' found")
}

cat("Number of WM measurement columns:", length(msmt_cols), "\n")

if (anyNA(exposome_df_filt[, msmt_cols, drop = FALSE])) {
  stop("WM feature matrix contains NA values; Step 1 required-data filter should prevent this")
}

data_exposome <- data.matrix(exposome_df_filt[, msmt_cols, drop = FALSE])
storage.mode(data_exposome) <- "double"
data_exposome <- t(data_exposome)  # features x subjects
cat(
  "WM matrix dimensions (features x subjects):",
  paste(dim(data_exposome), collapse = " x "), "\n"
)

# -------------------------
# CovBat preservation model
# -------------------------
# These covariates are preserved during harmonization. This does not mean
# they must also be included in the downstream General Exposome regression.
if (analysis_type == "main") {
  mod_formula <- ~ age + sex +
    General_SES + School + Family_Values + Family_Turmoil +
    Dense_Urban_Poverty + Extracurriculars + Screen_Time
}

if (analysis_type == "cognition") {
  mod_formula <- ~ age + sex +
    General_SES + School + Family_Values + Family_Turmoil +
    Dense_Urban_Poverty + Extracurriculars + Screen_Time +
    neurocog_pc1.bl + neurocog_pc2.bl + neurocog_pc3.bl
}

if (analysis_type == "income") {
  mod_formula <- ~ age + sex + parental_education + income +
    le_l_adi__addr1__national_prcnt
}

mod_exposome <- model.matrix(mod_formula, data = exposome_df_filt)
cat(
  "CovBat design matrix dimensions:",
  paste(dim(mod_exposome), collapse = " x "), "\n"
)

if (nrow(mod_exposome) != nrow(exposome_df_filt)) {
  stop("model.matrix changed the row count; likely an unexpected missing covariate")
}
if (anyNA(mod_exposome)) {
  stop("CovBat design matrix contains NA values")
}

# -------------------------
# Full-dataset CovBat
# -------------------------
# train=NULL means CovBat estimates all harmonization parameters from the
# complete filtered dataset.
cat("Running CovBat on the FULL filtered dataset...\n")
gc()

covbat_full <- covbat(
  dat = data_exposome,
  bat = exposome_batch_filt,
  mod = mod_exposome,
  train = NULL
)

gc()

# -------------------------
# Reassemble output dataframe
# -------------------------
harmonized_full <- t(covbat_full$dat.covbat)

if (nrow(harmonized_full) != nrow(exposome_df_filt)) {
  stop("Full harmonized output row mismatch")
}
if (ncol(harmonized_full) != length(msmt_cols)) {
  stop("Full harmonized output feature mismatch")
}

non_msmt <- setdiff(names(exposome_df_filt), msmt_cols)

df_harmonized_full <- cbind(
  exposome_df_filt[, non_msmt, drop = FALSE],
  as.data.frame(harmonized_full, check.names = FALSE)
)

names(df_harmonized_full)[
  (ncol(df_harmonized_full) - length(msmt_cols) + 1):ncol(df_harmonized_full)
] <- msmt_cols

# -------------------------
# Final QC by original split label
# -------------------------
A_downstream <- df_harmonized_full[
  df_harmonized_full[[GROUP_COL]] == 1,
  ,
  drop = FALSE
]

B_downstream <- df_harmonized_full[
  df_harmonized_full[[GROUP_COL]] == 2,
  ,
  drop = FALSE
]

if (nrow(A_downstream) != n_A_global) {
  stop("Full harmonized output lost Group-A participants unexpectedly")
}
if (nrow(B_downstream) != n_B_global) {
  stop("Full harmonized output lost Group-B participants unexpectedly")
}

cat("Final full-dataset harmonized N:", nrow(df_harmonized_full), "\n")
cat("Final Group A N:", nrow(A_downstream), "\n")
cat("Final Group B N:", nrow(B_downstream), "\n")

sample_flow <- rbind(
  sample_flow,
  data.frame(
    stage = "03_final_full_harmonized_counts",
    total_n = nrow(df_harmonized_full),
    group1_n = nrow(A_downstream),
    group2_n = nrow(B_downstream),
    stringsAsFactors = FALSE
  )
)

# -------------------------
# Save harmonized file + QC
# -------------------------
outputfile_full <- file.path(
  HARMONIZED_DIR,
  output_files_full[[analysis_type]]
)

write.csv(
  df_harmonized_full,
  outputfile_full,
  row.names = FALSE
)

flow_file <- file.path(
  QC_DIR,
  paste0(analysis_type, "_step2_FULL_sample_flow.csv")
)
write.csv(sample_flow, flow_file, row.names = FALSE)

cat("\nSample flow:\n")
print(sample_flow)
cat("\n[SAVE] FULL ->", outputfile_full, "\n")
cat("[SAVE] QC ->", flow_file, "\n")
