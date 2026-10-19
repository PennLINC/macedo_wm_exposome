#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(CovBat)
  library(data.table)
  library(dplyr)
  library(tidyr)
})

# ============================================================================
# Step 2 — Split-aware CovBat harmonization with explicit sample-flow QC
# ============================================================================
#
# Key changes relative to the older script:
#   * explicit input/output/QC paths;
#   * checks one row per subject and matched_group in {1,2};
#   * records globally tiny-batch exclusions with subject IDs;
#   * verifies that all A-training participants survive the A-trained run and
#     all B-training participants survive the B-trained run;
#   * records cross-split subjects omitted from each harmonization run because
#     their batch was absent from that training split;
#   * preserves the current CovBat biological-covariate model. Inclusion of the
#     six exposome subfactors here means their variation is preserved during
#     harmonization; it is NOT the same as adjusting for them in the downstream
#     General Exposome -> WM regression.
#
# Usage:
#   Rscript Step2_covbat_harmonization_clean_qc.R main
#   Rscript Step2_covbat_harmonization_clean_qc.R cognition
#   Rscript Step2_covbat_harmonization_clean_qc.R income
# ============================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript Step2_covbat_harmonization_clean_qc.R <main|cognition|income>")
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
QC_DIR <- file.path(HARMONIZED_DIR, "qc")

dir.create(HARMONIZED_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(QC_DIR, recursive = TRUE, showWarnings = FALSE)

input_files <- c(
  main = "exposome_FINAL_clean_qc.csv",
  cognition = "exposome_FINAL_cognition_clean_qc.csv",
  income = "parental_edu_income_sensitivity_clean_qc.csv"
)

output_files_A <- c(
  main = "df_harmonized_exposome_A_clean_qc.csv",
  cognition = "df_harmonized_exposome_cognition_A_clean_qc.csv",
  income = "df_income_sens_A_clean_qc.csv"
)

output_files_B <- c(
  main = "df_harmonized_exposome_B_clean_qc.csv",
  cognition = "df_harmonized_exposome_cognition_B_clean_qc.csv",
  income = "df_income_sens_B_clean_qc.csv"
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
  dup <- exposome_df[duplicated(exposome_df[[ID_COL]]) | duplicated(exposome_df[[ID_COL]], fromLast = TRUE), ]
  dup_file <- file.path(QC_DIR, paste0(analysis_type, "_step2_duplicate_subject_rows.csv"))
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

global_drop <- exposome_df[!keep_global, c(ID_COL, GROUP_COL, BATCH_COL), drop = FALSE]
if (nrow(global_drop) > 0) {
  global_drop$exclusion_reason <- "global_batch_size_lt_2"
  global_drop_file <- file.path(QC_DIR, paste0(analysis_type, "_step2_global_tiny_batch_exclusions.csv"))
  write.csv(global_drop, global_drop_file, row.names = FALSE)
  cat("Dropped globally tiny batches (<2):", nrow(global_drop), "subjects\n")
  print(global_drop)
} else {
  cat("No subjects dropped by global batch-size filter.\n")
}

exposome_df_filt <- exposome_df[keep_global, , drop = FALSE]
exposome_batch_filt <- droplevels(exposome_batch[keep_global])

train_A_filt <- exposome_df_filt[[GROUP_COL]] == 1
train_B_filt <- exposome_df_filt[[GROUP_COL]] == 2

n_A_global <- sum(train_A_filt)
n_B_global <- sum(train_B_filt)
cat("After global batch filter — total:", nrow(exposome_df_filt),
    "A:", n_A_global, "B:", n_B_global, "\n")

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
cat("WM matrix dimensions (features x subjects):", paste(dim(data_exposome), collapse = " x "), "\n")

# -------------------------
# CovBat preservation model
# -------------------------
# NOTE: These covariates are preserved during harmonization. This does not mean
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
  mod_formula <- ~ age + sex + parental_education + income + le_l_adi__addr1__national_prcnt
}

mod_exposome <- model.matrix(mod_formula, data = exposome_df_filt)
cat("CovBat design matrix dimensions:", paste(dim(mod_exposome), collapse = " x "), "\n")

if (nrow(mod_exposome) != nrow(exposome_df_filt)) {
  stop("model.matrix changed the row count; likely an unexpected missing covariate")
}
if (anyNA(mod_exposome)) {
  stop("CovBat design matrix contains NA values")
}

# -------------------------
# Helper: prepare and audit a training-specific CovBat run
# -------------------------
prepare_run <- function(train_mask, train_label) {
  batches_in_train <- unique(as.character(exposome_batch_filt[train_mask]))
  keep_run <- as.character(exposome_batch_filt) %in% batches_in_train

  dropped <- exposome_df_filt[!keep_run, c(ID_COL, GROUP_COL, BATCH_COL), drop = FALSE]
  if (nrow(dropped) > 0) {
    dropped$exclusion_reason <- paste0("batch_absent_from_", train_label, "_training_set")
    drop_file <- file.path(
      QC_DIR,
      paste0(analysis_type, "_step2_dropped_from_", train_label, "_trained_run.csv")
    )
    write.csv(dropped, drop_file, row.names = FALSE)
    cat("Subjects omitted from", train_label, "-trained run because batch absent from training set:",
        nrow(dropped), "\n")
  }

  # Critical leakage-prevention assertion: every subject belonging to the
  # training split must necessarily have a batch represented in that split.
  if (any(train_mask & !keep_run)) {
    stop("Internal error: training-set subjects were removed from their own CovBat run")
  }

  list(
    keep = keep_run,
    data = data_exposome[, keep_run, drop = FALSE],
    bat = droplevels(exposome_batch_filt[keep_run]),
    mod = mod_exposome[keep_run, , drop = FALSE],
    train = train_mask[keep_run],
    df = exposome_df_filt[keep_run, , drop = FALSE]
  )
}

# -------------------------
# A-trained CovBat
# -------------------------
run_A <- prepare_run(train_A_filt, "A")
cat("Running CovBat with A as training set...\n")
gc()
covbat_A <- covbat(
  dat = run_A$data,
  bat = run_A$bat,
  mod = run_A$mod,
  train = run_A$train
)
gc()

# -------------------------
# B-trained CovBat
# -------------------------
run_B <- prepare_run(train_B_filt, "B")
cat("Running CovBat with B as training set...\n")
gc()
covbat_B <- covbat(
  dat = run_B$data,
  bat = run_B$bat,
  mod = run_B$mod,
  train = run_B$train
)
gc()

# -------------------------
# Reassemble output dataframes
# -------------------------
harmonized_A <- t(covbat_A$dat.covbat)
harmonized_B <- t(covbat_B$dat.covbat)

if (nrow(harmonized_A) != nrow(run_A$df)) stop("A output row mismatch")
if (nrow(harmonized_B) != nrow(run_B$df)) stop("B output row mismatch")
if (ncol(harmonized_A) != length(msmt_cols)) stop("A output feature mismatch")
if (ncol(harmonized_B) != length(msmt_cols)) stop("B output feature mismatch")

non_msmt_A <- setdiff(names(run_A$df), msmt_cols)
non_msmt_B <- setdiff(names(run_B$df), msmt_cols)

df_harmonized_A <- cbind(
  run_A$df[, non_msmt_A, drop = FALSE],
  as.data.frame(harmonized_A, check.names = FALSE)
)
df_harmonized_B <- cbind(
  run_B$df[, non_msmt_B, drop = FALSE],
  as.data.frame(harmonized_B, check.names = FALSE)
)

names(df_harmonized_A)[(ncol(df_harmonized_A) - length(msmt_cols) + 1):ncol(df_harmonized_A)] <- msmt_cols
names(df_harmonized_B)[(ncol(df_harmonized_B) - length(msmt_cols) + 1):ncol(df_harmonized_B)] <- msmt_cols

# Verify the downstream leakage-prevention subsets have the expected post-global-filter counts.
A_downstream <- df_harmonized_A[df_harmonized_A[[GROUP_COL]] == 1, , drop = FALSE]
B_downstream <- df_harmonized_B[df_harmonized_B[[GROUP_COL]] == 2, , drop = FALSE]

if (nrow(A_downstream) != n_A_global) {
  stop("A-trained output lost A-training participants unexpectedly")
}
if (nrow(B_downstream) != n_B_global) {
  stop("B-trained output lost B-training participants unexpectedly")
}

cat("Final downstream A split N:", nrow(A_downstream), "\n")
cat("Final downstream B split N:", nrow(B_downstream), "\n")
cat("Final downstream total N:", nrow(A_downstream) + nrow(B_downstream), "\n")

sample_flow <- rbind(
  sample_flow,
  data.frame(
    stage = "03_final_downstream_split_counts",
    total_n = nrow(A_downstream) + nrow(B_downstream),
    group1_n = nrow(A_downstream),
    group2_n = nrow(B_downstream),
    stringsAsFactors = FALSE
  )
)

# -------------------------
# Save harmonized files + QC
# -------------------------
outputfileA <- file.path(HARMONIZED_DIR, output_files_A[[analysis_type]])
outputfileB <- file.path(HARMONIZED_DIR, output_files_B[[analysis_type]])

write.csv(df_harmonized_A, outputfileA, row.names = FALSE)
write.csv(df_harmonized_B, outputfileB, row.names = FALSE)

flow_file <- file.path(QC_DIR, paste0(analysis_type, "_step2_sample_flow.csv"))
write.csv(sample_flow, flow_file, row.names = FALSE)

cat("\nSample flow:\n")
print(sample_flow)
cat("\n[SAVE] A ->", outputfileA, "\n")
cat("[SAVE] B ->", outputfileB, "\n")
cat("[SAVE] QC ->", flow_file, "\n")
