#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(splines)
  library(ComBatFamQC)
  library(limma)
  library(mgcv)
})

ROOT_DIR <- "/mnt/isilon/bgdlab_hbcd/projects/macedo_wm_exposome/macedo_wm_exposome"

CLEANED_DIR <- file.path(ROOT_DIR, "output_data", "cleaned")
HBN_DIR <- file.path(CLEANED_DIR, "HBN")
HARMONIZED_DIR <- file.path(ROOT_DIR, "output_data", "harmonized", "HBN")

if (!dir.exists(HARMONIZED_DIR)) {
  dir.create(HARMONIZED_DIR, recursive = TRUE)
}

message("[INFO] Loading data...")
hbn_raw  <- read.csv(file.path(HBN_DIR, "final_hbn_dataset_12_23.csv"), check.names = FALSE)
abcd_raw <- read.csv(file.path(CLEANED_DIR, "exposome_FINAL_cognition_11_10.csv"), check.names = FALSE)

bad_idx <- which(is.na(names(hbn_raw)) | trimws(names(hbn_raw)) == "")
if (length(bad_idx) > 0) {
  names(hbn_raw)[bad_idx] <- paste0("unnamed_col_", bad_idx)
}

hbn_df <- hbn_raw %>%
  rename(
    age = Age,
    sex = Sex,
    qc_prediction = t1_neighbor_corr
  )

abcd_df <- abcd_raw

feature_cols <- grep("^FEATURES_", names(hbn_df), value = TRUE)

make_bundle_name <- function(col) {
  rest <- sub("^FEATURES_", "", col)
  parts <- strsplit(rest, "_", fixed = TRUE)[[1]]
  
  if (length(parts) < 3) return(col)
  
  bundle_id <- paste(parts[1:2], collapse = "_")
  metric <- paste(parts[3:length(parts)], collapse = "_")
  
  paste0("bundle_", bundle_id, "_", metric)
}

new_names <- vapply(feature_cols, make_bundle_name, character(1))
names(hbn_df)[match(feature_cols, names(hbn_df))] <- new_names

hbn_df  <- hbn_df  %>% mutate(dataset = "HBN",  batch = paste0("HBN_", site))
abcd_df <- abcd_df %>% mutate(dataset = "ABCD", batch = paste0("ABCD_", site))

ref_batch <- "ABCD_16"

hbn_msmt_cols  <- grep("^bundle_", names(hbn_df), value = TRUE)
abcd_msmt_cols <- grep("^bundle_", names(abcd_df), value = TRUE)
msmt_cols <- intersect(hbn_msmt_cols, abcd_msmt_cols)

message("[INFO] HBN msmt cols: ", length(hbn_msmt_cols))
message("[INFO] ABCD msmt cols: ", length(abcd_msmt_cols))
message("[INFO] Intersect msmt_cols: ", length(msmt_cols))

get_bundle <- function(x) {
  sub("^bundle_([^_]+_[^_]+)_.*$", "\\1", x)
}

hbn_bundles  <- sort(unique(get_bundle(hbn_msmt_cols)))
abcd_bundles <- sort(unique(get_bundle(abcd_msmt_cols)))

message("[INFO] Bundles in ABCD not HBN:")
print(setdiff(abcd_bundles, hbn_bundles))

message("[INFO] Bundles in HBN not ABCD:")
print(setdiff(hbn_bundles, abcd_bundles))

hbn_df$site <- as.character(hbn_df$site)
abcd_df$site <- as.character(abcd_df$site)

all_df <- bind_rows(hbn_df, abcd_df)
all_df$sex <- as.factor(all_df$sex)

all_df$batch <- factor(trimws(all_df$batch))
ref_batch <- trimws(as.character(ref_batch))

message("[INFO] Running CovBat harmonization...")
harm <- covfam(
  data = as.matrix(all_df[, msmt_cols]),
  bat = all_df$batch,
  covar = all_df[, c("age", "sex", "General_SES")],
  model = mgcv::gam,
  formula = y ~ s(age, k = 3) + sex + General_SES,
  eb = TRUE,
  ref.batch = ref_batch
)

X_harm <- harm$dat.covbat
all_df_harm <- all_df
all_df_harm[, msmt_cols] <- X_harm

hbn_out <- all_df_harm[all_df_harm$dataset == "HBN", , drop = FALSE] %>%
  as.data.frame()

keep_meta <- c(
  "participant_id",
  "site",
  "batch",
  "dataset",
  "age",
  "sex",
  "qc_prediction",
  "General_SES",
  "matched_group"
)
keep_meta <- intersect(keep_meta, names(hbn_out))

hbn_out <- hbn_out[, c(keep_meta, msmt_cols), drop = FALSE]

output_path1 <- file.path(HARMONIZED_DIR, "HBN_harmonized_refABCD16_01_20.csv")
write.csv(hbn_out, file = output_path1, row.names = FALSE)
message("[SAVE] -> ", output_path1)

message("[INFO] Running nuisance regression...")
hbn_out$sex <- droplevels(as.factor(hbn_out$sex))

Z <- model.matrix(~ ns(age, df = 3) + sex + qc_prediction, data = hbn_out)
Y <- t(as.matrix(hbn_out[, msmt_cols]))

fit <- lmFit(Y, Z)
Y_hat <- fit$coefficients %*% t(Z)
Y_resid <- Y - Y_hat
hbn_out[, msmt_cols] <- t(Y_resid)

output_path2 <- file.path(HARMONIZED_DIR, "HBN_harmonized_refABCD16_then_nuisanceResid_12_23.csv")
write.csv(hbn_out, file = output_path2, row.names = FALSE)
message("[SAVE] -> ", output_path2)

message("[DONE]")