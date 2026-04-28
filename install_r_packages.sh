#!/usr/bin/env bash
# Install R packages into the macedo-rep conda/mamba environment
# needed for Step2_covbat_harmonization_ses.R and Figure_6_HBN_harmonization.R.
#
# Usage:
#   bash install_r_packages.sh
#
# Requires: mamba or conda, env "macedo-rep" already created (see macedo-em-exposome-env.yml).

set -euo pipefail

ENV_NAME="macedo-rep"
if command -v mamba >/dev/null 2>&1; then
  CONDA_EXE="mamba"
elif command -v conda >/dev/null 2>&1; then
  CONDA_EXE="conda"
else
  echo "Error: neither mamba nor conda found on PATH." >&2
  exit 1
fi

R_SNIPPET="$(mktemp --suffix=-install_r_packages.R)"
cleanup() { rm -f "${R_SNIPPET}"; }
trap cleanup EXIT

cat > "${R_SNIPPET}" <<'RS'
options(repos = c(CRAN = "https://cloud.r-project.org"))
install.packages(c("remotes", "BiocManager"), dependencies = TRUE)
install.packages(c(
  "data.table", "dplyr", "tidyr", "ggplot2", "mgcv", "ComBatFamQC"
), dependencies = TRUE)
BiocManager::install("limma", ask = FALSE, update = FALSE)
remotes::install_github("andy1764/CovBat_Harmonization/R", upgrade = "never")
cat("\n[INFO] Verifying load...\n")
invisible(library(CovBat))
invisible(library(data.table))
invisible(library(dplyr))
invisible(library(tidyr))
invisible(library(ggplot2))
invisible(library(ComBatFamQC))
invisible(library(limma))
invisible(library(mgcv))
invisible(library(splines))
cat("[OK] R packages loaded.\n")
RS

echo "[INFO] Installing R packages into ${ENV_NAME} (using ${CONDA_EXE})..."
# Avoid multiline Rscript -e with mamba run (breaks its exec wrapper). No --no-capture-output
# (some mamba builds mishandle it with Rscript).
"${CONDA_EXE}" run -n "${ENV_NAME}" Rscript "${R_SNIPPET}"

echo "[DONE] Activate with:  conda activate ${ENV_NAME}"
echo "       (or: mamba activate ${ENV_NAME})"
