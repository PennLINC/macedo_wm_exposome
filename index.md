# White matter reflects the childhood exposome

# Replication Guide

1. TOC {:toc}

## Abstract

The childhood environment is critical for brain development and contributes to cognitive outcomes. However, most neuroimaging studies examine a single environmental measure (e.g., socioeconomic status) or a limited set of exposures, obscuring how additive exposures jointly influence brain development. Here we investigated how white matter shape and tissue properties in youth are linked to the childhood exposome, a multidimensional measure capturing a wide variety of environmental exposures. Using multi-shell diffusion MRI from 8,183 children (ages 9-10) in the ABCD study, we quantified microstructural and macrostructural properties across 62 person-specific white matter tracts. The exposome showed widespread and highly replicable associations with both white matter microstructure and macrostructure. The strength of these effects was related to the degree a tract spanned the cortical hierarchy defined by the sensorimotor-association axis. Multivariate models demonstrated that patterns of white matter features explained 25% of the variance in the exposome in unseen individuals (out of sample r=0.50). Notably, white matter-based prediction of cognition was markedly reduced after accounting for the exposome (r=0.37 to r=0.16; an ~82% reduction in explained variance), indicating that brain-cognition associations overlap substantially with variance captured by the exposome. These findings generalized to an independent cohort, the Healthy Brain Network (n=869), which differs substantially from ABCD in MRI acquisition, participant selection, and childhood environments. Together, these results suggest that the environment is reflected in developing white matter architecture in childhood.

## Project Lead

Briana Macedo

## Faculty Lead

Theodore D. Satterthwaite

## Analytic Replicator

Joëlle Bagautdinova

## Collaborators

Joëlle Bagautdinova, Steven L. Meisler, Matthew Cieslak, Lucinda M. Sisk, Christos Davatazikos, Alexandre R. Franco, Meike D. Hettwer, Arielle S. Keller, Gregory Kiar, Audrey C. Luo, Allyson P. Mackey, Michael P. Milham, Tyler M. Moore, Valerie J. Sydnor, Kevin Y. Sun, Fang-Cheng Yeh, Ran Barzilay, Damien A. Fair, Russell T. Shinohara, Aaron Alexander-Bloch, Theodore D. Satterthwaite

## Project Start Date

June 2025

## Dataset

ABCD, with replication in HBN

## **Github Repository**

[https://github.com/PennLINC/macedo_wm_exposome](https://github.com/PennLINC/macedo_wm_exposome)

## Respublica Project Directory

/mnt/isilon/bgdlab_hbcd/projects/macedo_wm/exposome

## Slack Channel

#macedo_tractometry

# Directory Structure

```jsx
project_root/
├── code/
├── input_data/
│   ├── HBN/
│   └── ABCD (files at root)
├── output_data/
│   ├── cleaned/
│   │   ├── HBN/
│   │   └── ABCD (files at root)
│   ├── harmonized/
│   │   ├── HBN/
│   │   └── ABCD (files at root)
│   ├── model_outputs/
│   └── results/
│       ├── main_figures/
│       │   ├── figure2/
│       │   ├── figure3/
│       │   ├── etc
│       └── supplemental_figures/
```

# Code Documentation

## Step 0: Required Software

To access required Python software, you can download the `macedo-em-exposome-env.yml` file from GitHub: https://github.com/PennLINC/macedo_wm_exposome/blob/main/macedo-em-exposome-env.yml 

Then, you can use anaconda/mamba to create an environment with the required software packages.

- `conda env create -f macedo-em-exposome-env.yml`

To access required R software, you can download `install_r_packages.sh` file from GitHub and then run
- chmod +x install_r_packages.sh
- bash install_r_packages.sh

To generate tracts, we use DSIStudio: https://dsi-studio.labsolver.org/download.html for MacOS15+.

## Step 1: Prepare Dataset

### **Notebook:** `Step1_Data_Prep.ipynb`

### **Goal**

Create a single ABCD baseline arm dataframe that merges:

- white matter features
- split-half assignment
- exposome factor scores
- cognition principal components

### **Inputs**

1.  dMRI processed data from Steven: `/mnt/isilon/bgdlab_hbcd/projects/meisler_abcd_dmri_new/data/raw_data/merged_data_meisler_analyses.parquet`
2. ABCD Reproducible Matched Samples:  `participants.tsv` from ABCD to determine demographically matched split halves using the `matched_group` variable. Also contains data for sensitivity analyses: namely the categorical `parental_education` and `income` variables. Sets values 777 (Don’t Know) and 999 (Declined to answer) to NA when creating the sample for the sensitivity analysis.
3. Exposome scores from Arielle: `abcd_longitudinal_factor_scores_7f_bifactor.csv`
    1. [Keller AS, Moore TM, Luo A, et al. A general exposome factor explains individual differences in functional brain network topography and cognition in youth. *Dev Cogn Neurosci*. 2024;66:101370. doi:10.1016/j.dcn.2024.101370](https://www.zotero.org/google-docs/?z0C1Tb) 
4. Area Deprivation Index (ADI; for sensitivity analysis) from Steven, downloaded from LASSO
    1. `/mnt/isilon/bgdlab_hbcd/shared_data/ABCD/LASSO_tabular/dairc/rawdata/phenotype/le_l_adi.parquet`
5. Cognition data from Kevin: `cognition_data.csv` 
    1. [Thompson WK, Barch DM, Bjork JM, et al. The structure of cognition in 9 and 10 year-old children and associations with problem behaviors: Findings from the ABCD study’s baseline neurocognitive battery. *Dev Cogn Neurosci*. 2019;36:100606. doi:10.1016/j.dcn.2018.12.004](https://www.zotero.org/google-docs/?z0C1Tb) 
6. Combine all of this data using subject IDs from baseline arm (ages 9-10).

### **Feature exclusion**

Remove:

- Bundles:
    - msmt_ProjectionBrainstem_DentatorubrothalamicTract-lr
    - msmt_ProjectionBrainstem_DentatorubrothalamicTract-rl
    - msmt_Commissure_AnteriorCommissure
    - msmt_ProjectionBrainstem_CorticobulbarTractL
    - msmt_ProjectionBrainstem_CorticobulbarTractR
- Metrics:
    - number_of_tracts
    - NODDI_directions

### **Outputs**

1. Primary merged dataset (exposome + WM + cognition + matched_group) -> add your csv name here `exposome_FINAL_11_3.csv`
2. Primary merged cognition dataset (exposome + WM + cognition + matched_group)
    1. `exposome_FINAL_cognition_11_10.csv`
3. Sensitivity dataset (WM + cognition + matched_group + parental_education + income)
    1. `parental_edu_income_sensitivity_df.csv`

## Step 2: Covbat Harmonization

### **R Script (preferred):** `Step2_covbat_harmonization_ses.R`

Run this script in the terminal with: 

`srun -c 4 --mem=32G -t 24:00:00 ~/miniforge3/envs/macedo-wm-exposome-env/bin/Rscript /mnt/isilon/bgdlab_hbcd/projects/macedo_wm_exposome/macedo_wm_exposome/code/Step2_covbat_harmonization.R <analysis_type>`

Analysis types are: main, cognition, and income. Each one takes a little while to run!

### **Goal**

Harmonize white matter features across ABCD sites using CovBat with:

- **Batch variable:** batch_device_software (site, machine, software)
- **Protected covariates:** always age and sex, plus additional variables depending on analysis (see below)
- **Leakage prevention:** run CovBat twice:
    - once using **split half A** as the training subset (harmonization parameters learned from A)
    - once using **split half B** as the training subset (harmonization parameters learned from B)

These two harmonized datasets are later used depending on which split half is treated as the **training** set in downstream analyses.

### **Input**

- Output of Step 1:
    - for main analysis: `exposome_FINAL_11_3.csv`
    - for cognition analysis: `exposome_FINAL_cognition_11_10.csv`
    - for SES-proxy sensitivity analysis: `parental_edu_income_sensitivity_analysis_df.csv`
- All columns beginning with `bundle_` prefix are harmonized (WM features x subjects matrix).
- Other columns (demographics, exposome, cognition, etc.) are not harmonized and are re-attached afterward.

### **Covariates protected**

**Main analysis:** we protect age, sex, and all of the exposome scores and subfactors:

```jsx
model.matrix(~ age + sex + General_SES + School + 
							Family_Values + Family_Turmoil + 
							Dense_Urban_Poverty + Extracurriculars + 
							Screen_Time, data = exposome_df_filt)
```

**Cognition analysis:** we also protect for cognition variables:

```jsx
model.matrix(~ age + sex + General_SES + School + 
							Family_Values + Family_Turmoil + 
							Dense_Urban_Poverty + Extracurriculars + 
							Screen_Time + neurocog_pc1.bl + 
							neurocog_pc2.bl + neurocog_pc3.bl,
							data = exposome_df_filt)
```

**SES-proxy sensitivity analysis:** we protect for parental education and income rather than exposome scores:

```jsx
mod_exposome <- model.matrix(~ age + sex + parental_education + income + le_l_adi_addr1_national_prcnt, 
                              data = exposome_df_filt)
```

### **Output**

Main analysis

- `../output_data/df_harmonized_exposome_cognition_A_12_10.csv`
- `../output_data/df_harmonized_exposome_cognition_B_12_10.csv`

Cognition analysis

- `../output_data/df_harmonized_exposome_cognition_A_12_10.csv`
- `../output_data/df_harmonized_exposome_cognition_B_12_10.csv`

Income sensitivity analysis

- `../output_data/df_income_sens_A_11_30.csv`
- `../output_data/df_income_sens_B_11_30.csv`

**NOTE: on clusters/Respublica, it is easier to run the R script from the command line rather than using the notebook**

- When running the Rscript, you specify in the command line which analysis (main, cognition, income) you would like to run:
    - Step2_covbat_harmonization_ses.R main
    - Step2_covbat_harmonization_ses.R cognition
    - Step2_covbat_harmonization_ses.R income
- Run the following code in Respublica for all three analyses (e.g., main, cognition, income)
- srun -c 4 --mem=32G -t 24:00:00 ~/miniforge3/envs/macedo-wm-exposome-env/bin/Rscript /mnt/isilon/bgdlab_hbcd/projects/macedo_wm_exposome/macedo_wm_exposome/code/Step2_covbat_harmonization_ses.R main

## Step 3: Subset data

### **Python Notebook:** `Step3_abcd_prep_harmonized_dfs`

### **Goal**

Create analysis-specific feature subsets from each harmonized dataframe (A-trained and B-trained), producing versions that include:

- **macrostructure only**
- **NODDI microstructure only**
- **DKI microstructure only**
- **macro + NODDI**
- **macro + DKI**

These subsets are used in downstream analysis.

### **Inputs**

Outputs of Step 2.

### **Outputs**

From dfA:

- dfA_macro_12_10.csv
- dfA_NODDI_12_10.csv
- dfA_macro_plus_NODDI_12_10.csv
- dfA_DKI_12_10.csv
- dfA_macro_plus_DKI_12_10.csv

From dfB:

- same pattern with prefix dfB_...

For cognition-harmonized versions we additionally output:

- dfA_<label>_cog_12_10.csv
- dfB_<label>_cog_12_10.csv

## Step 4: Mass Univariate Analysis and PCA

### **Python Notebook:** `Figure_2_Mass_Univariate.ipynb`

### **Helper module:** `partial_r.py`

### **Goal**

1. Run **mass-univariate** linear regression models testing associations between the **General Exposome factor** and each WM metric (bundle x metric), controlling for covariates.
2. Quantify association strength using **partial *r*** and compute **FDR-corrected p-values**.
3. Summarize the tract-by-metric association pattern using **PCA** applied to a tract-by-metric matrix of partial *r* values.

### **Inputs**

From Step 3 (harmonized + subset dataframes).

- `dfA_macro_plus_NODDI_12_10.csv`
- `dfB_macro_plus_NODDI_12_10.csv`

### **Covariates**

The main covariate set used in mass-univariate models:

```jsx
covariates = ["age", "sex", "t1post_dwi_contrast",
							"General_SES", "School", "Family_Values", "Family_Turmoil",
							"Dense_Urban_Poverty", "Extracurriculars", "Screen_Time"]
```

### **Mass-univariate model**

For each WM feature:

- **Outcome:** standardized feature: y = *z*-score(feature)
- **Full model:** y ~ z(covariates), OLS regression
- **Reduced model:** same model **excluding** General Exposome score
- **Total models:** 1,302 models (62 bundles x 21 metrics, 3 NODDI and 18 macrostructure)
- **Significance testing**:  F-test (anova_lm(reduced, full))
- **Multiple testing correction:** Benjamini–Hochberg FDR across all features
- **Partial *r* formula:**

$$
\text{partial } R^2 = \frac{R^2_{\text{full}} - R^2_{\text{reduced}}}{1 - R^2_{\text{reduced}}}
$$

$$
\text{partial } r = \text{sign}(\beta_{\text{General\_SES}})\sqrt{|\text{partial } R^2|}
$$

### **PCA on bundle-by-metric partial *r* matrix**

**PCA input matrix construction**

1. Compute partial r results separately in split half A and split half B.
2. Average partial r values **across split halves** (A and B)
3. **Z-score across bundles** (scaling is applied so metrics are comparable before PCA).
4. Fit PCA

### **PCA outputs**

```
  ../output_data/pca_ALL_LRkept_bundle_scores_PC12.csv
  ../output_data/pca_ALL_LRkept_metric_loadings_PC12.csv
  ../output_data/pca_ALL_LRkept_explained_var_PC12.csv
```

## Step 5: SA Axis Annotation

### **Python Notebook:** `Figure_3_SA_Axis_Annotation.ipynb`

### **Goal**

Test whether tract-level PCA results (PC1 bundle scores from Step 4) relate to **sensorimotor–association (S-A) axis range** annotations for each tract.

### **Input data**

- Outputs from Step 4:
    - `pca_ALL_LRkept_bundle_scores_PC12.csv`
    - `pca_ALL_LRkept_explained_var_PC12.csv`
- Tract annotation files from Joelle:
    - `abbreviations.xlsx`
    - `tract_sa_axis_ranges_thresh50.csv`

We read in the data, ensure that bundle names are consistent across datasets, then compute a Pearson correlation.

## Step 6: **Multivariate Prediction of Exposome Scores from White Matter Features in Held-Out Data**

### **Python Notebook:** `Figure_4_Multivariate.ipynb`

### **Helper script:** `regression_functions.py`

### **Goal**

Test whether **person-specific white matter features** can **predict General Exposome** in **held-out split-half data**, using **multivariate ridge regressions**.

We evaluate prediction across three feature sets:

1. **NODDI** metrics only
2. **Macrostructure** metrics only
3. **Macrostructure + NODDI** (combined)

We then visualize predicted vs. actual performance in both split directions and quantify robustness using **permutation tests**.

### **Inputs**

Harmonized feature datasets (from Step 3)

Each file is already harmonized and contains columns:

- covariates: age, sex, t1post_dwi_contrast, optionally estimated_brain_volume
- target: General_SES
- features: all columns starting with msmt_ (WM features)
- split label: matched_group (values 1/2 for the split halves)

### **Split-half structure and training/testing directions**

Each harmonized dataset (A-harmonized, B-harmonized) contains both matched groups, but the *harmonization* was trained on one half only (**Step 2**).

Within `rf.run_model_comparison(...)`, the split is:

- In **A-harmonized** dataframe (df_A):
    - Train, matched_group = 1/A
    - Test, matched_group = 2/B
- In **B-harmonized** dataframe (df_B):
    - Train, matched_group = 2/B
    - Test, matched_group = 1/A

### **Models**

**Main model: Ridge regression**

- **Model:** sklearn.linear_model.RidgeCV
- **Performance metric for reporting:** Pearson correlation r(y_true, y_pred)

**Supplementary model: Partial Least Squares (PLS)**

- **Model:** sklearn.cross_decomposition.PLSRegression
- n_components = 17 selected from CV tuning

### **Feature handling and preprocessing**

**Train-only nuisance regression**

Implemented in `rf.prepare_direction_data(...)`:

1. Extract raw feature matrix X and nuisance matrix Z from train/test
2. Fit nuisance model **on training only**:
    - Linear regression: X_train ~ Z_train
3. Residualize:
    - X_train_resid = X_train - Ŷ_train
    - X_test_resid = X_test - Ŷ_test using the **train-fit** nuisance model

**Train-only scaling**

After residualization:

- StandardScaler() is fit on training only for X and y
- both train/test are transformed using the train-fit scalers

### **Outputs from `rf.run_model_comparison(...)`**

**Performance dictionary**

- r_A and r_B, error metrics (mse, r2, medae), permutation p-values (p_val_A, p_val_B)

**Haufe weights**

- For interpretability, coefficients are Haufe transformed: haufe = cov(X_train) @ weights

### **Permutation testing**

**Goal:** Quantify whether observed prediction performance exceeds chance.

**Procedure**

1. Compute test correlation r_true
2. Repeat n_perm = 1000 times:
    - permute y_train (training labels only)
    - refit model on permuted training labels
    - evaluate on the test set
3. Calculation of *p*-value from permutation test:

$$
p = \frac{\#\{ r_{\text{null}} \ge r_{\text{true}} \} + 1}{n_{\text{perm}} + 1}

$$

Note: permutation testing takes a while to run (about 16 hours)!

## Step 7: Cognition vs. Exposome — Unique Multivariate and Mass-Univariate Effects

### **Python Notebook:** `Figure_5_Cognition.ipynb`

### Helper scripts: `partial_r.py` and `regression_functions.py`

### Goal

Evaluates whether white-matter features explain **unique variance** in:

- **General cognition** beyond the general exposome (SES) and
- **General exposome (SES)** beyond cognition

using both:

1. multivariate ridge regression and
2. mass-univariate partial *r* (feature-level effects)

### **Input data**

Harmonized split-half ABCD datasets, cognition and SES variables (Step 3)

e.g. `dfA_macro_plus_NODDI_cog_12_10.csv)`

### **Targets and confound structures**

| **Analysis** | **Target** | **Confounds** | **Interpretation** |
| --- | --- | --- | --- |
| Cognition (base) | neurocog_pc1.bl | age, sex, QC | WM → cognition |
| Cognition | SES | neurocog_pc1.bl | age, sex, QC, **General_SES** | WM explains cognition **unique of SES** |
| SES (base) | General_SES | age, sex, QC | WM → SES |
| SES | Cognition | General_SES | age, sex, QC, **neurocog_pc1.bl** | WM explains SES **unique of cognition** |

### **Multivariate Prediction (Ridge Regression)**

**Procedure**

- WM features are nuisance-regressed and scaled using training data only
- Ridge models are trained on split A and tested on split B, and vice versa.
- Performance is summarized using Pearson *r* between predicted and observed outcomes.

### **Mass-Univariate Partial *r***

**Covariate sets**

cog_covariates      = [age, sex, t1post_dwi_contrastqc_prediction, neurocog_pc1.bl]

ses_covariates      = [age, sex, t1post_dwi_contrastqc_predictiont1post_dwi_contrastqrediction, General_SES]

combined_covariates = [age, sex, t1post_dwi_contrastqc_prediction, neurocog_pc1.bl, General_SES]

Note: this takes a while to run (about 40 minutes).

## Step 8: Prep data for HBN Replication

### **Python Notebook:** `Figure_6_HBN_replication_data_prep.ipynb`

### **Goal**

Assemble an HBN dataset containing:

- bundle-level **NODDI** metrics
- bundle-level **macrostructure** metrics
- QC measure (t1_neighbor_corr)
- demographics (age, sex, site)
- **General SES** exposome score

### **Inputs**

**NODDI bundle scalarstats (subject-level TSVs)**

Local folder: `../input_data/HBN/noddi/`

Generated from CUBIC zipped derivatives with:

```jsx
bash unzip_files.sh \
/cbica/projects/pennlinc_rbc/datasets/LINC_HBN/derivatives/QSIRECON-1-1-0_BUNDLE-STATS_zipped \
/cbica/projects/macedo_long_wm/data/noddi \
"qsirecon/derivatives/qsirecon-wmNODDI/sub-*/ses-1/dwi/sub-*_bundles-DSIStudio_scalarstats*"
```

Refer to Tien’s A12D guide for downloading HBN data:

- https://pennlinc.github.io/AI2D/docs/datasets/HBN/
- https://pennlinc.github.io/AI2D/docs/get_data#31-get-data-without-datalad

We  keep only variables {icvf, isovf, od} and take the mean value per bundle per metric (the `mean` column)

**Macrostructure bundle stats (subject-level CSVs)**

Local folder:  `../input_data/HBN/macro/`

Generated from CUBIC zipped derivatives with:

```jsx
bash unzip_files.sh \ 
/cbica/projects/pennlinc_rbc/datasets/LINC_HBN/derivatives/QSIRECON-1-1-0_BUNDLE-STATS_zipped \
/cbica/projects/macedo_long_wm/data/macro \
"qsirecon/derivatives/qsirecon-MSMTAutoTrack/sub-*/ses-1/dwi/sub-*msmt_bundlestats*"
```

**QC files (subject-level TSVs)**

Local folder: `../input_data/HBN/qc/`

Generated from:

```jsx
bash unzip_files.sh \
/cbica/projects/pennlinc_rbc/datasets/LINC_HBN/derivatives/QSIPREP-1-0-1_zipped \
/cbica/projects/macedo_long_wm/data/qc \
"qsiprep/sub-*/ses-1/dwi/sub-*desc-image_qc*"
```

We extract `t1_neighbor_corr` from each file.

**Demographics / site metadata**

Local folder: `../input_data/HBN/participants.tsv`

Location on CUBIC: `/cbica/projects/pennlinc_rbc/datasets/LINC_HBN/BIDS/participants.tsv`

Columns: participant_id, Sex, Age, site

**Exposome score for HBN**

Local folder `../input_data/HBN/HBN_General_SES.csv`

Location on CUBIC: `/cbica/projects/thalamocortical_development/sample_info/HBN/HBN_General_SES.csv`

### **Output:** `../output_data/cleanedinput_data/HBN/final_hbn_dataset_12_23.csv`

## Step 9: Covbat Harmonization of HBN data and Nuisance Regression

### **R markdown:** `Figure_6_HBN_harmonization.Rmd`

Run this with: 

srun -c 4 --mem=32G -t 24:00:00 ~/miniforge3/envs/macedo-wm-exposome-env/bin/Rscript /mnt/isilon/bgdlab_hbcd/projects/macedo_wm_exposome/macedo_wm_exposome/code/Figure_6_HBN_harmonization.R

### **Goal**

Harmonize HBN dataset to the ABCD feature space for replication analyses, by:

1. combining ABCD and HBN into a single harmonization dataset
2. harmonizing features with CovBat (**site** as batch), **anchored to ABCD site 16**
3. retaining only HBN rows after harmonization
4. nuisance regression HBN features for **age (natural spline, df=3)** and **sex**

### **Input data**

From `HBN_replication_regression.ipynb` and `Step1_Data_Prep.ipynb`, respectively

```
hbn_raw  <- read.csv("../input_data/HBN/final_hbn_dataset_12_23.csv")
abcd_raw <- read.csv("../output_data/exposome_FINAL_cognition_11_10.csv")
```

We combine these datasets and only include features that are in both ABCD and HBN. Then we harmonize, with reference site set to ABCD site 16 (the site with the largest sample), with software and scanner version`anon0b7.Siemens_VE11B` (harmonization procedure is otherwise the same as in Step 2).

### **Protected covariates (kept in signal during harmonization)**

- ns(age, df=3) + sex + General_SES

### **Nuisance regression on HBN (post-harmonization)**

After harmonization, run nuisance regression in HBN only:

- nuisance design: ~ ns(age, df=3) + sex + qc_prediction
- fit all features simultaneously via limma::lmFit

This produces harmonized and residualized features suitable for replication analyses.

### Output:`../output_data/HBN_harmonized_refABCD16_then_nuisanceResid_12_23.csv`

## Step 10: HBN Replication

Rmd: Figure_6_HBN_GAM.Rmd

### Python Notebook: `Figure_6_HBN_replication_regression.ipynb`

**Helper scripts**

- `regression_functions.py` (prediction  of held out data + permutation testing)
- `partial_r.py` (mass-univariate partial r)

### **Goal**

1. Train a multivariate model in ABCD and test **out-of-sample** on HBN (external generalization).
2. Characterize feature-level HBN associations with SES with mass-univariate analysis.

### **Input data**

- **HBN harmonized dataset (from Step 9):**
    
    `../output_data/HBN_harmonized_refABCD16_then_nuisanceResid_12_23.csv`
    
- **ABCD harmonized split-half datasets (from Steps 2–3):**
    
    `../output_data/dfA_macro_plus_NODDI_12_10.csv`
    
    `../output_data/dfB_macro_plus_NODDI_12_10.csv`
    

### Multivariate prediction on held out data (ABCD → HBN)

**Target:** `General_SES`

**Model:** ridge regression

**Confounds in ABCD model training:** `['age', 'sex', 't1post_dwi_contrastqc_prediction']`

**Feature set:** `bundle_msmt_cols` (macrostructure + NODDI features, columns starting with `msmt_`)

**External testing dataset: HBN**: `df_HBN` (already harmonized + nuisance regressed in Step 9)

Note: HBN features are already nuisance regressed in Step 9, so the external testing call uses:

`test_features_already_residualized=True`. This prevents redoing nuisance regression.

### Permutation testing (ABCD → HBN)

Same procedure as in Step 6. Permutes **training labels only.**

**Saved results:** Stored as `.npz` bundles containing `r_true`, `null`, and `p_val`:

- `../figures/perm_results_external/ABCD_A_to_HBN_macroNODDI_perm.npz`
- `../figures/perm_results_external/ABCD_B_to_HBN_macroNODDI_perm.npz`

### Within-HBN repeated cross-validation (sanity check)

**Goal**: Estimate how well SES is *predictable within HBN* using the same feature set.

**Method:** Repeated 5-fold CV (20 repeats) within HBN, fits ridge regression in each fold.

**Output:** Fold-level and repeat-level performance.

### HBN mass-univariate partial r (feature-level SES effects)

**Goal**: Identify which bundles/metrics in HBN show significant SES associations.

**Target covariate:** `General_SES`

**Covariates in model:** `["General_SES", "age", "sex", "qc_prediction"]`

**Multiple-comparisons control:** FDR across all features
