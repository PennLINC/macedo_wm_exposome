import re
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr
from matplotlib import patheffects as path_effects
from matplotlib.colors import TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

def run_partial_r(df, msmt_cols, covariates, include_etiv=False, target_cov="General_SES"):
    predictors = covariates + (["eTIV"] if include_etiv else [])
    scaler = StandardScaler()
    out_rows = []

    for feat in msmt_cols:
        y = StandardScaler().fit_transform(df[[feat]]).ravel()

        # full model
        X_full_scaled = scaler.fit_transform(df[predictors])
        X_full = pd.DataFrame(X_full_scaled, columns=predictors, index=df.index)
        X_full = sm.add_constant(X_full)
        model_full = sm.OLS(y, X_full).fit()

        # reduced model
        reduced_predictors = [c for c in predictors if c != target_cov]
        X_reduced_scaled = scaler.fit_transform(df[reduced_predictors])
        X_reduced = pd.DataFrame(X_reduced_scaled, columns=reduced_predictors, index=df.index)
        X_reduced = sm.add_constant(X_reduced)
        model_reduced = sm.OLS(y, X_reduced).fit()

        # compute partial r 
        r2_full, r2_red = model_full.rsquared, model_reduced.rsquared
        partial_r2 = (r2_full - r2_red) / (1 - r2_red)

        # sign from correct coefficient
        sign = np.sign(model_full.params[target_cov])
        partial_r = sign * np.sqrt(abs(partial_r2))

        # F-test for nested models
        anova_res = sm.stats.anova_lm(model_reduced, model_full)
        p_val = anova_res["Pr(>F)"][1]

        out_rows.append({"feature": feat, "covariate": target_cov, "partial_r": partial_r, "partial_R2": partial_r2, "p_value": p_val})

    df_out = pd.DataFrame(out_rows)
    _, p_fdr, _, _ = multipletests(df_out["p_value"], method="fdr_bh") # FDR correction
    df_out["p_fdr"] = p_fdr
    
    return df_out

def plot_replicability_scatter(dfA, dfB, covariate, ax):
    dfA_cov = dfA[dfA["covariate"] == covariate]
    dfB_cov = dfB[dfB["covariate"] == covariate]
    merged = pd.merge(dfA_cov, dfB_cov, on="feature", suffixes=("_A", "_B"))

    ax.scatter(merged["partial_r_A"], merged["partial_r_B"], s=12, alpha=0.7)
    lim = np.nanmax(np.abs(merged[["partial_r_A", "partial_r_B"]].to_numpy()))
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Partial r (Split A)")
    ax.set_ylabel("Partial r (Split B)")
    ax.set_title(f"{covariate} — Split-Half Replicability")

    r = np.corrcoef(merged["partial_r_A"], merged["partial_r_B"])[0, 1]
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, fontsize=12, verticalalignment="top", 
            bbox=dict(facecolor="white", alpha=0.8, lw=0))

prefixes = ['Association_', 'Cerebellum_', 'Commissure_', 'ProjectionBrainstem_', 'ProjectionBasalGanglia_']

def clean_bundle_name_noLR(name):
    name = re.sub(r'(L|R)$', '', name) 
    for p in prefixes:
        if name.startswith(p):
            name = name[len(p):]
            break
    name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
    return name.strip()

def collapse_lr(df):
    df = df.copy()

    parts = df["feature"].str.split("_")
    df = df[parts.str.len() >= 4].copy()
    parts = df["feature"].str.split("_")

    df["bundle"] = (
        parts.str[1] + "_" + parts.str[2]
    ).str.replace(r"(?i)(l|r)$", "", regex=True)

    df["metric"] = parts.str[3:].str.join("_")

    return df

def get_bundle_type(bundle_name):
    if bundle_name.startswith("Association_"):
        return "Association_"
    elif bundle_name.startswith("ProjectionBasalGanglia_"):
        return "ProjectionBasalGanglia_"
    elif bundle_name.startswith("ProjectionBrainstem_"):
        return "ProjectionBrainstem_"
    elif bundle_name.startswith("Cerebellum_"):
        return "Cerebellum_"
    elif bundle_name.startswith("Commissure_"):
        return "Commissure_"
    else:
        return None

def draw_star(ax, x, y, s, fontsize=5.5):
    ax.annotate(s, (x, y), xytext=(0, -0.9), textcoords="offset points", fontsize=fontsize,
                ha="center", va="center", fontweight="bold", clip_on=True, zorder=10)

def make_partial_r_heatmap_split_half(
    dfA, dfB, covariate, ax, cmap,
    noddi_macro_metrics_dict=None,
    dki_macro_metrics_dict=None,
    norm=None,
    alphas=[0.05, 0.01, 0.001],
    bundle_palette=None
):
    """
    • Collapses L/R bundles
    • Upper-left triangle = A
    • Lower-right triangle = B
    • Full gray cell unless BOTH halves significant
    • Multi-level stars (*, **, ***)
    """

    # collapse and avg p and partial r across l/r bundles
    dfA = collapse_lr(dfA[dfA["covariate"] == covariate])
    dfB = collapse_lr(dfB[dfB["covariate"] == covariate])

    dfA = dfA.groupby(["bundle", "metric", "covariate"], as_index=False).agg(
        {"partial_r": "mean", "p_fdr": "mean"}
    )
    dfB = dfB.groupby(["bundle", "metric", "covariate"], as_index=False).agg(
        {"partial_r": "mean", "p_fdr": "mean"}
    )

    # metric ordering logic
    unique_metrics = dfA["metric"].unique().tolist()

    if any(m.startswith("NODDI") for m in unique_metrics):
        lookup_dict = noddi_macro_metrics_dict
        main_metrics = sorted([m for m in unique_metrics if m.startswith("NODDI")])
        other_metrics = [m for m in unique_metrics if not m.startswith("NODDI")]
    else:
        lookup_dict = dki_macro_metrics_dict
        main_metrics = sorted([m for m in unique_metrics if m.startswith("DKI")])
        other_metrics = [m for m in unique_metrics if not m.startswith("DKI")]

    macro_order = [
        "1st_quarter_volume_mm3", "2nd_and_3rd_quarter_volume_mm3", "4th_quarter_volume_mm3",
        "volume_of_end_branches_1", "volume_of_end_branches_2", "total_volume_mm3",
        "area_of_end_region_1_mm2", "area_of_end_region_2_mm2", "total_area_of_end_regions_mm2",
        "total_surface_area_mm2", "radius_of_end_region_1_mm", "radius_of_end_region_2_mm",
        "total_radius_of_end_regions_mm", "irregularity", "curl", "elongation",
        "mean_length_mm", "span_mm"
    ]

    macro_metrics = [m for m in macro_order if m in other_metrics]
    metric_order = main_metrics + macro_metrics

    matA = dfA.pivot(index="bundle", columns="metric", values="partial_r").reindex(columns=metric_order)
    pA   = dfA.pivot(index="bundle", columns="metric", values="p_fdr").reindex_like(matA)

    matB = dfB.pivot(index="bundle", columns="metric", values="partial_r").reindex(columns=metric_order)
    pB   = dfB.pivot(index="bundle", columns="metric", values="p_fdr").reindex_like(matA)

    bundles = matA.index.tolist()
    metrics = matA.columns.tolist()

    # color normalization
    if norm is None:
        combined_vals = np.concatenate([matA.values.ravel(), matB.values.ravel()])
        vmax = np.nanpercentile(np.abs(combined_vals), 98)
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    def star(p):
        if p < alphas[2]:
            return "***"
        elif p < alphas[1]:
            return "**"
        elif p < alphas[0]:
            return "*"
        else:
            return ""

    # triangles and gray cells
    ax.set_xlim(0, len(metrics))
    ax.set_ylim(0, len(bundles))
    ax.invert_yaxis()

    for i, bundle in enumerate(bundles):
        for j, metric in enumerate(metrics):

            valA = matA.loc[bundle, metric]
            valB = matB.loc[bundle, metric]
            pvalA = pA.loc[bundle, metric]
            pvalB = pB.loc[bundle, metric]

            sigA = (not pd.isna(pvalA)) and (pvalA < alphas[0])
            sigB = (not pd.isna(pvalB)) and (pvalB < alphas[0])

            # must be significant in both
            if not (sigA and sigB):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color="gainsboro"))
                continue

            # A – upper-left triangle
            ax.add_patch(
                plt.Polygon(
                    [(j, i + 1), (j, i), (j + 1, i)],
                    facecolor=cmap(norm(valA)),
                    edgecolor="none"
                )
            )
            sA = star(pvalA)
            if sA:
                draw_star(ax, j + 1/3, i + 1/3, sA)

            # B – lower-right triangle
            ax.add_patch(
                plt.Polygon(
                    [(j + 1, i), (j + 1, i + 1), (j, i + 1)],
                    facecolor=cmap(norm(valB)),
                    edgecolor="none"
                )
            )
            sB = star(pvalB)
            if sB:
                draw_star(ax, j + 2/3, i + 2/3, sB)

    # ---- ticks / labels ----
    ax.set_xticks(np.arange(len(metrics)) + 0.5)
    ax.set_yticks(np.arange(len(bundles)) + 0.5)

    ax.set_xticklabels(
        [lookup_dict.get(m, m) for m in metrics],
        rotation=35, ha="right", va="top"
    )
    ax.tick_params(axis="x", pad=1)
    ax.margins(x=0)

    bundles_clean = [clean_bundle_name_noLR(b) for b in bundles]
    ax.set_yticklabels(bundles_clean)

    for tick_label, raw_bundle in zip(ax.get_yticklabels(), bundles):
        bundle_type = get_bundle_type(raw_bundle)
        if bundle_palette is not None and bundle_type in bundle_palette:
            tick_label.set_color(bundle_palette[bundle_type])
            tick_label.set_fontweight("bold")

    # divider between main metrics and macro metrics
    divider_index = len(main_metrics)
    if divider_index > 0:
        ax.axvline(divider_index, color="black", lw=2.0)

    # -------------------------
    # right-side layout:
    # shorter colorbar near top
    # dataset legend below it
    # -------------------------
    fig = ax.figure
    pos = ax.get_position()

    # Shrink only a little to preserve near-square heatmap
    gap = 0.010
    strip_w = 0.060
    cbar_w = 0.012

    ax.set_position([pos.x0, pos.y0, pos.width - (strip_w + gap), pos.height])
    pos = ax.get_position()

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    # colorbar: shorter and near the top
    cbar_h = 0.58 * pos.height
    cbar_x = pos.x1 + gap
    cbar_y = pos.y1 - cbar_h

    cax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(r"Partial $r$ (General Exposome)", labelpad=8)

    # dataset legend below colorbar
    # legend directly below colorbar, same left edge
    leg_x = cbar_x - 0.02
    
    # vertical gap between colorbar and legend
    v_gap = 0.05
    
    leg_w = 0.16
    leg_h = 0.16
    leg_y = cbar_y - leg_h - v_gap

    lax = fig.add_axes([leg_x, leg_y, leg_w, leg_h])
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.set_aspect("equal")
    lax.axis("off")

    # title
    x0, y0, s = 0.00, 0.25, 0.22
    lax.text(0.00, 0.90, "Dataset", ha="left", va="top", fontsize=9)

    # split-square key
    x0, y0, s = 0.00, 0.22, 0.22
    lax.add_patch(
        plt.Rectangle((x0, y0), s, s, facecolor="white", edgecolor="black", lw=0.8)
    )

    # upper-left triangle
    lax.add_patch(
        plt.Polygon(
            [(x0, y0 + s), (x0, y0), (x0 + s, y0 + s)],
            facecolor="lightgray", edgecolor="none"
        )
    )

    # lower-right triangle
    lax.add_patch(
        plt.Polygon(
            [(x0 + s, y0 + s), (x0 + s, y0), (x0, y0)],
            facecolor="darkgray", edgecolor="none"
        )
    )

    # text with larger vertical spacing
    text_x = x0 + s + 0.12
    lax.text(text_x, y0 + 0.18, "Upper = Discovery", ha="left", va="center", fontsize=8)
    lax.text(text_x, y0 + 0.02, "Lower = Replication", ha="left", va="center", fontsize=8)

    # clean spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    return ax
    
def _prepare_partial_r_heatmap_data(
    df,
    covariate,
    value_col,
    p_col="p_fdr",
    noddi_macro_metrics_dict=None,
    dki_macro_metrics_dict=None
):
    """
    Shared preprocessing for partial-r heatmaps:
      • filter covariate
      • collapse L/R
      • average across L/R
      • infer metric family
      • enforce metric order
      • build value and p-value matrices
    """

    df = collapse_lr(df[df["covariate"] == covariate])

    df = df.groupby(["bundle", "metric", "covariate"], as_index=False).agg(
        {value_col: "mean", p_col: "mean"}
    )

    unique_metrics = df["metric"].unique().tolist()

    if any(m.startswith("NODDI") for m in unique_metrics):
        lookup_dict = noddi_macro_metrics_dict if noddi_macro_metrics_dict is not None else {}
        main_metrics = sorted([m for m in unique_metrics if m.startswith("NODDI")])
        other_metrics = [m for m in unique_metrics if not m.startswith("NODDI")]

    elif any(m.startswith("DKI") for m in unique_metrics):
        lookup_dict = dki_macro_metrics_dict if dki_macro_metrics_dict is not None else {}
        main_metrics = sorted([m for m in unique_metrics if m.startswith("DKI")])
        other_metrics = [m for m in unique_metrics if not m.startswith("DKI")]

    else:
        lookup_dict = {}
        main_metrics = []
        other_metrics = unique_metrics

    macro_order = [
        "1st_quarter_volume_mm3", "2nd_and_3rd_quarter_volume_mm3", "4th_quarter_volume_mm3",
        "volume_of_end_branches_1", "volume_of_end_branches_2", "total_volume_mm3",
        "area_of_end_region_1_mm2", "area_of_end_region_2_mm2", "total_area_of_end_regions_mm2",
        "total_surface_area_mm2", "radius_of_end_region_1_mm", "radius_of_end_region_2_mm",
        "total_radius_of_end_regions_mm", "irregularity", "curl", "elongation",
        "mean_length_mm", "span_mm"
    ]

    macro_metrics = [m for m in macro_order if m in other_metrics]
    metric_order = main_metrics + macro_metrics

    mat = df.pivot(index="bundle", columns="metric", values=value_col).reindex(columns=metric_order)
    pmap = df.pivot(index="bundle", columns="metric", values=p_col).reindex_like(mat)

    bundles = mat.index.tolist()
    metrics = mat.columns.tolist()

    return {
        "df": df,
        "mat": mat,
        "pmap": pmap,
        "bundles": bundles,
        "metrics": metrics,
        "lookup_dict": lookup_dict,
        "main_metrics": main_metrics,
        "metric_order": metric_order,
        "unique_metrics": unique_metrics,
    }

def make_partial_r_heatmap_with_fdr_stars(
    df,
    covariate,
    ax,
    cmap,
    noddi_macro_metrics_dict=None,
    dki_macro_metrics_dict=None,
    norm=None,
    alphas=[0.05, 0.01, 0.001],
    cbar_label=r"Partial $r$ (General Exposome)",
    bundle_palette=None
):
    """
    Single-dataset partial-r heatmap with:
      • same preprocessing / ordering / labels as split-half version
      • one full square per cell
      • FDR stars on significant cells
    """

    prep = _prepare_partial_r_heatmap_data(
        df=df,
        covariate=covariate,
        value_col="partial_r",
        p_col="p_fdr",
        noddi_macro_metrics_dict=noddi_macro_metrics_dict,
        dki_macro_metrics_dict=dki_macro_metrics_dict,
    )

    mat = prep["mat"]
    pmap = prep["pmap"]
    bundles = prep["bundles"]
    metrics = prep["metrics"]
    lookup_dict = prep["lookup_dict"]
    main_metrics = prep["main_metrics"]
    unique_metrics = prep["unique_metrics"]

    if len(metrics) == 0:
        raise ValueError(
            f"No plottable metrics found for covariate={covariate!r}. "
            f"Parsed unique metrics: {sorted(unique_metrics)}"
        )

    if norm is None:
        vals = mat.values.ravel()
        vals = vals[~np.isnan(vals)]
        vmax = np.nanpercentile(np.abs(vals), 98) if len(vals) else 1.0
        if vmax == 0:
            vmax = 1.0
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    def star(p):
        if p < alphas[2]:
            return "***"
        elif p < alphas[1]:
            return "**"
        elif p < alphas[0]:
            return "*"
        else:
            return ""

    ax.set_xlim(0, len(metrics))
    ax.set_ylim(0, len(bundles))
    ax.invert_yaxis()

    for i, bundle in enumerate(bundles):
        for j, metric in enumerate(metrics):
            val = mat.loc[bundle, metric]
            pval = pmap.loc[bundle, metric]
            sig = (not pd.isna(pval)) and (pval < alphas[0])

            if (not sig) or pd.isna(val):
                ax.add_patch(
                    plt.Rectangle((j, i), 1, 1, facecolor="gainsboro", edgecolor="none")
                )
                continue

            ax.add_patch(
                plt.Rectangle((j, i), 1, 1, facecolor=cmap(norm(val)), edgecolor="none")
            )

            s = star(pval)
            if s:
                draw_star(ax, j + 0.5, i + 0.5, s, fontsize=6.5)

    ax.set_xticks(np.arange(len(metrics)) + 0.5)
    ax.set_yticks(np.arange(len(bundles)) + 0.5)

    ax.set_xticklabels(
        [lookup_dict.get(m, m) for m in metrics],
        rotation=35, ha="right", va="top"
    )
    ax.tick_params(axis="x", pad=1)
    ax.margins(x=0)

    bundles_clean = [clean_bundle_name_noLR(b) for b in bundles]
    ax.set_yticklabels(bundles_clean)

    for tick_label, raw_bundle in zip(ax.get_yticklabels(), bundles):
        bundle_type = get_bundle_type(raw_bundle)
        if bundle_palette is not None and bundle_type in bundle_palette:
            tick_label.set_color(bundle_palette[bundle_type])
            tick_label.set_fontweight("bold")

    divider_index = len(main_metrics)
    if divider_index > 0:
        ax.axvline(divider_index, color="black", lw=2.0)

    fig = ax.figure
    pos = ax.get_position()

    gap = 0.010
    strip_w = 0.060
    cbar_w = 0.012

    ax.set_position([pos.x0, pos.y0, pos.width - (strip_w + gap), pos.height])
    pos = ax.get_position()

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    cbar_h = 0.58 * pos.height
    cbar_x = pos.x1 + gap
    cbar_y = pos.y1 - cbar_h

    cax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(cbar_label, labelpad=8)

    for spine in ax.spines.values():
        spine.set_visible(False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    return ax


def make_interaction_t_heatmap_split_half(
    dfA,
    dfB,
    covariate,
    ax,
    cmap,
    noddi_macro_metrics_dict=None,
    dki_macro_metrics_dict=None,
    norm=None,
    alpha=0.05,
    use_fdr=True,
    sig_mode="both",   # "both", "either", "none"
    bundle_palette=None,
    gray_color="gainsboro",
    cbar_label="Sex × General SES interaction (t-statistic)"
):
    """
    Split-half interaction heatmap:
      upper-left triangle = A
      lower-right triangle = B
      non-significant cells grayed out depending on sig_mode
    """

    dfA = collapse_lr(dfA[dfA["covariate"] == covariate])
    dfB = collapse_lr(dfB[dfB["covariate"] == covariate])

    p_col = "p_fdr" if use_fdr else "p_value"

    dfA = dfA.groupby(["bundle", "metric", "covariate"], as_index=False).agg(
        {"t_stat": "mean", p_col: "mean"}
    )
    dfB = dfB.groupby(["bundle", "metric", "covariate"], as_index=False).agg(
        {"t_stat": "mean", p_col: "mean"}
    )

    unique_metrics = dfA["metric"].unique().tolist()

    if any(m.startswith("NODDI") for m in unique_metrics):
        lookup_dict = noddi_macro_metrics_dict if noddi_macro_metrics_dict is not None else {}
        main_metrics = sorted([m for m in unique_metrics if m.startswith("NODDI")])
        other_metrics = [m for m in unique_metrics if not m.startswith("NODDI")]
    else:
        lookup_dict = dki_macro_metrics_dict if dki_macro_metrics_dict is not None else {}
        main_metrics = sorted([m for m in unique_metrics if m.startswith("DKI")])
        other_metrics = [m for m in unique_metrics if not m.startswith("DKI")]

    macro_order = [
        "1st_quarter_volume_mm3", "2nd_and_3rd_quarter_volume_mm3", "4th_quarter_volume_mm3",
        "volume_of_end_branches_1", "volume_of_end_branches_2", "total_volume_mm3",
        "area_of_end_region_1_mm2", "area_of_end_region_2_mm2", "total_area_of_end_regions_mm2",
        "total_surface_area_mm2", "radius_of_end_region_1_mm", "radius_of_end_region_2_mm",
        "total_radius_of_end_regions_mm", "irregularity", "curl", "elongation",
        "mean_length_mm", "span_mm"
    ]
    macro_metrics = [m for m in macro_order if m in other_metrics]
    metric_order = main_metrics + macro_metrics

    matA = dfA.pivot(index="bundle", columns="metric", values="t_stat").reindex(columns=metric_order)
    pA   = dfA.pivot(index="bundle", columns="metric", values=p_col).reindex_like(matA)

    matB = dfB.pivot(index="bundle", columns="metric", values="t_stat").reindex(columns=metric_order)
    pB   = dfB.pivot(index="bundle", columns="metric", values=p_col).reindex_like(matA)

    bundles = matA.index.tolist()
    metrics = matA.columns.tolist()

    if norm is None:
        combined_vals = np.concatenate([matA.values.ravel(), matB.values.ravel()])
        combined_vals = combined_vals[~np.isnan(combined_vals)]
        vmax = np.nanpercentile(np.abs(combined_vals), 98) if len(combined_vals) else 1.0
        if vmax == 0:
            vmax = 1.0
        norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    ax.set_xlim(0, len(metrics))
    ax.set_ylim(0, len(bundles))
    ax.invert_yaxis()

    for i, bundle in enumerate(bundles):
        for j, metric in enumerate(metrics):

            valA = matA.loc[bundle, metric]
            valB = matB.loc[bundle, metric]
            pvalA = pA.loc[bundle, metric]
            pvalB = pB.loc[bundle, metric]

            sigA = (not pd.isna(pvalA)) and (pvalA < alpha)
            sigB = (not pd.isna(pvalB)) and (pvalB < alpha)

            if sig_mode == "both":
                show_cell = sigA and sigB
            elif sig_mode == "either":
                show_cell = sigA or sigB
            elif sig_mode == "none":
                show_cell = True
            else:
                raise ValueError("sig_mode must be one of: 'both', 'either', 'none'")

            if (not show_cell) or (pd.isna(valA) and pd.isna(valB)):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=gray_color, ec="none"))
                continue

            # upper-left triangle = A
            if not pd.isna(valA):
                ax.add_patch(
                    plt.Polygon(
                        [(j, i + 1), (j, i), (j + 1, i)],
                        facecolor=cmap(norm(valA)),
                        edgecolor="none"
                    )
                )
            else:
                ax.add_patch(
                    plt.Polygon(
                        [(j, i + 1), (j, i), (j + 1, i)],
                        facecolor=gray_color,
                        edgecolor="none"
                    )
                )

            # lower-right triangle = B
            if not pd.isna(valB):
                ax.add_patch(
                    plt.Polygon(
                        [(j + 1, i), (j + 1, i + 1), (j, i + 1)],
                        facecolor=cmap(norm(valB)),
                        edgecolor="none"
                    )
                )
            else:
                ax.add_patch(
                    plt.Polygon(
                        [(j + 1, i), (j + 1, i + 1), (j, i + 1)],
                        facecolor=gray_color,
                        edgecolor="none"
                    )
                )

    ax.set_xticks(np.arange(len(metrics)) + 0.5)
    ax.set_yticks(np.arange(len(bundles)) + 0.5)

    ax.set_xticklabels(
        [lookup_dict.get(m, m) for m in metrics],
        rotation=35, ha="right", va="top"
    )
    ax.tick_params(axis="x", pad=1)
    ax.margins(x=0)

    bundles_clean = [clean_bundle_name_noLR(b) for b in bundles]
    ax.set_yticklabels(bundles_clean)

    for tick_label, raw_bundle in zip(ax.get_yticklabels(), bundles):
        bundle_type = get_bundle_type(raw_bundle)
        if bundle_palette is not None and bundle_type in bundle_palette:
            tick_label.set_color(bundle_palette[bundle_type])
            tick_label.set_fontweight("bold")

    divider_index = len(main_metrics)
    if divider_index > 0:
        ax.axvline(divider_index, color="black", lw=2.0)

    fig = ax.figure
    pos = ax.get_position()

    gap = 0.010
    strip_w = 0.060
    cbar_w = 0.012

    ax.set_position([pos.x0, pos.y0, pos.width - (strip_w + gap), pos.height])
    pos = ax.get_position()

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    cbar_h = 0.58 * pos.height
    cbar_x = pos.x1 + gap
    cbar_y = pos.y1 - cbar_h

    cax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(cbar_label, labelpad=8)

    # dataset legend
    leg_x = cbar_x - 0.02
    v_gap = 0.05
    leg_w = 0.16
    leg_h = 0.16
    leg_y = cbar_y - leg_h - v_gap

    lax = fig.add_axes([leg_x, leg_y, leg_w, leg_h])
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.set_aspect("equal")
    lax.axis("off")

    lax.text(0.00, 0.90, "Dataset", ha="left", va="top", fontsize=9)

    x0, y0, s = 0.00, 0.22, 0.22
    lax.add_patch(plt.Rectangle((x0, y0), s, s, facecolor="white", edgecolor="black", lw=0.8))
    lax.add_patch(
        plt.Polygon([(x0, y0 + s), (x0, y0), (x0 + s, y0 + s)],
                    facecolor="lightgray", edgecolor="none")
    )
    lax.add_patch(
        plt.Polygon([(x0 + s, y0 + s), (x0 + s, y0), (x0, y0)],
                    facecolor="darkgray", edgecolor="none")
    )

    text_x = x0 + s + 0.12
    lax.text(text_x, y0 + 0.18, "Upper = Discovery", ha="left", va="center", fontsize=8)
    lax.text(text_x, y0 + 0.02, "Lower = Replication", ha="left", va="center", fontsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    return ax

def run_interaction_tstats(
    df,
    msmt_cols,
    covariates,
    exposure="General_SES",
    sex_var="sex",
    include_etiv=False,
    etiv_var="eTIV"
):
    """
    Run one linear model per feature:
        y ~ sex + exposure + sex:exposure + other covariates (+ eTIV)

    Returns:
        DataFrame with columns:
        feature, covariate, interaction_term, t_stat, beta, p_value, p_fdr
    """

    df = df.copy()

    # covariates to include as main effects besides sex/exposure
    extra_covs = [c for c in covariates if c not in [sex_var, exposure]]
    if include_etiv and etiv_var not in extra_covs:
        extra_covs = extra_covs + [etiv_var]

    out_rows = []

    for feat in msmt_cols:
        cols_needed = [feat, sex_var, exposure] + extra_covs
        sub = df[cols_needed].dropna().copy()

        if sub.empty:
            out_rows.append({
                "feature": feat,
                "covariate": exposure,
                "interaction_term": f"{sex_var}:{exposure}",
                "beta": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan
            })
            continue

        # z-score outcome for comparability across features
        y = sub[feat]
        y_sd = y.std(ddof=0)
        if pd.isna(y_sd) or y_sd == 0:
            out_rows.append({
                "feature": feat,
                "covariate": exposure,
                "interaction_term": f"{sex_var}:{exposure}",
                "beta": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan
            })
            continue

        sub[feat] = (y - y.mean()) / y_sd

        # formula
        rhs_terms = [sex_var, exposure, f"{sex_var}:{exposure}"] + extra_covs
        formula = f"{feat} ~ " + " + ".join(rhs_terms)

        try:
            model = smf.ols(formula=formula, data=sub).fit()

            # statsmodels can name the interaction either sex:exposure or exposure:sex
            interaction_name_1 = f"{sex_var}:{exposure}"
            interaction_name_2 = f"{exposure}:{sex_var}"

            if interaction_name_1 in model.params.index:
                term = interaction_name_1
            elif interaction_name_2 in model.params.index:
                term = interaction_name_2
            else:
                term = interaction_name_1  # fallback label

            beta = model.params.get(term, np.nan)
            t_stat = model.tvalues.get(term, np.nan)
            p_val = model.pvalues.get(term, np.nan)

        except Exception:
            beta = np.nan
            t_stat = np.nan
            p_val = np.nan
            term = f"{sex_var}:{exposure}"

        out_rows.append({
            "feature": feat,
            "covariate": exposure,
            "interaction_term": term,
            "beta": beta,
            "t_stat": t_stat,
            "p_value": p_val
        })

    df_out = pd.DataFrame(out_rows)

    valid = df_out["p_value"].notna()
    p_fdr = np.full(len(df_out), np.nan)
    if valid.sum() > 0:
        _, p_fdr_valid, _, _ = multipletests(df_out.loc[valid, "p_value"], method="fdr_bh")
        p_fdr[valid] = p_fdr_valid

    df_out["p_fdr"] = p_fdr
    return df_out