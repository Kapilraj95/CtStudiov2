import io
import re
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns  # styling

# ==============================
# Global scientific plot styling
# ==============================
plt.style.use("default")
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.linewidth": 1.2,
    "axes.edgecolor": "#2e2e2e",
    "axes.labelcolor": "#111827",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "grid.color": "#f0f0f0",
    "grid.linewidth": 0.8,
    "font.family": "Inter",
    "figure.figsize": (7, 5),
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

# ---------------------------------------------
# Page config FIRST
# ---------------------------------------------
st.set_page_config(page_title="CtStudio", page_icon="🧪", layout="wide")

# ---------------------------------------------
# Minimal CSS + faint footer color
# ---------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  font-size: 15px;
  background: #f8fafc;
  color: #1f2937;
}
h1, h2, h3 { font-weight: 700; color: #111827; }
h2, h3 { border-left: 4px solid #3b82f6; padding-left: 10px; }

/* Buttons */
div.stButton>button {
  background: linear-gradient(90deg, #2563eb, #3b82f6);
  color: white; border-radius: 8px; border: none; font-weight: 600;
}
.stDownloadButton button {
  background: linear-gradient(90deg, #10b981, #34d399);
  color: white; border-radius: 8px; border: none; font-weight: 600;
}

/* Containers */
.stDataFrame { border: 1px solid #e5e7eb; border-radius: 10px; background: white; }
[data-testid="stExpander"] {
  background: white !important; border-radius: 10px !important;
  border: 1px solid #e5e7eb !important; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* Footer (faint) */
.ctstudio-footer {
  margin-top: 24px;
  padding: 10px 12px;
  text-align: right;
  color: #9ca3af; /* faint gray */
  font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# Header
# ---------------------------------------------
st.markdown("""
<div style="display:flex; align-items:center; justify-content:center; margin-top:8px; margin-bottom:20px;">
  <div style="background:linear-gradient(135deg,#2563eb,#3b82f6,#10b981);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;
              font-size:42px; font-weight:800; letter-spacing:-0.5px;">🧪 CtStudio</div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# Helpers (with caching for speed)
# =====================================================
def find_results_header_row(df_headerless: pd.DataFrame) -> Optional[int]:
    for i in range(min(400, len(df_headerless))):
        row = df_headerless.iloc[i].astype(str).str.strip().tolist()
        low = [c.lower() for c in row]
        if len(row) >= 5 and low[0] == "well" and "sample name" in low and "target name" in low:
            return i
    return None

@st.cache_data(show_spinner=False)
def _read_results_from_bytes(file_bytes: bytes, sheet_name: str = "Results") -> pd.DataFrame:
    """Parse QuantStudio 'Results' from raw bytes (cached)."""
    bio = io.BytesIO(file_bytes)
    raw = pd.read_excel(bio, sheet_name=sheet_name, header=None, engine="openpyxl")
    header_idx = find_results_header_row(raw)
    if header_idx is None:
        raise ValueError("Could not locate the 'Results' header row in sheet 'Results'.")
    bio2 = io.BytesIO(file_bytes)
    df = pd.read_excel(bio2, sheet_name=sheet_name, header=header_idx, engine="openpyxl")
    return df

def read_quantstudio_results(upload, sheet_name="Results") -> pd.DataFrame:
    name = getattr(upload, "name", "uploaded.xlsx")
    df = _read_results_from_bytes(upload.getvalue(), sheet_name)
    cols = {str(c).strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        lower = {str(c).strip().lower(): c for c in df.columns}
        for n in names:
            if n.lower() in lower: return lower[n.lower()]
        return None

    well   = pick("Well", "Well Position")
    sample = pick("Sample Name", "Sample")
    gene   = pick("Target Name", "Target", "Assay Name", "Gene")
    ct     = pick("CT", "Ct", "Cq", "Ct Mean", "Cq Mean")
    if not (sample and gene and ct):
        raise ValueError("Missing required columns in 'Results' (need Sample Name, Target Name, CT/Ct/Cq).")

    keep = [x for x in [sample, gene, ct, well] if x]
    out = df[keep].copy()
    out.columns = ["Sample", "Gene", "Ct", "Well"][:len(keep)]
    out["Ct"] = pd.to_numeric(out["Ct"], errors="coerce")
    out = out.dropna(subset=["Ct"])
    out["File"] = name

    sort_cols = ["File", "Sample", "Gene"]
    if "Well" in out.columns: sort_cols.append("Well")
    out = out.sort_values(sort_cols)
    out["Replicate"] = out.groupby(["File", "Sample", "Gene"]).cumcount() + 1
    return out

def clean_gene_names(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    if not mapping: return df
    df = df.copy()
    m = {k.lower(): v for k, v in mapping.items() if v}
    df["Gene"] = df["Gene"].apply(lambda g: m.get(str(g).lower(), g))
    return df

def assign_groups_blockwise(df: pd.DataFrame, reps_per_sample: int, samples_per_group: int, group_names: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["__order__"] = np.arange(len(df))
    if "Well" in df.columns:
        df = df.sort_values(["File", "Gene", "Well", "Replicate", "__order__"])
    else:
        df = df.sort_values(["File", "Gene", "Sample", "Replicate", "__order__"])
    df["SampleIdx"] = df.groupby(["File", "Gene"]).cumcount() // reps_per_sample

    def idx_to_group(i):
        block = i // samples_per_group
        return group_names[block] if block < len(group_names) else f"Group_{block+1}"

    df["Group"] = df["SampleIdx"].apply(idx_to_group)
    return df.drop(columns=["__order__"])

def qc_flags_per_triplicate(df: pd.DataFrame, reps_per_sample: int, sd_cut: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = df.copy()
    stats = x.groupby(["File", "Gene", "SampleIdx"])["Ct"].agg(["mean", "std", "count"]).reset_index()
    stats["Flag_SD>cut"] = (stats["std"] > sd_cut)
    x = x.merge(stats[["File", "Gene", "SampleIdx", "std", "Flag_SD>cut"]],
                on=["File", "Gene", "SampleIdx"], how="left")
    return x, stats

def aggregate_mean_after_manual_exclusion(df: pd.DataFrame, exclude_mask_col: str) -> pd.DataFrame:
    x = df[~df[exclude_mask_col]].copy()
    agg = x.groupby(["File", "Gene", "SampleIdx", "Group"], as_index=False)["Ct"].mean()
    agg = agg.rename(columns={"Ct": "Ct_agg"})
    return agg

def df_to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, index=False, sheet_name=name[:31])
    buf.seek(0)
    return buf.getvalue()

# =========================
# Cached helpers for speed
# =========================
@st.cache_data(show_spinner=False)
def _shorten_file_labels(unique_files: Tuple[str, ...]) -> Dict[str, str]:
    def _shorten(fname: str) -> str:
        s = re.sub(r"\.xlsx$", "", str(fname), flags=re.I)
        s = re.sub(r"_QuantStudio.*$", "", s)
        return (s[:40] + "…") if len(s) > 41 else s
    return {f: _shorten(f) for f in unique_files}

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    uploads = st.file_uploader("📂 Upload one or more .xlsx exports", type=["xlsx"], accept_multiple_files=True)

    st.divider()
    reps_per_sample = st.number_input("🔬 Replicates per sample", min_value=2, max_value=10, value=3, step=1)
    samples_per_group = st.number_input("🧫 Samples per group (block size)", min_value=1, max_value=100, value=3, step=1)
    group_names_text = st.text_area("🧪 Group names (one per line, in order)", value="Control\nTreatment1\nTreatment2")

    st.divider()
    sd_cut = st.number_input("📏 Triplicate SD threshold (flag if > this)", min_value=0.0, value=0.25, step=0.05)
    manual_exclusion = st.checkbox("🖐️ Enable manual per-replicate exclusion", value=True)

    st.divider()
    calib_mode = st.selectbox("⚖️ Calibrator mode", ["By group mean", "By sample"])
    calib_group = st.text_input("If 'By group mean', which group?", value="Control")

group_names = [g.strip() for g in group_names_text.splitlines() if g.strip()]
normalize_gene_case = True

# Persist per-gene sample exclusion state for Visualization
if "exclude_samples_state" not in st.session_state:
    st.session_state.exclude_samples_state = {}

# =====================================================
# Pipeline
# =====================================================
if uploads:
    # ---------- Read all files ----------
    tidies, errors = [], []
    for up in uploads:
        try:
            t = read_quantstudio_results(up, sheet_name="Results")
            tidies.append(t)
        except Exception as e:
            errors.append(f"{getattr(up,'name','file')}: {e}")
    if errors:
        st.warning("Some files could not be read:\n- " + "\n- ".join(errors))
    if not tidies:
        st.stop()

    tidy = pd.concat(tidies, ignore_index=True)

    # ---------- File label mapping (cached) ----------
    unique_files = tuple(tidy["File"].unique().tolist())
    file_short = _shorten_file_labels(unique_files)
    tidy["FileLabel"] = tidy["File"].map(file_short)

    # Normalize gene case
    if normalize_gene_case:
        tidy["Gene"] = tidy["Gene"].astype(str).str.strip().str.lower().str.title()

    # ---------------------------
    # 📄 Raw CT (preview)
    # ---------------------------
    with st.expander("📄 Raw CT (tidy across files)", expanded=False):
        show_cols = [c for c in ["FileLabel", "Sample", "Gene", "Well", "Replicate", "Ct"] if c in tidy.columns]
        st.dataframe(tidy[show_cols].head(150), use_container_width=True)

    # ---------------------------
    # 🧹 Gene name cleanup / renaming
    # ---------------------------
    with st.expander("🧹 Gene name cleanup / renaming", expanded=False):
        genes_unique = sorted(tidy["Gene"].astype(str).unique())
        rename_df = pd.DataFrame({"Detected": genes_unique, "RenameTo": genes_unique})
        rename_df = st.data_editor(rename_df, num_rows="dynamic", use_container_width=True)
        rename_map = {row["Detected"]: row["RenameTo"] for _, row in rename_df.iterrows()
                      if row["Detected"] != row["RenameTo"]}
        tidy = clean_gene_names(tidy, rename_map)
        st.caption("Renaming applies immediately throughout the pipeline.")

    # ---------------------------
    # Group assignment + QC flags
    # ---------------------------
    tidy = assign_groups_blockwise(
        tidy,
        reps_per_sample=reps_per_sample,
        samples_per_group=samples_per_group,
        group_names=group_names,
    )
    qc_df, trip_stats = qc_flags_per_triplicate(tidy, reps_per_sample=reps_per_sample, sd_cut=sd_cut)

    # ---------------------------
    # 🧪 Replicates with QC flags (manual exclude)
    # ---------------------------
    with st.expander("🧪 Replicates with QC flags", expanded=True):
        if manual_exclusion:
            # Ensure Exclude column exists (default False)
            qc_df["Exclude"] = False

            # --------- FAST auto-suggest exclusion (vectorized) ---------
            group_keys = ["File", "Gene", "SampleIdx"]
            flagged_groups = trip_stats.loc[trip_stats["Flag_SD>cut"], group_keys].drop_duplicates()

            if not flagged_groups.empty:
                key_df = qc_df.merge(flagged_groups.assign(__flag_group__=1), on=group_keys, how="left")
                is_flagged_row = key_df["__flag_group__"].fillna(0).astype(bool)
                mean_ct = qc_df.groupby(group_keys)["Ct"].transform("mean")
                abs_dev = (qc_df["Ct"] - mean_ct).abs()
                abs_dev_masked = abs_dev.where(is_flagged_row, other=np.nan)
                idx_max_per_group = (
                    qc_df.assign(_abs_dev_=abs_dev_masked)
                         .groupby(group_keys)["_abs_dev_"]
                         .idxmax()              # pandas<2.2 safe
                         .dropna()
                         .astype(int)
                         .tolist()
                )
                if idx_max_per_group:
                    qc_df.loc[idx_max_per_group, "Exclude"] = True

            # Display editor with FileLabel
            qcd_disp = qc_df.copy()
            qcd_disp["FileLabel"] = qcd_disp["File"].map(file_short)
            qcd_disp = qcd_disp[["FileLabel", "Gene", "Group", "SampleIdx", "Replicate", "Ct", "std", "Flag_SD>cut", "Exclude"]]
            qc_df = st.data_editor(qcd_disp, use_container_width=True)

            # Rebuild 'File' if editor dropped it
            if "File" not in qc_df.columns and "FileLabel" in qc_df.columns:
                inv = {v: k for k, v in file_short.items()}
                qc_df["File"] = qc_df["FileLabel"].map(inv)

        else:
            qc_df["Exclude"] = False
            qcd_disp = qc_df.copy()
            qcd_disp["FileLabel"] = qcd_disp["File"].map(file_short)
            st.dataframe(
                qcd_disp[["FileLabel", "Gene", "Group", "SampleIdx", "Replicate", "Ct", "std", "Flag_SD>cut"]],
                use_container_width=True,
            )

    # ---------------------------
    # Aggregate means per sample (after manual exclusion)
    # ---------------------------
    agg = aggregate_mean_after_manual_exclusion(qc_df, exclude_mask_col="Exclude")

    # ---------------------------
    # Housekeeping & Calibrator
    # ---------------------------
    genes_now = sorted(agg["Gene"].unique())
    if not genes_now:
        st.error("No genes detected. Check uploads/renaming."); st.stop()
    default_gene = "EF1A" if "EF1A" in genes_now else genes_now[0]
    hk = st.selectbox("Housekeeping gene (post-renaming)", options=genes_now, index=genes_now.index(default_gene))

    if calib_mode == "By group mean":
        calibrator_value = calib_group
    else:
        # Build choices across Group | SampleIdx (file-agnostic)
        choices = agg.sort_values(["Group", "SampleIdx"])[["Group", "SampleIdx"]].drop_duplicates().to_dict("records")
        def _fmt(rec): return f"{rec['Group']} | {int(rec['SampleIdx'])}"
        sel = st.selectbox("Choose calibrator sample", options=choices, format_func=_fmt)
        calibrator_value = f"{sel['Group']} | {int(sel['SampleIdx'])}"

    # ---------------------------
    # ΔCt / ΔΔCt / FoldChange (full table)
    # ---------------------------
    def ddct_pipeline(agg: pd.DataFrame, ref_gene: str, calibrator_mode: str, calibrator_value: str) -> pd.DataFrame:
        hk  = agg[agg["Gene"] == ref_gene].rename(columns={"Ct_agg": "Ct_ref"})
        tgt = agg[agg["Gene"] != ref_gene].rename(columns={"Ct_agg": "Ct_tgt"})
        ref_same   = hk[["File", "Group", "SampleIdx", "Ct_ref"]]
        ref_pooled = hk.groupby(["Group", "SampleIdx"], as_index=False)["Ct_ref"].mean().rename(columns={"Ct_ref": "Ct_ref_pooled"})
        m = tgt.merge(ref_same, on=["File", "Group", "SampleIdx"], how="left")
        m = m.merge(ref_pooled, on=["Group", "SampleIdx"], how="left")
        m["Ct_ref"] = m["Ct_ref"].where(m["Ct_ref"].notna(), m["Ct_ref_pooled"])
        m.drop(columns=["Ct_ref_pooled"], inplace=True)
        m["dCt"] = m["Ct_tgt"] - m["Ct_ref"]

        if calibrator_mode == "By group mean":
            cal_group = str(calibrator_value).strip()
            if cal_group not in m["Group"].unique():
                cal_group = m["Group"].iloc[0]
            ctrl = m[m["Group"] == cal_group].groupby("Gene", as_index=False)["dCt"].mean()
        else:
            parts = [p.strip() for p in str(calibrator_value).split("|")]
            if len(parts) == 1:
                g, idx = parts[0], 0
            elif len(parts) == 2:
                g, idx = parts
            else:
                g, idx = m["Group"].iloc[0], 0
            try:
                idx = int(idx)
            except Exception:
                idx = 0
            ctrl = m[(m["Group"] == g) & (m["SampleIdx"] == idx)][["Gene", "dCt"]]
            if ctrl.empty:
                g_fallback = m["Group"].iloc[0]
                ctrl = m[m["Group"] == g_fallback].groupby("Gene", as_index=False)["dCt"].mean()

        ctrl = ctrl.rename(columns={"dCt": "dCt_cal"})
        out = m.merge(ctrl, on="Gene", how="left")
        out["ddCt"] = out["dCt"] - out["dCt_cal"]
        out["FoldChange"] = np.power(2.0, -out["ddCt"])
        return out[["File", "Group", "SampleIdx", "Gene", "Ct_ref", "Ct_tgt", "dCt", "dCt_cal", "ddCt", "FoldChange"]] \
                 .sort_values(["Gene", "Group", "SampleIdx", "File"])

    ddct = ddct_pipeline(agg, ref_gene=hk, calibrator_mode=calib_mode, calibrator_value=calibrator_value)

    with st.expander("📈 ΔCt / ΔΔCt / FoldChange (full table)", expanded=False):
        ddct_disp = ddct.copy()
        ddct_disp["FileLabel"] = ddct_disp["File"].map(file_short)
        st.dataframe(
            ddct_disp[["FileLabel", "Group", "SampleIdx", "Gene", "Ct_ref", "Ct_tgt", "dCt", "dCt_cal", "ddCt", "FoldChange"]],
            use_container_width=True,
        )

    # =====================================================
    # 📊 Visualization (Mean ± SEM) + raw points
    # =====================================================
    with st.expander("📊 Visualization (Mean ± SEM)", expanded=False):
        plot_gene = st.selectbox("🧬 Select gene to plot", options=sorted(ddct["Gene"].unique()), key="plot_gene_select")

        # Per-gene sample exclusion (persisted)
        sub = ddct[ddct["Gene"] == plot_gene][["File", "Group", "SampleIdx", "Gene", "FoldChange"]].copy()
        sub = sub.sort_values(["Group", "File", "SampleIdx"]).reset_index(drop=True)
        if plot_gene in st.session_state.exclude_samples_state:
            prev = st.session_state.exclude_samples_state[plot_gene]
            sub = sub.merge(prev[["File", "Group", "SampleIdx", "Exclude"]], on=["File", "Group", "SampleIdx"], how="left")
            sub["Exclude"] = sub["Exclude"].fillna(False)
        else:
            sub["Exclude"] = False

        edited_samples = st.data_editor(
            sub[["File", "Group", "SampleIdx", "FoldChange", "Exclude"]],
            use_container_width=True,
            num_rows="dynamic",
            key=f"sample_exclude_{plot_gene}",
        )
        st.session_state.exclude_samples_state[plot_gene] = edited_samples[["File", "Group", "SampleIdx", "Exclude"]].copy()
        sub_filtered = edited_samples[edited_samples["Exclude"] == False].copy()

        # Summary stats
        summary = sub_filtered.groupby("Group", as_index=False)["FoldChange"].agg(mean="mean", std="std", count="count")
        order = [g for g in group_names if g in summary["Group"].values]
        summary["Group"] = pd.Categorical(summary["Group"], categories=order, ordered=True)
        summary = summary.sort_values("Group")
        summary["SEM"] = summary["std"] / np.sqrt(summary["count"].replace(0, np.nan))

        # Reference group selection for relative plot
        if calib_mode == "By group mean" and str(calib_group) in summary["Group"].astype(str).values:
            default_ref = calib_group
        elif "Control" in summary["Group"].astype(str).values:
            default_ref = "Control"
        else:
            default_ref = str(summary["Group"].astype(str).iloc[0])

        group_labels = list(summary["Group"].astype(str))
        ref_group = st.selectbox("📍 Reference group (set to 1.0)", options=group_labels,
                                 index=group_labels.index(str(default_ref)), key="ref_group_select")
        ref_mean = float(summary.loc[summary["Group"].astype(str) == ref_group, "mean"].iloc[0])
        summary["RelMean"] = summary["mean"] / ref_mean
        summary["RelSEM"] = summary["SEM"] / ref_mean

        # Plot (bars + SEM + raw points)
        def plot_relative_expression_bar(summary_df, raw_df, gene, group_order, ref_group, ref_mean_val):
            fig, ax = plt.subplots(figsize=(7, 5))
            palette = sns.color_palette(["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"])
            order_local = [g for g in group_order if g in summary_df["Group"].astype(str).tolist()]
            summary_df = summary_df.copy()
            summary_df["Group"] = summary_df["Group"].astype(str)
            summary_df = summary_df.set_index("Group").reindex(order_local).reset_index()

            sns.barplot(
                data=summary_df, x="Group", y="RelMean",
                order=order_local, palette=palette, edgecolor="black",
                linewidth=1.0, ci=None, ax=ax,
            )
            x_positions = np.arange(len(order_local))
            y_means = summary_df["RelMean"].to_numpy()
            y_sems = summary_df["RelSEM"].fillna(0.0).to_numpy()
            ax.errorbar(x_positions, y_means, yerr=y_sems, fmt="none",
                        ecolor="#111827", elinewidth=1.2, capsize=4, capthick=1.2)

            raw_df = raw_df.copy()
            raw_df["Group"] = raw_df["Group"].astype(str)
            raw_df["RelFoldChange"] = raw_df["FoldChange"] / ref_mean_val
            sns.stripplot(data=raw_df, x="Group", y="RelFoldChange",
                          order=order_local, color="#111827", size=5, jitter=0.15, alpha=0.7, ax=ax)

            ax.set_xlabel("")
            ax.set_ylabel("Relative Expression (2⁻ΔΔCt)", fontsize=13, labelpad=10)
            ax.set_title(f"{gene} — normalized to {hk} | Calibrator: {ref_group}",
                         fontsize=14, fontweight="bold", pad=15)
            ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.7, color="#e5e7eb")
            ax.set_xticklabels([g.replace(" ", "\n") if len(g) > 12 else g for g in order_local])
            sns.despine(ax=ax, trim=True)
            plt.tight_layout()
            st.pyplot(fig)

        plot_relative_expression_bar(
            summary_df=summary[["Group", "RelMean", "RelSEM"]],
            raw_df=sub_filtered[["Group", "FoldChange"]],
            gene=plot_gene,
            group_order=group_names,
            ref_group=ref_group,
            ref_mean_val=ref_mean,
        )

    # =====================================================
    # 🧫 Prism-ready table (per-sample values)
    # =====================================================
    with st.expander("🧫 Prism-ready table (per-sample values)", expanded=False):
        current_gene = st.session_state.get("plot_gene_select", sorted(ddct["Gene"].unique())[0])
        current_ref_group = st.session_state.get(
            "ref_group_select",
            group_names[0] if group_names else str(ddct["Group"].iloc[0])
        )

        ps = ddct[ddct["Gene"] == current_gene][["File", "Group", "SampleIdx", "FoldChange"]].reset_index(drop=True)
        ps["Group"] = pd.Categorical(ps["Group"], categories=group_names, ordered=True)
        ps = ps.sort_values(["Group", "File", "SampleIdx"]).reset_index(drop=True)
        ps["FileLabel"] = ps["File"].map(file_short)

        # Reference mean for normalization
        gmeans = ps.groupby("Group", as_index=False)["FoldChange"].mean().rename(columns={"FoldChange": "GroupMean"})
        if str(current_ref_group) not in gmeans["Group"].astype(str).values:
            current_ref_group = str(gmeans["Group"].iloc[0])
        ref_mean_val = float(gmeans.loc[gmeans["Group"].astype(str) == str(current_ref_group), "GroupMean"].iloc[0])

        # Relative values
        ps[f"Relative (/{current_ref_group})"] = ps["FoldChange"] / ref_mean_val
        ps["Sample"] = ps.groupby("Group").cumcount() + 1

        tab1, tab2 = st.tabs(["📋 Detailed per-sample table", "📊 Prism matrix view"])
        with tab1:
            ps_display = ps[["FileLabel", "Group", "Sample", "FoldChange", f"Relative (/{current_ref_group})"]]
            st.dataframe(ps_display, use_container_width=True)
        with tab2:
            prism_wide = (
                ps.pivot_table(index="Sample", columns="Group",
                               values=f"Relative (/{current_ref_group})", aggfunc="mean")
                  .reindex(columns=group_names, fill_value=np.nan)
                  .reset_index(drop=False)
            )
            st.markdown(f"**Relative fold change normalized to:** `{current_ref_group}`")
            st.dataframe(prism_wide, use_container_width=True)

        # Prepare CSV bytes
        ps_csv = ps_display.to_csv(index=False).encode("utf-8")
        prism_csv = prism_wide.to_csv(index=False).encode("utf-8")

    # =====================================================
    # 💾 Export Results
    # =====================================================
    with st.expander("💾 Export Results", expanded=False):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button(
                "📊 Download Prism table (CSV)",
                data=prism_csv,
                file_name=f"prism_matrix_{st.session_state.get('plot_gene_select', 'gene')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            out_xlsx = df_to_excel_bytes(
                {
                    "CT_tidy": tidy,
                    "Replicates_QC": qc_df,
                    "Triplicate_stats": trip_stats,
                    "Ct_means": agg,
                    "DDCt": ddct,
                }
            )
            st.download_button(
                "📘 Download full results (Excel)",
                data=out_xlsx,
                file_name="qpcr_lab_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    # ---------- Footer ----------
    st.markdown('<div class="ctstudio-footer">Created by <strong>Kapilraj Periyasamy</strong></div>', unsafe_allow_html=True)

# =====================================================
# ELSE: no upload yet
# =====================================================
else:
    st.info("Upload one or more **QuantStudio Results** .xlsx exports to begin.")
    st.markdown('<div class="ctstudio-footer">Created by <strong>Kapilraj Periyasamy</strong></div>', unsafe_allow_html=True)
