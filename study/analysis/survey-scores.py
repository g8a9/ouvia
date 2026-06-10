import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import json
    from pathlib import Path
    import statsmodels.formula.api as smf
    import numpy as np
    from patsy import build_design_matrices
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    import itertools

    return Path, itertools, json, mo, multipletests, norm, np, pd


@app.cell
def _():
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    sns.set_context(
        "paper",
        rc={
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            "figure.titlesize": 15,
        },
    )
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.figsize": (6.5, 4.0),
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.2,
            "axes.edgecolor": "black",
            "axes.grid": True,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.35,
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )
    return plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Survey scores

    Loads the two production snapshots into a single dataframe, expands the final
    survey payload, and checks that the survey score columns are complete.
    """)
    return


@app.cell
def _(Path):
    data_store = {
        "us_black": {
            "snapshot_file": Path("../../data/snapshots/snapshot_20260224_120529_us_black.json")
        },
        "us_white": {
            "snapshot_file": Path("../../data/snapshots/snapshot_20260308_190010_us_white.json")
        },
        "hindi": {
            "snapshot_file": Path("../../data/snapshots/snapshot_20260519_094000_hindi.json")
        },
        "final_survey_cols": ["satisfaction_score", "trust_score", "reliance_score"],
        "translation_model_mapping": {
            "desta2": "DeSTA2",
            "phi4mm": "Phi 4",
            "voxtralsmall": "Voxtral",
            "tower-plus-9b": "Tower+"
        },
        "language_group_mapping": {
            "us_black": "US Black",
            "us_white": "US White",
            "hindi": "Hindi"
        }
    }
    data_store["baseline_cols"] = [f"baseline_{col}" for col in data_store["final_survey_cols"]]
    return (data_store,)


@app.cell
def _(Path, data_store, json, pd):
    def _load_snapshot(snapshot_file: Path) -> pd.DataFrame:
        with snapshot_file.open() as f:
            snapshot = json.load(f)
        return pd.DataFrame(snapshot)

    def _unpack_final_survey_dict(df: pd.DataFrame) -> pd.DataFrame:
        survey_dicts = df["final_survey"]
        survey_df = pd.json_normalize(survey_dicts)
        return pd.concat([df.drop(columns=["final_survey"]), survey_df], axis=1)

    def _discard_entries(df, entries_to_discard_tsv: Path) -> pd.DataFrame:
        print(f"Discarding entries in {entries_to_discard_tsv} from dataframe with {len(df)} entries...")
        entries_to_discard = pd.read_csv(entries_to_discard_tsv, sep="\t")
        print("Initial shape of dataframe:", df.shape)
        df = df.merge(
            entries_to_discard[["entry_id", "sender_unique_id"]].drop_duplicates(),
            on=["entry_id", "sender_unique_id"],
            how="left",
            indicator=True,
        )
        df = df[df["_merge"] == "left_only"].drop(columns="_merge")
        print("Shape of dataframe after discarding entries:", df.shape)
        return df

    def _add_high_low_stakes(df: pd.DataFrame) -> pd.DataFrame:
        df["stakes"] = df["conversation_topic"].map({"Health": "high", "Everyday": "low"})
        return df

    # Load US White Entries
    us_white_df = _load_snapshot(data_store["us_white"]["snapshot_file"])
    us_white_df = _unpack_final_survey_dict(us_white_df)
    us_white_df = _discard_entries(us_white_df, Path("../quality-checks/manual-checks/white_entries_to_discard.tsv"))
    us_white_df = _add_high_low_stakes(us_white_df)
    us_white_df["group"] = "us_white"

    # Load US Black Entries
    us_black_df = _load_snapshot(data_store["us_black"]["snapshot_file"])
    us_black_df = _unpack_final_survey_dict(us_black_df)
    us_black_df = _discard_entries(us_black_df, Path("../quality-checks/manual-checks/black_entries_to_discard.tsv"))
    us_black_df = _add_high_low_stakes(us_black_df)
    us_black_df["group"] = "us_black"

    hindi_df = _load_snapshot(data_store["hindi"]["snapshot_file"])
    hindi_df = _unpack_final_survey_dict(hindi_df)
    hindi_df = _add_high_low_stakes(hindi_df)
    hindi_df["group"] = "hindi"
    return hindi_df, us_black_df, us_white_df


@app.cell
def _(hindi_df):
    hindi_df.head()
    return


@app.cell
def _(data_store, hindi_df, pd, us_black_df, us_white_df):
    full_data = pd.concat([us_white_df, us_black_df, hindi_df], ignore_index=True)

    def _assert_all_not_null(df: pd.DataFrame, cols: list[str]):
        for col in cols:
            assert df[col].notnull().all(), f"Column {col} contains null values"

    _assert_all_not_null(full_data, data_store["final_survey_cols"])

    # Keep for now only records where sender_gender is woman or man
    full_data = full_data[full_data["sender_gender"].isin(["woman", "man"])]

    # Define two new columns: the average of the final survey scores, and the average of the baseline survey scores
    full_data["survey_score"] = full_data[data_store["final_survey_cols"]].mean(axis=1)
    full_data["baseline_survey_score"] = full_data[[f"baseline_{col}" for col in data_store["final_survey_cols"]]].mean(axis=1)

    # Translation model renaming
    full_data["translation_model"] = full_data["translation_model"].map(data_store["translation_model_mapping"])
    full_data["group"] = full_data["group"].map(data_store["language_group_mapping"])

    # Normalize MetricX scores to [0, 1] range
    metricx_col = "google--metricx-24-hybrid-xl-v2p6"
    full_data[metricx_col] = 1 - full_data[metricx_col] / 25

    # Add the ratio between number of validator corrections and total questions
    full_data["correction_ratio"] = full_data["validator_corrections"] / full_data["validator_question_total"]

    # Score shift
    full_data["score_shift"] = full_data["survey_score"] - full_data["baseline_survey_score"]
    # QA Score (higher is better correction quality)
    full_data["qa_score"] = 1 - full_data["correction_ratio"]
    return (full_data,)


@app.cell
def _(full_data):
    full_data["conversation_source"].value_counts()
    return


@app.cell
def _(data_store, full_data):
    summary = {
        "rows": len(full_data),
        "groups": full_data["group"].value_counts().to_dict(),
        "final_survey_cols": data_store["final_survey_cols"],
    }
    summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### All columns
    """)
    return


@app.cell
def _(full_data):
    full_data.columns.tolist()
    return


@app.cell
def _(full_data):
    full_data.head(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset Statistics
    """)
    return


@app.cell
def _(full_data, mo):
    _outputs = []
    _n_rows = len(full_data)
    _outputs.append(mo.md(f"**Total rows:** {_n_rows}"))

    # Rows by group
    _by_group = full_data["group"].value_counts().to_frame("n")
    _by_group.index.name = "group"
    _outputs.append(mo.md("**Rows by group:**"))
    _outputs.append(_by_group)

    # Rows by gender × group
    _by_gxg = (
        full_data.groupby(["sender_gender", "group"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    _outputs.append(mo.md("**Rows by gender × group:**"))
    _outputs.append(mo.ui.table(_by_gxg))

    # Unique IDs
    _id_cols = ["sender_unique_id", "receiver_unique_id", "validator_unique_id", "conversation_id"]
    if "final_sender_unique_id" in full_data.columns:
        _id_cols.append("final_sender_unique_id")

    for _c in _id_cols:
        _n = full_data[_c].nunique()
        _outputs.append(mo.md(f"Unique **{_c}**: {_n}"))

    # final_sender_unique_id not in sender_unique_id
    if "final_sender_unique_id" in full_data.columns:
        _final_set = set(full_data["final_sender_unique_id"].dropna())
        _sender_set = set(full_data["sender_unique_id"].dropna())
        _only_final = _final_set - _sender_set
        _outputs.append(mo.md(f"final_sender_unique_id **not** in sender_unique_id: {len(_only_final)}"))

    # Sender record counts
    _sender_counts = full_data["sender_unique_id"].value_counts()
    _count_dist = _sender_counts.value_counts().sort_index(ascending=False)
    _count_dist = _count_dist.rename_axis("records_per_sender").rename("n_senders")
    _outputs.append(mo.md("**Senders by number of records:**"))
    _outputs.append(_count_dist)

    mo.vstack(_outputs)
    return


@app.cell
def _(full_data, mo):
    _q_outputs = []
    _q_col = "questions" if "questions" in full_data.columns else None

    if _q_col is not None:
        _q_lens = full_data[_q_col].dropna().apply(len)
        _q_outputs.append(mo.md(f"**Avg questions per entry:** {_q_lens.mean():.1f}"))
        _q_outputs.append(mo.md(f"**Total questions:** {_q_lens.sum()}"))

        _q_len_dist = _q_lens.value_counts().sort_index()
        _q_len_dist = _q_len_dist.rename_axis("n_questions").rename("n_entries")
        _q_outputs.append(mo.md("**Entries by number of questions:**"))
        _q_outputs.append(_q_len_dist)
    else:
        _q_outputs.append(mo.md("Column 'questions' not found in dataset."))

    mo.vstack(_q_outputs)
    return


@app.cell
def _(mo, full_data):
    _outputs_ge4 = []
    _ge4 = full_data[full_data["survey_score"] >= 4]
    _n_ge4 = len(_ge4)
    _pct_ge4 = _n_ge4 / len(full_data) * 100
    _outputs_ge4.append(mo.md(f"**Rows with $u \\geq 4$:** {_n_ge4} ({_pct_ge4:.1f}%)"))

    _by_group_ge4 = _ge4["group"].value_counts()
    _by_group_all = full_data["group"].value_counts()
    for _g in ["Hindi", "US Black", "US White"]:
        _n = _by_group_ge4.get(_g, 0)
        _tot = _by_group_all.get(_g, 1)
        _outputs_ge4.append(mo.md(f"{_g}: {_n} / {_tot} ({_n / _tot * 100:.1f}%)"))

    mo.vstack(_outputs_ge4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Validator Scores
    """)
    return


@app.cell
def _(np):
    def plot_dot_ci_h(ax, df, group_col, value_col="validator_translation_score",
                      order=None, label_map=None):
        """Horizontal dot + 95% CI for each level of *group_col*."""
        _summ = (
            df.groupby(group_col, observed=True)[value_col]
            .agg(mean="mean", std="std", n="count")
        )
        _summ["se"] = _summ["std"] / np.sqrt(_summ["n"])
        _summ["ci95"] = 1.96 * _summ["se"]

        if order is not None:
            _summ = _summ.reindex(order)
        if label_map is not None:
            _summ.index = [_summ.index.map(label_map)]

        _y = list(range(len(_summ)))
        for _i, (_idx, _row) in enumerate(_summ.iterrows()):
            ax.errorbar(
                _row["mean"], _i,
                xerr=_row["ci95"],
                fmt="o", color="black", capsize=4, markersize=7,
                markerfacecolor="black",
            )
            ax.text(
                _row["mean"] + _row["ci95"] + 1.2,
                _i,
                f"{_row['mean']:.1f}",
                va="center", ha="left",
            )
        ax.set_yticks(_y)
        ax.set_yticklabels(_summ.index.tolist())
        ax.set_xlim(0, 100)

    # Display labels and orders
    MODEL_ORDER =    ["DeSTA2", "Phi 4", "Voxtral", "Tower+"]
    TOPIC_ORDER =    ["Everyday", "Health"]
    GROUP_ORDER =    ["US White", "US Black", "Hindi"]
    GENDER_ORDER =   ["man", "woman"]

    GROUP_LABELS = {
        "Hindi":    "Hindi",
        "US Black": r"US$_\mathregular{B}$",
        "US White": r"US$_\mathregular{W}$",
    }
    GENDER_LABELS = {"man": "Man", "woman": "Woman"}
    TOPIC_LABELS =  {"Everyday": "Everyday", "Health": "Health"}
    return (
        GENDER_LABELS,
        GENDER_ORDER,
        GROUP_LABELS,
        GROUP_ORDER,
        MODEL_ORDER,
        TOPIC_ORDER,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Combined Topic x Gender Chart
    """)
    return


@app.cell
def _(MODEL_ORDER, TOPIC_ORDER, full_data, np, plt):
    # Figure (a): validator score by model × topic
    _vscore_col = "validator_translation_score"

    _vt_summary_mt = (
        full_data.groupby(["translation_model", "conversation_topic"], observed=True)[_vscore_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    _vt_summary_mt["se"] = _vt_summary_mt["std"] / np.sqrt(_vt_summary_mt["n"])
    _vt_summary_mt["ci95"] = 1.96 * _vt_summary_mt["se"]

    _vt_palette_mt = {"first": "#222222", "second": "#888888"}
    _vt_markers_mt  = {"Everyday": "o", "Health": "D"}
    _vt_offsets_mt  = {"Everyday": -0.13, "Health": 0.13}

    _fig_vt1, _ax_vt1 = plt.subplots(figsize=(4, 4))

    _x_m = list(range(len(MODEL_ORDER)))
    for _i, _vt_topic in enumerate(TOPIC_ORDER):
        _curr = (
            _vt_summary_mt[_vt_summary_mt["conversation_topic"] == _vt_topic]
            .set_index("translation_model")
            .reindex(MODEL_ORDER)
            .reset_index()
        )
        _x = [_v + _vt_offsets_mt[_vt_topic] for _v in _x_m]
        _clr = _vt_palette_mt["first"] if _i == 0 else _vt_palette_mt["second"]
        _ax_vt1.errorbar(
            _x, _curr["mean"], yerr=_curr["ci95"],
            linestyle="None",
            marker=_vt_markers_mt[_vt_topic], markersize=9, capsize=5,
            color=_clr, label=_vt_topic,
        )
        _y_sign = -1 if _i == 0 else 1
        _va = "top" if _i == 0 else "bottom"
        for _xi, (_mv, _ci) in enumerate(zip(_curr["mean"], _curr["ci95"])):
            _ax_vt1.text(
                _x[_xi], _mv + _y_sign * (_ci + 2),
                f"{_mv:.1f}",
                ha="center", va=_va, fontsize=13, color=_clr, zorder=10,
            )
    _ax_vt1.set_xticks(_x_m)
    _ax_vt1.set_xticklabels(MODEL_ORDER)
    _ax_vt1.set_ylabel("")
    _ax_vt1.set_xlabel("")
    _ax_vt1.set_ylim(45, 100)
    _ax_vt1.set_xmargin(0.1)
    _ax_vt1.text(0.95, 0.25, "Validator\nTranslation\nScore", transform=_ax_vt1.transAxes,
                 va="top", ha="right", fontsize=14)
    _ax_vt1.legend(title=None, loc="upper left", frameon=False, handletextpad=0.02)

    _fig_vt1.tight_layout()
    _fig_vt1.savefig("fig_validator_score_model_topic.pdf", dpi=300, bbox_inches="tight")
    _fig_vt1
    return


@app.cell
def _(
    GENDER_LABELS,
    GENDER_ORDER,
    GROUP_LABELS,
    GROUP_ORDER,
    full_data,
    np,
    plt,
):
    # Figure (b): validator score by group × gender — no y-axis, mirrors left
    _vscore_col = "validator_translation_score"

    _vt_summary_gg = (
        full_data.groupby(["group", "sender_gender"], observed=True)[_vscore_col]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    _vt_summary_gg["se"] = _vt_summary_gg["std"] / np.sqrt(_vt_summary_gg["n"])
    _vt_summary_gg["ci95"] = 1.96 * _vt_summary_gg["se"]

    _vt_palette_gg = {"first": "#355070", "second": "#E56B6F"}
    _vt_markers_gg  = {"man": "s", "woman": "^"}
    _vt_offsets_gg  = {"man": -0.09, "woman": 0.09}

    _fig_vt2, _ax_vt2 = plt.subplots(figsize=(2.8, 3.2))

    _x_g = list(range(len(GROUP_ORDER)))
    for _i, _vt_gender in enumerate(GENDER_ORDER):
        _curr = (
            _vt_summary_gg[_vt_summary_gg["sender_gender"] == _vt_gender]
            .set_index("group")
            .reindex(GROUP_ORDER)
            .reset_index()
        )
        _x = [_v + _vt_offsets_gg[_vt_gender] for _v in _x_g]
        _clr = _vt_palette_gg["first"] if _i == 0 else _vt_palette_gg["second"]
        _ax_vt2.errorbar(
            _x, _curr["mean"], yerr=_curr["ci95"],
            linestyle="None",
            marker=_vt_markers_gg[_vt_gender], markersize=7, capsize=5,
            color=_clr, label=GENDER_LABELS[_vt_gender],
        )
    _ax_vt2.set_xticks(_x_g)
    _ax_vt2.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER])
    _ax_vt2.set_ylim(45, 100)
    _ax_vt2.set_xlabel("")
    _ax_vt2.set_ylabel("")
    _ax_vt2.tick_params(axis="y", left=False, labelleft=False)
    _ax_vt2.spines["left"].set_visible(False)
    _ax_vt2.legend(title=None, loc="upper right", frameon=False, handletextpad=0.02)

    _fig_vt2.tight_layout()
    _fig_vt2.savefig("fig_validator_score_group_gender.pdf", dpi=300, bbox_inches="tight")
    _fig_vt2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Mean validator translation score ($\pm$95% CI) across four translation models and two conversation topics (left), and across three language groups and gender (right). Circles and diamonds distinguish topic/gender levels; darker markers indicate the Everyday topic (left) and men (right).
    """)
    return


@app.cell
def _(full_data, pd):
    # Build a descriptive-statistics table for key numerical columns
    _num_cols = [
        "survey_score",
        "satisfaction_score",
        "trust_score",
        "reliance_score",
        "baseline_survey_score",
        "baseline_satisfaction_score",
        "baseline_trust_score",
        "baseline_reliance_score",
        "score_shift",
        "validator_translation_score",
        "validator_corrections",
        "validator_question_total",
        "correction_ratio",
        "Unbabel--XCOMET-XL",
        "Unbabel--wmt22-comet-da",
        "Unbabel--wmt22-cometkiwi-da",
        "Unbabel--wmt23-cometkiwi-da-xl",
        "google--metricx-24-hybrid-xl-v2p6",
    ]
    _col_labels = {
        "survey_score": "Usability ($u$)",
        "satisfaction_score": "Satisfaction",
        "trust_score": "Trust",
        "reliance_score": "Reliance",
        "baseline_survey_score": "$u$ (Baseline)",
        "baseline_satisfaction_score": "Satisfaction (Baseline)",
        "baseline_trust_score": "Trust (Baseline)",
        "baseline_reliance_score": "Reliance (Baseline)",
        "score_shift": "$\\Delta_u$",
        "validator_translation_score": "Validator Score",
        "validator_corrections": "Validator Corrections",
        "validator_question_total": "Validator Questions Total",
        "qa_score": "QA Score",
        "Unbabel--XCOMET-XL": "XCOMET-XL",
        "Unbabel--wmt22-comet-da": "COMET-22 DA",
        "Unbabel--wmt22-cometkiwi-da": "COMET-Kiwi-22 DA",
        "Unbabel--wmt23-cometkiwi-da-xl": "COMET-Kiwi-23 XL",
        "google--metricx-24-hybrid-xl-v2p6": "MetricX-24",
    }

    def _desc_block(df, group_label):
        rows = []
        for _col in _num_cols:
            _s = df[_col].dropna()
            rows.append({
                "Measure": _col_labels.get(_col, _col),
                "Group": group_label,
                "N": len(_s),
                "Mean": _s.mean(),
                "SD": _s.std(),
                "Min": _s.min(),
                "Max": _s.max(),
            })
        return pd.DataFrame(rows)

    _blocks = [_desc_block(full_data, "Overall")]

    for _grp, _label in [("man", "Man"), ("woman", "Woman")]:
        _blocks.append(_desc_block(full_data[full_data["sender_gender"] == _grp], _label))

    for _grp in ["Hindi", "US Black", "US White"]:
        _blocks.append(_desc_block(full_data[full_data["group"] == _grp], _grp))

    for _g, _gen in [("Hindi", "man"), ("Hindi", "woman"),
                       ("US Black", "man"), ("US Black", "woman"),
                       ("US White", "man"), ("US White", "woman")]:
        _sub = full_data[(full_data["group"] == _g) & (full_data["sender_gender"] == _gen)]
        _blocks.append(_desc_block(_sub, f"{_g} × {_gen}"))

    for _top in ["Everyday", "Health"]:
        _blocks.append(_desc_block(
            full_data[full_data["conversation_topic"] == _top], _top,
        ))

    for _model in ["DeSTA2", "Phi 4", "Voxtral", "Tower+"]:
        _blocks.append(_desc_block(
            full_data[full_data["translation_model"] == _model], _model,
        ))

    desc_table = pd.concat(_blocks, ignore_index=True)
    desc_table = desc_table.round(3)
    desc_table.to_csv("descriptive_stats.csv", index=False)
    desc_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LMER and Forest Plots

    We refit the same LMER structure with validator assessments as dependent
    variables to test whether language groups differ in how validators evaluate
    their translations.
    """)
    return


@app.cell
def _(analysis_df, lmer, pl):
    def _r_safe(s: str) -> str:
        return s.replace(" ", "_").replace("+", "plus")

    _vlmer_df = analysis_df.assign(
        group=analysis_df["group"].astype(str).map(_r_safe),
        gender=analysis_df["gender"].astype(str),
        translation_model=analysis_df["translation_model"].astype(str).map(_r_safe),
        conversation_topic=analysis_df["conversation_topic"].astype(str),
    )
    _vlmer_polars = pl.from_pandas(_vlmer_df)

    _vts_formula = (
        "validator_translation_score ~ group * gender + conversation_topic"
        " + translation_model"
        " + conversation_source"
        " + (1 | validator_unique_id)"
        " + (1 | receiver_unique_id)"
        " + (1 | participant_id) + (1 | conv_id)"
    )

    vts_lmer = lmer(_vts_formula, data=_vlmer_polars)
    vts_lmer.fit(REML=True, summarize=True)
    vts_lmer.summary()

    _cr_formula = (
        "correction_ratio ~ group * gender + conversation_topic"
        " + translation_model"
        " + conversation_source"
        " + (1 | validator_unique_id)"
        " + (1 | receiver_unique_id)"
        " + (1 | participant_id) + (1 | conv_id)"
    )

    cr_lmer = lmer(_cr_formula, data=_vlmer_polars)
    cr_lmer.fit(REML=True, summarize=True)
    cr_lmer.summary()
    return cr_lmer, vts_lmer


@app.cell
def _(pd, plt, vts_lmer):
    _rf_vts = vts_lmer.result_fit.to_pandas().set_index("term")

    _coef_vts = pd.DataFrame({
        "coef":   _rf_vts["estimate"],
        "ci_low": _rf_vts["conf_low"],
        "ci_high": _rf_vts["conf_high"],
        "pval":   _rf_vts["p_value"],
    })

    _coef_vts = _coef_vts.drop(index="(Intercept)", errors="ignore")
    _coef_vts = _coef_vts[~_coef_vts.index.str.contains("conversation_source", case=False)]
    _coef_vts = _coef_vts.sort_values("coef", ascending=True)

    _vts_label_map = {
        "groupUS_Black":                       r"$\mathregular{US_B}$",
        "groupUS_White":                       r"$\mathregular{US_W}$",
        "genderwoman":                         "Woman",
        "conversation_topicHealth":            "Health",
        "conversation_sourceGemini":           "Source: Gemini",
        "conversation_sourcePatientPhysician": "Source: MED-MT",
        "conversation_sourceQwen3":            "Source: Qwen 3",
        "translation_modelPhi_4":              "Phi 4",
        "translation_modelTowerplus":          "Tower+",
        "translation_modelVoxtral":            "Voxtral",
        "baseline_survey_score":               r"$u$ (Baseline)",
        "groupUS_Black:genderwoman":           r"$\mathregular{US_B} \times$ (W)",
        "groupUS_White:genderwoman":           r"$\mathregular{US_W} \times$ (W)",
    }
    _vts_labels = [_vts_label_map.get(idx, idx) for idx in _coef_vts.index]

    fig_vts, ax_vts = plt.subplots(figsize=(4, 4))
    _y_vts = list(range(len(_coef_vts)))

    for _yi, (_, _row) in zip(_y_vts, _coef_vts.iterrows()):
        _p = _row["pval"]
        if 0.05 <= _p < 0.1:
            _fmt, _mfc = "D", "white"
        else:
            _fmt, _mfc = "o", "black" if _p < 0.05 else "white"
        ax_vts.errorbar(
            _row["coef"], _yi,
            xerr=[[_row["coef"] - _row["ci_low"]], [_row["ci_high"] - _row["coef"]]],
            fmt=_fmt,
            capsize=4,
            color="black",
            ecolor="black",
            markerfacecolor=_mfc,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )
        ax_vts.text(
            _row["ci_high"] + 1,
            _yi,
            f"{_row['coef']:.2f}",
            va="center",
            ha="left",
        )

    ax_vts.axvline(0, linestyle="--", color="gray", alpha=0.3)
    ax_vts.set_yticks(_y_vts)
    ax_vts.set_yticklabels(_vts_labels)
    ax_vts.set_xlabel("")
    ax_vts.set_xticks([-10, 0, 10, 20])
    _xl, _xr = ax_vts.get_xlim()
    ax_vts.set_xlim(_xl * 1.2, _xr * 1.4)
    ax_vts.margins(y=0.1)

    fig_vts.tight_layout()
    fig_vts.savefig("fig_coef_validator_translation_score.pdf", dpi=300, bbox_inches="tight")
    fig_vts
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Validator translation scores. (a) Descriptive means ($\pm$95\% CI) by translation model and conversation topic. (b) Descriptive means by language group and gender. (c) Fixed-effect coefficients from a linear mixed-effects model predicting validator translation score, with random intercepts for participants, conversations, validators, and receivers. Reference categories: Hindi men, DeSTA2 model, Everyday conversations, PatientPhysician source. Filled circles: $p < 0.05$; diamonds: $0.05 \leq p < 0.1$; hollow circles: non-significant.
    """)
    return


@app.cell
def _(analysis_df, lmer_df, np, pd, plt, vts_lmer):
    # Model-adjusted interaction plot for VTS (group × gender)
    import rpy2.robjects as _ro_vts

    _rf_vts_emm = vts_lmer.result_fit.to_pandas().set_index("term")
    _fe_names_vts = list(_rf_vts_emm.index)
    _fe_idx_vts = {t: i for i, t in enumerate(_fe_names_vts)}
    _beta_vts = _rf_vts_emm["estimate"].to_numpy()

    _vcov_r_vts = _ro_vts.r["as.matrix"](_ro_vts.r["vcov"](vts_lmer.r_model))
    _cov_fe_vts = np.array(_vcov_r_vts).reshape(len(_beta_vts), len(_beta_vts))

    # Marginal weights
    _topic_weights_vts = (
        lmer_df["conversation_topic"].value_counts(normalize=True).sort_index()
    )
    _model_weights_vts = (
        lmer_df["translation_model"].value_counts(normalize=True).sort_index()
    )
    _source_weights_vts = (
        lmer_df["conversation_source"].value_counts(normalize=True).sort_index()
    )
    _topic_levels_vts = list(_topic_weights_vts.index)
    _model_levels_vts = list(_model_weights_vts.index)
    _source_levels_vts = list(_source_weights_vts.index)

    def _r_safe_vts(s):
        return s.replace(" ", "_").replace("+", "plus")

    _vts_group_levels = list(analysis_df["group"].cat.categories)
    _vts_gender_levels = list(analysis_df["gender"].cat.categories)

    def _vts_design_vec(group_r, gender_r, topic, model_r, source):
        v = np.zeros(len(_fe_idx_vts))
        if "(Intercept)" in _fe_idx_vts:
            v[_fe_idx_vts["(Intercept)"]] = 1.0
        if group_r == "US_Black" and "groupUS_Black" in _fe_idx_vts:
            v[_fe_idx_vts["groupUS_Black"]] = 1.0
        if group_r == "US_White" and "groupUS_White" in _fe_idx_vts:
            v[_fe_idx_vts["groupUS_White"]] = 1.0
        if gender_r == "woman" and "genderwoman" in _fe_idx_vts:
            v[_fe_idx_vts["genderwoman"]] = 1.0
        if topic == "Health" and "conversation_topicHealth" in _fe_idx_vts:
            v[_fe_idx_vts["conversation_topicHealth"]] = 1.0
        _tm_key = f"translation_model{model_r}"
        if _tm_key in _fe_idx_vts:
            v[_fe_idx_vts[_tm_key]] = 1.0
        if source == "Gemini" and "conversation_sourceGemini" in _fe_idx_vts:
            v[_fe_idx_vts["conversation_sourceGemini"]] = 1.0
        if source == "Qwen3" and "conversation_sourceQwen3" in _fe_idx_vts:
            v[_fe_idx_vts["conversation_sourceQwen3"]] = 1.0
        if source == "PatientPhysician" and "conversation_sourcePatientPhysician" in _fe_idx_vts:
            v[_fe_idx_vts["conversation_sourcePatientPhysician"]] = 1.0
        if group_r == "US_Black" and gender_r == "woman" and "groupUS_Black:genderwoman" in _fe_idx_vts:
            v[_fe_idx_vts["groupUS_Black:genderwoman"]] = 1.0
        if group_r == "US_White" and gender_r == "woman" and "groupUS_White:genderwoman" in _fe_idx_vts:
            v[_fe_idx_vts["groupUS_White:genderwoman"]] = 1.0
        return v

    _vts_rows = []
    for _g in _vts_group_levels:
        for _gen in _vts_gender_levels:
            _gr = _r_safe_vts(_g)
            _cv = np.zeros(len(_beta_vts))
            for _top in _topic_levels_vts:
                for _mod in _model_levels_vts:
                    for _src in _source_levels_vts:
                        _dv = _vts_design_vec(_gr, _gen, _top, _mod, _src)
                        _w = (float(_topic_weights_vts.get(_top, 0))
                              * float(_model_weights_vts.get(_mod, 0))
                              * float(_source_weights_vts.get(_src, 0)))
                        _cv += _dv * _w
            _est = float(_cv @ _beta_vts)
            _se = float(np.sqrt(_cv @ _cov_fe_vts @ _cv))
            _vts_rows.append({
                "group": _g, "gender": _gen,
                "estimated_mean": _est,
                "std_error": _se,
                "ci_low": _est - 1.96 * _se,
                "ci_high": _est + 1.96 * _se,
            })

    _vts_plot_table = pd.DataFrame(_vts_rows)
    _vts_plot_table["group"] = pd.Categorical(
        _vts_plot_table["group"],
        categories=["US White", "US Black", "Hindi"],
        ordered=True,
    )
    _vts_plot_table["gender"] = pd.Categorical(
        _vts_plot_table["gender"],
        categories=_vts_gender_levels,
        ordered=True,
    )

    _vts_gender_style = {
        "man":   {"marker": "s", "offset": -0.08, "ha": "right"},
        "woman": {"marker": "^", "offset":  0.08, "ha": "left"},
    }
    _vts_group_order = ["US White", "US Black", "Hindi"]
    _vts_group_label_map = {
        "Hindi":    "Hindi",
        "US Black": r"$\mathrm{US}_\mathrm{B}$",
        "US White": r"$\mathrm{US}_\mathrm{W}$",
    }
    _vts_tick_labels = [_vts_group_label_map[g] for g in _vts_group_order]
    _vts_x_base = np.arange(len(_vts_group_order))

    _vts_palette = {
        str(_vts_gender_levels[0]): "#355070",
        str(_vts_gender_levels[1]): "#E56B6F",
    }

    _fig_vts_int, _ax_vts_int = plt.subplots(figsize=(4, 4))

    for _gen in _vts_gender_levels:
        _curr = (
            _vts_plot_table[_vts_plot_table["gender"] == _gen]
            .sort_values("group")
            .reset_index(drop=True)
        )
        _style = _vts_gender_style[str(_gen)]
        _x_pos = _vts_x_base + _style["offset"]
        _ax_vts_int.errorbar(
            _x_pos,
            _curr["estimated_mean"],
            yerr=1.96 * _curr["std_error"],
            marker=_style["marker"],
            markersize=9,
            linewidth=2,
            linestyle=":",
            capsize=4,
            label=str(_gen),
            color=_vts_palette[str(_gen)],
        )
        _x_sign = -1 if _gen == "man" else 1
        _y_sign = -1 if _gen == "man" else 1
        _va = "top" if _gen == "man" else "bottom"
        for _xi, (_mv, _ci) in enumerate(zip(_curr["estimated_mean"], 1.96 * _curr["std_error"])):
            _ax_vts_int.text(
                _x_pos[_xi] + _x_sign * 0.1, _mv + _y_sign * 0.05,
                f"{_mv:.1f}",
                ha=_style["ha"], va=_va, fontsize=13, color=_vts_palette[str(_gen)],
            )

    _ax_vts_int.set_xticks(_vts_x_base)
    _ax_vts_int.set_xticklabels(_vts_tick_labels)
    _ax_vts_int.set_xlabel("")
    _ax_vts_int.set_ylabel("")
    _ax_vts_int.text(0.95, 0.95, "Validator\nTranslation\nScore", transform=_ax_vts_int.transAxes,
                     va="top", ha="right", fontsize=14)
    _ax_vts_int.legend(title=None, loc="best")
    _ax_vts_int.margins(x=0.3)

    _fig_vts_int.tight_layout()
    _fig_vts_int.savefig("fig_interaction_vts_group_gender.pdf", dpi=300, bbox_inches="tight")
    _fig_vts_int
    return


@app.cell
def _(cr_lmer, pd, plt):
    _rf_cr = cr_lmer.result_fit.to_pandas().set_index("term")

    _coef_cr = pd.DataFrame({
        "coef":   _rf_cr["estimate"],
        "ci_low": _rf_cr["conf_low"],
        "ci_high": _rf_cr["conf_high"],
        "pval":   _rf_cr["p_value"],
    })

    _coef_cr = _coef_cr.drop(index="(Intercept)", errors="ignore")
    _coef_cr = _coef_cr.sort_values("coef", ascending=True)

    _cr_label_map = {
        "groupUS_Black":                       r"$\mathregular{US_B}$",
        "groupUS_White":                       r"$\mathregular{US_W}$",
        "genderwoman":                         "Woman",
        "conversation_topicHealth":            "Topic: Health",
        "conversation_sourceGemini":           "Source: Gemini",
        "conversation_sourcePatientPhysician": "Source: MED-MT",
        "conversation_sourceQwen3":            "Source: Qwen 3",
        "translation_modelPhi_4":              "Model: Phi 4",
        "translation_modelTowerplus":          "Model: Tower+",
        "translation_modelVoxtral":            "Model: Voxtral",
        "baseline_survey_score":               r"$u$ (Baseline)",
        "groupUS_Black:genderwoman":           r"$\mathregular{US_B} \times$ Woman",
        "groupUS_White:genderwoman":           r"$\mathregular{US_W} \times$ Woman",
    }
    _cr_labels = [_cr_label_map.get(idx, idx) for idx in _coef_cr.index]

    fig_cr, ax_cr = plt.subplots(figsize=(5.5, 0.42 * len(_coef_cr)))
    _y_cr = list(range(len(_coef_cr)))

    for _yi, (_, _row) in zip(_y_cr, _coef_cr.iterrows()):
        _p = _row["pval"]
        if 0.05 <= _p < 0.1:
            _fmt, _mfc = "D", "white"
        else:
            _fmt, _mfc = "o", "black" if _p < 0.05 else "white"
        ax_cr.errorbar(
            _row["coef"], _yi,
            xerr=[[_row["coef"] - _row["ci_low"]], [_row["ci_high"] - _row["coef"]]],
            fmt=_fmt,
            capsize=4,
            color="black",
            ecolor="black",
            markerfacecolor=_mfc,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )
        ax_cr.text(
            _row["ci_high"] + 0.02,
            _yi,
            f"{_row['coef']:.3f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax_cr.axvline(0, linestyle="--", color="gray", alpha=0.3)
    ax_cr.set_yticks(_y_cr)
    ax_cr.set_yticklabels(_cr_labels)
    ax_cr.set_xlabel("Effect on Correction Ratio")
    ax_cr.margins(x=0.25)

    fig_cr.tight_layout()
    fig_cr.savefig("fig_coef_correction_ratio.pdf", dpi=300, bbox_inches="tight")
    fig_cr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Fixed-effect coefficients from the linear mixed-effects model predicting correction ratio (validator corrections / total questions). Reference categories: Hindi men, DeSTA2 model, Everyday conversations, PatientPhysician source. Filled circles: $p < 0.05$; diamonds: $0.05 \leq p < 0.1$; hollow circles: non-significant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Survival Plots Exploration
    """)
    return


@app.cell
def _(full_data, np):
    # Build CDF dataframe with all needed columns
    _cdf_cols = [
        "sender_gender", "group",
        "validator_translation_score", "correction_ratio",
        "Unbabel--XCOMET-XL", "google--metricx-24-hybrid-xl-v2p6",
        "survey_score", "baseline_survey_score", "reliance_score",
        "qa_score",
    ]
    cdf_df = full_data[_cdf_cols].copy()
    cdf_df = cdf_df.rename(columns={"sender_gender": "gender"})

    # Min-max rescale all metrics except qa_score (already [0,1])
    _rescale_cols = [
        "validator_translation_score", "Unbabel--XCOMET-XL",
        "google--metricx-24-hybrid-xl-v2p6",
        "survey_score", "baseline_survey_score", "reliance_score",
    ]
    for _col in _rescale_cols:
        _min, _max = cdf_df[_col].min(), cdf_df[_col].max()
        cdf_df[_col] = (cdf_df[_col] - _min) / (_max - _min)

    def ecdf(series):
        x = np.sort(series.dropna().to_numpy())
        y = 1 - np.arange(1, len(x) + 1) / len(x)
        return x, y

    def plot_cdf_overlay(ax, df, value_col, group_col, palette, *, alpha=0.18, linewidth=1.5):
        """Plot ECDFs for each group with filled area + line."""
        for grp in sorted(df[group_col].dropna().unique()):
            series = df[df[group_col] == grp][value_col]
            x, y = ecdf(series)
            c = palette.get(grp, "black")
            ax.fill_between(x, 0, y, alpha=alpha, color=c)
            ax.plot(x, y, color=c, linewidth=linewidth, label=str(grp))

    METRIC_SPECS = {
        "validator_translation_score": "Validator Translation Score",
        "qa_score": "QA Score",
        "Unbabel--XCOMET-XL": "XCOMET-XL",
        "google--metricx-24-hybrid-xl-v2p6": "MetricX-24",
        "survey_score": "Usability ($u$)",
        "baseline_survey_score": "$u$ (Baseline)",
        "reliance_score": "Reliance",
    }
    CDF_METRICS = list(METRIC_SPECS.keys())
    METRIC_COLORS = {
        "validator_translation_score": "#355070",
        "qa_score": "#606C37",
        "Unbabel--XCOMET-XL": "#DDA15E",
        "google--metricx-24-hybrid-xl-v2p6": "#E56B6F",
        "survey_score": "#5F0F40",
        "baseline_survey_score": "#7D6B91",
        "reliance_score": "#333333",
    }
    GENDER_PALETTE = {"man": "#355070", "woman": "#E56B6F"}
    GG_PALETTE = {
        ("Hindi", "man"):       "#355070",
        ("Hindi", "woman"):     "#E56B6F",
        ("US Black", "man"):    "#606C37",
        ("US Black", "woman"):  "#5F0F40",
        ("US White", "man"):    "#DDA15E",
        ("US White", "woman"):  "#7D6B91",
    }
    GG_LABEL_MAP = {
        ("Hindi", "man"):       "Hindi (M)",
        ("Hindi", "woman"):     "Hindi (W)",
        ("US Black", "man"):    r"$\mathregular{US_B}$ (M)",
        ("US Black", "woman"):  r"$\mathregular{US_B}$ (W)",
        ("US White", "man"):    r"$\mathregular{US_W}$ (M)",
        ("US White", "woman"):  r"$\mathregular{US_W}$ (W)",
    }
    GG_ORDER = [
        ("Hindi", "man"), ("Hindi", "woman"),
        ("US Black", "man"), ("US Black", "woman"),
        ("US White", "man"), ("US White", "woman"),
    ]
    return (
        CDF_METRICS,
        GENDER_PALETTE,
        GG_LABEL_MAP,
        GG_ORDER,
        GG_PALETTE,
        METRIC_COLORS,
        METRIC_SPECS,
        cdf_df,
        ecdf,
        plot_cdf_overlay,
    )


@app.cell
def _(CDF_METRICS, METRIC_COLORS, METRIC_SPECS, cdf_df, ecdf, plt):
    # Overlay CDFs of all four metrics on a single plot (filled areas)
    _cdf_fig_all, _cdf_ax_all = plt.subplots(figsize=(6, 4.2))

    for _cdf_col in CDF_METRICS:
        _cdf_x, _cdf_y = ecdf(cdf_df[_cdf_col])
        _cdf_clr = METRIC_COLORS[_cdf_col]
        _cdf_ax_all.fill_between(_cdf_x, 0, _cdf_y, alpha=0.15, color=_cdf_clr)
        _cdf_ax_all.plot(_cdf_x, _cdf_y, color=_cdf_clr, linewidth=1.8, label=METRIC_SPECS[_cdf_col])

    _cdf_ax_all.set_xlabel("Score")
    _cdf_ax_all.set_ylabel("Share ≥ score")
    _cdf_ax_all.legend(title=None, loc="lower left", framealpha=0.6, facecolor="white", edgecolor="none")
    _cdf_ax_all.set_xlim(0, 1)
    _cdf_ax_all.set_ylim(0, 1)

    _cdf_fig_all.tight_layout()
    _cdf_fig_all.savefig("fig_cdf_all_metrics.pdf", dpi=300, bbox_inches="tight")
    _cdf_fig_all
    return


@app.cell
def _(
    CDF_METRICS,
    GENDER_PALETTE,
    METRIC_SPECS,
    cdf_df,
    plot_cdf_overlay,
    plt,
):
    # CDF by gender for each metric (2x2 subplots)
    _cdf_fig_gen, _cdf_axes_gen = plt.subplots(4, 2, figsize=(12, 15))
    for _cdf_col, _cdf_ax in zip(CDF_METRICS, _cdf_axes_gen.flat):
        plot_cdf_overlay(_cdf_ax, cdf_df, _cdf_col, "gender", GENDER_PALETTE)
        _cdf_ax.set_title(METRIC_SPECS[_cdf_col])
        _cdf_ax.set_xlabel("Score")
        _cdf_ax.set_ylabel("Share ≥ score")
        _cdf_ax.set_xlim(0, 1)
        _cdf_ax.set_ylim(0, 1)
    _cdf_axes_gen.flat[-1].set_visible(False)

    _cdf_handles, _cdf_labels = _cdf_axes_gen.flat[0].get_legend_handles_labels()
    _cdf_fig_gen.legend(_cdf_handles, _cdf_labels, title=None, loc="lower center", ncol=2, fontsize=12, frameon=False,
                        bbox_to_anchor=(0.5, -0.03))
    _cdf_fig_gen.tight_layout()
    _cdf_fig_gen.savefig("fig_cdf_by_gender.pdf", dpi=300, bbox_inches="tight")
    _cdf_fig_gen
    return


@app.cell
def _(
    CDF_METRICS,
    GG_LABEL_MAP,
    GG_ORDER,
    GG_PALETTE,
    METRIC_SPECS,
    cdf_df,
    ecdf,
    plt,
):
    # CDF by gender × group for each metric (2x2 subplots, 6 lines each)
    _cdf_gg = cdf_df.copy()
    _cdf_gg["cell"] = list(zip(cdf_df["group"], cdf_df["gender"]))

    _cdf_fig_gg, _cdf_axes_gg = plt.subplots(4, 2, figsize=(12, 15))
    for _cdf_col, _cdf_ax in zip(CDF_METRICS, _cdf_axes_gg.flat):
        for _cell_key in GG_ORDER:
            _mask = _cdf_gg["cell"] == _cell_key
            if not _mask.any():
                continue
            _x, _y = ecdf(_cdf_gg.loc[_mask, _cdf_col])
            _cdf_ax.fill_between(_x, 0, _y, alpha=0.15, color=GG_PALETTE[_cell_key])
            _cdf_ax.plot(_x, _y, color=GG_PALETTE[_cell_key], linewidth=1.3, label=GG_LABEL_MAP[_cell_key])
        _cdf_ax.set_title(METRIC_SPECS[_cdf_col])
        _cdf_ax.set_xlabel("Score")
        _cdf_ax.set_ylabel("Share ≥ score")
        _cdf_ax.set_xlim(0, 1)
        _cdf_ax.set_ylim(0, 1)
    _cdf_axes_gg.flat[-1].set_visible(False)

    _cdf_handles, _cdf_labels = _cdf_axes_gg.flat[0].get_legend_handles_labels()
    _cdf_fig_gg.legend(_cdf_handles, _cdf_labels, title=None, loc="lower center", ncol=3, fontsize=11, frameon=False,
                       bbox_to_anchor=(0.5, -0.04))
    _cdf_fig_gg.tight_layout()
    _cdf_fig_gg.savefig("fig_cdf_by_gender_group.pdf", dpi=300, bbox_inches="tight")
    _cdf_fig_gg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Threshold Exploration
    """)
    return


@app.cell
def _(CDF_METRICS, mo):
    mo.md("### Threshold filter explorer")
    thresh_col = mo.ui.dropdown(
        options=CDF_METRICS, value="Unbabel--XCOMET-XL", label="Filter on",
    )
    thresh_val = mo.ui.slider(
        start=0.0, stop=1.0, step=0.01, value=0.8, label="Threshold",
    )
    target_col = mo.ui.dropdown(
        options=CDF_METRICS, value="survey_score", label="Show",
    )
    mo.vstack([thresh_col, target_col, thresh_val])
    return target_col, thresh_col, thresh_val


@app.cell
def _(
    GENDER_PALETTE,
    GG_LABEL_MAP,
    GG_ORDER,
    GG_PALETTE,
    cdf_df,
    ecdf,
    mo,
    pd,
    plt,
    target_col,
    thresh_col,
    thresh_val,
):
    # Filter
    _filt = cdf_df[cdf_df[thresh_col.value] >= thresh_val.value]
    _x_max = _filt[target_col.value].max()
    _n_total = len(cdf_df)
    _n_filt = len(_filt)

    mo.md(f"Retained **{_n_filt}** / {_n_total} records ({_n_filt / _n_total:.1%}) above {thresh_val.value:.2f}.")

    # -- Descriptive stats --
    GROUP_PALETTE_CDF = {
        "Hindi": "#355070", "US Black": "#606C37", "US White": "#DDA15E",
    }

    def _filt_desc(df, label):
        return {
            "Group": label,
            "N": len(df),
            "Mean": df[target_col.value].mean(),
            "SD": df[target_col.value].std(),
            "Min": df[target_col.value].min(),
            "Max": df[target_col.value].max(),
        }

    _filt_rows = [_filt_desc(_filt, "Overall")]
    for _g in ["man", "woman"]:
        _filt_rows.append(_filt_desc(_filt[_filt["gender"] == _g], _g))
    for _g in ["Hindi", "US Black", "US White"]:
        _filt_rows.append(_filt_desc(_filt[_filt["group"] == _g], _g))
    for _g, _gen in GG_ORDER:
        _sub = _filt[(_filt["group"] == _g) & (_filt["gender"] == _gen)]
        _filt_rows.append(_filt_desc(_sub, f"{_g} × {_gen}"))

    _filt_stats = pd.DataFrame(_filt_rows).round(3)
    _table = mo.ui.table(_filt_stats)

    # -- Survival plots --
    _fig_th, (_ax_gender, _ax_group, _ax_gg) = plt.subplots(
        ncols=3, figsize=(16, 4.5),
    )

    # By gender
    for _g in ["man", "woman"]:
        _s = _filt[_filt["gender"] == _g][target_col.value]
        _x, _y = ecdf(_s)
        _ax_gender.fill_between(_x, 0, _y, alpha=0.15, color=GENDER_PALETTE[_g])
        _ax_gender.plot(_x, _y, color=GENDER_PALETTE[_g], linewidth=1.5, label=_g)
    _ax_gender.set_title("By gender")
    _ax_gender.set_xlabel("Score"); _ax_gender.set_ylabel("Share ≥ score")
    _ax_gender.set_xlim(0, _x_max); _ax_gender.set_ylim(0, 1)
    _ax_gender.legend(frameon=False)

    # By group
    for _g in ["Hindi", "US Black", "US White"]:
        _s = _filt[_filt["group"] == _g][target_col.value]
        _x, _y = ecdf(_s)
        _c = GROUP_PALETTE_CDF[_g]
        _ax_group.fill_between(_x, 0, _y, alpha=0.15, color=_c)
        _ax_group.plot(_x, _y, color=_c, linewidth=1.5, label=_g)
    _ax_group.set_title("By group")
    _ax_group.set_xlabel("Score")
    _ax_group.set_xlim(0, _x_max); _ax_group.set_ylim(0, 1)
    _ax_group.legend(frameon=False)

    # By gender × group
    for _cell_key in GG_ORDER:
        _mask = _filt["group"] == _cell_key[0]
        _mask &= _filt["gender"] == _cell_key[1]
        if not _mask.any():
            continue
        _x, _y = ecdf(_filt.loc[_mask, target_col.value])
        _c = GG_PALETTE[_cell_key]
        _ax_gg.fill_between(_x, 0, _y, alpha=0.15, color=_c)
        _ax_gg.plot(_x, _y, color=_c, linewidth=1.3, label=GG_LABEL_MAP[_cell_key])
    _ax_gg.set_title("By gender × group")
    _ax_gg.set_xlabel("Score")
    _ax_gg.set_xlim(0, _x_max); _ax_gg.set_ylim(0, 1)
    _ax_gg.legend(fontsize=8, frameon=False, ncol=2)

    _fig_th.tight_layout()
    mo.vstack([_table, _fig_th])
    return


@app.cell
def _(full_data, plt):
    _group_pal = {"Hindi": "#355070", "US Black": "#606C37", "US White": "#DDA15E"}
    _group_mrk = {"Hindi": "o", "US Black": "s", "US White": "^"}
    _group_ord = ["Hindi", "US Black", "US White"]

    _psm1 = full_data.copy()
    _sender1 = (
        _psm1.groupby("sender_unique_id", observed=True)
        .agg(survey_score=("survey_score", "mean"),
             qa_score=("qa_score", "mean"),
             group=("group", "first"))
        .reset_index()
    )
    _fig_psm1, _ax_psm1 = plt.subplots(figsize=(3.15, 3.15))
    for _g in _group_ord:
        _sub = _sender1[_sender1["group"] == _g]
        _ax_psm1.scatter(_sub["qa_score"], _sub["survey_score"],
                          alpha=0.4, s=14,
                          color=_group_pal[_g], marker=_group_mrk[_g],
                          label=_g)
    _ax_psm1.axvline(0.8, linestyle="--", color="gray", alpha=0.6)
    _above = _sender1[(_sender1["qa_score"] >= 0.8) & (_sender1["survey_score"] >= 4)]
    _counts1 = _above["group"].value_counts().reindex(_group_ord, fill_value=0)
    _txt1 = "\n".join([f"{_g}: {_counts1.get(_g, 0)}" for _g in _group_ord])
    _ax_psm1.text(0.02, 0.02, _txt1, transform=_ax_psm1.transAxes,
                   va="bottom", ha="left")
    _ax_psm1.set_xlabel("QA Score")
    _ax_psm1.set_ylabel("Usability ($u$)")
    _fig_psm1.tight_layout()
    _fig_psm1.savefig("fig_per_sender_qa.pdf", dpi=300, bbox_inches="tight")
    _fig_psm1
    return


@app.cell
def _(full_data, plt):
    _group_pal2 = {"Hindi": "#355070", "US Black": "#606C37", "US White": "#DDA15E"}
    _group_mrk2 = {"Hindi": "o", "US Black": "s", "US White": "^"}
    _group_ord2 = ["Hindi", "US Black", "US White"]

    _psm2 = full_data.copy()
    _sender2 = (
        _psm2.groupby("sender_unique_id", observed=True)
        .agg(survey_score=("survey_score", "mean"),
             comet=("Unbabel--wmt22-comet-da", "mean"),
             group=("group", "first"))
        .reset_index()
    )
    _fig_psm2, _ax_psm2 = plt.subplots(figsize=(3.15, 3.15))
    for _g in _group_ord2:
        _sub = _sender2[_sender2["group"] == _g]
        _ax_psm2.scatter(_sub["comet"], _sub["survey_score"],
                          alpha=0.4, s=14,
                          color=_group_pal2[_g], marker=_group_mrk2[_g],
                          label=_g)
    _ax_psm2.axvline(0.8, linestyle="--", color="gray", alpha=0.6)
    _above2 = _sender2[(_sender2["comet"] >= 0.8) & (_sender2["survey_score"] >= 3)]
    _counts2 = _above2["group"].value_counts().reindex(_group_ord2, fill_value=0)
    _txt2 = "\n".join([f"{_g}: {_counts2.get(_g, 0)}" for _g in _group_ord2])
    _ax_psm2.text(0.02, 0.02, _txt2, transform=_ax_psm2.transAxes,
                   va="bottom", ha="left")
    _ax_psm2.set_xlabel("COMET")
    _ax_psm2.set_ylabel("Usability ($u$)")
    _fig_psm2.tight_layout()
    _fig_psm2.savefig("fig_per_sender_comet.pdf", dpi=300, bbox_inches="tight")
    _fig_psm2
    return


@app.cell
def _(mo):
    mo.md("### Threshold split")
    u_threshold = mo.ui.slider(
        start=1.0, stop=5.0, step=0.1, value=3.0, label="Usability threshold",
    )
    mo.hstack([u_threshold])
    return (u_threshold,)


@app.cell
def _(full_data, np, plt, u_threshold):
    _groups = ["US White", "US Black", "Hindi"]
    _pal = {"US White": "#666666", "US Black": "#333333", "Hindi": "#AAAAAA"}
    _mrk = {"US White": "o", "US Black": "s", "Hindi": "^"}
    _lbl = {"US White": r"$\mathrm{US_W}$", "US Black": r"$\mathrm{US_B}$", "Hindi": "Hindi"}

    _df_ts = full_data.copy()
    _thr = u_threshold.value

    _df_ts["above"] = _df_ts["survey_score"] >= _thr
    _splits = {"Below": False, "Above": True}

    _fig_ts, _axes_ts = plt.subplots(ncols=2, figsize=(5, 3.15), sharey=True)

    for (_label, _mask), _ax_ts in zip(_splits.items(), _axes_ts):
        _sub = _df_ts[_df_ts["above"] == _mask]
        _x_base = [0, 1]
        for _i, _g in enumerate(_groups):
            _gs = _sub[_sub["group"] == _g]
            _mean_qa = _gs["qa_score"].mean()
            _mean_co = _gs["Unbabel--wmt22-comet-da"].mean()
            _ci_qa = 1.96 * _gs["qa_score"].std() / np.sqrt(len(_gs))
            _ci_co = 1.96 * _gs["Unbabel--wmt22-comet-da"].std() / np.sqrt(len(_gs))
            _vals = [_mean_qa, _mean_co]
            _errs = [_ci_qa, _ci_co]
            _x = [v + (_i - 1) * 0.08 for v in _x_base]
            _ax_ts.errorbar(
                _x, _vals, yerr=_errs,
                marker=_mrk[_g], markersize=6, capsize=3,
                color=_pal[_g], linestyle="None", label=_lbl[_g],
            )
        _ax_ts.set_xticks(_x_base)
        _ax_ts.set_xticklabels(["QA Score", "COMET"])
        _ax_ts.set_title(f"{_label} $u \\geq {_thr:.1f}$")
        _ax_ts.set_ylim(0.5, 1)

    _handles_ts, _labels_ts = _axes_ts[0].get_legend_handles_labels()
    _fig_ts.legend(_handles_ts, _labels_ts, title=None, ncol=3,
                   loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.06))
    _fig_ts.tight_layout()
    _fig_ts
    return


@app.cell
def _(full_data, np, pd, plt):
    _groups_sw = ["US White", "US Black", "Hindi"]
    _pal_sw = {"US White": "#666666", "US Black": "#333333", "Hindi": "#AAAAAA"}
    _mrk_sw = {"US White": "o", "US Black": "s", "Hindi": "^"}
    _lbl_sw = {"US White": r"$\mathrm{US_W}$", "US Black": r"$\mathrm{US_B}$", "Hindi": "Hindi"}

    _thresholds = np.arange(1.0, 5.1, 0.5)
    _rows_sw = []

    for _t in _thresholds:
        _above = full_data[full_data["survey_score"] >= _t]
        for _g in _groups_sw:
            _gs = _above[_above["group"] == _g]
            _n = len(_gs)
            _m_qa = _gs["qa_score"].mean() if _n > 0 else np.nan
            _m_co = _gs["Unbabel--wmt22-comet-da"].mean() if _n > 0 else np.nan
            _ci_qa = 1.96 * _gs["qa_score"].std() / np.sqrt(_n) if _n > 0 else np.nan
            _ci_co = 1.96 * _gs["Unbabel--wmt22-comet-da"].std() / np.sqrt(_n) if _n > 0 else np.nan
            _rows_sw.append({
                "threshold": _t, "group": _g,
                "qa_mean": _m_qa, "qa_ci": _ci_qa,
                "comet_mean": _m_co, "comet_ci": _ci_co,
            })

    _df_sw = pd.DataFrame(_rows_sw)

    _fig_sw, (_ax_qa, _ax_co) = plt.subplots(ncols=2, figsize=(8, 3.15), sharey=False)

    for _g in _groups_sw:
        _sub = _df_sw[_df_sw["group"] == _g].sort_values("threshold")
        _ax_qa.errorbar(
            _sub["threshold"], _sub["qa_mean"], yerr=_sub["qa_ci"],
            marker=_mrk_sw[_g], markersize=6, capsize=3,
            color=_pal_sw[_g], linestyle=":", label=_lbl_sw[_g],
        )
        _ax_co.errorbar(
            _sub["threshold"], _sub["comet_mean"], yerr=_sub["comet_ci"],
            marker=_mrk_sw[_g], markersize=6, capsize=3,
            color=_pal_sw[_g], linestyle=":", label=_lbl_sw[_g],
        )

    _ax_qa.set_xlabel("Usability threshold ($u \\geq$)")
    _ax_qa.set_ylabel("QA Score")
    _ax_qa.set_ylim(0.7, 1)
    _ax_co.set_xlabel("Usability threshold ($u \\geq$)")
    _ax_co.set_ylabel("COMET")
    _ax_co.set_ylim(0.75, 0.85)

    _h_sw, _l_sw = _ax_qa.get_legend_handles_labels()
    _fig_sw.legend(_h_sw, _l_sw, title=None, ncol=3,
                   loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.06))
    _fig_sw.tight_layout()
    _fig_sw
    return


@app.cell
def _(full_data, np, pd, plt):
    _thresholds_ov = np.arange(1.0, 5.1, 0.5)
    _rows_ov = []

    for _t in _thresholds_ov:
        _above = full_data[full_data["survey_score"] >= _t]
        _n = len(_above)
        _m_qa = _above["qa_score"].mean() if _n > 0 else np.nan
        _m_co = _above["Unbabel--wmt22-comet-da"].mean() if _n > 0 else np.nan
        _ci_qa = 1.96 * _above["qa_score"].std() / np.sqrt(_n) if _n > 0 else np.nan
        _ci_co = 1.96 * _above["Unbabel--wmt22-comet-da"].std() / np.sqrt(_n) if _n > 0 else np.nan
        _rows_ov.append({
            "threshold": _t,
            "qa_mean": _m_qa, "qa_ci": _ci_qa,
            "comet_mean": _m_co, "comet_ci": _ci_co,
        })

    _df_ov = pd.DataFrame(_rows_ov)

    _fig_ov, (_ax_ov_qa, _ax_ov_co) = plt.subplots(ncols=2, figsize=(6.5, 2.8))

    for _ax_ov, _col in [(_ax_ov_qa, "qa_mean"), (_ax_ov_co, "comet_mean")]:
        _ci_col = _col.replace("_mean", "_ci")
        # All markers in dark gray
        _ax_ov.errorbar(
            _df_ov["threshold"], _df_ov[_col], yerr=_df_ov[_ci_col],
            marker="o", markersize=6, capsize=3,
            color="#777777", ecolor="#777777", linestyle=":",
        )
        # Overlay thresholds 3, 4, 5 in black
        _hl = _df_ov[_df_ov["threshold"].isin([3.0, 4.0, 5.0])]
        _ax_ov.errorbar(
            _hl["threshold"], _hl[_col], yerr=_hl[_ci_col],
            marker="o", markersize=7, capsize=4,
            color="black", ecolor="black", linestyle="None",
        )

    for _ax_ov, _col, _offset in [(_ax_ov_qa, "qa_mean", 0.015), (_ax_ov_co, "comet_mean", 0.007)]:
        _ax_ov.set_xticks(np.arange(1, 6))
        _ax_ov.set_xticklabels(["1", "2", "3", "4", "5"])
        _ax_ov.set_xticks(np.arange(1.5, 5, 0.5), minor=True)
        _ax_ov.grid(axis="x", which="minor", alpha=0.3, linestyle="--")
        _ax_ov.grid(axis="x", which="major", alpha=0.6)
        for _t_ann in [3, 4, 5]:
            _row = _df_ov[_df_ov["threshold"] == float(_t_ann)]
            if len(_row):
                _val = _row[_col].values[0]
                _ax_ov.text(
                    _t_ann - 0.3, _val + _offset,
                    f"{_val:.3f}", ha="center", va="bottom",
                )

    for _ax_ov in [_ax_ov_qa, _ax_ov_co]:
        _ax_ov.margins(x=0.1, y=0.1)

    # _ax_ov_qa.set_xlabel("($u \\geq$)")
    _ax_ov_qa.set_xlabel("")
    _ax_ov_qa.set_ylabel("Mean QA Score")
    _ax_ov_qa.set_ylim(0.75, 1)
    # _ax_ov_co.set_xlabel("($u \\geq$)")
    _ax_ov_co.set_xlabel("")
    _ax_ov_co.set_ylabel("Mean COMET")
    _ax_ov_co.set_ylim(0.78, 0.85)

    _fig_ov.tight_layout()
    _fig_ov.savefig("fig_sweep_thresholds.pdf", dpi=300, bbox_inches="tight")
    _fig_ov
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Mean QA Score (left) and COMET (right) among records
    whose usability meets or exceeds each threshold value on the
    x-axis. Error bars are 95\% confidence intervals. Black markers
    highlight thresholds 3, 4, and 5.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quality-Usability Correlation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scatterplots vs. Usability ($u$)
    """)
    return


@app.cell
def _(full_data, plt):
    _metrics_q = [
        "validator_translation_score",
        "qa_score",
        "Unbabel--XCOMET-XL",
        "google--metricx-24-hybrid-xl-v2p6",
    ]
    _plot_q = full_data.copy()
    _plot_q["qa_score"] = 1 - _plot_q["correction_ratio"]
    _metric_labels_q = {
        "validator_translation_score": "Validator Score",
        "qa_score": "QA Score",
        "Unbabel--XCOMET-XL": "XCOMET XL",
        "google--metricx-24-hybrid-xl-v2p6": "MetricX 24",
    }
    _group_palette_q = {
        "Hindi": "#355070", "US Black": "#606C37", "US White": "#DDA15E",
    }
    _group_markers_q = {
        "Hindi": "o", "US Black": "s", "US White": "^",
    }
    _group_order_q = ["Hindi", "US Black", "US White"]

    _fig_q, _axes_q = plt.subplots(2, 2, figsize=(8, 6))
    for (_col, _title), _ax_q in zip(
        [(_m, _metric_labels_q[_m]) for _m in _metrics_q], _axes_q.flat,
    ):
        for _g in _group_order_q:
            _sub = _plot_q[_plot_q["group"] == _g]
            _ax_q.scatter(
                _sub[_col], _sub["survey_score"],
                alpha=0.3, s=12,
                color=_group_palette_q[_g],
                marker=_group_markers_q[_g],
                label=_g,
            )
        _ax_q.set_xlabel(_title)
        _ax_q.set_ylabel("Usability ($u$)")
        _ax_q.set_xlim(0)
        _ax_q.set_ylim(1, 5)

    _handles_q, _labels_q = _axes_q.flat[0].get_legend_handles_labels()
    _fig_q.legend(_handles_q, _labels_q, title=None, loc="lower center",
                  ncol=3, frameon=True, framealpha=0.7, facecolor="white",
                  bbox_to_anchor=(0.5, -0.04))
    _fig_q.tight_layout()
    _fig_q.savefig("fig_scatter_quality_usability.pdf", dpi=300, bbox_inches="tight")
    _fig_q
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Scatterplots of each quality metric against usability ($u$).
    Points are colored and shaped by language group.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scatterplots vs. Baseline Usability ($u$)
    """)
    return


@app.cell
def _(full_data, plt):
    _plot_qb = full_data.copy()
    _plot_qb["qa_score"] = 1 - _plot_qb["correction_ratio"]
    _metrics_qb = [
        "validator_translation_score",
        "qa_score",
        "Unbabel--XCOMET-XL",
        "google--metricx-24-hybrid-xl-v2p6",
    ]
    _metric_labels_qb = {
        "validator_translation_score": "Validator Score",
        "qa_score": "QA Score",
        "Unbabel--XCOMET-XL": "XCOMET XL",
        "google--metricx-24-hybrid-xl-v2p6": "MetricX 24",
    }
    _group_palette_qb = {
        "Hindi": "#355070", "US Black": "#606C37", "US White": "#DDA15E",
    }
    _group_markers_qb = {
        "Hindi": "o", "US Black": "s", "US White": "^",
    }
    _group_order_qb = ["Hindi", "US Black", "US White"]

    _fig_qb, _axes_qb = plt.subplots(2, 2, figsize=(8, 6))
    for (_col, _title), _ax_qb in zip(
        [(_m, _metric_labels_qb[_m]) for _m in _metrics_qb], _axes_qb.flat,
    ):
        for _g in _group_order_qb:
            _sub = _plot_qb[_plot_qb["group"] == _g]
            _ax_qb.scatter(
                _sub[_col], _sub["baseline_survey_score"],
                alpha=0.3, s=12,
                color=_group_palette_qb[_g],
                marker=_group_markers_qb[_g],
                label=_g,
            )
        _ax_qb.set_xlabel(_title)
        _ax_qb.set_ylabel("$u$ (Baseline)")
        _ax_qb.set_xlim(0)
        _ax_qb.set_ylim(1, 5)

    _handles_qb, _labels_qb = _axes_qb.flat[0].get_legend_handles_labels()
    _fig_qb.legend(_handles_qb, _labels_qb, title=None, loc="lower center",
                   ncol=3, frameon=True, framealpha=0.7, facecolor="white",
                   bbox_to_anchor=(0.5, -0.04))
    _fig_qb.tight_layout()
    _fig_qb.savefig("fig_scatter_quality_baseline.pdf", dpi=300, bbox_inches="tight")
    _fig_qb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Scatterplots of each quality metric against baseline usability
    ($u$ baseline). Points are colored and shaped by language group.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boxplots by Usability Level
    """)
    return


@app.cell
def _(full_data, plt):
    _metrics_bp = [
        "validator_translation_score",
        "qa_score",
        "Unbabel--XCOMET-XL",
        "google--metricx-24-hybrid-xl-v2p6",
    ]
    _metric_labels_bp = {
        "validator_translation_score": "Validator Score",
        "qa_score": "QA Score",
        "Unbabel--XCOMET-XL": "XCOMET XL",
        "google--metricx-24-hybrid-xl-v2p6": "MetricX 24",
    }
    _bp_df = full_data.copy()
    _bp_df["qa_score"] = 1 - _bp_df["correction_ratio"]
    _bp_df["u_bin"] = _bp_df["survey_score"].round().astype(int)

    _fig_bp, _axes_bp = plt.subplots(2, 2, figsize=(8, 6))
    for (_col, _title), _ax_bp in zip(
        [(_m, _metric_labels_bp[_m]) for _m in _metrics_bp], _axes_bp.flat,
    ):
        _data_bp = [_bp_df[_bp_df["u_bin"] == b][_col].dropna() for b in range(1, 6)]
        _bp = _ax_bp.boxplot(_data_bp, positions=range(1, 6), widths=0.5,
                              patch_artist=True,
                              boxprops=dict(facecolor="#CCCCCC", alpha=0.6),
                              medianprops=dict(color="black"),
                              flierprops=dict(marker=".", markersize=3, alpha=0.3))
        _ax_bp.set_xlabel("Usability ($u$)")
        _ax_bp.set_ylabel(_title)

    _fig_bp.tight_layout()
    _fig_bp.savefig("fig_boxplot_quality_usability.pdf", dpi=300, bbox_inches="tight")
    _fig_bp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Boxplots of each quality metric by rounded usability score
    (1--5). Boxes show medians and quartiles; whiskers extend to 1.5× IQR.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boxplots by Baseline Level
    """)
    return


@app.cell
def _(full_data, plt):
    _metrics_bpb = [
        "validator_translation_score",
        "qa_score",
        "Unbabel--XCOMET-XL",
        "google--metricx-24-hybrid-xl-v2p6",
    ]
    _metric_labels_bpb = {
        "validator_translation_score": "Validator Score",
        "qa_score": "QA Score",
        "Unbabel--XCOMET-XL": "XCOMET XL",
        "google--metricx-24-hybrid-xl-v2p6": "MetricX 24",
    }
    _bpb_df = full_data.copy()
    _bpb_df["qa_score"] = 1 - _bpb_df["correction_ratio"]
    _bpb_df["ub_bin"] = _bpb_df["baseline_survey_score"].round().astype(int)

    _fig_bpb, _axes_bpb = plt.subplots(2, 2, figsize=(8, 6))
    for (_col, _title), _ax_bpb in zip(
        [(_m, _metric_labels_bpb[_m]) for _m in _metrics_bpb], _axes_bpb.flat,
    ):
        _data_bpb = [_bpb_df[_bpb_df["ub_bin"] == b][_col].dropna() for b in range(1, 6)]
        _ax_bpb.boxplot(_data_bpb, positions=range(1, 6), widths=0.5,
                         patch_artist=True,
                         boxprops=dict(facecolor="#CCCCCC", alpha=0.6),
                         medianprops=dict(color="black"),
                         flierprops=dict(marker=".", markersize=3, alpha=0.3))
        _ax_bpb.set_xlabel("$u$ (Baseline)")
        _ax_bpb.set_ylabel(_title)

    _fig_bpb.tight_layout()
    _fig_bpb.savefig("fig_boxplot_quality_baseline.pdf", dpi=300, bbox_inches="tight")
    _fig_bpb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Boxplots of each quality metric by rounded baseline usability
    score (1--5).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-User Means
    """)
    return


@app.cell
def _(full_data, plt):
    _metrics_pu = [
        "validator_translation_score",
        "qa_score",
        "Unbabel--XCOMET-XL",
        "google--metricx-24-hybrid-xl-v2p6",
    ]
    _metric_labels_pu = {
        "validator_translation_score": "Validator Score",
        "qa_score": "QA Score",
        "Unbabel--XCOMET-XL": "XCOMET XL",
        "google--metricx-24-hybrid-xl-v2p6": "MetricX 24",
    }
    _group_palette_pu = {
        "Hindi": "#355070", "US Black": "#606C37", "US White": "#DDA15E",
    }
    _group_markers_pu = {
        "Hindi": "o", "US Black": "s", "US White": "^",
    }
    _group_order_pu = ["Hindi", "US Black", "US White"]

    _pu_src = full_data.copy()
    _pu_src["qa_score"] = 1 - _pu_src["correction_ratio"]
    _agg_dict_pu = {m: "mean" for m in _metrics_pu}
    _agg_dict_pu.update({
        "survey_score": "mean",
        "baseline_survey_score": "mean",
        "group": "first",
    })
    _pu_df = (
        _pu_src.groupby("sender_unique_id", observed=True)
        .agg(_agg_dict_pu)
    )
    _pu_df = _pu_df.reset_index()

    _fig_pu, _axes_pu = plt.subplots(2, 2, figsize=(8, 6))
    for (_col, _title), _ax_pu in zip(
        [(_m, _metric_labels_pu[_m]) for _m in _metrics_pu], _axes_pu.flat,
    ):
        for _g in _group_order_pu:
            _sub = _pu_df[_pu_df["group"] == _g]
            _ax_pu.scatter(
                _sub[_col], _sub["survey_score"],
                alpha=0.4, s=18,
                color=_group_palette_pu[_g],
                marker=_group_markers_pu[_g],
                label=_g,
            )
        _ax_pu.set_xlabel(_title)
        _ax_pu.set_ylabel("Usability ($u$)")
        _ax_pu.set_ylim(1, 5)

    _handles_pu, _labels_pu = _axes_pu.flat[0].get_legend_handles_labels()
    _fig_pu.legend(_handles_pu, _labels_pu, title=None, loc="lower center",
                   ncol=3, frameon=True, framealpha=0.7, facecolor="white",
                   bbox_to_anchor=(0.5, -0.04))
    _fig_pu.tight_layout()
    _fig_pu.savefig("fig_scatter_peruser_usability.pdf", dpi=300, bbox_inches="tight")
    _fig_pu
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Per-user means: each point is one participant's average
    across all their conversations. Quality metrics vs. usability ($u$).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Per-User Means vs. Baseline
    """)
    return


@app.cell
def _(full_data, plt):
    _metrics_pub = [
        "validator_translation_score",
        "qa_score",
        "Unbabel--XCOMET-XL",
        "google--metricx-24-hybrid-xl-v2p6",
    ]
    _metric_labels_pub = {
        "validator_translation_score": "Validator Score",
        "qa_score": "QA Score",
        "Unbabel--XCOMET-XL": "XCOMET XL",
        "google--metricx-24-hybrid-xl-v2p6": "MetricX 24",
    }
    _group_palette_pub = {
        "Hindi": "#355070", "US Black": "#606C37", "US White": "#DDA15E",
    }
    _group_markers_pub = {
        "Hindi": "o", "US Black": "s", "US White": "^",
    }
    _group_order_pub = ["Hindi", "US Black", "US White"]

    _pub_src = full_data.copy()
    _pub_src["qa_score"] = 1 - _pub_src["correction_ratio"]
    _agg_dict_pub = {m: "mean" for m in _metrics_pub}
    _agg_dict_pub.update({
        "survey_score": "mean",
        "baseline_survey_score": "mean",
        "group": "first",
    })
    _pub_df = (
        _pub_src.groupby("sender_unique_id", observed=True)
        .agg(_agg_dict_pub)
    )
    _pub_df.columns = [c[0] if c[0] else c for c in _pub_df.columns]
    _pub_df = _pub_df.reset_index()

    _fig_pub, _axes_pub = plt.subplots(2, 2, figsize=(8, 6))
    for (_col, _title), _ax_pub in zip(
        [(_m, _metric_labels_pub[_m]) for _m in _metrics_pub], _axes_pub.flat,
    ):
        for _g in _group_order_pub:
            _sub = _pub_df[_pub_df["group"] == _g]
            _ax_pub.scatter(
                _sub[_col], _sub["baseline_survey_score"],
                alpha=0.4, s=18,
                color=_group_palette_pub[_g],
                marker=_group_markers_pub[_g],
                label=_g,
            )
        _ax_pub.set_xlabel(_title)
        _ax_pub.set_ylabel("$u$ (Baseline)")
        _ax_pub.set_ylim(1, 5)

    _handles_pub, _labels_pub = _axes_pub.flat[0].get_legend_handles_labels()
    _fig_pub.legend(_handles_pub, _labels_pub, title=None, loc="lower center",
                    ncol=3, frameon=True, framealpha=0.7, facecolor="white",
                    bbox_to_anchor=(0.5, -0.04))
    _fig_pub.tight_layout()
    _fig_pub.savefig("fig_scatter_peruser_baseline.pdf", dpi=300, bbox_inches="tight")
    _fig_pub
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Per-user means: each point is one participant's average
    across all their conversations. Quality metrics vs. baseline usability.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Z-Normalized Per-Sender Quadrants
    """)
    return


@app.cell
def _(full_data, plt):
    _z_metrics = [
        "qa_score",
        "validator_translation_score",
        "Unbabel--XCOMET-XL",
        "google--metricx-24-hybrid-xl-v2p6",
    ]
    import seaborn as _sns_z
    _z_labels = {
        "qa_score": "QA Score",
        "validator_translation_score": "Validator Score",
        "Unbabel--XCOMET-XL": "XCOMET XL",
        "google--metricx-24-hybrid-xl-v2p6": "MetricX 24",
    }
    _group_palette_z = {
        "Hindi": "#355070", "US Black": "#606C37", "US White": "#DDA15E",
    }
    _group_markers_z = {
        "Hindi": "o", "US Black": "s", "US White": "^",
    }
    _group_order_z = ["Hindi", "US Black", "US White"]

    _z_src = full_data.copy()
    _z_src["qa_score"] = 1 - _z_src["correction_ratio"]

    # Within-group percentile rank (0-100) for each metric
    for _col in ["survey_score"] + _z_metrics:
        _z_src[f"{_col}_z"] = (
            _z_src.groupby("group", observed=True)[_col]
            .transform(lambda x: x.rank(pct=True) * 100)
        )

    # Per-sender means of z-scores
    _z_agg = {f"{_col}_z": "mean" for _col in ["survey_score"] + _z_metrics}
    _z_agg["group"] = "first"
    _z_sender = (
        _z_src.groupby("sender_unique_id", observed=True)
        .agg(_z_agg)
        .reset_index()
    )

    _fig_z, _axes_z = plt.subplots(2, 2, figsize=(8, 7))
    for (_col, _title), _ax_z in zip(
        [(_m, _z_labels[_m]) for _m in _z_metrics], _axes_z.flat,
    ):
        _uz = "survey_score_z"
        _mz = f"{_col}_z"
        for _g in _group_order_z:
            _sub = _z_sender[_z_sender["group"] == _g]
            _ax_z.scatter(
                _sub[_mz], _sub[_uz],
                alpha=0.5, s=16,
                color=_group_palette_z[_g],
                marker=_group_markers_z[_g],
                label=_g,
            )
        _ax_z.axhline(50, linestyle="--", color="gray", alpha=0.5)
        _ax_z.axvline(50, linestyle="--", color="gray", alpha=0.5)
        # KDE density contours
        try:
            _sns_z.kdeplot(
                x=_z_sender[_mz], y=_z_sender[_uz],
                ax=_ax_z, levels=5, linewidths=0.8, color="black", alpha=0.3,
            )
        except Exception:
            pass
        _ax_z.set_xlabel(f"{_title} (pct)")
        _ax_z.set_ylabel("Usability ($u$) (pct)")
        _ax_z.set_aspect("equal")
        _ax_z.set_xlim(0, 100)
        _ax_z.set_ylim(0, 100)

    _h_z, _l_z = _axes_z.flat[0].get_legend_handles_labels()
    _fig_z.legend(_h_z, _l_z, title=None, loc="lower center",
                  ncol=3, frameon=True, framealpha=0.7, facecolor="white",
                  bbox_to_anchor=(0.5, -0.03))
    _fig_z.tight_layout()
    _fig_z.savefig("fig_zscore_per_sender_quadrants.pdf", dpi=300, bbox_inches="tight")
    _fig_z
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Per-sender mean percentile ranks: usability vs. each quality
    metric, ranked within language group (0--100). Each point is one
    participant. Quadrant lines at the 50th percentile separate above- and
    below-median senders.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hexbin by Sender
    """)
    return


@app.cell
def _(full_data, plt):
    _hx_metrics = [
        "qa_score",
        "validator_translation_score",
        "Unbabel--XCOMET-XL",
        "google--metricx-24-hybrid-xl-v2p6",
    ]
    _hx_labels = {
        "qa_score": "QA Score",
        "validator_translation_score": "Validator Score",
        "Unbabel--XCOMET-XL": "XCOMET XL",
        "google--metricx-24-hybrid-xl-v2p6": "MetricX 24",
    }

    _hx_src = full_data.copy()
    _hx_src["qa_score"] = 1 - _hx_src["correction_ratio"]

    for _col in ["survey_score"] + _hx_metrics:
        _hx_src[f"{_col}_z"] = (
            _hx_src.groupby("group", observed=True)[_col]
            .transform(lambda x: x.rank(pct=True) * 100)
        )

    _hx_agg = {f"{_col}_z": "mean" for _col in ["survey_score"] + _hx_metrics}
    _hx_sender = (
        _hx_src.groupby("sender_unique_id", observed=True)
        .agg(_hx_agg)
        .reset_index()
    )

    _fig_hx, _axes_hx = plt.subplots(2, 2, figsize=(8, 7))
    for (_col, _title), _ax_hx in zip(
        [(_m, _hx_labels[_m]) for _m in _hx_metrics], _axes_hx.flat,
    ):
        _uz = "survey_score_z"
        _mz = f"{_col}_z"
        _ax_hx.hexbin(
            _hx_sender[_mz], _hx_sender[_uz],
            gridsize=20, cmap="Greys", mincnt=1,
        )
        _ax_hx.axhline(50, linestyle="--", color="black", alpha=0.4)
        _ax_hx.axvline(50, linestyle="--", color="black", alpha=0.4)
        _ax_hx.set_xlabel(f"{_title} (pct)")
        _ax_hx.set_ylabel("Usability ($u$) (pct)")
        _ax_hx.set_aspect("equal")
        _ax_hx.set_xlim(0, 100)
        _ax_hx.set_ylim(0, 100)

    _fig_hx.tight_layout()
    _fig_hx.savefig("fig_hexbin_per_sender.pdf", dpi=300, bbox_inches="tight")
    _fig_hx
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Hexbin plots of per-sender mean percentile ranks (0--100).
    Darker cells indicate more senders at that position. Quadrant lines
    at the 50th percentile.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Usability Scores
    """)
    return


@app.cell
def _(data_store, full_data, pd):
    def get_table_with_score_statistics(df: pd.DataFrame):

        agg = []
        for gname in ["US Black", "US White", "Hindi"]:

            baseline_d = {} 
            final_d = {} 
            group_df = df[df["group"] == gname]
            baseline_d["group"] = gname
            final_d["group"] = gname

            # Get two dfs with final_survey_cols and baseline_ columns
            final_survey_df = group_df[["entry_id", "survey_score"] + data_store["final_survey_cols"]]
            baseline_df = group_df[["entry_id", "baseline_survey_score"] + data_store["baseline_cols"]]

            baseline_df = baseline_df.rename(columns={f"baseline_{col}": col for col in data_store["final_survey_cols"]})
            baseline_df = baseline_df.rename(columns={"baseline_survey_score": "survey_score"})

            # Pre aggregation standard deviation of the average between the survey outputs (i.e., the "survey_score" column)
            final_survey_score_std = final_survey_df["survey_score"].std()
            baseline_survey_score_std = baseline_df["survey_score"].std()

            # Average difference between the final survey scores and the baseline survey scores
            # Reported only for the lines relative to final scores
            avg_diff = final_survey_df["survey_score"] - baseline_df["survey_score"]
            avg_diff_mean = avg_diff.mean()

            baseline_df = baseline_df.mean(axis=0)
            final_survey_df = final_survey_df.mean(axis=0)

            agg.append({
                "group": gname,
                "S": final_survey_df["satisfaction_score"],
                "T": final_survey_df["trust_score"],
                "R": final_survey_df["reliance_score"],
                "A": final_survey_df["survey_score"],
                "$\\sigma$": final_survey_score_std,
                "$\\Delta$": avg_diff_mean
            })
            agg.append({
                "group": "$\\text{b}\\rfloor$",
                "S": baseline_df["satisfaction_score"],
                "T": baseline_df["trust_score"],
                "R": baseline_df["reliance_score"],
                "A": baseline_df["survey_score"],
                "$\\sigma$": baseline_survey_score_std
            })

        new_df = pd.DataFrame(agg)
        return new_df.round(2)


    def format_table(df):
        """Changes applied by this method
        - Every even row gets cells wrapped in {\small {value}}
        """
        for col in ["group", "S", "T", "R", "A", "$\\sigma$"]:
            df[col] = df[col].astype(object)
        for i in range(len(df)):
            if i % 2 == 1:
                for col in ["group", "S", "T", "R", "A", "$\\sigma$"]:
                    df.at[i, col] = f"{{\\small {df.at[i, col]}}}"
        return df


    tab1 = get_table_with_score_statistics(full_data)
    tab1 = format_table(tab1)
    tab1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    5-point Likert Survey Scores by (S)atisfaction, (T)rust, and (R)eliance. We compute "u", the average between S, T, and R and report is group mean (A), std ($\Sigma$), and mean difference between the final and baseline scores.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fig: Usability by Model and Stakes
    """)
    return


@app.cell
def _(full_data, plt, sns):
    _model_order = ["DeSTA2", "Phi 4", "Voxtral", "Tower+"]
    _hatch_patterns = ["", "//"]

    fig_model, ax_model = plt.subplots(figsize=(7.0, 3))


    sns.barplot(
        data=full_data,
        x="translation_model",
        y="survey_score",
        hue="conversation_topic",
        order=_model_order,
        ax=ax_model,
    )

    # Apply hatch patterns per hue group and add value labels on top
    for container, hatch in zip(ax_model.containers, _hatch_patterns):
        for bar in container:
            bar.set_hatch(hatch)
        ax_model.bar_label(container, fmt="%.2f", padding=10, fontsize=10)

    ax_model.set_xlabel("")
    ax_model.set_ylabel("Usability ($u$)")
    ax_model.set_ylim(0, 5)

    _model_handles, _model_labels = ax_model.get_legend_handles_labels()
    for _model_handle, hatch in zip(_model_handles, _hatch_patterns):
        _model_handle.set_hatch(hatch)
    ax_model.legend(
        _model_handles,
        _model_labels,
        title=None,
        ncols=2,
        loc="upper left",
        bbox_to_anchor=(0, 1.02),
    )

    fig_model.tight_layout()
    fig_model.savefig("fig_usability_by_model_and_stakes.pdf", dpi=300, bbox_inches="tight")
    fig_model
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Usability by Model and Domain
    """)
    return


@app.cell
def _(full_data, plt):
    _alt_model_order = ["DeSTA2", "Phi 4", "Voxtral", "Tower+"]
    _alt_topic_order = ["Everyday", "Health"]
    _alt_palette = {
        "Everyday": "#222222",
        "Health": "#888888",
    }
    _alt_markers = {
        "Everyday": "o",
        "Health": "D",
    }
    _alt_topic_offsets = {
        "Everyday": -0.09,
        "Health": 0.09,
    }

    _alt_summary = (
        full_data.groupby(["translation_model", "conversation_topic"], observed=True)["survey_score"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    _alt_summary["se"] = _alt_summary["std"] / (_alt_summary["n"] ** 0.5)
    _alt_summary["ci95"] = 1.96 * _alt_summary["se"]

    fig_model_alt, (_ax_mean, _ax_delta) = plt.subplots(
        ncols=2,
        figsize=(8, 3.4),
        gridspec_kw={"width_ratios": [1.5, 1.2]},
    )

    # Panel 1: model-by-topic interaction (mean +/- 95% CI)
    _x_base = list(range(len(_alt_model_order)))
    for _alt_topic in _alt_topic_order:
        _curr = (
            _alt_summary[_alt_summary["conversation_topic"] == _alt_topic]
            .set_index("translation_model")
            .reindex(_alt_model_order)
            .reset_index()
        )
        _x = [_v + _alt_topic_offsets[_alt_topic] for _v in _x_base]
        _ax_mean.errorbar(
            _x,
            _curr["mean"],
            yerr=_curr["ci95"],
            linestyle="None",
            marker=_alt_markers[_alt_topic],
            markersize=7,
            capsize=5,
            color=_alt_palette[_alt_topic],
            label=_alt_topic,
        )
        _va = "top" if _alt_topic == "Everyday" else "bottom"
        _y_sign = -1 if _alt_topic == "Everyday" else 1
        for _xi, (_mv, _ci) in enumerate(zip(_curr["mean"], _curr["ci95"])):
            _ax_mean.text(
                _x[_xi], _mv + _y_sign * (_ci + 0.12),
                f"{_mv:.2f}",
                ha="center", va=_va, color=_alt_palette[_alt_topic],
            )

    _ax_mean.set_xticks(_x_base)
    _ax_mean.set_xticklabels(_alt_model_order)
    _ax_mean.set_ylim(1, 5)
    _ax_mean.margins(x=0.15)
    _ax_mean.set_xlabel("")
    _ax_mean.set_ylabel("Usability ($u$)")
    _ax_mean.legend(
        title=None,
        ncols=2,
        loc="lower right",
        frameon=False,
        columnspacing=0.6,
        handletextpad=0.02,
    )

    # Panel 2: explicit topic difference per model, Delta = Health - Everyday
    _alt_mean_pivot = (
        _alt_summary.pivot(index="translation_model", columns="conversation_topic", values="mean")
        .reindex(_alt_model_order)
    )
    _alt_se_pivot = (
        _alt_summary.pivot(index="translation_model", columns="conversation_topic", values="se")
        .reindex(_alt_model_order)
    )

    _alt_delta = _alt_mean_pivot["Health"] - _alt_mean_pivot["Everyday"]
    _alt_delta_ci95 = 1.96 * ((_alt_se_pivot["Health"] ** 2 + _alt_se_pivot["Everyday"] ** 2) ** 0.5)
    _alt_y = list(range(len(_alt_model_order)))

    _ax_delta.errorbar(
        _alt_delta,
        _alt_y,
        xerr=_alt_delta_ci95,
        fmt="o",
        markersize=7,
        capsize=5,
        color="#333333",
    )
    for _yi, (_dv, _ci) in enumerate(zip(_alt_delta, _alt_delta_ci95)):
        _ax_delta.text(
            _dv + _ci + 0.03, _alt_y[_yi],
            f"{_dv:.2f}",
            va="center", ha="left", color="#333333",
        )
    _ax_delta.axvline(0, linestyle="--", color="gray", alpha=0.7)
    _ax_delta.set_yticks(_alt_y)
    _ax_delta.set_yticklabels(_alt_model_order)
    _ax_delta.set_xlabel("$\\Delta_{u}$ (Health - Everyday)")
    _ax_delta.set_ylabel("")
    _ax_delta.set_xlim(-0.25, 1.15)

    fig_model_alt.tight_layout()
    fig_model_alt.savefig("fig_usability_model_topic_alternative.pdf", dpi=300, bbox_inches="tight")
    fig_model_alt
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fig: Score Distribution by Group
    """)
    return


@app.cell
def _(full_data, np, plt):
    _group_order = ["US White", "US Black", "Hindi"]
    _group_palette = {
        "US White": "#DDA15E", "US Black": "#606C37", "Hindi": "#355070",
    }
    _group_styles = {
        "US White": "-", "US Black": "--", "Hindi": "-.",
    }
    _group_labels = {
        "US White": r"$\mathrm{US_W}$",
        "US Black": r"$\mathrm{US_B}$",
        "Hindi": "Hindi",
    }

    def _ecdf_us(series):
        x = np.sort(series.dropna().to_numpy())
        y = 1 - np.arange(1, len(x) + 1) / len(x)
        return x, y

    _fig_dist, _ax_dist = plt.subplots(figsize=(4, 3.15))

    for _g in _group_order:
        _s = full_data[full_data["group"] == _g]["survey_score"]
        _x, _y = _ecdf_us(_s)
        _c = _group_palette[_g]
        _ls = _group_styles[_g]
        _ax_dist.fill_between(_x, 0, _y, alpha=0.12, color=_c)
        _ax_dist.plot(_x, _y, color=_c, linestyle=_ls, linewidth=1.8,
                       label=_group_labels.get(_g, _g))

    # _ax_dist.set_xlabel("Usability ($u$)")
    _ax_dist.set_xlabel("")
    _ax_dist.set_ylabel("Share ≥ score")
    _ax_dist.set_xlim(1, 5)
    _ax_dist.set_ylim(0, 1)
    _ax_dist.text(0.63, 0.95, "Usability ($u$)", transform=_ax_dist.transAxes,
                va="top", ha="left", fontsize=13)
    _ax_dist.legend(title=None, loc="lower left", frameon=True, framealpha=1, facecolor="white", edgecolor="black")

    # Annotate share ≥ 4 for each group
    _annot_spec = {
        "US White": {"y_off": 0.06, "x": 4.02, "ha": "left"},
        "US Black": {"y_off": 0.0,  "x": 4.02, "ha": "left"},
        "Hindi":    {"y_off": -0.06, "x": 3.98, "ha": "right"},
    }
    _ax_dist.axvline(4, linestyle=":", color="gray", alpha=0.4, zorder=0)
    for _g in _group_order:
        _share_ge4 = (full_data[full_data["group"] == _g]["survey_score"] >= 4).mean()
        _spec = _annot_spec[_g]
        _ax_dist.text(
            _spec["x"], _share_ge4 + _spec["y_off"],
            f"{_share_ge4:.1%}",
            color=_group_palette[_g], fontweight="bold",
            va="center", ha=_spec["ha"],
        )

    _fig_dist.tight_layout()
    _fig_dist.savefig("fig_score_distribution_by_group.pdf", dpi=300, bbox_inches="tight")
    _fig_dist
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Survival curves of usability scores by language group.
    Each curve shows the fraction of the group whose usability rating
    reaches or exceeds a given score on the x-axis. Annotations at
    $u=4$ report the share of each group scoring 4 or above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fig: Impact of Gender and Language Group
    """)
    return


@app.cell
def _(full_data, pd):
    # Chart 3.

    keep_cols = [
        "sender_unique_id",
        "validator_unique_id",
        "receiver_unique_id",
        "conversation_id",
        "conversation_topic",
        "conversation_source",
        "group",
        "sender_gender",
        "translation_model",
        "baseline_survey_score",
        "survey_score",
        "validator_translation_score",
        "correction_ratio",
        "score_shift",
        "reliance_score",
        "satisfaction_score",
        "trust_score",
        "Unbabel--XCOMET-XL"
    ]

    analysis_df = full_data[keep_cols].copy()

    analysis_df = analysis_df.rename(columns={
        "sender_unique_id": "participant_id",
        "conversation_id": "conv_id",
        "sender_gender": "gender",
    })

    for _col in ["gender", "translation_model", "group"]:
        analysis_df[_col] = pd.Categorical(analysis_df[_col])

    print(analysis_df.groupby(["translation_model", "group"]).size().rename("n_records"))
    print(analysis_df.columns.tolist())
    return (analysis_df,)


@app.cell
def _():
    import os
    os.environ["R_HOME"] = '/Users/gattanasio/miniconda3/envs/py310/lib/R'
    return


@app.cell
def _():
    from pymer4.models import lmer
    import pymer4
    import polars as pl
    pymer4.config.Rhome()
    return lmer, pl


@app.cell
def _(analysis_df, lmer, pl):
    def r_safe(s: str) -> str:
        return s.replace(" ", "_").replace("+", "plus")

    lmer_df = analysis_df.assign(
        group=analysis_df["group"].astype(str).map(r_safe),
        gender=analysis_df["gender"].astype(str),
        translation_model=analysis_df["translation_model"].astype(str).map(r_safe),
        conversation_topic=analysis_df["conversation_topic"].astype(str),
    )
    _lmer_df_polars = pl.from_pandas(lmer_df)

    lmer_result = lmer(
        "survey_score ~ group * gender + translation_model"
        " + baseline_survey_score"
        " + conversation_topic + conversation_source"
        " + (1 | validator_unique_id)"
        " + (1 | receiver_unique_id)"
        " + (1 | participant_id)"
        "+ (1 | conv_id)",
        data=_lmer_df_polars,
    )
    lmer_result.fit(REML=True, summarize=True)
    return lmer_df, lmer_result, r_safe


@app.cell
def _(lmer_result):
    lmer_result.summary()
    return


@app.cell
def _(lmer_result):
    # Export pymer4 fixed-effects table (paper-ready)
    # result_fit is a polars DataFrame; convert to pandas for easy export
    _lmer_coefs_export = (
        lmer_result.result_fit
        .to_pandas()
        .set_index("term")[["estimate", "std_error", "t_stat", "p_value", "conf_low", "conf_high"]]
        .rename(columns={
            "estimate": "Estimate",
            "std_error": "Std. Error",
            "t_stat": "t",
            "p_value": "p",
            "conf_low": "CI 2.5%",
            "conf_high": "CI 97.5%",
        })
        .round(4)
    )

    # Format index for better readability
    index_formatting_map = {
        "Intercept": "Intercept",
        "groupUS_Black": "Group: US\\textsubscript{B}",
        "groupUS_White": "Group: US\\textsubscript{W}",
        "genderwoman": "Gender: Woman",
        "conversation_topicHealth": "Topic: Health",
        "conversation_sourceGemini": "Source: Gemini",
        "conversation_sourcePatientPhysician": "Source: MED-MT",
        "conversation_sourceQwen3": "Source: Qwen 3",
        "translation_modelPhi_4": "Model: Phi 4",
        "translation_modelTowerplus": "Model: Tower+",
        "translation_modelVoxtral": "Model: Voxtral",
        "baseline_survey_score": "$u$ (Baseline)",
        "validator_translation_score": "Validator Translation Score",
        "qa_score": "QA Score",
        "groupUS_Black:genderwoman": "Group: US\\textsubscript{B} × Gender: Woman",
        "groupUS_White:genderwoman": "Group: US\\textsubscript{W} × Gender: Woman"
    }

    _lmer_coefs_export = _lmer_coefs_export.rename(index=index_formatting_map)
    _lmer_coefs_export.to_csv("lmer_coefs_export.csv")
    _lmer_coefs_export
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Table.** Fixed-effect estimates from a linear mixed-effects model (REML) predicting usability ($u$) across $N=1{,}738$ observations. The model includes group, gender, and their interaction as focal predictors, with conversation topic, translation model, baseline usability, validator translation score, and correction ratio as covariates. Random intercepts are included for participants and conversations. The reference category is Hindi men with DeSTA2 as the translation model on an Everyday conversation. CIs are 95\% Wald intervals.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Coeff plot
    """)
    return


@app.cell
def _(lmer_result, pd, plt):
    _rf = lmer_result.result_fit.to_pandas().set_index("term")

    _coef_df = pd.DataFrame({
        "coef":   _rf["estimate"],
        "ci_low": _rf["conf_low"],
        "ci_high": _rf["conf_high"],
        "pval":   _rf["p_value"],
    })

    _coef_df = _coef_df.drop(index="(Intercept)", errors="ignore")
    # keep only index with group, gender, conversation_topic, and translation_model effects
    _coef_df = _coef_df[
        _coef_df.index.str.contains("group|gender|conversation_topic|translation_model")
    ]
    _coef_df = _coef_df.sort_values("coef", ascending=True)

    _label_map = {
        "genderwoman":               "(W)",
        "groupUS_Black":             r"$\mathrm{US}_\mathrm{B}$",
        "groupUS_White":             r"$\mathrm{US}_\mathrm{W}$",
        "groupUS_Black:genderwoman": r"$\mathrm{US}_\mathrm{B}$ $\times$ (W)",
        "groupUS_White:genderwoman": r"$\mathrm{US}_\mathrm{W}$ $\times$ (W)",
        "conversation_topicHealth":            "Health",
        "conversation_sourceGemini":           "Source: Gemini",
        "conversation_sourcePatientPhysician": "Source: MED-MT",
        "conversation_sourceQwen3":            "Source: Qwen 3",
        "translation_modelPhi_4":              "Phi 4",
        "translation_modelTowerplus":          "Tower+",
        "translation_modelVoxtral":            "Voxtral",
        "baseline_survey_score":               r"$u$ (Baseline)",
    }
    _display_labels = [_label_map.get(idx, idx) for idx in _coef_df.index]

    fig_coef, ax = plt.subplots(figsize=(3.8, 3.15))
    _y = list(range(len(_coef_df)))

    # Plot each coefficient individually: filled circle if p < 0.05, diamond if 0.05 ≤ p < 0.1, hollow otherwise
    for _yi, (_, _row) in zip(_y, _coef_df.iterrows()):
        _p = _row["pval"]
        if 0.05 <= _p < 0.1:
            _fmt = "D"
            _mfc = "white"
        else:
            _fmt = "o"
            _mfc = "black" if _p < 0.05 else "white"
        ax.errorbar(
            _row["coef"], _yi,
            xerr=[[_row["coef"] - _row["ci_low"]], [_row["ci_high"] - _row["coef"]]],
            fmt=_fmt,
            capsize=4,
            color="black",
            ecolor="black",
            markerfacecolor=_mfc,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )
        ax.text(
            _row["ci_high"] + 0.03,
            _yi,
            f"{_row['coef']:.2f}",
            va="center",
            ha="left",
            fontsize=11,
        )

    ax.axvline(0, linestyle="--", color="gray", alpha=0.3)
    ax.set_yticks(_y)
    ax.set_xlim(-0.7, 1.2)
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    ax.set_yticklabels(_display_labels)
    ax.set_xlabel("")
    # ax.set_xlabel("Effect on Usability ($u$)")
    ax.margins(x=0.2)
    # Reference category box (placed outside axes; captured by bbox_inches="tight")
    # _ref_text = "Reference:\ngender: man\ngroup: Hindi"
    # ax.text(
    #     1.04, 0.5, _ref_text,
    #     transform=ax.transAxes,
    #     va="center", ha="left",
    #     fontsize=10,
    #     clip_on=False,
    #     bbox=dict(boxstyle="round,pad=0.45", facecolor="#fffde7", edgecolor="#9e9e9e", alpha=0.9),
    # )

    fig_coef.tight_layout()
    fig_coef.savefig("fig_coef_gender_group.pdf", dpi=300, bbox_inches="tight")
    fig_coef
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Fixed-effect coefficients for group and gender from a linear mixed-effects model predicting usability ($u$), with participant and conversation random effects. The reference category is Hindi men. Points show the estimated coefficient; horizontal bars show 95\% confidence intervals. Filled circles indicate $p < 0.05$ (uncorrected); diamonds indicate $0.05 \leq p < 0.1$; hollow circles are non-significant. Coefficient values are annotated to the right of each interval.
    """)
    return


@app.cell
def _(lmer_result, pd, plt):
    _rf_all = lmer_result.result_fit.to_pandas().set_index("term")

    _coef_all_df = pd.DataFrame({
        "coef":   _rf_all["estimate"],
        "ci_low": _rf_all["conf_low"],
        "ci_high": _rf_all["conf_high"],
        "pval":   _rf_all["p_value"],
    })

    _coef_all_df = _coef_all_df.drop(index="(Intercept)", errors="ignore")
    _coef_all_df = _coef_all_df.sort_values("coef", ascending=True)

    _all_label_map = {
        # Main effects
        "groupUS_Black":                       r"$\mathregular{US_B}$",
        "groupUS_White":                       r"$\mathregular{US_W}$",
        "genderwoman":                         "Woman",
        "conversation_topicHealth":            "Topic: Health",
        "conversation_sourceGemini":           "Source: Gemini",
        "conversation_sourcePatientPhysician": "Source: MED-MT",
        "conversation_sourceQwen3":            "Source: Qwen 3",
        "translation_modelPhi_4":              "Model: Phi 4",
        "translation_modelTowerplus":          "Model: Tower+",
        "translation_modelVoxtral":            "Model: Voxtral",
        "baseline_survey_score":               r"$u$ (Baseline)",
        # Interactions
        "groupUS_Black:genderwoman":           r"$\mathregular{US_B} \times$ Woman",
        "groupUS_White:genderwoman":           r"$\mathregular{US_W} \times$ Woman",
    }
    _all_display_labels = [_all_label_map.get(idx, idx) for idx in _coef_all_df.index]

    fig_coef_all, ax_all = plt.subplots(figsize=(5.5, 0.42 * len(_coef_all_df)))
    _y_all = list(range(len(_coef_all_df)))

    for _yi, (_, _row) in zip(_y_all, _coef_all_df.iterrows()):
        _p = _row["pval"]
        if 0.05 <= _p < 0.1:
            _fmt = "D"
            _mfc = "white"
        else:
            _fmt = "o"
            _mfc = "black" if _p < 0.05 else "white"
        ax_all.errorbar(
            _row["coef"], _yi,
            xerr=[[_row["coef"] - _row["ci_low"]], [_row["ci_high"] - _row["coef"]]],
            fmt=_fmt,
            capsize=4,
            color="black",
            ecolor="black",
            markerfacecolor=_mfc,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )
        ax_all.text(
            _row["ci_high"] + 0.04,
            _yi,
            f"{_row['coef']:.2f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax_all.axvline(0, linestyle="--", color="gray", alpha=0.6)
    ax_all.set_yticks(_y_all)
    ax_all.set_yticklabels(_all_display_labels)
    ax_all.set_xlabel("Effect on Usability ($u$)")
    ax_all.margins(x=0.25)

    fig_coef_all.tight_layout()
    fig_coef_all.savefig("fig_coef_all.pdf", dpi=300, bbox_inches="tight")
    fig_coef_all
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Fixed-effect coefficients from the linear mixed-effects model predicting usability ($u$). Reference categories: Hindi men, DeSTA2 model, Everyday conversations, PatientPhysician source. Points show the estimated coefficient with 95\% confidence intervals. Filled circles indicate $p < 0.05$ (uncorrected); diamonds indicate $0.05 \leq p < 0.1$; hollow circles are non-significant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Group x Gender Differences Usability
    """)
    return


@app.cell
def _(analysis_df, itertools, lmer_df, lmer_result, np, pd, r_safe):
    # Point 1: model-adjusted cell means for the 6 group × gender combinations
    # Builds a manual R treatment-coding design vector from lmer_result (no patsy dependency).
    # Continuous covariates are marginalised at their grand mean.
    # Reference levels: group=Hindi, gender=man, topic=Everyday, model=DeSTA2.

    _rf_emm = lmer_result.result_fit.to_pandas().set_index("term")
    _fe_names_emm = list(_rf_emm.index)
    _beta_emm = _rf_emm["estimate"].to_numpy()
    _fe_idx_emm = {t: i for i, t in enumerate(_fe_names_emm)}

    # Continuous covariate grand means
    _baseline_mean_emm = float(lmer_df["baseline_survey_score"].mean())
    _vts_mean_emm = float(lmer_df["validator_translation_score"].mean())
    _cr_mean_emm = float(lmer_df["correction_ratio"].mean())

    # Observed marginal distributions (use lmer_df which carries R-safe names)
    topic_weights = (
        lmer_df["conversation_topic"]
        .value_counts(normalize=True)
        .sort_index()
    )
    _model_weights_emm = (
        lmer_df["translation_model"]
        .value_counts(normalize=True)
        .sort_index()
    )

    group_levels = list(analysis_df["group"].cat.categories)   # display names
    gender_levels = list(analysis_df["gender"].cat.categories)
    topic_levels = list(topic_weights.index)
    _r_model_levels_emm = list(_model_weights_emm.index)        # R-safe model names

    # Exported: used by the contrasts cell (Point 2) and downstream cells
    lmer_fe_idx = _fe_idx_emm


    def lmer_design_vec(group_r, gender_r, topic, model_r,
                        fe_idx, baseline_mean, vts_mean, cr_mean):
        """Build a fixed-effect design vector using R treatment coding.

        group_r and model_r must be R-safe (spaces→_, +→plus).
        Reference levels: Hindi, man, Everyday, DeSTA2.
        """
        v = np.zeros(len(fe_idx))
        if "(Intercept)" in fe_idx:
            v[fe_idx["(Intercept)"]] = 1.0
        if group_r == "US_Black" and "groupUS_Black" in fe_idx:
            v[fe_idx["groupUS_Black"]] = 1.0
        if group_r == "US_White" and "groupUS_White" in fe_idx:
            v[fe_idx["groupUS_White"]] = 1.0
        if gender_r == "woman" and "genderwoman" in fe_idx:
            v[fe_idx["genderwoman"]] = 1.0
        if topic == "Health" and "conversation_topicHealth" in fe_idx:
            v[fe_idx["conversation_topicHealth"]] = 1.0
        tm_key = f"translation_model{model_r}"
        if tm_key in fe_idx:
            v[fe_idx[tm_key]] = 1.0
        if "baseline_survey_score" in fe_idx:
            v[fe_idx["baseline_survey_score"]] = baseline_mean
        if "validator_translation_score" in fe_idx:
            v[fe_idx["validator_translation_score"]] = vts_mean
        if "correction_ratio" in fe_idx:
            v[fe_idx["correction_ratio"]] = cr_mean
        if group_r == "US_Black" and gender_r == "woman" and "groupUS_Black:genderwoman" in fe_idx:
            v[fe_idx["groupUS_Black:genderwoman"]] = 1.0
        if group_r == "US_White" and gender_r == "woman" and "groupUS_White:genderwoman" in fe_idx:
            v[fe_idx["groupUS_White:genderwoman"]] = 1.0
        return v


    # Build prediction grid: all group × gender × topic × model combinations
    _r_group_levels_emm = [r_safe(g) for g in group_levels]
    pred_grid = pd.DataFrame(
        list(itertools.product(_r_group_levels_emm, gender_levels, topic_levels, _r_model_levels_emm)),
        columns=["group_r", "gender", "conversation_topic", "translation_model_r"],
    )

    pred_grid["pred"] = [
        float(lmer_design_vec(
            row["group_r"], row["gender"], row["conversation_topic"], row["translation_model_r"],
            _fe_idx_emm, _baseline_mean_emm, _vts_mean_emm, _cr_mean_emm,
        ) @ _beta_emm)
        for _, row in pred_grid.iterrows()
    ]

    # Weight by observed marginal frequencies (topic ⊥ model assumed for marginalization)
    pred_grid["topic_weight"] = pred_grid["conversation_topic"].map(topic_weights).astype(float)
    pred_grid["model_weight"] = pred_grid["translation_model_r"].map(_model_weights_emm).astype(float)
    pred_grid["weighted_pred"] = pred_grid["pred"] * pred_grid["topic_weight"] * pred_grid["model_weight"]

    # Map R-safe group names back to display names for the output table
    _r_to_disp_group = {r_safe(g): g for g in group_levels}
    pred_grid["group"] = pred_grid["group_r"].map(_r_to_disp_group)

    emm_table = (
        pred_grid.groupby(["group", "gender"], observed=True)
        .agg(estimated_mean=("weighted_pred", "sum"))
        .reset_index()
        .sort_values("estimated_mean", ascending=False)
        .reset_index(drop=True)
    )

    emm_table["rank"] = emm_table.index + 1
    emm_table["cell"] = (
        emm_table["group"].astype(str) + " × " + emm_table["gender"].astype(str)
    )

    emm_table[["rank", "cell", "estimated_mean"]]
    return (
        gender_levels,
        group_levels,
        lmer_design_vec,
        lmer_fe_idx,
        topic_levels,
        topic_weights,
    )


@app.cell
def _(
    lmer_design_vec,
    lmer_df,
    lmer_result,
    multipletests,
    norm,
    np,
    pd,
    r_safe,
    topic_levels,
    topic_weights,
):
    # Point 2: simple contrasts for the 3-group × 2-gender design, with Holm correction
    import rpy2.robjects as _ro

    _rf_cont = lmer_result.result_fit.to_pandas().set_index("term")
    _fe_names_cont = list(_rf_cont.index)
    _fe_idx_cont = {t: i for i, t in enumerate(_fe_names_cont)}
    beta = _rf_cont["estimate"].to_numpy()

    # Variance-covariance matrix of fixed effects from R's vcov()
    # lme4::vcov() returns a dpoMatrix (Matrix package); as.matrix() converts it to
    # a plain base-R matrix so that rpy2 / np.array() can expand it correctly.
    _vcov_r = _ro.r["as.matrix"](_ro.r["vcov"](lmer_result.r_model))
    cov_fe = np.array(_vcov_r).reshape(len(beta), len(beta))

    # Continuous covariate grand means
    _baseline_mean_cont = float(lmer_df["baseline_survey_score"].mean())
    _vts_mean_cont = float(lmer_df["validator_translation_score"].mean())
    _cr_mean_cont = float(lmer_df["correction_ratio"].mean())

    _model_weights_d = (
        lmer_df["translation_model"]
        .value_counts(normalize=True)
        .sort_index()
    )
    _model_levels_d = list(_model_weights_d.index)  # R-safe names


    def _cell_exog(group, gender):
        """Weighted-average fixed-effect vector for a group × gender cell (display names)."""
        group_r = r_safe(group)
        cell_vec = np.zeros(len(_fe_names_cont))
        for _topic in topic_levels:
            for _model_r in _model_levels_d:
                v = lmer_design_vec(
                    group_r, gender, _topic, _model_r,
                    _fe_idx_cont, _baseline_mean_cont, _vts_mean_cont, _cr_mean_cont,
                )
                w = float(topic_weights.get(_topic, 0.0)) * float(_model_weights_d.get(_model_r, 0.0))
                cell_vec += v * w
        return cell_vec


    def _contrast_result(label, left_group, left_gender, right_group, right_gender):
        left = _cell_exog(left_group, left_gender)
        right = _cell_exog(right_group, right_gender)
        contrast = left - right

        estimate = float(contrast @ beta)
        se = float(np.sqrt(contrast @ cov_fe @ contrast))
        z_value = estimate / se
        p_value = 2 * norm.sf(abs(z_value))
        ci_low = estimate - 1.96 * se
        ci_high = estimate + 1.96 * se

        return {
            "contrast": label,
            "estimate": estimate,
            "std_error": se,
            "z": z_value,
            "p_uncorrected": p_value,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }


    contrast_results = pd.DataFrame([
        # Gender gap within each group
        _contrast_result(
            "Women - men within Hindi",
            "Hindi", "woman",
            "Hindi", "man",
        ),
        _contrast_result(
            "Women - men within US Black",
            "US Black", "woman",
            "US Black", "man",
        ),
        _contrast_result(
            "Women - men within US White",
            "US White", "woman",
            "US White", "man",
        ),
        # Group differences within men
        _contrast_result(
            "US Black - Hindi within men",
            "US Black", "man",
            "Hindi", "man",
        ),
        _contrast_result(
            "US White - Hindi within men",
            "US White", "man",
            "Hindi", "man",
        ),
        _contrast_result(
            "US White - US Black within men",
            "US White", "man",
            "US Black", "man",
        ),
        # Group differences within women
        _contrast_result(
            "US Black - Hindi within women",
            "US Black", "woman",
            "Hindi", "woman",
        ),
        _contrast_result(
            "US White - Hindi within women",
            "US White", "woman",
            "Hindi", "woman",
        ),
        _contrast_result(
            "US White - US Black within women",
            "US White", "woman",
            "US Black", "woman",
        ),
    ])

    reject, p_holm, _, _ = multipletests(
        contrast_results["p_uncorrected"],
        method="holm",
        alpha=0.05,
    )

    contrast_results["p_holm"] = p_holm
    contrast_results["reject_holm_0_05"] = reject

    contrast_results = contrast_results[
        [
            "contrast",
            "estimate",
            "std_error",
            "z",
            "p_uncorrected",
            "p_holm",
            "ci_low",
            "ci_high",
            "reject_holm_0_05",
        ]
    ].round(4)

    contrast_results
    return beta, cov_fe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interaction Plot: Usability vs. Group, Gender
    """)
    return


@app.cell
def _(
    beta,
    cov_fe,
    gender_levels,
    group_levels,
    lmer_design_vec,
    lmer_df,
    lmer_fe_idx,
    np,
    pd,
    plt,
    r_safe,
    topic_levels,
    topic_weights,
):
    # Point 3: interaction plot with model-adjusted means and 95% CIs

    _model_weights_p3 = (
        lmer_df["translation_model"]
        .value_counts(normalize=True)
        .sort_index()
    )
    _model_levels_p3 = list(_model_weights_p3.index)  # R-safe names

    _baseline_mean_p3 = float(lmer_df["baseline_survey_score"].mean())
    _vts_mean_p3 = float(lmer_df["validator_translation_score"].mean())
    _cr_mean_p3 = float(lmer_df["correction_ratio"].mean())

    plot_rows = []

    for group in group_levels:
        for gender in gender_levels:
            group_r = r_safe(group)
            cell_vec = np.zeros(len(beta))
            for _topic in topic_levels:
                for _model_r in _model_levels_p3:
                    v = lmer_design_vec(
                        group_r, gender, _topic, _model_r,
                        lmer_fe_idx, _baseline_mean_p3, _vts_mean_p3, _cr_mean_p3,
                    )
                    w = float(topic_weights.get(_topic, 0.0)) * float(_model_weights_p3.get(_model_r, 0.0))
                    cell_vec += v * w

            estimate = float(cell_vec @ beta)
            std_error = float(np.sqrt(cell_vec @ cov_fe @ cell_vec))

            plot_rows.append({
                "group": group,
                "gender": gender,
                "estimated_mean": estimate,
                "std_error": std_error,
                "ci_low": estimate - 1.96 * std_error,
                "ci_high": estimate + 1.96 * std_error,
            })

    plot_table = pd.DataFrame(plot_rows)
    plot_table["group"] = pd.Categorical(
        plot_table["group"],
        categories=list(reversed(group_levels)),
        ordered=True,
    )
    plot_table["gender"] = pd.Categorical(
        plot_table["gender"],
        categories=gender_levels,
        ordered=True,
    )

    _gender_style = {
        "man":   {"marker": "s", "offset": -0.08, "ha": "right"},
        "woman": {"marker": "^", "offset":  0.08, "ha": "left"},
    }

    _group_order_int = [str(g) for g in reversed(group_levels)]
    _group_label_map = {
        "Hindi":    "Hindi",
        "US Black": r"$\mathrm{US}_\mathrm{B}$",
        "US White": r"$\mathrm{US}_\mathrm{W}$",
    }
    _group_tick_labels = [_group_label_map.get(g, g) for g in _group_order_int]
    _x_base_int = np.arange(len(_group_order_int))

    palette = {
        str(gender_levels[0]): "#355070",
        str(gender_levels[1]): "#E56B6F",
    }

    fig_interaction, ax_int = plt.subplots(figsize=(3.5, 3.15))

    for gender in gender_levels:
        curr = (
            plot_table[plot_table["gender"] == gender]
            .sort_values("group")
            .reset_index(drop=True)
        )
        style = _gender_style[str(gender)]
        x_pos = _x_base_int + style["offset"]
        ax_int.errorbar(
            x_pos,
            curr["estimated_mean"],
            yerr=1.96 * curr["std_error"],
            marker=style["marker"],
            markersize=9,
            linewidth=2,
            linestyle=":",
            capsize=4,
            label=str(gender),
            color=palette[str(gender)],
        )
        _x_sign = -1 if gender == "man" else 1
        _y_sign = -1 if gender == "man" else 1
        _va = "top" if gender == "man" else "bottom"
        for _xi, (_mv, _ci) in enumerate(zip(curr["estimated_mean"], 1.96 * curr["std_error"])):
            ax_int.text(
                x_pos[_xi] + _x_sign * 0.1, _mv + _y_sign * 0.05,
                f"{_mv:.2f}",
                ha=style["ha"], va=_va, fontsize=13, color=palette[str(gender)],
            )

    ax_int.set_xticks(_x_base_int)
    ax_int.set_xticklabels(_group_tick_labels)
    ax_int.tick_params(axis="y")
    ax_int.set_xlabel("")
    ax_int.set_ylabel("")
    ax_int.text(0.6, 0.95, "Usability ($u$)", transform=ax_int.transAxes,
                va="top", ha="left", fontsize=13)
    ax_int.legend(title=None, loc="best")
    ax_int.margins(x=0.3)

    fig_interaction.tight_layout()
    fig_interaction.savefig("fig_interaction_group_gender.pdf", dpi=300, bbox_inches="tight")
    fig_interaction
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Estimated marginal means (±95% CI) of usability ($u$) by language group and gender, averaged over conversation topic and translation model. Circles = man; diamonds = woman.
    """)
    return


@app.cell
def _(
    beta,
    cov_fe,
    itertools,
    lmer_design_vec,
    lmer_df,
    lmer_fe_idx,
    multipletests,
    norm,
    np,
    pd,
    r_safe,
    topic_levels,
    topic_weights,
):
    # Point 4: all 15 pairwise comparisons among the 6 intersectional cells
    # (3 groups × 2 genders); Holm correction across all 15 tests
    # marimo-safe: localize temporary names inside a function

    _model_weights_p4 = (
        lmer_df["translation_model"]
        .value_counts(normalize=True)
        .sort_index()
    )
    _model_levels_p4 = list(_model_weights_p4.index)  # R-safe names

    _baseline_mean_p4 = float(lmer_df["baseline_survey_score"].mean())
    _vts_mean_p4 = float(lmer_df["validator_translation_score"].mean())
    _cr_mean_p4 = float(lmer_df["correction_ratio"].mean())


    def build_pairwise_results_table():
        pairwise_cell_specs = [
            ("Hindi", "man"),
            ("Hindi", "woman"),
            ("US Black", "man"),
            ("US Black", "woman"),
            ("US White", "man"),
            ("US White", "woman"),
        ]

        def pairwise_cell_vector(group, gender):
            group_r = r_safe(group)
            cell_vec = np.zeros(len(beta))
            for _topic in topic_levels:
                for _model_r in _model_levels_p4:
                    v = lmer_design_vec(
                        group_r, gender, _topic, _model_r,
                        lmer_fe_idx, _baseline_mean_p4, _vts_mean_p4, _cr_mean_p4,
                    )
                    w = float(topic_weights.get(_topic, 0.0)) * float(_model_weights_p4.get(_model_r, 0.0))
                    cell_vec += v * w
            return cell_vec

        pairwise_cell_vectors = {
            f"{group} × {gender}": pairwise_cell_vector(group, gender)
            for group, gender in pairwise_cell_specs
        }

        pairwise_rows_local = []

        for pairwise_left_label, pairwise_right_label in itertools.combinations(pairwise_cell_vectors.keys(), 2):
            pairwise_left_vec = pairwise_cell_vectors[pairwise_left_label]
            pairwise_right_vec = pairwise_cell_vectors[pairwise_right_label]
            pairwise_contrast_vec = pairwise_left_vec - pairwise_right_vec

            pairwise_estimate = float(pairwise_contrast_vec @ beta)
            pairwise_std_error = float(np.sqrt(pairwise_contrast_vec @ cov_fe @ pairwise_contrast_vec))
            pairwise_z_value = pairwise_estimate / pairwise_std_error
            pairwise_p_value = 2 * norm.sf(abs(pairwise_z_value))
            pairwise_ci_low = pairwise_estimate - 1.96 * pairwise_std_error
            pairwise_ci_high = pairwise_estimate + 1.96 * pairwise_std_error

            pairwise_rows_local.append({
                "contrast": f"{pairwise_left_label} - {pairwise_right_label}",
                "estimate": pairwise_estimate,
                "std_error": pairwise_std_error,
                "z": pairwise_z_value,
                "p_uncorrected": pairwise_p_value,
                "ci_low": pairwise_ci_low,
                "ci_high": pairwise_ci_high,
            })

        pairwise_results_local = pd.DataFrame(pairwise_rows_local)

        pairwise_reject, pairwise_p_holm, _, _ = multipletests(
            pairwise_results_local["p_uncorrected"],
            method="holm",
            alpha=0.05,
        )

        pairwise_results_local["p_holm"] = pairwise_p_holm
        pairwise_results_local["reject_holm_0_05"] = pairwise_reject

        return (
            pairwise_results_local[
                [
                    "contrast",
                    "estimate",
                    "std_error",
                    "z",
                    "p_uncorrected",
                    "p_holm",
                    "ci_low",
                    "ci_high",
                    "reject_holm_0_05",
                ]
            ]
            .sort_values("p_holm")
            .reset_index(drop=True)
            .round(4)
        )

    pairwise_results = build_pairwise_results_table()
    pairwise_results
    return (pairwise_results,)


@app.cell
def _(np, pairwise_results, plt):
    # Pairwise contrast forest plot
    # Uses the pairwise_results table already computed

    p4_group_abbrev = {
        "US Black": r"$\mathrm{US}_\mathrm{B}$",
        "US White": r"$\mathrm{US}_\mathrm{W}$",
        "Hindi":    "H",
    }
    p4_gender_abbrev = {
        "man": "(M)",
        "woman": "(W)",
    }


    def fmt_p4_label(raw_label):
        """Convert 'Group × gender - Group × gender' to abbreviated LaTeX label."""
        left, right = raw_label.split(" - ", 1)

        def _fmt_cell(cell):
            group, gender = cell.split(" × ", 1)
            return f"{p4_group_abbrev.get(group, group)} {p4_gender_abbrev.get(gender, gender)}"

        return f"{_fmt_cell(left)} − {_fmt_cell(right)}"


    forest_plot_df = pairwise_results.copy()
    forest_plot_df = forest_plot_df.sort_values("estimate", ascending=True).reset_index(drop=True)

    forest_fig, forest_ax = plt.subplots(
        figsize=(8.0, max(3.6, 0.55 * len(forest_plot_df)))
    )

    forest_y = np.arange(len(forest_plot_df))

    for forest_idx, (_, forest_row) in enumerate(forest_plot_df.iterrows()):
        _p = forest_row["p_uncorrected"]
        if 0.05 <= _p < 0.1:
            _fmt = "D"
            _mfc = "white"
        else:
            _fmt = "o"
            _mfc = "black" if _p < 0.05 else "white"
        forest_ax.errorbar(
            forest_row["estimate"],
            forest_idx,
            xerr=[
                [forest_row["estimate"] - forest_row["ci_low"]],
                [forest_row["ci_high"] - forest_row["estimate"]],
            ],
            fmt=_fmt,
            color="black",
            ecolor="black",
            elinewidth=2,
            capsize=4,
            markersize=6,
            markerfacecolor=_mfc,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=2,
        )

    _p4_tick_labels = [fmt_p4_label(c) for c in forest_plot_df["contrast"]]

    forest_ax.axvline(0, linestyle="--", color="gray", alpha=0.7)
    forest_ax.set_yticks(forest_y)
    forest_ax.set_yticklabels(_p4_tick_labels)
    forest_ax.set_xlabel("Pairwise difference in usability ($u$)")
    forest_ax.set_title(
        "Intersectional pairwise contrasts\n95% CI, Holm-corrected significance"
    )

    forest_fig.tight_layout()
    forest_fig.savefig("fig_forest_pairwise_contrasts.pdf", dpi=300, bbox_inches="tight")
    forest_fig
    return fmt_p4_label, forest_plot_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Forest plot of all 15 pairwise contrasts in model-adjusted usability ($u$) among the six intersectional cells ($3~\text{groups} \times 2~\text{genders}$), estimated via a linear mixed-effects model with participant and conversation random effects. Points show the estimated difference (left cell minus right cell); horizontal bars show 95\% confidence intervals. Filled circles indicate contrasts that are statistically significant after Holm correction ($p_\text{Holm} < 0.05$); diamonds indicate $0.05 \leq p_\text{uncorrected} < 0.1$; hollow circles are non-significant. Contrasts are sorted by effect size in ascending order.
    """)
    return


@app.cell
def _(fmt_p4_label, forest_plot_df, np, plt):
    # Filtered forest plot — significant contrasts only, with estimate labels
    # Reuses forest_plot_df and fmt_p4_label from the cell above

    _sig_df = forest_plot_df[forest_plot_df["reject_holm_0_05"]].reset_index(drop=True)
    _n_sig = len(_sig_df)

    # y positions: bottom entries unchanged; top entry shifted up by 1 to make room
    # for "..." between the first and second entry from the top
    _sig_y = np.arange(_n_sig, dtype=float)
    _sig_y[-1] += 1.0          # top entry at y = _n_sig (was _n_sig-1)
    _dots_y = float(_n_sig) - 1.0   # "..." at midpoint between second-from-top and top

    _sig_fig, _sig_ax = plt.subplots(
        figsize=(5.5, max(1.5, 0.38 * (_n_sig + 1.5)))
    )

    for _si, (_, _sr) in enumerate(_sig_df.iterrows()):
        _p = _sr["p_uncorrected"]
        if 0.05 <= _p < 0.1:
            _fmt = "D"
            _mfc = "white"
        else:
            _fmt = "o"
            _mfc = "black" if _p < 0.05 else "white"
        _sig_ax.errorbar(
            _sr["estimate"],
            _sig_y[_si],
            xerr=[
                [_sr["estimate"] - _sr["ci_low"]],
                [_sr["ci_high"] - _sr["estimate"]],
            ],
            fmt=_fmt,
            color="black",
            ecolor="black",
            elinewidth=2,
            capsize=4,
            markersize=6,
            markerfacecolor=_mfc,
            markeredgecolor="black",
            markeredgewidth=1.5,
            zorder=2,
        )
        _sig_ax.text(
            _sr["ci_high"] + 0.02,
            _sig_y[_si],
            f"{_sr['estimate']:.2f}",
            va="center",
            ha="left",
            fontsize=11,
        )

    # Insert "..." between second-from-top and top in both labels and tick positions
    _sig_tick_labels = (
        [fmt_p4_label(c) for c in _sig_df["contrast"].iloc[:-1]]
        + ["..."]
        + [fmt_p4_label(_sig_df["contrast"].iloc[-1])]
    )
    _sig_tick_pos = list(_sig_y[:-1]) + [_dots_y, _sig_y[-1]]

    _sig_ax.axvline(0, linestyle="--", color="gray", alpha=0.3)
    _sig_ax.set_yticks(_sig_tick_pos)
    _sig_ax.set_yticklabels(_sig_tick_labels)
    _sig_ax.set_ylim(-0.6, _sig_y[-1] + 0.5)
    _sig_ax.set_xlabel("Pairwise difference in usability ($u$)")
    _sig_ax.margins(x=0.2)

    _sig_fig.tight_layout()
    _sig_fig.savefig("fig_pairwise_sig_forest.pdf", dpi=300, bbox_inches="tight")
    _sig_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Significant pairwise contrasts in model-adjusted usability ($u$) among the six intersectional cells ($3~\text{groups} \times 2~\text{genders}$). Only the $p_\text{Holm} < 0.05$ contrasts are shown; ``$\ldots$'' indicates omitted non-significant comparisons. Points show the estimated difference with 95\% confidence intervals; numerical estimates are annotated to the right of each interval.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Caption for the merged figure:

    **Figure.** Results from a linear mixed-effects model predicting usability ($u$), with random intercepts for participants and conversations. All estimates use Hindi men as the reference category.
    **(a)** Fixed-effect coefficients for group and gender. Points show the estimate with 95\% confidence intervals; filled circles indicate $p < 0.05$ (uncorrected), diamonds indicate $0.05 \leq p < 0.1$, hollow circles are non-significant; values are annotated to the right of each interval.
    **(b)** Estimated marginal means ($\pm$95\% CI) by language group and gender, averaged over conversation topic and translation model. Circles~=~man; diamonds~=~woman.
    **(c)** Significant pairwise contrasts in model-adjusted $u$ among the six intersectional cells ($3~\text{groups} \times 2~\text{genders}$); only $p_\text{Holm} < 0.05$ contrasts are shown ($\ldots$ indicates omitted non-significant comparisons). Points show the estimated difference with 95\% confidence intervals; numerical estimates are annotated to the right of each interval.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Automatic Translation Quality (XCOMET)

    We refit the same LMER structure with the XCOMET-XL automatic quality score
    as the dependent variable.
    """)
    return


@app.cell
def _(analysis_df, lmer, pl):
    def _r_safe(s: str) -> str:
        return s.replace(" ", "_").replace("+", "plus")

    _xcomet_df = analysis_df.rename(columns={"Unbabel--XCOMET-XL": "xcomet_score"}).assign(
        group=analysis_df["group"].astype(str).map(_r_safe),
        gender=analysis_df["gender"].astype(str),
        translation_model=analysis_df["translation_model"].astype(str).map(_r_safe),
        conversation_topic=analysis_df["conversation_topic"].astype(str),
    )
    _xcomet_polars = pl.from_pandas(_xcomet_df)

    _xcomet_formula = (
        "xcomet_score ~ group * gender + conversation_topic"
        " + translation_model"
        " + conversation_source"
        " + (1 | validator_unique_id)"
        " + (1 | receiver_unique_id)"
        " + (1 | participant_id) + (1 | conv_id)"
    )

    xcomet_lmer = lmer(_xcomet_formula, data=_xcomet_polars)
    xcomet_lmer.fit(REML=True, summarize=True)
    xcomet_lmer.summary()
    return (xcomet_lmer,)


@app.cell
def _(pd, plt, xcomet_lmer):
    _rf_xc = xcomet_lmer.result_fit.to_pandas().set_index("term")

    _coef_xc = pd.DataFrame({
        "coef":   _rf_xc["estimate"],
        "ci_low": _rf_xc["conf_low"],
        "ci_high": _rf_xc["conf_high"],
        "pval":   _rf_xc["p_value"],
    })

    _coef_xc = _coef_xc.drop(index="(Intercept)", errors="ignore")
    _coef_xc = _coef_xc.sort_values("coef", ascending=True)

    _xc_label_map = {
        "groupUS_Black":                       r"$\mathregular{US_B}$",
        "groupUS_White":                       r"$\mathregular{US_W}$",
        "genderwoman":                         "Woman",
        "conversation_topicHealth":            "Topic: Health",
        "conversation_sourceGemini":           "Source: Gemini",
        "conversation_sourcePatientPhysician": "Source: MED-MT",
        "conversation_sourceQwen3":            "Source: Qwen 3",
        "translation_modelPhi_4":              "Model: Phi 4",
        "translation_modelTowerplus":          "Model: Tower+",
        "translation_modelVoxtral":            "Model: Voxtral",
        "baseline_survey_score":               r"$u$ (Baseline)",
        "groupUS_Black:genderwoman":           r"$\mathregular{US_B} \times$ Woman",
        "groupUS_White:genderwoman":           r"$\mathregular{US_W} \times$ Woman",
    }
    _xc_labels = [_xc_label_map.get(idx, idx) for idx in _coef_xc.index]

    fig_xc, ax_xc = plt.subplots(figsize=(5.5, 0.42 * len(_coef_xc)))
    _y_xc = list(range(len(_coef_xc)))

    for _yi, (_, _row) in zip(_y_xc, _coef_xc.iterrows()):
        _p = _row["pval"]
        if 0.05 <= _p < 0.1:
            _fmt, _mfc = "D", "white"
        else:
            _fmt, _mfc = "o", "black" if _p < 0.05 else "white"
        ax_xc.errorbar(
            _row["coef"], _yi,
            xerr=[[_row["coef"] - _row["ci_low"]], [_row["ci_high"] - _row["coef"]]],
            fmt=_fmt,
            capsize=4,
            color="black",
            ecolor="black",
            markerfacecolor=_mfc,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )
        ax_xc.text(
            _row["ci_high"] + 0.04,
            _yi,
            f"{_row['coef']:.3f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax_xc.axvline(0, linestyle="--", color="gray", alpha=0.6)
    ax_xc.set_yticks(_y_xc)
    ax_xc.set_yticklabels(_xc_labels)
    ax_xc.set_xlabel("Effect on XCOMET Score")
    ax_xc.margins(x=0.25)

    fig_xc.tight_layout()
    fig_xc.savefig("fig_coef_xcomet.pdf", dpi=300, bbox_inches="tight")
    fig_xc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Figure.** Fixed-effect coefficients from the linear mixed-effects model predicting XCOMET-XL automatic translation quality score. Reference categories: Hindi men, DeSTA2 model, Everyday conversations, PatientPhysician source. Filled circles: $p < 0.05$; diamonds: $0.05 \leq p < 0.1$; hollow circles: non-significant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Other Analyses
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Survey scores

    In this section of the notebook we study `survey_score`.
    """)
    return


@app.cell
def _(full_data, pd, plt, sns):
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(8.5, 4.0))

    def _survey_score_by_translation_model(data: pd.DataFrame, ax: plt.Axes):
        sns.boxplot(data=full_data, x="translation_model", y="survey_score", ax=ax)
        ax.set_xlabel("")
        return ax

    def _survey_score_by_group(data: pd.DataFrame, ax: plt.Axes):
        sns.violinplot(
            data=full_data,
            x="group",
            y="survey_score",
            ax=ax,
            hue="sender_gender",
            split=True,
            gap=.1,
            inner="quart"
        )
        ax.set_xlabel("")
        return ax

    # Chart 1.
    _survey_score_by_translation_model(full_data, axes[0])

    # Chart 2.
    _survey_score_by_group(full_data, axes[1])

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend_.remove()

    axes[1].legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.25, 0.98, 0.73, 0.12),
        ncol=2,
        mode="expand",
        title=None,
        frameon=False,
        borderaxespad=0,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    return


@app.cell
def _(full_data):
    # Compute median, mean, and std of survey_score by group and gender
    stats = (
        full_data
        .groupby(['group', 'sender_gender'])['survey_score']
        .agg(['mean', 'median', 'std'])
        .reset_index()
    )
    stats
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Quality Metrics and Usability

    Which quality metrics best predict perceived usability? We compare human
    judgments (validator score, correction ratio) against automatic metrics
    (XCOMET, COMET variants, MetricX), both globally and across quality regimes.
    """)
    return


@app.cell
def _():
    METRIC_LABELS = {
        "survey_score": "Usability ($u$)",
        "validator_translation_score": "Translation Score",
        "qa_score": "QA Score",
        "Unbabel--wmt23-cometkiwi-da-xl": "COMET Kiwi XL",
        "Unbabel--wmt22-cometkiwi-da": "COMET Kiwi",
        "Unbabel--wmt22-comet-da": "COMET",
        "Unbabel--XCOMET-XL": "XCOMET XL",
        "google--metricx-24-hybrid-xl-v2p6": "MetricX 24",
    }
    # R-safe versions for pymer4 cells
    METRIC_LABELS_SAFE = {
        **METRIC_LABELS,
        "Unbabel_XX_wmt23_cometkiwi_da_xl": "COMET Kiwi XL",
        "Unbabel_XX_wmt22_cometkiwi_da": "COMET Kiwi",
        "Unbabel_XX_wmt22_comet_da": "COMET",
        "Unbabel_XX_XCOMET_XL": "XCOMET XL",
        "google_XX_metricx_24_hybrid_xl_v2p6": "MetricX 24",
    }
    QUALITY_METRICS = [
        "validator_translation_score",
        "qa_score",
        "Unbabel--wmt23-cometkiwi-da-xl",
        "Unbabel--wmt22-cometkiwi-da",
        "Unbabel--wmt22-comet-da",
        "Unbabel--XCOMET-XL",
        "google--metricx-24-hybrid-xl-v2p6",
    ]
    return METRIC_LABELS, METRIC_LABELS_SAFE, QUALITY_METRICS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Correlations
    """)
    return


@app.cell
def _(METRIC_LABELS, full_data, plt):
    # Correlation matrix of quality metrics and survey_score
    full_data["qa_score"] = 1 - full_data["correction_ratio"]
    _corr_cols = [
        "survey_score",
        "validator_translation_score",
        "qa_score",
        "Unbabel--wmt23-cometkiwi-da-xl",
        "Unbabel--wmt22-cometkiwi-da",
        "Unbabel--wmt22-comet-da",
        "Unbabel--XCOMET-XL",
        "google--metricx-24-hybrid-xl-v2p6",
    ]
    _corr_df = full_data[_corr_cols].dropna()
    _corr_matrix = _corr_df.corr(method="spearman")
    _labels = [METRIC_LABELS.get(c, c) for c in _corr_cols]

    _fig_corr, _ax_corr = plt.subplots(figsize=(7.5, 6))
    _im = _ax_corr.imshow(_corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    _ax_corr.set_xticks(range(len(_labels)))
    _ax_corr.set_yticks(range(len(_labels)))
    _ax_corr.set_xticklabels(_labels, rotation=45, ha="right")
    _ax_corr.set_yticklabels(_labels)
    for _i in range(len(_labels)):
        for _j in range(len(_labels)):
            _ax_corr.text(_j, _i, f"{_corr_matrix.iloc[_i, _j]:.2f}",
                          ha="center", va="center",
                          color="white" if abs(_corr_matrix.iloc[_i, _j]) > 0.5 else "black")
    _ax_corr.set_title("Spearman correlation")
    _fig_corr.colorbar(_im, shrink=0.8)
    _fig_corr.tight_layout()
    _fig_corr.savefig("fig_correlation_matrix.pdf", dpi=300, bbox_inches="tight")
    _fig_corr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Predictive Models

    We fit one LMER per quality metric, adding it as a fixed-effect predictor
    of usability alongside the standard covariates. This isolates the unique
    contribution of each metric.
    """)
    return


@app.cell
def _(QUALITY_METRICS, full_data, lmer, pl):
    # Fit one LMER per quality metric; extract its coefficient summary
    def _r_safe_pred(s):
        return s.replace(" ", "_").replace("+", "plus")

    _metric_raw_names = [m for m in QUALITY_METRICS if m != "qa_score"]

    # R-safe metric names (double dashes, single dashes break R formulas)
    def _sanitize(name):
        return name.replace("--", "_XX_").replace("-", "_").replace(".", "_dot_")

    _metric_safe_map = {m: _sanitize(m) for m in _metric_raw_names}
    _metric_safe_list = [m for m in QUALITY_METRICS if m not in _metric_safe_map] + list(_metric_safe_map.values())

    _keep_cols_pred = [
        "sender_unique_id", "validator_unique_id", "receiver_unique_id",
        "conversation_id", "conversation_topic", "conversation_source",
        "group", "sender_gender", "translation_model",
        "baseline_survey_score", "survey_score",
        "correction_ratio",
    ] + _metric_raw_names

    _pred_df = full_data[_keep_cols_pred].copy()
    _pred_df["qa_score"] = 1 - _pred_df["correction_ratio"]

    # Min-max rescale all raw metrics (except qa_score) to [0,1]
    for _m in _metric_raw_names:
        _min, _max = _pred_df[_m].min(), _pred_df[_m].max()
        _pred_df[_m] = (_pred_df[_m] - _min) / (_max - _min)

    _pred_df = _pred_df.rename(columns={
        **{m: s for m, s in _metric_safe_map.items()},
        "sender_unique_id": "participant_id",
        "sender_gender": "gender",
        "conversation_id": "conv_id",
    })
    _pred_df["group"] = _pred_df["group"].astype(str).map(_r_safe_pred)
    _pred_df["gender"] = _pred_df["gender"].astype(str)
    _pred_df["translation_model"] = _pred_df["translation_model"].astype(str).map(_r_safe_pred)
    _pred_df["conversation_topic"] = _pred_df["conversation_topic"].astype(str)
    _pred_pl = pl.from_pandas(_pred_df)

    _base_covariates = (
        "group * gender + conversation_topic + translation_model"
        " + baseline_survey_score + conversation_source"
    )
    _random_effects = (
        "(1 | validator_unique_id) + (1 | receiver_unique_id)"
        " + (1 | participant_id) + (1 | conv_id)"
    )

    pred_results = {}
    for _metric in _metric_safe_list:
        _formula = f"survey_score ~ {_metric} + {_base_covariates} + {_random_effects}"
        _m = lmer(_formula, data=_pred_pl)
        _m.fit(REML=True, summarize=False)
        pred_results[_metric] = _m
    return (pred_results,)


@app.cell
def _(METRIC_LABELS_SAFE, pd, plt, pred_results):
    # Extract coefficient, SE, CI, and p for each metric from its model
    _rows = []

    for _metric, _model in pred_results.items():
        _fe = _model.result_fit.to_pandas().set_index("term")
        _row = _fe.loc[_metric]
        _rows.append({
            "metric": METRIC_LABELS_SAFE.get(_metric, _metric),
            "coef": _row["estimate"],
            "se": _row["std_error"],
            "ci_low": _row["conf_low"],
            "ci_high": _row["conf_high"],
            "t_stat": _row["t_stat"],
            "p_value": _row["p_value"],
        })

    pred_summary = pd.DataFrame(_rows).sort_values("coef", ascending=True)

    # Forest plot
    _fig_pred, _ax_pred = plt.subplots(figsize=(4.5, 4))
    _y_pred = list(range(len(pred_summary)))
    for _yi, (_, _r) in enumerate(pred_summary.iterrows()):
        _p = _r["p_value"]
        if 0.05 <= _p < 0.1:
            _fmt, _mfc = "D", "white"
        else:
            _fmt, _mfc = "o", "black" if _p < 0.05 else "white"
        _ax_pred.errorbar(
            _r["coef"], _yi,
            xerr=[[_r["coef"] - _r["ci_low"]], [_r["ci_high"] - _r["coef"]]],
            fmt=_fmt, capsize=4, color="black", ecolor="black",
            markerfacecolor=_mfc, markeredgecolor="black", markeredgewidth=1.5,
        )
        _ax_pred.text(
            _r["ci_high"] + 0.2, _yi,
            f"{_r['coef']:.3f}",
            va="center", ha="left",
        )
    _ax_pred.axvline(0, linestyle="--", color="gray", alpha=0.6)
    _ax_pred.set_yticks(_y_pred)
    _ax_pred.set_yticklabels(pred_summary["metric"])
    _ax_pred.set_xlabel("Effect on Usability ($u$)")
    _ax_pred.margins(x=0.3, y=0.1   )
    _fig_pred.tight_layout()
    _fig_pred.savefig("fig_metric_coefficients.pdf", dpi=300, bbox_inches="tight")
    _fig_pred
    return (pred_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quality Regimes

    We split the data into tertiles of validator translation score (low, medium,
    high) and re-fit the predictive model for each metric within each regime.
    """)
    return


@app.cell
def _(METRIC_LABELS, QUALITY_METRICS, full_data, np, pd, plt):
    # Split into tertiles by validator_translation_score
    _q_cuts = full_data["validator_translation_score"].dropna()
    _tertiles = np.percentile(_q_cuts, [33.33, 66.67])
    _tertile_labels = ["Low", "Medium", "High"]

    full_data["quality_regime"] = np.select(
        [full_data["validator_translation_score"] <= _tertiles[0],
         full_data["validator_translation_score"] <= _tertiles[1]],
        [0, 1],
        default=2,
    )

    # Collect regime-specific coefficients (re-fitting within regimes)
    _regime_rows = []

    import statsmodels.formula.api as _smf_reg

    for _regime_idx, _regime_label in enumerate(_tertile_labels):
        _sub = full_data[full_data["quality_regime"] == _regime_idx].copy()
        _sub["qa_score"] = 1 - _sub["correction_ratio"]
        _sub = _sub.rename(columns={"sender_gender": "gender"})
        # Min-max rescale metrics to [0,1] in this subset
        for _m in QUALITY_METRICS:
            if _m == "qa_score":
                continue
            _min, _max = _sub[_m].min(), _sub[_m].max()
            if _max > _min:
                _sub[_m] = (_sub[_m] - _min) / (_max - _min)
        for _metric in QUALITY_METRICS:
            _formula = (
                f"survey_score ~ Q('{_metric}') + C(group) + C(gender)"
                f" + C(group):C(gender) + C(conversation_topic)"
                f" + C(translation_model) + baseline_survey_score"
                f" + C(conversation_source)"
            )
            _fit = _smf_reg.ols(_formula, data=_sub).fit()
            _params_key = _metric if _metric in _fit.params.index else f"Q('{_metric}')"
            _ci = _fit.conf_int().loc[_params_key]
            _regime_rows.append({
                "regime": _regime_label,
                "metric": METRIC_LABELS.get(_metric, _metric),
                "coef": _fit.params[_params_key],
                "se": _fit.bse[_params_key],
                "ci_low": _ci[0],
                "ci_high": _ci[1],
                "p_value": _fit.pvalues[_params_key],
                "n": len(_sub),
            })

    regime_summary = pd.DataFrame(_regime_rows)
    regime_summary["coef"] = pd.to_numeric(regime_summary["coef"], errors="coerce")

    # Plot regime coefficients
    _regime_palette = {"Low": "#222222", "Medium": "#666666", "High": "#AAAAAA"}
    _regime_markers = {"Low": "o", "Medium": "s", "High": "D"}
    _regime_offsets = {"Low": -0.18, "Medium": 0.0, "High": 0.18}
    _metric_order_reg = (
        regime_summary.groupby("metric")["coef"].mean().sort_values().index.tolist()
    )

    _fig_regime, _ax_regime = plt.subplots(figsize=(7, 3.5))
    _xr_base = list(range(len(_metric_order_reg)))
    for _reg in _tertile_labels:
        _curr = (
            regime_summary[regime_summary["regime"] == _reg]
            .set_index("metric")
            .reindex(_metric_order_reg)
            .reset_index()
        )
        _xr = [_v + _regime_offsets[_reg] for _v in _xr_base]
        _ax_regime.errorbar(
            _xr, _curr["coef"], yerr=(_curr["ci_high"] - _curr["ci_low"]) / 2,
            linestyle="None", marker=_regime_markers[_reg], markersize=6, capsize=4,
            color=_regime_palette[_reg], label=_reg,
        )
    _ax_regime.set_xticks(_xr_base)
    _ax_regime.set_xticklabels(_metric_order_reg, rotation=30, ha="right")
    _ax_regime.axhline(0, linestyle="--", color="gray", alpha=0.5)
    _ax_regime.set_ylabel("Effect on Usability ($u$)")
    _ax_regime.legend(title="Quality regime", frameon=False, loc="best")
    _fig_regime.tight_layout()
    _fig_regime.savefig("fig_regime_coefficients.pdf", dpi=300, bbox_inches="tight")
    _fig_regime
    return (regime_summary,)


@app.cell
def _(pred_summary, regime_summary):
    # Summary table: LMER + regime coefficients per metric
    _lmer = pred_summary.set_index("metric")[["coef", "p_value"]].rename(
        columns={"coef": "coef_lmer", "p_value": "p_lmer"}
    )

    _regime = (
        regime_summary.pivot(index="metric", columns="regime", values=["coef", "p_value"])
    )
    _regime.columns = [f"{v}_{r.lower()}" for v, r in _regime.columns]

    _summary_table = _lmer.join(_regime).reset_index()
    _summary_table = _summary_table.rename(columns={"index": "Metric"})
    _summary_table = _summary_table.round(3)
    _summary_table.to_csv("metric_summary.csv", index=False)
    _summary_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Table.** Each metric's estimated effect on usability ($u$) from a
    linear mixed model (LMER column) and from OLS models fitted within
    low, medium, and high validator-score tertiles. All metrics rescaled
    to $[0,1]$. A coefficient of $k$ means that moving from the
    worst- to the best-observed quality on that metric is associated
    with a $k$-point shift in usability (1--5 scale), holding other
    covariates constant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Human vs. Automatic Metrics

    Do human judgments (validator score, correction ratio) and automatic
    metrics form distinct families, and does correction ratio carry unique
    predictive information?
    """)
    return


@app.cell
def _(METRIC_LABELS, full_data, np, pd):
    # Does QA Score add predictive power beyond automatic metrics?
    _hum_v_auto_cols = [
        "survey_score", "validator_translation_score", "qa_score",
    ]
    _auto_cols = [
        "Unbabel--wmt23-cometkiwi-da-xl",
        "Unbabel--wmt22-cometkiwi-da",
        "Unbabel--wmt22-comet-da",
        "Unbabel--XCOMET-XL",
        "google--metricx-24-hybrid-xl-v2p6",
    ]

    # Partial correlations controlling for group, gender, model, topic
    import statsmodels.formula.api as _smf_part

    _part_rows = []
    for _auto in _auto_cols:
        # residualize
        _df_part = full_data[[
            "survey_score", "validator_translation_score", "qa_score", _auto,
            "group", "sender_gender", "translation_model", "conversation_topic", "conversation_source",
            "baseline_survey_score",
        ]].rename(columns={"sender_gender": "gender"}).dropna()

        # Model 1: usability ~ auto  |  covariates
        _m1 = _smf_part.ols(
            f"survey_score ~ Q('{_auto}') + group + gender + translation_model"
            f" + conversation_topic + conversation_source + baseline_survey_score",
            data=_df_part,
        ).fit()

        # Model 2: usability ~ auto + qa_score  |  covariates
        _m2 = _smf_part.ols(
            f"survey_score ~ Q('{_auto}') + qa_score + group + gender"
            f" + translation_model + conversation_topic + conversation_source"
            f" + baseline_survey_score",
            data=_df_part,
        ).fit()

        _auto_param = f"Q('{_auto}')" if f"Q('{_auto}')" in _m2.params.index else _auto
        _part_rows.append({
            "metric": METRIC_LABELS.get(_auto, _auto),
            "auto_only_coef": _m1.params.get(_auto_param, _m1.params.get(_auto, np.nan)),
            "auto_only_p": _m1.pvalues.get(_auto_param, _m1.pvalues.get(_auto, np.nan)),
            "with_correction_coef": _m2.params.get(_auto_param, _m2.params.get(_auto, np.nan)),
            "with_correction_p": _m2.pvalues.get(_auto_param, _m2.pvalues.get(_auto, np.nan)),
            "qa_score_coef": _m2.params.get("qa_score", np.nan),
            "qa_score_p": _m2.pvalues.get("qa_score", np.nan),
        })

    _part_table = pd.DataFrame(_part_rows).round(4)
    _part_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    How to read the Human vs. Automatic Metrics table:

    The analysis fits two OLS models for each automatic metric:

    Model 1 (auto_only): usability ~ metric + covariates — the auto metric's standalone predictive power
    Model 2 (with QA): usability ~ metric + QA Score + covariates — what happens when QA Score joins
    Key signals to look for:

    auto_only_coef significant, with_QA_coef shrinks — the auto metric and QA Score share predictive information. QA Score captures some of what the auto metric was picking up.
    qa_score_coef significant — QA Score adds unique predictive power beyond that specific auto metric. If this holds across all auto metrics, QA Score is not redundant with any of them — it's providing independent information.
    auto_only_coef stays significant after adding QA Score — the auto metric still matters. Both contribute independently.
    If QA Score is never significant when paired with a given auto metric, they're capturing the same underlying signal (likely overall translation quality).
    Practically: you want to know whether qa_score (your novel measure) is just re-measuring what COMET/XCOMET already measure, or whether it contributes something new. Significant qa_score_p values alongside still-significant auto metrics = complementary signals.
    """)
    return


if __name__ == "__main__":
    app.run()
