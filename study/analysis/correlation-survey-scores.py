import marimo

__generated_with = "0.23.6"
app = marimo.App()

with app.setup:
    import marimo as mo
    import pandas as pd
    import json
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Survey score correlation analysis

    **Examines whether the three final survey scores — satisfaction, trust, and reliance
    — capture distinct constructs or are effectively redundant**. Uses a within/between
    decomposition to separate participant-level and interaction-level variance, then
    runs exploratory factor analysis (EFA) with parallel analysis on each component
    independently.
    """)
    return


@app.cell
def _():
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
    return data_store, full_data


@app.cell
def _(full_data):
    print(full_data.shape)
    full_data.columns.tolist()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Factor Analysis
    """)
    return


@app.cell
def _(data_store, full_data):
    survey_cols = data_store["final_survey_cols"]
    user_col = "sender_unique_id"

    # Between component: one row per user (120 rows)
    between = full_data.groupby(user_col)[survey_cols].mean().reset_index(drop=True)

    # Within component: interaction scores minus the user's own mean (1200 rows)
    user_means = full_data.groupby(user_col)[survey_cols].transform("mean")
    within = (full_data[survey_cols] - user_means).reset_index(drop=True)
    return between, survey_cols, user_col, within


@app.cell
def _(between, survey_cols, within):
    short_labels = ["Sat.", "Trust", "Rel."]
    fig_corr, axes_corr = plt.subplots(1, 2, figsize=(9, 4))

    for ax, data, title in zip(
        axes_corr,
        [within[survey_cols], between[survey_cols]],
        ["Within-user (n=1200)", "Between-user (n=120)"],
    ):
        corr = data.corr()
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            ax=ax, square=True, xticklabels=short_labels, yticklabels=short_labels,
        )
        ax.set_title(title)

    fig_corr.tight_layout()
    fig_corr
    return (short_labels,)


@app.cell
def _(between, survey_cols, within):
    def _parallel_analysis(data, n_iter=500, percentile=95, seed=42):
        rng = np.random.default_rng(seed)
        n, p = data.shape
        obs_ev = np.linalg.eigvalsh(np.corrcoef(data.T))[::-1]
        rand_evs = np.array([
            np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)).T))[::-1]
            for _ in range(n_iter)
        ])
        threshold = np.percentile(rand_evs, percentile, axis=0)
        n_factors = int(np.sum(obs_ev > threshold))
        return n_factors, obs_ev, threshold

    efa_results = {}
    for _label, _data in [("within", within[survey_cols]), ("between", between[survey_cols])]:
        _chi2, _p = calculate_bartlett_sphericity(_data)
        _kmo_all, _kmo_model = calculate_kmo(_data)
        _n_factors, _obs_ev, _pa_thresh = _parallel_analysis(_data)
        efa_results[_label] = {
            "bartlett": (_chi2, _p),
            "kmo": _kmo_model,
            "n_factors": _n_factors,
            "obs_ev": _obs_ev,
            "pa_threshold": _pa_thresh,
        }
        print(f"--- {_label.upper()} COMPONENT ---")
        print(f"  Bartlett χ²={_chi2:.1f}, p={_p:.4f}")
        print(f"  KMO={_kmo_model:.3f}")
        print(f"  Parallel analysis → {_n_factors} factor(s)\n")
    return (efa_results,)


@app.cell
def _(efa_results, survey_cols):
    fig_scree, axes_scree = plt.subplots(1, 2, figsize=(10, 4))

    for ax2, label2 in zip(axes_scree, ["within", "between"]):
        r = efa_results[label2]
        x = range(1, len(survey_cols) + 1)
        ax2.plot(x, r["obs_ev"], "o-", label="Observed")
        ax2.plot(x, r["pa_threshold"], "s--", color="red", label="PA 95th pct.")
        ax2.axhline(1, linestyle=":", color="gray", alpha=0.5, label="Eigenvalue = 1")
        ax2.set_xticks(list(x))
        ax2.set_xlabel("Factor")
        ax2.set_ylabel("Eigenvalue")
        ax2.set_title(f"{label2.capitalize()} — scree + parallel analysis")
        ax2.legend(fontsize=8)

    fig_scree.tight_layout()
    fig_scree
    return


@app.cell
def _(between, efa_results, short_labels, survey_cols, within):
    component_data = {"within": within[survey_cols], "between": between[survey_cols]}

    print("HELLO")
    for label, curr_data in component_data.items():
        n_factors = max(efa_results[label]["n_factors"], 1)
        rotation = "oblimin" if n_factors > 1 else None
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
        fa.fit(curr_data)

        factor_cols = [f"F{i+1}" for i in range(n_factors)]
        loadings_df = pd.DataFrame(fa.loadings_, index=short_labels, columns=factor_cols)
        communalities = pd.Series(fa.get_communalities(), index=short_labels, name="h²")
        summary = pd.concat([loadings_df, communalities], axis=1).round(3)

        var_explained = fa.get_factor_variance()
        cum_var = sum(var_explained[1])

        print(f"--- {label.upper()} | {n_factors} factor(s), rotation={rotation} ---")
        print(summary.to_string())
        print(f"Cumulative variance explained: {cum_var:.1%}\n")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Same analysis for `baseline_survey_score`
    """)
    return


@app.cell
def _(data_store, full_data, user_col):
    # Repeat the same analysis but for baseline survey scores
    baseline_survey_cols = data_store["baseline_final_survey_cols"]
    short_labels_baseline = ["Base Sat.", "Base Trust", "Base Rel."]

    # Between component: one row per user (120 rows) for baseline scores
    between_baseline = full_data.groupby(user_col)[baseline_survey_cols].mean().reset_index(drop=True)

    # Within component: interaction scores minus the user's own mean (1200 rows) for baseline scores
    user_means_baseline = full_data.groupby(user_col)[baseline_survey_cols].transform("mean")
    within_baseline = (full_data[baseline_survey_cols] - user_means_baseline).reset_index(drop=True)

    # Correlation heatmaps for baseline scores
    fig_corr_baseline, axes_corr_baseline = plt.subplots(1, 2, figsize=(9, 4))

    for ax, data, title in zip(
        axes_corr_baseline,
        [within_baseline[baseline_survey_cols], between_baseline[baseline_survey_cols]],
        ["Within-user (n=1200)", "Between-user (n=120)"],
    ):
        corr = data.corr()
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            ax=ax, square=True, xticklabels=short_labels_baseline, yticklabels=short_labels_baseline,
        )
        ax.set_title(title)

    fig_corr_baseline.tight_layout()
    fig_corr_baseline
    return (
        baseline_survey_cols,
        between_baseline,
        short_labels_baseline,
        within_baseline,
    )


@app.cell
def _(baseline_survey_cols, between_baseline, within_baseline):
    # Bartlett's test, KMO, and parallel analysis for baseline scores
    efa_results_baseline = {}
    for _label, _data in [("within", within_baseline[baseline_survey_cols]), ("between", between_baseline[baseline_survey_cols])]:
        _chi2, _p = calculate_bartlett_sphericity(_data)
        _kmo_all, _kmo_model = calculate_kmo(_data)
        _n_factors, _obs_ev, _pa_thresh = _parallel_analysis(_data)
        efa_results_baseline[_label] = {
            "bartlett": (_chi2, _p),
            "kmo": _kmo_model,
            "n_factors": _n_factors,
            "obs_ev": _obs_ev,
            "pa_threshold": _pa_thresh,
        }
        print(f"--- {_label.upper()} COMPONENT (BASELINE) ---")
        print(f"  Bartlett χ²={_chi2:.1f}, p={_p:.4f}")
        print(f"  KMO={_kmo_model:.3f}")
        print(f"  Parallel analysis → {_n_factors} factor(s)\n")
    return (efa_results_baseline,)


@app.cell
def _(baseline_survey_cols, efa_results_baseline):
    # Scree plots for baseline scores
    fig_scree_baseline, axes_scree_baseline = plt.subplots(1, 2, figsize=(10, 4))

    for ax2, label2 in zip(axes_scree_baseline, ["within", "between"]):
        r = efa_results_baseline[label2]
        x = range(1, len(baseline_survey_cols) + 1)
        ax2.plot(x, r["obs_ev"], "o-", label="Observed")
        ax2.plot(x, r["pa_threshold"], "s--", color="red", label="PA 95th pct.")
        ax2.axhline(1, linestyle=":", color="gray", alpha=0.5, label="Eigenvalue = 1")
        ax2.set_xticks(list(x))
        ax2.set_xlabel("Factor")
        ax2.set_ylabel("Eigenvalue")
        ax2.set_title(f"{label2.capitalize()} — scree + parallel analysis (Baseline)")
        ax2.legend(fontsize=8)

    fig_scree_baseline.tight_layout()
    fig_scree_baseline
    return


@app.cell
def _(
    baseline_survey_cols,
    between_baseline,
    efa_results_baseline,
    short_labels_baseline,
    within_baseline,
):
    # Factor analysis for baseline scores
    component_data_baseline = {"within": within_baseline[baseline_survey_cols], "between": between_baseline[baseline_survey_cols]}

    for label, curr_data in component_data_baseline.items():
        n_factors = max(efa_results_baseline[label]["n_factors"], 1)
        rotation = "oblimin" if n_factors > 1 else None
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
        fa.fit(curr_data)

        factor_cols = [f"F{i+1}" for i in range(n_factors)]
        loadings_df = pd.DataFrame(fa.loadings_, index=short_labels_baseline, columns=factor_cols)
        communalities = pd.Series(fa.get_communalities(), index=short_labels_baseline, name="h²")
        summary = pd.concat([loadings_df, communalities], axis=1).round(3)

        var_explained = fa.get_factor_variance()
        cum_var = sum(var_explained[1])

        print(f"--- {label.upper()} (BASELINE) | {n_factors} factor(s), rotation={rotation} ---")
        print(summary.to_string())
        print(f"Cumulative variance explained: {cum_var:.1%}\n")
    return


if __name__ == "__main__":
    app.run()
