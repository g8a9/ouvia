import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.formula.api as smf
    from pathlib import Path

    return Path, json, mo, pd, plt, smf, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First- vs. third-person perspective analysis

    Fits a linear mixed-effects model (REML) to test whether subjective usability
    scores (`score_avg`, the mean of satisfaction, trust, and reliance ratings)
    differ depending on whether the assessment was given by the original sender
    (first-person) or a replacement participant (third-person). Fixed effects include
    perspective, translation quality, comprehension rate, gender, translation model,
    and language group. Random intercepts are included for participant and conversation.
    """)
    return


@app.cell
def _():
    # Known column names (from data inspection)
    PARTICIPANT_ID_COL = "sender_unique_id"
    CONV_ID_COL = "conversation_id"
    TRANSLATION_QUALITY_COL = "validator_translation_score"
    CORRECTIONS_COL = "validator_corrections"
    QUESTIONS_COL = "validator_question_total"
    GENDER_COL = "sender_gender"
    MODEL_COL = "translation_model"
    SURVEY_COLS = ["satisfaction_score", "trust_score", "reliance_score"]
    return (
        CONV_ID_COL,
        CORRECTIONS_COL,
        GENDER_COL,
        MODEL_COL,
        PARTICIPANT_ID_COL,
        QUESTIONS_COL,
        SURVEY_COLS,
        TRANSLATION_QUALITY_COL,
    )


@app.cell
def _(Path, json, pd):
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
    return (full_data,)


@app.cell
def _(full_data):
    full_data.shape
    return


@app.cell
def _(
    CONV_ID_COL,
    CORRECTIONS_COL,
    GENDER_COL,
    MODEL_COL,
    PARTICIPANT_ID_COL,
    QUESTIONS_COL,
    SURVEY_COLS,
    TRANSLATION_QUALITY_COL,
    full_data,
    pd,
):
    keep_cols = [
        PARTICIPANT_ID_COL,
        CONV_ID_COL,
        "final_sender_unique_id",
        "group",
        TRANSLATION_QUALITY_COL,
        CORRECTIONS_COL,
        QUESTIONS_COL,
        GENDER_COL,
        MODEL_COL,
    ] + SURVEY_COLS

    analysis_df = full_data[keep_cols].copy()

    # Derive perspective: first-person if the sender assessed their own data,
    # third-person if they assessed someone else's.
    _is_first = (
        analysis_df["final_sender_unique_id"].isna()
        | (analysis_df[PARTICIPANT_ID_COL] == analysis_df["final_sender_unique_id"])
    )
    analysis_df["perspective"] = _is_first.map({True: "first", False: "third"})

    # Comprehension rate: proportion of questions answered correctly
    analysis_df["comprehension_rate"] = (
        (analysis_df[QUESTIONS_COL] - analysis_df[CORRECTIONS_COL])
        / analysis_df[QUESTIONS_COL]
    )

    # Dependent variable: average of the three correlated survey scores
    analysis_df["score_avg"] = analysis_df[SURVEY_COLS].mean(axis=1)

    # Rename to short, formula-friendly names
    analysis_df = analysis_df.rename(columns={
        PARTICIPANT_ID_COL: "participant_id",
        CONV_ID_COL: "conv_id",
        TRANSLATION_QUALITY_COL: "translation_quality",
        GENDER_COL: "gender",
        MODEL_COL: "translation_model",
    })

    # Cast categoricals (lme4 will treat these as factors)
    for _col in ["gender", "translation_model", "perspective", "group"]:
        analysis_df[_col] = pd.Categorical(analysis_df[_col])

    print(analysis_df.groupby("perspective").size().rename("n_records"))
    print()
    print(analysis_df[["translation_quality", "comprehension_rate", "score_avg"]].describe())
    return (analysis_df,)


@app.cell
def _(analysis_df, plt, sns):
    fig_dist, axes_dist = plt.subplots(1, 2, figsize=(10, 4))

    sns.histplot(analysis_df["score_avg"], bins=9, ax=axes_dist[0], kde=True)
    axes_dist[0].set_title("Distribution of score_avg")
    axes_dist[0].set_xlabel("score_avg (1–5)")

    if "perspective" in analysis_df.columns:
        sns.boxplot(data=analysis_df, x="perspective", y="score_avg", ax=axes_dist[1])
        axes_dist[1].set_title("score_avg by perspective")
    else:
        sns.boxplot(data=analysis_df, x="gender", y="score_avg", ax=axes_dist[1])
        axes_dist[1].set_title("score_avg by gender (perspective TBD)")

    fig_dist.tight_layout()
    fig_dist
    return


@app.cell
def _(analysis_df, smf):
    # Primary grouping: random intercept per participant (via groups=)
    # Second random intercept per conversation: added as a variance component
    # via vc_formula, equivalent to (1|conv_id) in lme4 notation.
    formula = (
        "score_avg ~ perspective + translation_quality + comprehension_rate"
        " + C(gender) + C(translation_model) + C(group)"
    )
    _model = smf.mixedlm(
        formula,
        data=analysis_df,
        groups=analysis_df["participant_id"],
        vc_formula={"conv_id": "0 + C(conv_id)"},
    )
    lme_result = _model.fit(reml=True)
    print(lme_result.summary())

    # Note: conv_id variance component may show wide CIs with only ~4 observations
    # per conversation (one per group×gender cell). Low identifiability is expected.
    print(f"\nconv_id variance:  {lme_result.vcomp[0]:.4f}")
    print(f"Residual variance: {lme_result.scale:.4f}")
    return (lme_result,)


@app.cell
def _(lme_result, pd, plt):
    _fe = lme_result.fe_params
    _ci = lme_result.conf_int()
    _coef_df = pd.DataFrame({
        "coef": _fe,
        "ci_low": _ci[0],
        "ci_high": _ci[1],
    })
    # Drop intercept and variance component rows (not fixed effects)
    _coef_df = _coef_df[~_coef_df.index.str.endswith(" Var")]
    _coef_df = _coef_df.drop(index="Intercept", errors="ignore")

    fig_coef, ax = plt.subplots(figsize=(7, max(3, len(_coef_df) * 0.5)))
    _y = range(len(_coef_df))
    ax.errorbar(
        _coef_df["coef"], _y,
        xerr=[
            _coef_df["coef"] - _coef_df["ci_low"],
            _coef_df["ci_high"] - _coef_df["coef"],
        ],
        fmt="o", capsize=4,
    )
    ax.axvline(0, linestyle="--", color="gray", alpha=0.6)
    ax.set_yticks(list(_y))
    ax.set_yticklabels(_coef_df.index)
    ax.set_xlabel("Coefficient (effect on score_avg)")
    ax.set_title("Fixed effects — 95% CI\n(positive = higher score_avg)")
    fig_coef.tight_layout()
    fig_coef
    return


if __name__ == "__main__":
    app.run()
