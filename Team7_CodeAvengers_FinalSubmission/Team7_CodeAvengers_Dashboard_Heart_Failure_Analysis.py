import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay


import base64
from pathlib import Path


# ============================================================
# ORIGINAL NOTEBOOK — PREDICTIVE QUESTION 1 ROC FUNCTIONS
# Preserved from the notebook. They are rendered only when the
# original notebook's y_test/log_prob/rf_prob arrays are available.
# ============================================================
from sklearn.metrics import RocCurveDisplay

st.set_page_config(page_title="Team 7 CodeAvengers — Heart Failure Analysis Dashboard", layout="wide")

st.markdown(
    """
    <style>
    /* ==========================================
       SIDEBAR FILTER LABELS
       Gender, Age Category, NYHA Class, etc.
       ========================================== */
    
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 22px !important;
    font-weight: 700 !important;
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        font-size: 30px !important;
        font-weight: 700 !important;
    }



    /* Filter label font size */
    [data-testid="stSidebar"] label {
        font-size: 30px !important;
        font-weight: 600 !important;
    }

    /* Selected value inside dropdown */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        font-size: 22px !important;
    }

    /* Multiselect selected values */
    [data-testid="stSidebar"] [data-baseweb="tag"] {
        font-size: 16px !important;
    }

    /* Sidebar normal text */
    [data-testid="stSidebar"] p {
        font-size: 20px !important;
    }

    /* Sidebar filter labels */
    [data-testid="stSidebar"] label p {
        font-size: 22px !important;
        font-weight: 600 !important;
    }

    /* Dropdown / multiselect text */
    [data-testid="stSidebar"] [data-baseweb="select"] * {
        font-size: 20px !important;
    }

    /* Tab font size */
    button[data-baseweb="tab"] p {
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Exact source data paths from the notebook
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "kamsandhya-cyber/Team7_CodeAvengers_PythonHackathon_SEP2026/"
    "main/cleaned_data/"
)

@st.cache_data
def load_data():
    demo = pd.read_csv(DATA_URL + "Team7_CodeAvengers_Category1_DataCleaning_demography.csv")
    labs = pd.read_csv(DATA_URL + "Team7_CodeAvengers_Category1_DataCleaning_labs.csv")
    cc   = pd.read_csv(DATA_URL + "Team7_CodeAvengers_Category1_DataCleaning_cardiac_complications.csv")
    resp = pd.read_csv(DATA_URL + "Team7_CodeAvengers_Category1_DataCleaning_responsivenes.csv")
    ph   = pd.read_csv(DATA_URL + "Team7_CodeAvengers_Category1_DataCleaning_patienthistory_cleaned.csv")
    hd   = pd.read_csv(DATA_URL + "Team7_CodeAvengers_Category1_DataCleaning_hospitalization_discharge.csv")
    pp   = pd.read_csv(DATA_URL + "Team7_CodeAvengers_Category1_DataCleaning_patient_precriptions.csv")
    return demo,labs,cc,resp,ph,hd,pp

demo,labs,cc,resp,ph,hd,pp = load_data()

demo_d = demo[["inpatient_number","agecat","gender"]].drop_duplicates("inpatient_number")
resp_d = resp[["inpatient_number","consciousness"]].drop_duplicates("inpatient_number")
ph_d = ph[["inpatient_number","cci_score"]].drop_duplicates("inpatient_number")
cc_d = cc[["inpatient_number","nyha_cardiac_function_classification","type_of_heart_failure","lvef"]].drop_duplicates("inpatient_number")
labs_d = labs[["inpatient_number","brain_natriuretic_peptide","creatinine_enzymatic_method","glomerular_filtration_rate",
               "high_sensitivity_troponin","hemoglobin","albumin","glucose_blood_gas","urea"]].drop_duplicates("inpatient_number")
hd_d = hd[["inpatient_number","re_admission_within_6_months"]].drop_duplicates("inpatient_number")

dashboard_df = (demo_d.merge(resp_d,on="inpatient_number",how="left")
 .merge(ph_d,on="inpatient_number",how="left").merge(cc_d,on="inpatient_number",how="left")
 .merge(labs_d,on="inpatient_number",how="left").merge(hd_d,on="inpatient_number",how="left"))

for col in ["cci_score","lvef","brain_natriuretic_peptide","creatinine_enzymatic_method",
            "glomerular_filtration_rate","re_admission_within_6_months"]:
    dashboard_df[col] = pd.to_numeric(dashboard_df[col],errors="coerce")
dashboard_df["Abnormal_Response"] = dashboard_df["consciousness"].astype(str).str.strip().str.lower().ne("clear")

def create_q1_roc_curve(y_test, log_prob, rf_prob, log_auc_q1=None, rf_auc_q1=None):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    log_name = f"Logistic Regression (AUC={log_auc_q1:.3f})" if log_auc_q1 is not None else "Logistic Regression"
    rf_name = f"Random Forest (AUC={rf_auc_q1:.3f})" if rf_auc_q1 is not None else "Random Forest"
    RocCurveDisplay.from_predictions(y_test, log_prob, name=log_name, ax=ax)
    RocCurveDisplay.from_predictions(y_test, rf_prob, name=rf_name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_title("ROC Curve: Readmission Prediction", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.20)
    plt.tight_layout()
    return fig

def create_roc_curve(y_test, log_prob, rf_prob):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    RocCurveDisplay.from_predictions(y_test, log_prob, name="Logistic Regression", ax=ax)
    RocCurveDisplay.from_predictions(y_test, rf_prob, name="Random Forest", ax=ax)
    ax.set_title("ROC Curve: Readmission Prediction", fontsize=15, color="#16395F", pad=15)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig



def create_egfr_chart(filtered_df):

    # Make a copy of filtered dashboard data
    tmp = filtered_df[
        [
            "agecat",
            "gender",
            "glomerular_filtration_rate"
        ]
    ].copy()

    # Convert eGFR to numeric
    tmp["glomerular_filtration_rate"] = pd.to_numeric(
        tmp["glomerular_filtration_rate"],
        errors="coerce"
    )

    # Remove missing values
    tmp = tmp.dropna(
        subset=[
            "agecat",
            "gender",
            "glomerular_filtration_rate"
        ]
    )

    # Calculate average eGFR
    egfr_by_group = (
        tmp
        .groupby(
            ["agecat", "gender"],
            observed=True
        )["glomerular_filtration_rate"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # Create line chart
    fig = px.line(
        egfr_by_group,
        x="agecat",
        y="mean",
        color="gender",
        markers=True,

        title="Average eGFR by Age Category and Gender",

        labels={
            "agecat": "Age Category",
            "mean": "Average eGFR",
            "gender": "Gender"
        },

        color_discrete_sequence=[
            "#B576A5",
            "#147D50"
        ],

        hover_data=["count"]
    )

    # Line formatting
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8)
    )

    # Overall average
    overall_avg = (
        tmp["glomerular_filtration_rate"].mean()
    )

    fig.add_hline(
        y=overall_avg,
        line_dash="dash",
        line_color="#5B9DB8",
        annotation_text=(
            f"Overall Avg: {overall_avg:.1f}"
        ),
        annotation_position="top left"
    )

    # Dashboard formatting
    fig.update_layout(

        # IMPORTANT:
        # No fixed width
        autosize=True,

        height=500,

        plot_bgcolor="#F2FBFD",
        paper_bgcolor="white",

        title_x=0.5,

        title_font=dict(
            size=16
        ),

        xaxis_title="Age Category",
        yaxis_title="Average eGFR",

        legend_title="Gender",

        font=dict(size=11),

        margin=dict(
            l=45,
            r=20,
            t=70,
            b=45
        ),

        xaxis=dict(
            showgrid=False
        ),

        yaxis=dict(
            showgrid=False
        )
    )

    return fig

def create_radar_chart(filtered_df):

    # Make copy
    df = filtered_df.copy()

    # -------------------------------------------------------
    # Create responsiveness status
    # -------------------------------------------------------

    df["Response_Status"] = (
        df["consciousness"]
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(
            lambda x:
            "Normal"
            if x == "clear"
            else "Abnormal"
        )
    )

    # -------------------------------------------------------
    # Clinical markers
    # -------------------------------------------------------

    markers = [
        "cci_score",
        "lvef",
        "brain_natriuretic_peptide",
        "creatinine_enzymatic_method",
        "glomerular_filtration_rate"
    ]

    # Convert to numeric
    for col in markers:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # -------------------------------------------------------
    # Normalize markers 0–1
    # -------------------------------------------------------

    normalized_df = df.copy()

    for col in markers:

        min_val = df[col].min()
        max_val = df[col].max()

        if (
            pd.notna(min_val)
            and pd.notna(max_val)
        ):

            if max_val != min_val:

                normalized_df[col] = (
                    (df[col] - min_val)
                    /
                    (max_val - min_val)
                )

            else:

                normalized_df[col] = 0

    # -------------------------------------------------------
    # Median marker values
    # -------------------------------------------------------

    radar_data = (
        normalized_df
        .groupby("Response_Status")[markers]
        .median()
    )

    # Readable labels
    labels = [
        "CCI Score",
        "LVEF",
        "BNP",
        "Creatinine",
        "GFR"
    ]

    # -------------------------------------------------------
    # Check groups exist
    # -------------------------------------------------------

    if (
        "Normal" not in radar_data.index
        or
        "Abnormal" not in radar_data.index
    ):

        fig = go.Figure()

        fig.add_annotation(
            text=(
                "Normal and Abnormal groups are "
                "required for this filter selection."
            ),
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14)
        )

        fig.update_layout(
            height=500,
            autosize=True
        )

        return fig

    # -------------------------------------------------------
    # Get values
    # -------------------------------------------------------

    normal_values = (
        radar_data
        .loc["Normal", markers]
        .tolist()
    )

    abnormal_values = (
        radar_data
        .loc["Abnormal", markers]
        .tolist()
    )

    # Close radar
    radar_labels = labels + [labels[0]]

    normal_values = (
        normal_values
        + [normal_values[0]]
    )

    abnormal_values = (
        abnormal_values
        + [abnormal_values[0]]
    )

    # -------------------------------------------------------
    # Create figure
    # -------------------------------------------------------

    fig = go.Figure()

    # Normal
    fig.add_trace(

        go.Scatterpolar(

            r=normal_values,

            theta=radar_labels,

            fill="toself",

            name="Normal",

            line=dict(
                color="#2196F3",
                width=3
            ),

            marker=dict(
                color="#2196F3",
                size=7
            ),

            fillcolor=(
                "rgba(33,150,243,0.15)"
            ),

            hovertemplate=(
                "<b>Normal</b><br>"
                "Marker: %{theta}<br>"
                "Normalized Median: %{r:.3f}"
                "<extra></extra>"
            )
        )
    )

    # Abnormal
    fig.add_trace(

        go.Scatterpolar(

            r=abnormal_values,

            theta=radar_labels,

            fill="toself",

            name="Abnormal",

            line=dict(
                color="#E53935",
                width=3
            ),

            marker=dict(
                color="#E53935",
                size=7
            ),

            fillcolor=(
                "rgba(229,57,53,0.15)"
            ),

            hovertemplate=(
                "<b>Abnormal</b><br>"
                "Marker: %{theta}<br>"
                "Normalized Median: %{r:.3f}"
                "<extra></extra>"
            )
        )
    )

    # -------------------------------------------------------
    # Dashboard formatting
    # -------------------------------------------------------

    fig.update_layout(

        title=dict(
            text=(
                "Clinical Markers: "
                "Normal vs Abnormal Responsiveness"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=16)
        ),

        polar=dict(

            bgcolor="white",

            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[
                    0,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1
                ],
                gridcolor="lightgray"
            ),

            angularaxis=dict(
                gridcolor="lightgray"
            )
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),

        # IMPORTANT
        # No width=800
        autosize=True,

        # Same as chart 1
        height=500,

        template="plotly_white",

        margin=dict(
            l=40,
            r=40,
            t=70,
            b=70
        )
    )

    return fig

def create_biomarker_boxplots(filtered_df):

    # Work on filtered dashboard data
    df = filtered_df.copy()

    # Convert required columns to numeric
    df["brain_natriuretic_peptide"] = pd.to_numeric(
        df["brain_natriuretic_peptide"],
        errors="coerce"
    )

    df["high_sensitivity_troponin"] = pd.to_numeric(
        df["high_sensitivity_troponin"],
        errors="coerce"
    )

    # Convert NYHA to string
    df["nyha_cardiac_function_classification"] = (
        df["nyha_cardiac_function_classification"]
        .astype(str)
        .str.replace(".0", "", regex=False)
    )

    # Keep NYHA 2, 3 and 4
    df = df[
        df["nyha_cardiac_function_classification"]
        .isin(["2", "3", "4"])
    ].copy()

    order = ["2", "3", "4"]

    # ---------------------------------------------------
    # Create figure
    # ---------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    # ---------------------------------------------------
    # BNP
    # ---------------------------------------------------

    sns.boxplot(
        data=df,
        x="nyha_cardiac_function_classification",
        y="brain_natriuretic_peptide",
        hue="nyha_cardiac_function_classification",
        order=order,
        hue_order=order,
        ax=axes[0],
        palette="Blues_d",
        legend=False,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "7"
        }
    )

    axes[0].set_yscale("log")

    axes[0].set_title(
        "BNP Distribution by NYHA Class",
        fontsize=12,
        fontweight="bold"
    )

    axes[0].set_xlabel("NYHA Class")
    axes[0].set_ylabel("BNP (pg/mL) - Log Scale")


    # ---------------------------------------------------
    # Troponin
    # ---------------------------------------------------

    sns.boxplot(
        data=df,
        x="nyha_cardiac_function_classification",
        y="high_sensitivity_troponin",
        hue="nyha_cardiac_function_classification",
        order=order,
        hue_order=order,
        ax=axes[1],
        palette="Reds_d",
        legend=False,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "7"
        }
    )

    axes[1].set_yscale("log")

    axes[1].set_title(
        "High-Sensitivity Troponin Distribution by NYHA Class",
        fontsize=12,
        fontweight="bold"
    )

    axes[1].set_xlabel("NYHA Class")
    axes[1].set_ylabel("Troponin (ng/mL) - Log Scale")


    # ---------------------------------------------------
    # Overall title
    # ---------------------------------------------------

    fig.suptitle(
        "Cardiac Biomarker Distributions across NYHA Classes",
        fontsize=15,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()

    return fig

def create_cardiac_priority_chart(filtered_df):

    # -------------------------------------------------------
    # Work with filtered dashboard data
    # -------------------------------------------------------
    df_cardiac = filtered_df.copy()

    # Convert BNP and LVEF to numeric
    df_cardiac["brain_natriuretic_peptide"] = pd.to_numeric(
        df_cardiac["brain_natriuretic_peptide"],
        errors="coerce"
    )

    df_cardiac["lvef"] = pd.to_numeric(
        df_cardiac["lvef"],
        errors="coerce"
    )

    df_cardiac = df_cardiac.dropna(
        subset=[
            "brain_natriuretic_peptide",
            "lvef",
            "agecat",
            "gender"
        ]
    )

    # -------------------------------------------------------
    # Higher BNP and Lower LVEF thresholds
    # -------------------------------------------------------
    bnp_q75 = df_cardiac[
        "brain_natriuretic_peptide"
    ].quantile(0.75)

    lvef_q25 = df_cardiac[
        "lvef"
    ].quantile(0.25)

    # -------------------------------------------------------
    # Identify priority patients
    # -------------------------------------------------------
    df_cardiac["Higher_BNP"] = (
        df_cardiac["brain_natriuretic_peptide"]
        >= bnp_q75
    )

    df_cardiac["Lower_LVEF"] = (
        df_cardiac["lvef"]
        <= lvef_q25
    )

    df_cardiac["Cardiac_Priority"] = (
        df_cardiac["Higher_BNP"]
        &
        df_cardiac["Lower_LVEF"]
    ).astype(int)

    # -------------------------------------------------------
    # Calculate patient-group percentages
    # -------------------------------------------------------
    polar_data = (
        df_cardiac
        .groupby(
            ["agecat", "gender"],
            observed=True
        )
        .agg(
            Total_Patients=(
                "inpatient_number",
                "count"
            ),

            Priority_Patients=(
                "Cardiac_Priority",
                "sum"
            ),

            Priority_Percent=(
                "Cardiac_Priority",
                "mean"
            ),

            Median_BNP=(
                "brain_natriuretic_peptide",
                "median"
            ),

            Median_LVEF=(
                "lvef",
                "median"
            )
        )
        .reset_index()
    )

    polar_data["Priority_Percent"] = (
        polar_data["Priority_Percent"] * 100
    )

    polar_data = polar_data.sort_values(
        ["agecat", "gender"]
    )

    # Patient group labels
    polar_data["Patient_Group"] = (
        polar_data["agecat"].astype(str)
        + " | "
        + polar_data["gender"].astype(str)
    )

    # -------------------------------------------------------
    # Create Polar Chart
    # -------------------------------------------------------
    fig = go.Figure()

    fig.add_trace(
        go.Barpolar(

            r=polar_data["Priority_Percent"],

            theta=polar_data["Patient_Group"],

            marker=dict(

                color=polar_data["Priority_Percent"],

                colorscale=[
                    [0.00, "#FFFDE7"],
                    [0.25, "#FFE082"],
                    [0.50, "#FFB74D"],
                    [0.75, "#F4511E"],
                    [1.00, "#B71C1C"]
                ],

                colorbar=dict(
                    title="Priority %",
                    thickness=12
                ),

                line=dict(
                    color="white",
                    width=2
                )
            ),

            customdata=polar_data[
                [
                    "Total_Patients",
                    "Priority_Patients",
                    "Median_BNP",
                    "Median_LVEF"
                ]
            ],

            hovertemplate=(
                "<b>%{theta}</b><br><br>"
                "Higher BNP + Lower LVEF: %{r:.1f}%<br>"
                "Total Patients: %{customdata[0]}<br>"
                "Priority Patients: %{customdata[1]}<br>"
                "Median BNP: %{customdata[2]:.2f}<br>"
                "Median LVEF: %{customdata[3]:.2f}"
                "<extra></extra>"
            )
        )
    )

    # -------------------------------------------------------
    # Dashboard formatting
    # -------------------------------------------------------
    fig.update_layout(

        title=dict(
            text="Cardiac Assessment Priority by Age & Gender",
            x=0.5,
            xanchor="center",
            font=dict(size=16)
        ),

        polar=dict(

            radialaxis=dict(
                title="Priority %",
                ticksuffix="%",
                showgrid=True,
                gridcolor="lightgray"
            ),

            angularaxis=dict(
                direction="clockwise",
                rotation=90
            )
        ),

        template="plotly_white",

        # Do NOT use width=900
        autosize=True,

        height=600,

        margin=dict(
            l=30,
            r=70,
            t=75,
            b=40
        )
    )

    return fig

def create_lab_sankey(filtered_df):

    patient = filtered_df.copy()

    # --------------------------------------------------
    # Select available laboratory biomarkers
    # --------------------------------------------------

    lab_options = {
        "Hemoglobin": [
            "hemoglobin",
            "hemoglobin_blood_gas"
        ],

        "Albumin": [
            "albumin",
            "albumin_serum"
        ],

        "Glucose": [
            "glucose_blood_gas",
            "glucose"
        ],

        "Creatinine": [
            "creatinine_enzymatic_method",
            "creatinine"
        ],

        "Urea": [
            "urea"
        ]
    }

    selected_labs = {}

    for label, options in lab_options.items():

        col = next(
            (
                c for c in options
                if c in patient.columns
            ),
            None
        )

        if col:
            selected_labs[label] = col

    if not selected_labs:
        raise ValueError(
            "No laboratory biomarkers found."
        )

    # --------------------------------------------------
    # Result categories
    # --------------------------------------------------

    ABOVE = "Above Comparison Range"
    WITHIN = "Within Comparison Range"
    BELOW = "Below Comparison Range"

    result_categories = [
        ABOVE,
        WITHIN,
        BELOW
    ]

    # --------------------------------------------------
    # Prepare laboratory data
    # --------------------------------------------------

    results = []

    for label, col in selected_labs.items():

        tmp = patient[
            [
                "inpatient_number",
                "agecat",
                col
            ]
        ].copy()

        tmp[col] = pd.to_numeric(
            tmp[col],
            errors="coerce"
        )

        tmp[col] = tmp[col].replace(
            [np.inf, -np.inf],
            np.nan
        )

        tmp = tmp.dropna()

        if tmp.empty:
            continue

        tmp = (
            tmp.groupby(
                [
                    "inpatient_number",
                    "agecat"
                ],
                observed=True,
                as_index=False
            )[col]
            .median()
        )

        lower = tmp[col].quantile(1/3)
        upper = tmp[col].quantile(2/3)

        if lower >= upper:
            continue

        tmp["Lab Result"] = np.select(
            [
                tmp[col] > upper,
                tmp[col] < lower
            ],
            [
                ABOVE,
                BELOW
            ],
            default=WITHIN
        )

        tested = (
            tmp
            .groupby(
                "agecat",
                observed=True
            )
            .size()
            .rename("Tested Patients")
            .reset_index()
        )

        summary = (
            tmp
            .groupby(
                [
                    "agecat",
                    "Lab Result"
                ],
                observed=True
            )
            .size()
            .reset_index(
                name="Patient Count"
            )
        )

        summary = summary.merge(
            tested,
            on="agecat",
            how="left"
        )

        summary["Biomarker"] = label

        results.append(summary)

    if not results:
        raise ValueError(
            "No laboratory results available."
        )

    sankey_df = pd.concat(
        results,
        ignore_index=True
    )

    sankey_df = sankey_df.rename(
        columns={
            "agecat": "Age Category"
        }
    )

    sankey_df["Age Category"] = (
        sankey_df["Age Category"]
        .astype(str)
    )

    # --------------------------------------------------
    # Colors
    # --------------------------------------------------

    lab_colors = {
        "Hemoglobin": "#00A896",
        "Albumin": "#3A86FF",
        "Glucose": "#FFBE0B",
        "Creatinine": "#EF476F",
        "Urea": "#9B5DE5"
    }

    age_colors = [
        "#1D3557",
        "#2A4D69",
        "#355C7D",
        "#436F8E",
        "#4F81A0",
        "#5C93B1",
        "#6AA5C2",
        "#78B7D3"
    ]

    result_colors = {
        ABOVE: "#E63946",
        WITHIN: "#2EC4B6",
        BELOW: "#FFB703"
    }

    # --------------------------------------------------
    # Sort age categories
    # --------------------------------------------------

    def age_sort_key(age):

        nums = re.findall(
            r"\d+",
            str(age)
        )

        return (
            int(nums[0])
            if nums
            else -1
        )

    age_nodes = sorted(
        sankey_df[
            "Age Category"
        ].unique(),
        key=age_sort_key,
        reverse=True
    )

    lab_nodes = [
        lab
        for lab in lab_options
        if lab in sankey_df[
            "Biomarker"
        ].unique()
    ]

    result_nodes = [
        r
        for r in result_categories
        if r in sankey_df[
            "Lab Result"
        ].unique()
    ]

    # --------------------------------------------------
    # Node IDs
    # --------------------------------------------------

    age_ids = [
        f"age:{x}"
        for x in age_nodes
    ]

    lab_ids = [
        f"lab:{x}"
        for x in lab_nodes
    ]

    result_ids = [
        f"result:{x}"
        for x in result_nodes
    ]

    all_ids = (
        age_ids
        + lab_ids
        + result_ids
    )

    all_labels = (
        age_nodes
        + lab_nodes
        + result_nodes
    )

    node_map = {
        node_id: i
        for i, node_id
        in enumerate(all_ids)
    }

    # --------------------------------------------------
    # Connections
    # --------------------------------------------------

    links_1 = (
        sankey_df
        .groupby(
            [
                "Age Category",
                "Biomarker"
            ],
            observed=True
        )["Patient Count"]
        .sum()
        .reset_index()
    )

    links_2 = (
        sankey_df
        .groupby(
            [
                "Biomarker",
                "Lab Result"
            ],
            observed=True
        )["Patient Count"]
        .sum()
        .reset_index()
    )

    source = (
        [
            node_map[f"age:{age}"]
            for age
            in links_1["Age Category"]
        ]
        +
        [
            node_map[f"lab:{lab}"]
            for lab
            in links_2["Biomarker"]
        ]
    )

    target = (
        [
            node_map[f"lab:{lab}"]
            for lab
            in links_1["Biomarker"]
        ]
        +
        [
            node_map[f"result:{result}"]
            for result
            in links_2["Lab Result"]
        ]
    )

    values = (
        links_1[
            "Patient Count"
        ].tolist()
        +
        links_2[
            "Patient Count"
        ].tolist()
    )

    # --------------------------------------------------
    # Node colors
    # --------------------------------------------------

    node_colors = []

    for label in all_labels:

        if label in age_nodes:

            node_colors.append(
                age_colors[
                    age_nodes.index(label)
                    % len(age_colors)
                ]
            )

        elif label in lab_nodes:

            node_colors.append(
                lab_colors[label]
            )

        else:

            node_colors.append(
                result_colors[label]
            )

    # --------------------------------------------------
    # Link colors
    # --------------------------------------------------

    def hex_to_rgba(
        hex_color,
        opacity=0.55
    ):

        hex_color = (
            hex_color.lstrip("#")
        )

        r = int(
            hex_color[0:2],
            16
        )

        g = int(
            hex_color[2:4],
            16
        )

        b = int(
            hex_color[4:6],
            16
        )

        return (
            f"rgba({r},{g},{b},{opacity})"
        )

    link_colors = []

    for lab in links_1["Biomarker"]:

        link_colors.append(
            hex_to_rgba(
                lab_colors[lab],
                0.45
            )
        )

    for result in links_2["Lab Result"]:

        link_colors.append(
            hex_to_rgba(
                result_colors[result],
                0.58
            )
        )

    # --------------------------------------------------
    # Create Sankey
    # --------------------------------------------------

    fig = go.Figure(

        go.Sankey(

            arrangement="snap",

            node=dict(

                pad=15,

                thickness=18,

                line=dict(
                    color="white",
                    width=1
                ),

                label=all_labels,

                color=node_colors
            ),

            link=dict(

                source=source,

                target=target,

                value=values,

                color=link_colors
            )
        )
    )

    # --------------------------------------------------
    # Dashboard formatting
    # --------------------------------------------------

    fig.update_layout(

        title=dict(
            text="Laboratory Result Flow Across Age Groups",
            x=0.5,
            xanchor="center",
            font=dict(
                size=16,
                color="#16395F"
            )
        ),

        # Smaller for dashboard
        height=600,

        # IMPORTANT — no fixed width
        autosize=True,

        paper_bgcolor="white",

        font=dict(
            size=10,
            color="#16395F"
        ),

        margin=dict(
            t=75,
            l=15,
            r=15,
            b=30
        )
    )

    return fig

def create_correlation_heatmap(filtered_df):

    # -------------------------------------------------------
    # STEP 1: Copy filtered dashboard data
    # -------------------------------------------------------
    df_corr = filtered_df.copy()

    # -------------------------------------------------------
    # STEP 2: Variables for correlation analysis
    # -------------------------------------------------------
    analysis_cols = [
        "cci_score",
        "lvef",
        "brain_natriuretic_peptide",
        "high_sensitivity_troponin",
        "creatinine_enzymatic_method",
        "glomerular_filtration_rate"
    ]

    # -------------------------------------------------------
    # STEP 3: Convert columns to numeric
    # -------------------------------------------------------
    for col in analysis_cols:

        df_corr[col] = pd.to_numeric(
            df_corr[col],
            errors="coerce"
        )

    # -------------------------------------------------------
    # STEP 4: Create Spearman correlation matrix
    # -------------------------------------------------------
    correlation_matrix = (
        df_corr[analysis_cols]
        .corr(method="spearman")
    )

    # -------------------------------------------------------
    # STEP 5: Readable column names
    # -------------------------------------------------------
    readable_names = {
        "cci_score": "CCI Score",
        "lvef": "LVEF",
        "brain_natriuretic_peptide": "BNP",
        "high_sensitivity_troponin": "Troponin",
        "creatinine_enzymatic_method": "Creatinine",
        "glomerular_filtration_rate": "GFR"
    }

    correlation_matrix = correlation_matrix.rename(
        index=readable_names,
        columns=readable_names
    )

    # -------------------------------------------------------
    # STEP 6: Create interactive heatmap
    # -------------------------------------------------------
    fig = go.Figure(

        data=go.Heatmap(

            z=correlation_matrix.values,

            x=correlation_matrix.columns,

            y=correlation_matrix.index,

            # Display correlation values
            text=np.round(
                correlation_matrix.values,
                2
            ),

            texttemplate="%{text:.2f}",

            textfont=dict(
                size=11
            ),

            # Diverging colors
            colorscale=[
                [0.00, "#2166AC"],
                [0.25, "#67A9CF"],
                [0.50, "#F7F7F7"],
                [0.75, "#EF8A62"],
                [1.00, "#B2182B"]
            ],

            zmin=-1,
            zmax=1,
            zmid=0,

            # Hover information
            hovertemplate=(
                "<b>Marker 1:</b> %{y}<br>"
                "<b>Marker 2:</b> %{x}<br>"
                "<b>Spearman Correlation:</b> %{z:.3f}"
                "<extra></extra>"
            ),

            colorbar=dict(
                title="Correlation",
                thickness=12,
                len=0.75
            )
        )
    )

    # -------------------------------------------------------
    # STEP 7: Dashboard formatting
    # -------------------------------------------------------
    fig.update_layout(

        title=dict(
            text=(
                "Relationships Between Clinical "
                "& Cardiac Markers"
            ),
            x=0.5,
            xanchor="center",
            font=dict(
                size=16
            )
        ),

        xaxis=dict(
            title="Clinical Marker",
            tickangle=-40
        ),

        yaxis=dict(
            title="Clinical Marker",
            autorange="reversed"
        ),

        # Responsive dashboard width
        autosize=True,

        # Same height as Sunburst
        height=520,

        template="plotly_white",

        margin=dict(
            l=75,
            r=55,
            t=70,
            b=90
        )
    )

    return fig

def create_sunburst_chart(filtered_df, prescriptions):

    # -------------------------------------------------------
    # STEP 1: Copy filtered dashboard data
    # -------------------------------------------------------
    patient = filtered_df.copy()

    ID_COL = "inpatient_number"
    AGE_COL = "agecat"
    GLUCOSE_COL = "glucose_blood_gas"


    # -------------------------------------------------------
    # STEP 2: Find medication column
    # -------------------------------------------------------
    med_options = [
        "drug_name",
        "medication_name",
        "medicine_name",
        "prescription_name",
        "generic_name",
        "drug",
        "medication",
        "medicine",
        "drug_class",
        "drug_code"
    ]

    MED_COL = next(
        (
            col for col in med_options
            if col in prescriptions.columns
        ),
        None
    )

    if MED_COL is None:
        raise ValueError(
            "Medication column not found in prescriptions table."
        )


    # -------------------------------------------------------
    # STEP 3: Check required patient columns
    # -------------------------------------------------------
    required_patient_cols = [
        ID_COL,
        AGE_COL,
        GLUCOSE_COL
    ]

    missing_patient_cols = [
        col for col in required_patient_cols
        if col not in patient.columns
    ]

    if missing_patient_cols:
        raise ValueError(
            f"Missing columns in filtered_df: {missing_patient_cols}"
        )

    if ID_COL not in prescriptions.columns:
        raise ValueError(
            f"'{ID_COL}' not found in prescriptions table."
        )


    # -------------------------------------------------------
    # STEP 4: Prepare glucose data
    # -------------------------------------------------------
    glucose_data = patient[
        [
            ID_COL,
            AGE_COL,
            GLUCOSE_COL
        ]
    ].copy()

    glucose_data[GLUCOSE_COL] = pd.to_numeric(
        glucose_data[GLUCOSE_COL],
        errors="coerce"
    )

    glucose_data[GLUCOSE_COL] = (
        glucose_data[GLUCOSE_COL]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )

    glucose_data = glucose_data.dropna(
        subset=[
            ID_COL,
            AGE_COL,
            GLUCOSE_COL
        ]
    )

    glucose_data[ID_COL] = (
        glucose_data[ID_COL]
        .astype(str)
        .str.strip()
    )


    # -------------------------------------------------------
    # One glucose value per patient
    # -------------------------------------------------------
    glucose_data = (
        glucose_data
        .groupby(
            ID_COL,
            as_index=False
        )
        .agg({
            AGE_COL: "first",
            GLUCOSE_COL: "median"
        })
    )


    # -------------------------------------------------------
    # STEP 5: Prepare medication data
    # -------------------------------------------------------
    med_data = prescriptions[
        [
            ID_COL,
            MED_COL
        ]
    ].copy()

    med_data = med_data.dropna(
        subset=[
            ID_COL,
            MED_COL
        ]
    )

    med_data[ID_COL] = (
        med_data[ID_COL]
        .astype(str)
        .str.strip()
    )

    med_data[MED_COL] = (
        med_data[MED_COL]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Remove empty medication names
    med_data = med_data[
        med_data[MED_COL] != ""
    ]


    # -------------------------------------------------------
    # STEP 6: Count unique medications per patient
    # -------------------------------------------------------
    med_counts = (
        med_data
        .groupby(ID_COL)[MED_COL]
        .nunique()
        .reset_index(
            name="Medication Count"
        )
    )


    # -------------------------------------------------------
    # STEP 7: Merge glucose + medication data
    # -------------------------------------------------------
    sunburst_data = glucose_data.merge(
        med_counts,
        on=ID_COL,
        how="left"
    )

    sunburst_data["Medication Count"] = (
        sunburst_data["Medication Count"]
        .fillna(0)
        .astype(int)
    )


    # -------------------------------------------------------
    # STEP 8: Create Medication Groups
    # -------------------------------------------------------
    def medication_group(count):

        if count == 0:
            return "No Prescription"

        elif count <= 2:
            return "1–2 Medications"

        elif count <= 5:
            return "3–5 Medications"

        else:
            return "6+ Medications"


    sunburst_data["Medication Group"] = (
        sunburst_data["Medication Count"]
        .apply(medication_group)
    )


    # -------------------------------------------------------
    # STEP 9: Create Glucose Categories
    # -------------------------------------------------------
    lower = (
        sunburst_data[GLUCOSE_COL]
        .quantile(1 / 3)
    )

    upper = (
        sunburst_data[GLUCOSE_COL]
        .quantile(2 / 3)
    )


    sunburst_data["Glucose Level"] = np.select(

        [
            sunburst_data[GLUCOSE_COL] < lower,
            sunburst_data[GLUCOSE_COL] > upper
        ],

        [
            "Lower Glucose",
            "Higher Glucose"
        ],

        default="Middle Glucose"
    )


    # -------------------------------------------------------
    # Create Age Category
    # -------------------------------------------------------
    sunburst_data["Age Category"] = (
        sunburst_data[AGE_COL]
        .astype(str)
    )


    # -------------------------------------------------------
    # STEP 10: Prepare summary
    # -------------------------------------------------------
    summary = (
        sunburst_data
        .groupby(
            [
                "Age Category",
                "Medication Group",
                "Glucose Level"
            ],
            observed=True
        )
        .size()
        .reset_index(
            name="Patient Count"
        )
    )


    # -------------------------------------------------------
    # Check for empty result
    # -------------------------------------------------------
    if summary.empty:

        fig = go.Figure()

        fig.add_annotation(
            text="No data available for the selected filters.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=15,
                color="#16395F"
            )
        )

        fig.update_layout(
            height=520,
            template="plotly_white"
        )

        return fig


    # -------------------------------------------------------
    # STEP 11: Create Sunburst
    # -------------------------------------------------------
    fig = px.sunburst(

        summary,

        path=[
            "Age Category",
            "Medication Group",
            "Glucose Level"
        ],

        values="Patient Count",

        color="Glucose Level",

        color_discrete_map={
            "Lower Glucose": "#FFBE0B",
            "Middle Glucose": "#2EC4B6",
            "Higher Glucose": "#EF476F"
        }
    )


    # =======================================================
    # STEP 12: FIX INNER RING COLORS
    # =======================================================

    # Inner ring — Age Category
    age_color = "#DCEAF7"

    # Middle ring — Medication Group
    medication_color = "#5B8DB8"

    # Outer ring — Glucose
    glucose_colors = {
        "Lower Glucose": "#FFBE0B",
        "Middle Glucose": "#2EC4B6",
        "Higher Glucose": "#EF476F"
    }


    # Get Sunburst trace
    trace = fig.data[0]

    new_colors = []


    # -------------------------------------------------------
    # Assign colors according to hierarchy depth
    # -------------------------------------------------------
    for label, node_id in zip(
        trace.labels,
        trace.ids
    ):

        # Convert to string safely
        label = str(label)
        node_id = str(node_id)

        # Plotly Express uses "/" for hierarchy
        depth = node_id.count("/")


        # INNER RING — Age Category
        if depth == 0:

            new_colors.append(
                age_color
            )


        # MIDDLE RING — Medication Group
        elif depth == 1:

            new_colors.append(
                medication_color
            )


        # OUTER RING — Glucose Level
        else:

            new_colors.append(
                glucose_colors.get(
                    label,
                    "#D9E2EC"
                )
            )


    # -------------------------------------------------------
    # Apply colors
    # -------------------------------------------------------
    fig.update_traces(

        marker=dict(
            colors=new_colors,

            line=dict(
                color="white",
                width=1.5
            )
        ),

        textinfo="label+percent parent",

        insidetextorientation="radial",

        hovertemplate=(
            "<b>%{label}</b><br>"
            "Patients: %{value}<br>"
            "Percentage: %{percentParent:.1%}"
            "<extra></extra>"
        )
    )


    # -------------------------------------------------------
    # STEP 13: Dashboard formatting
    # -------------------------------------------------------
    fig.update_layout(

        title=dict(
            text=(
                "Prescription Medication & "
                "Glucose Patterns by Age Group"
            ),

            x=0.5,
            xanchor="center",

            font=dict(
                size=16,
                color="#16395F"
            )
        ),

        autosize=True,

        height=520,

        template="plotly_white",

        paper_bgcolor="white",

        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20
        )
    )


    # -------------------------------------------------------
    # VERY IMPORTANT — RETURN FIGURE
    # -------------------------------------------------------
    return fig

def create_comorbidity_triad_chart(filtered_df):

    temp_df = filtered_df.copy()

    # ===================================================
    # REQUIRED COMORBIDITY COLUMNS
    # ===================================================

    condition_cols = [
        "chronic_obstructive_pulmonary_disease",
        "diabetes",
        "cerebrovascular_disease"
    ]

    # ===================================================
    # CHECK WHICH COLUMNS ARE MISSING
    # ===================================================

    missing_cols = [
        col for col in condition_cols
        if col not in temp_df.columns
    ]

    
    # ===================================================
    # IMPORTANT
    # Merge missing columns from patient-history dataframe
    # ===================================================

    if missing_cols:

        # CHANGE patient_history below if your dataframe
        # has a different name

        history_lookup = ph[
            ["inpatient_number"] + missing_cols
        ].copy()

        history_lookup = history_lookup.drop_duplicates(
            subset="inpatient_number"
        )

        # Match ID types
        temp_df["inpatient_number"] = (
            temp_df["inpatient_number"]
            .astype(str)
        )

        history_lookup["inpatient_number"] = (
            history_lookup["inpatient_number"]
            .astype(str)
        )

        # Merge
        temp_df = temp_df.merge(
            history_lookup,
            on="inpatient_number",
            how="left"
        )


    # ===================================================
    # CONVERT CONDITIONS TO NUMERIC
    # ===================================================

    for col in condition_cols:

        temp_df[col] = pd.to_numeric(
            temp_df[col],
            errors="coerce"
        ).fillna(0)


    # ===================================================
    # INDIVIDUAL CONDITION PREVALENCE
    # ===================================================

    q7_data = (
        temp_df
        .groupby(
            "agecat",
            observed=True
        )[condition_cols]
        .mean()
        .mul(100)
        .reset_index()
    )


    q7_data = q7_data.rename(
        columns={
            "chronic_obstructive_pulmonary_disease":
                "COPD",

            "diabetes":
                "Diabetes",

            "cerebrovascular_disease":
                "Cerebrovascular"
        }
    )


    # ===================================================
    # TRIAD COUNT
    # ===================================================

    temp_df["triad_count"] = (
        temp_df[
            "chronic_obstructive_pulmonary_disease"
        ]
        +
        temp_df["diabetes"]
        +
        temp_df[
            "cerebrovascular_disease"
        ]
    )


    # ===================================================
    # 2+ CONDITION OVERLAP
    # ===================================================

    triad_overlap = (
        temp_df
        .groupby(
            "agecat",
            observed=True
        )["triad_count"]
        .apply(
            lambda x:
            (x >= 2).mean() * 100
        )
        .reset_index(
            name="Triad_Overlap"
        )
    )


    # ===================================================
    # MERGE RESULTS
    # ===================================================

    plot_data = q7_data.merge(
        triad_overlap,
        on="agecat",
        how="left"
    )


    # ===================================================
    # CREATE CHART
    # ===================================================

    fig = make_subplots(
        specs=[
            [{"secondary_y": True}]
        ]
    )


    # COPD
    fig.add_trace(

        go.Bar(
            x=plot_data["agecat"],
            y=plot_data["COPD"],

            name="COPD",

            marker_color="#315B7D",

            hovertemplate=
            "<b>Age Group:</b> %{x}<br>"
            "<b>COPD:</b> %{y:.1f}%"
            "<extra></extra>"
        ),

        secondary_y=False
    )


    # Diabetes
    fig.add_trace(

        go.Bar(
            x=plot_data["agecat"],
            y=plot_data["Diabetes"],

            name="Diabetes",

            marker_color="#80CBC4",

            hovertemplate=
            "<b>Age Group:</b> %{x}<br>"
            "<b>Diabetes:</b> %{y:.1f}%"
            "<extra></extra>"
        ),

        secondary_y=False
    )


    # Cerebrovascular
    fig.add_trace(

        go.Bar(
            x=plot_data["agecat"],
            y=plot_data["Cerebrovascular"],

            name="Cerebrovascular Disease",

            marker_color="#F9D44A",

            hovertemplate=
            "<b>Age Group:</b> %{x}<br>"
            "<b>Cerebrovascular:</b> %{y:.1f}%"
            "<extra></extra>"
        ),

        secondary_y=False
    )


    # ===================================================
    # HIGH-RISK OVERLAP
    # ===================================================

    fig.add_trace(

        go.Scatter(
            x=plot_data["agecat"],
            y=plot_data["Triad_Overlap"],

            mode="lines+markers",

            name="High-Risk Triad Overlap (2+ Conditions)",

            line=dict(
                width=4,
                color="#C4162A"
            ),

            marker=dict(
                size=11,
                symbol="diamond",
                color="#C4162A"
            ),

            hovertemplate=
            "<b>Age Group:</b> %{x}<br>"
            "<b>2+ Conditions:</b> %{y:.1f}%"
            "<extra></extra>"
        ),

        secondary_y=True
    )


    # ===================================================
    # FORMATTING
    # ===================================================

    fig.update_layout(

        title=dict(
            text=(
                "<b>Individual Comorbidities vs. "
                "Combined Triad Overlap"
                "<br>for Discharge Intervention</b>"
            ),

            x=0.5,
            xanchor="center"
        ),

        template="plotly_white",

        barmode="group",

        hovermode="x unified",

        autosize=True,

        height=550,

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),

        margin=dict(
            l=70,
            r=70,
            t=120,
            b=70
        )
    )


    # ===================================================
    # AXES
    # ===================================================

    fig.update_xaxes(
        title_text="<b>Age Category</b>"
    )


    fig.update_yaxes(
        title_text=(
            "<b>Individual Condition "
            "Prevalence (%)</b>"
        ),

        secondary_y=False
    )


    fig.update_yaxes(
        title_text=(
            "<b>Multi-Condition "
            "Overlap Rate (%)</b>"
        ),

        secondary_y=True
    )


    return fig

def create_responsiveness_performance_chart(comparison_q2):

    chart_data = comparison_q2.melt(
        id_vars="Model",

        value_vars=[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],

        var_name="Metric",
        value_name="Score"
    )


    fig = px.bar(
        chart_data,

        x="Metric",
        y="Score",

        color="Model",

        barmode="group",

        text="Score",

        color_discrete_map={
            "Logistic Regression": "#457B9D",
            "Random Forest": "#E76F51"
        },

        title=(
            "Predictive Model Performance Comparison"
            "<br>"
            "<sup>Abnormal Responsiveness Prediction</sup>"
        )
    )


    fig.update_traces(
        texttemplate="%{y:.2f}",
        textposition="outside",
        cliponaxis=False
    )


    fig.update_layout(

        title=dict(
            x=0.5,
            xanchor="center",
            font=dict(
                size=17,
                color="#16395F"
            )
        ),

        height=500,

        autosize=True,

        template="plotly_white",

        xaxis_title="Evaluation Metric",

        yaxis=dict(
            title="Performance Score",
            range=[0, 1.10]
        ),

        legend=dict(
            title="Model",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),

        margin=dict(
            l=60,
            r=30,
            t=110,
            b=60
        )
    )


    return fig


def create_q3_model_comparison_chart():

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 4.8)
    )


    # ========================================================
    # CHART 1 — ROC CURVE
    # ========================================================

    fpr_log, tpr_log, _ = roc_curve(
        y_test_q3,
        log_prob_q3
    )

    fpr_rf, tpr_rf, _ = roc_curve(
        y_test_q3,
        rf_prob_q3
    )


    axes[0].plot(
        fpr_log,
        tpr_log,
        label=f"Logistic Regression ({q3_log_auc:.3f})"
    )

    axes[0].plot(
        fpr_rf,
        tpr_rf,
        label=f"Random Forest ({q3_rf_auc:.3f})"
    )

    axes[0].plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Chance"
    )

    axes[0].set_title(
        "ROC Curves",
        fontweight="bold"
    )

    axes[0].set_xlabel(
        "False Positive Rate"
    )

    axes[0].set_ylabel(
        "True Positive Rate"
    )

    axes[0].legend(
        fontsize=7
    )

    axes[0].grid(
        alpha=0.20
    )


    # ========================================================
    # CHART 2 — PRECISION-RECALL CURVE
    # ========================================================

    precision_log, recall_log, _ = (
        precision_recall_curve(
            y_test_q3,
            log_prob_q3
        )
    )

    precision_rf, recall_rf, _ = (
        precision_recall_curve(
            y_test_q3,
            rf_prob_q3
        )
    )


    axes[1].plot(
        recall_log,
        precision_log,
        label=f"Logistic Regression ({q3_log_ap:.3f})"
    )

    axes[1].plot(
        recall_rf,
        precision_rf,
        label=f"Random Forest ({q3_rf_ap:.3f})"
    )


    # Test prevalence baseline
    prevalence_q3 = y_test_q3.mean()

    axes[1].axhline(
        prevalence_q3,
        linestyle="--",
        label=f"Test prevalence ({prevalence_q3:.3f})"
    )


    axes[1].set_title(
        "Precision–Recall Curves",
        fontweight="bold"
    )

    axes[1].set_xlabel(
        "Recall (Sensitivity)"
    )

    axes[1].set_ylabel(
        "Precision"
    )

    axes[1].legend(
        fontsize=7
    )

    axes[1].grid(
        alpha=0.20
    )


    # ========================================================
    # CHART 3 — CALIBRATION CURVE
    # ========================================================

    obs_log, pred_log = calibration_curve(
        y_test_q3,
        log_prob_q3,
        n_bins=5,
        strategy="quantile"
    )


    obs_rf, pred_rf = calibration_curve(
        y_test_q3,
        rf_prob_q3,
        n_bins=5,
        strategy="quantile"
    )


    axes[2].plot(
        pred_log,
        obs_log,
        marker="o",
        label="Logistic Regression"
    )

    axes[2].plot(
        pred_rf,
        obs_rf,
        marker="o",
        label="Random Forest"
    )


    # Perfect calibration reference
    axes[2].plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Ideal"
    )


    axes[2].set_title(
        "Calibration (5 Quantile Bins)",
        fontweight="bold"
    )

    axes[2].set_xlabel(
        "Mean Predicted Probability"
    )

    axes[2].set_ylabel(
        "Observed MI Fraction"
    )

    axes[2].legend(
        fontsize=7
    )

    axes[2].grid(
        alpha=0.20
    )


    # ========================================================
    # MAIN TITLE
    # ========================================================

    fig.suptitle(
        "Question 3 — Myocardial Infarction Model Evaluation",
        fontsize=15,
        fontweight="bold"
    )


    plt.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    return fig

def create_responsiveness_parallel_chart(filtered_df):

    temp_df = filtered_df.copy()

    # ===================================================
    # STEP 1 — ADD BMI FROM DEMOGRAPHY DATA
    # ===================================================

    if "bmi" not in temp_df.columns:

        bmi_lookup = demo[
            ["inpatient_number", "bmi"]
        ].copy()

        bmi_lookup = bmi_lookup.drop_duplicates(
            subset="inpatient_number"
        )

        # Make sure Patient ID data types match
        temp_df["inpatient_number"] = (
            temp_df["inpatient_number"].astype(str)
        )

        bmi_lookup["inpatient_number"] = (
            bmi_lookup["inpatient_number"].astype(str)
        )

        temp_df = temp_df.merge(
            bmi_lookup,
            on="inpatient_number",
            how="left"
        )

    # Convert BMI to numeric
    temp_df["bmi"] = pd.to_numeric(
        temp_df["bmi"],
        errors="coerce"
    )


    # ===================================================
    # STEP 2 — NORMAL VS ABNORMAL RESPONSIVENESS
    # ===================================================

    temp_df["Responsiveness"] = (
        temp_df["consciousness"]
        .astype(str)
        .str.strip()
        .apply(
            lambda x:
            "Normal"
            if x.lower() == "clear"
            else "Abnormal"
        )
    )


    # ===================================================
    # STEP 3 — BMI CATEGORY
    # ===================================================

    temp_df["BMI Category"] = pd.cut(
        temp_df["bmi"],

        bins=[
            0,
            18.5,
            25,
            30,
            float("inf")
        ],

        labels=[
            "Underweight",
            "Normal",
            "Overweight",
            "Obese"
        ]
    )


    # ===================================================
    # STEP 4 — KEEP ABNORMAL PATIENTS
    # ===================================================

    abnormal_df = temp_df[
        temp_df["Responsiveness"] == "Abnormal"
    ].copy()


    # Remove missing chart dimensions
    abnormal_df = abnormal_df.dropna(
        subset=[
            "agecat",
            "gender",
            "BMI Category",
            "consciousness"
        ]
    )


    # ===================================================
    # STEP 5 — RESPONSIVENESS COLOR
    # ===================================================

    color_map = {
        "Responsive To Sound": 1,
        "Responsive To Pain": 2,
        "Nonresponsive": 3
    }

    abnormal_df["Response Color"] = (
        abnormal_df["consciousness"]
        .map(color_map)
        .fillna(1)
    )


    # ===================================================
    # STEP 6 — PARALLEL CATEGORIES CHART
    # ===================================================

    fig = px.parallel_categories(

        abnormal_df,

        dimensions=[
            "agecat",
            "gender",
            "BMI Category",
            "consciousness"
        ],

        color="Response Color",

        color_continuous_scale=[
            [0.00, "#2E86DE"],
            [0.50, "#F39C12"],
            [1.00, "#C0392B"]
        ],

        title=(
            "Demographic Profiles Associated with "
            "Abnormal Patient Responsiveness"
        )
    )


    # ===================================================
    # DASHBOARD FORMATTING
    # ===================================================

    fig.update_layout(

        autosize=True,

        height=600,

        title=dict(
            x=0.5,
            xanchor="center",

            font=dict(
                size=18,
                color="#16395F"
            )
        ),

        font=dict(
            size=12
        ),

        margin=dict(
            l=40,
            r=40,
            t=80,
            b=40
        )
    )

    return fig
# ============================================================
# PREDICTIVE ANALYSIS HELPER FUNCTIONS
# ============================================================

def _binary_target(series):
    """Convert common yes/no, true/false, 1/0 values to binary."""

    numeric = pd.to_numeric(series, errors="coerce")

    out = pd.Series(
        np.nan,
        index=series.index,
        dtype="float64"
    )

    valid_numeric = numeric.isin([0, 1])
    out.loc[valid_numeric] = numeric.loc[valid_numeric]

    text = series.astype(str).str.strip().str.lower()

    yes = {
        "yes",
        "y",
        "true",
        "1",
        "readmitted",
        "abnormal"
    }

    no = {
        "no",
        "n",
        "false",
        "0",
        "not readmitted",
        "normal"
    }

    out.loc[text.isin(yes)] = 1
    out.loc[text.isin(no)] = 0

    return out


# ============================================================
# FIT BINARY MODELS
# ============================================================

def _fit_binary_models(
    data,
    feature_cols,
    target,
    random_state=42
):

    work = data[feature_cols].copy()

    y = pd.Series(
        target,
        index=data.index,
        dtype="float64"
    )

    valid = y.notna()

    X = work.loc[valid].copy()
    y = y.loc[valid].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "The target must contain both 0 and 1 classes."
        )

    # Numeric features
    numeric_features = (
        X.select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    # Categorical features
    categorical_features = [
        col
        for col in X.columns
        if col not in numeric_features
    ]

    transformers = []

    # Numeric preprocessing
    if numeric_features:

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ])

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    # Categorical preprocessing
    if categorical_features:

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ])

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # Check class distribution
    class_counts = y.value_counts()

    if class_counts.min() >= 2:
        stratify_y = y
    else:
        stratify_y = None

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=stratify_y
    )

    # Logistic Regression
    logistic_model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state
            )
        )
    ])

    # Random Forest
    random_forest_model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            )
        )
    ])

    # Train models
    logistic_model.fit(
        X_train,
        y_train
    )

    random_forest_model.fit(
        X_train,
        y_train
    )

    # Predictions
    log_pred = logistic_model.predict(
        X_test
    )

    log_prob = logistic_model.predict_proba(
        X_test
    )[:, 1]

    rf_pred = random_forest_model.predict(
        X_test
    )

    rf_prob = random_forest_model.predict_proba(
        X_test
    )[:, 1]

    # Return everything
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "logistic_model": logistic_model,
        "random_forest_model": random_forest_model,
        "log_pred": log_pred,
        "log_prob": log_prob,
        "rf_pred": rf_pred,
        "rf_prob": rf_prob
    }


# ============================================================
# MODEL METRICS
# ============================================================

def _model_metrics(
    model_name,
    actual,
    prediction,
    probability
):

    return {
        "Model": model_name,

        "Accuracy": accuracy_score(
            actual,
            prediction
        ),

        "Precision": precision_score(
            actual,
            prediction,
            zero_division=0
        ),

        "Recall": recall_score(
            actual,
            prediction,
            zero_division=0
        ),

        "F1 Score": f1_score(
            actual,
            prediction,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            actual,
            probability
        )
    }


# ============================================================
# ROC CURVE
# ============================================================

def create_roc_curve(
    y_test,
    log_prob,
    rf_prob
):

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    RocCurveDisplay.from_predictions(
        y_test,
        log_prob,
        name="Logistic Regression",
        ax=ax
    )

    RocCurveDisplay.from_predictions(
        y_test,
        rf_prob,
        name="Random Forest",
        ax=ax
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Chance"
    )

    ax.set_title(
        "ROC Curve: Readmission Prediction",
        fontsize=15,
        color="#16395F",
        pad=10
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        loc="lower right"
    )

    fig.tight_layout()

    return fig

    

def safe_mean(df,column):
    if column not in df.columns: return np.nan
    return pd.to_numeric(df[column],errors="coerce").mean()

def percent_yes(series):
    if series is None: return np.nan
    s=series.dropna()
    if s.empty: return np.nan
    numeric=pd.to_numeric(s,errors="coerce")
    if numeric.notna().any():
        vals=numeric.dropna()
        return vals.mean()*100 if vals.between(0,1).all() else vals.mean()
    return s.astype(str).str.strip().str.lower().isin({"yes","y","true","1","readmitted"}).mean()*100

def kpi_html(df):
    total=df["inpatient_number"].nunique()
    abnormal=df.loc[df["Abnormal_Response"],"inpatient_number"].nunique()
    cci3=df.loc[df["cci_score"]>=3,"inpatient_number"].nunique()
    bnp=safe_mean(df,"brain_natriuretic_peptide"); lvef=safe_mean(df,"lvef"); read=percent_yes(df["re_admission_within_6_months"])
    vals=[("👥","Total Patients",f"{total:,}"),("♥","Abnormal Responsiveness",f"{abnormal:,}"),
          ("⚕","High Comorbidity",f"{cci3:,}"),("🧪","Average BNP","N/A" if pd.isna(bnp) else f"{bnp:,.1f}"),
          ("♥","Average LVEF","N/A" if pd.isna(lvef) else f"{lvef:.1f}%"),
          ("↻","6-Month Readmission","N/A" if pd.isna(read) else f"{read:.1f}%")]
    cards="".join([f'<div class="hf-kpi-card"><div class="hf-kpi-icon">{ic}</div><div><div class="hf-kpi-title"><b>{t}</b></div><div class="hf-kpi-value">{v}</div></div></div>' for ic,t,v in vals])
    return f'<div class="kpi-row">{cards}</div><div class="filtered">Filtered rows: {len(df):,}</div>'

st.markdown("""
<style>
.block-container {max-width:100%; padding-top:1rem; padding-left:1.5rem; padding-right:1.5rem;}
.stApp {background:#F7F9FC;}
.main-border {border:3px solid #C4162A;border-radius:20px;padding:18px;background:#F7F9FC;box-shadow:0 5px 18px rgba(0,0,0,.10);}
.kpi-row {display:flex;gap:12px;width:100%;margin-top:16px;flex-wrap:wrap;}
.hf-kpi-card {flex:1 1 150px;min-width:145px;display:flex;align-items:center;gap:10px;background:#fff;border:1px solid #E7EDF4;border-radius:13px;padding:13px 14px;box-shadow:0 3px 10px rgba(19,54,89,.07);}
.hf-kpi-icon {width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#FDE8EC;color:#C4162A;font-size:21px;}
.hf-kpi-title {font-size:18px;color:#44566B;margin-bottom:4px;}
.hf-kpi-value {font-size:23px;font-weight:800;color:#102F54;}
.filtered {margin-top:18px;color:#657080;font-size:18px;}
div[data-testid="stPlotlyChart"], div[data-testid="stImage"], div[data-testid="stPyplot"] {background:white;border:1px solid #E5EBF2;border-radius:12px;padding:7px;box-shadow:0 3px 10px rgba(19,54,89,.06);}
button[data-baseweb="tab"] {font-weight:700;font-size:18px;min-width:220px;padding:18px 20px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background: linear-gradient(90deg, #9E1B00, #641500);
    border-radius: 20px;
    padding: 24px 20px;
    margin: 5px 0 14px 0;
    text-align: center;
    color: white;
    font-family: Arial, sans-serif;
">
    <div style="font-size: 30px; font-weight: 700;">
        Team 7 CodeAvengers — Heart Failure Analysis
    </div>
    <div style="font-size: 17px; margin-top: 4px;">
        Descriptive + Prescriptive + Predictive Analysis
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown('<h3 style="color:#16395F;margin:8px 0;">Dashboard Filters</h3>',unsafe_allow_html=True)

def opts(s): return ["All"]+sorted(s.dropna().astype(str).unique().tolist())
c1,c2,c3,c4=st.columns(4)
with c1: age=st.selectbox("Age:",opts(dashboard_df["agecat"]))
with c2: gender=st.selectbox("Gender:",opts(dashboard_df["gender"]))
with c3: nyha=st.selectbox("NYHA:",opts(dashboard_df["nyha_cardiac_function_classification"]))
with c4: hf=st.selectbox("HF Type:",opts(dashboard_df["type_of_heart_failure"]))

filtered_df=dashboard_df.copy()
if age!="All": filtered_df=filtered_df[filtered_df["agecat"].astype(str)==age]
if gender!="All": filtered_df=filtered_df[filtered_df["gender"].astype(str)==gender]
if nyha!="All": filtered_df=filtered_df[filtered_df["nyha_cardiac_function_classification"].astype(str)==nyha]
if hf!="All": filtered_df=filtered_df[filtered_df["type_of_heart_failure"].astype(str)==hf]

st.markdown(kpi_html(filtered_df),unsafe_allow_html=True)


tabs=st.tabs(["Introduction","Descriptive Analysis","Prescriptive Analysis","Predictive Analysis","Conclusion"])
# ============================================================
# TAB 1 — INTRODUCTION
# PURE STREAMLIT — NO HTML / DIV
# ============================================================

with tabs[0]:

    st.title("Heart Failure Analysis")

    st.subheader("Introduction")

    st.write(
        """
        Heart failure is a complex clinical condition in which
        the heart is unable to pump blood effectively enough
        to meet the body's needs. Patient outcomes can be
        influenced by multiple factors, including age,
        comorbidities, cardiac function, laboratory findings,
        medications, and hospitalization history.
        """
    )

    st.write(
        """
        This dashboard analyzes heart failure patient data to
        identify important clinical patterns, risk factors,
        and outcomes that can support better patient monitoring
        and data-driven healthcare decisions.
        """
    )

    st.divider()


    # --------------------------------------------------------
    # DASHBOARD OBJECTIVE
    # --------------------------------------------------------

    st.subheader("Dashboard Objective")

    st.info(
        """
        The objective of this dashboard is to explore patient
        characteristics, identify clinically important patterns,
        evaluate risk factors, support patient prioritization,
        and examine predictive relationships across heart failure
        patient data.
        """
    )


    # --------------------------------------------------------
    # KEY AREAS
    # --------------------------------------------------------

    st.subheader("Key Areas of Analysis")


    # ROW 1
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):

            st.markdown("#### 👥 Patient Characteristics")

            st.write(
                "Age, gender and demographics"
            )


    with col2:
        with st.container(border=True):

            st.markdown("#### 🩺 Comorbidities")

            st.write(
                "Medical history and disease burden"
            )


    with col3:
        with st.container(border=True):

            st.markdown("#### ❤️ Cardiac Function")

            st.write(
                "NYHA class and LVEF"
            )


    # ROW 2
    col4, col5, col6 = st.columns(3)

    with col4:
        with st.container(border=True):

            st.markdown("#### 🧪 Laboratory Findings")

            st.write(
                "BNP, creatinine, troponin and GFR"
            )


    with col5:
        with st.container(border=True):

            st.markdown("#### 💊 Medications")

            st.write(
                "Patient treatment information"
            )


    with col6:
        with st.container(border=True):

            st.markdown("#### 🏥 Hospitalization")

            st.write(
                "Visits, readmission and outcomes"
            )
with tabs[1]:
    a,b=st.columns(2)
    with a: st.plotly_chart(create_egfr_chart(filtered_df),use_container_width=True,config={"displaylogo":False})
    with b: st.plotly_chart(create_radar_chart(filtered_df),use_container_width=True,config={"displaylogo":False})
    fig3=create_biomarker_boxplots(filtered_df); st.pyplot(fig3,use_container_width=True); plt.close(fig3)
    st.plotly_chart(create_responsiveness_parallel_chart(filtered_df),use_container_width=True,config={"displaylogo":False})
with tabs[2]:
    a,b=st.columns(2)
    with a: st.plotly_chart(create_cardiac_priority_chart(filtered_df),use_container_width=True,config={"displaylogo":False})
    with b: st.plotly_chart(create_lab_sankey(filtered_df),use_container_width=True,config={"displaylogo":False})
    a,b=st.columns(2)
    with a: st.plotly_chart(create_correlation_heatmap(filtered_df),use_container_width=True,config={"displaylogo":False})
    with b: st.plotly_chart(create_sunburst_chart(filtered_df,pp),use_container_width=True,config={"displaylogo":False})
    st.plotly_chart(create_comorbidity_triad_chart(filtered_df),use_container_width=True,config={"displaylogo":False})
with tabs[3]:

    st.header("Predictive Analysis")

    st.divider()

    # ========================================================
    # QUESTION 1
    # ========================================================

    st.subheader("Question 1 — Hospital Readmission Prediction")

    st.write(
        """
        Can demographic, medical-history, medication,
        laboratory, cardiac, responsiveness, and
        hospitalization factors predict which patients
        are at higher risk of hospital readmission?
        """
    )


    # --------------------------------------------------------
    # REASON
    # --------------------------------------------------------

    st.markdown("#### Reason")

    st.write(
        """
        Hospital readmission is an important outcome because
        it can reflect continuing clinical risk after discharge.
        This analysis combines information across all seven
        healthcare tables to identify patterns associated with
        28-day readmission.
        """
    )


    # --------------------------------------------------------
    # HYPOTHESIS
    # --------------------------------------------------------

    st.markdown("#### Hypothesis")

    st.write(
        """
        **H₀:** Demographic characteristics, comorbidity burden,
        medication use, laboratory markers, cardiac complications,
        and responsiveness measures do not significantly improve
        prediction of hospital readmission.
        """
    )

    st.write(
        """
        **H₁:** These patient characteristics and clinical markers
        provide useful predictive information for identifying
        patients at greater risk of hospital readmission.
        """
    )


       # ========================================================
    # QUESTION 1 — MODEL + ROC CURVE
    # ========================================================
    st.markdown("#### Model Performance")

    # --------------------------------------------------------
    # Select available predictor columns
    # --------------------------------------------------------

    q1_features = [
        c for c in [
            "agecat",
            "gender",
            "nyha_cardiac_function_classification",
            "type_of_heart_failure",
            "cci_score",
            "lvef",
            "brain_natriuretic_peptide",
            "creatinine_enzymatic_method",
            "glomerular_filtration_rate",
            "high_sensitivity_troponin",
            "hemoglobin",
            "albumin",
            "glucose_blood_gas",
            "urea",
            "consciousness"
        ]
        if c in dashboard_df.columns
    ]

    # --------------------------------------------------------
    # Train Q1 models
    # --------------------------------------------------------

    try:

        # Create target
        q1_target = _binary_target(
            dashboard_df["re_admission_within_6_months"]
        )

        # Train models
        q1 = _fit_binary_models(
            dashboard_df,
            q1_features,
            q1_target
        )

        # ----------------------------------------------------
        # Model comparison
        # ----------------------------------------------------

        q1cmp = pd.DataFrame([
            _model_metrics(
                "Logistic Regression",
                q1["y_test"],
                q1["log_pred"],
                q1["log_prob"]
            ),

            _model_metrics(
                "Random Forest",
                q1["y_test"],
                q1["rf_pred"],
                q1["rf_prob"]
            )
        ])

        st.markdown("##### Model Comparison")

        st.dataframe(
            q1cmp.round(3),
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # ROC Curve
        # ----------------------------------------------------

        st.markdown("##### ROC Curve")

        fig_q1 = create_roc_curve(
            q1["y_test"],
            q1["log_prob"],
            q1["rf_prob"]
        )

        st.pyplot(
            fig_q1,
            use_container_width=True
        )

        plt.close(fig_q1)

    except Exception as e:

        st.warning(
            f"Question 1 could not be calculated: {e}"
        )
    # ========================================================
    # KEY INSIGHT
    # ========================================================

    st.markdown("#### Key Insight")

    st.info(
        """
        Readmission was uncommon (**140 of 2,008 patients**).

        Random Forest produced a slightly higher ROC-AUC
        (**0.618**) than Logistic Regression (**0.601**),
        indicating only modest discrimination.

        Logistic Regression captured more readmissions
        (recall **0.607**), but with low precision
        (**0.091**), so the model should not be judged
        by accuracy alone.
        """
    )
          # ============================================================
    # QUESTION 2 — MODEL PERFORMANCE
    # ============================================================

    st.markdown("#### Model Performance")

    # Select available predictor columns
    q2_features = [
        c for c in [
            "agecat",
            "gender",
            "nyha_cardiac_function_classification",
            "type_of_heart_failure",
            "cci_score",
            "lvef",
            "brain_natriuretic_peptide",
            "creatinine_enzymatic_method",
            "glomerular_filtration_rate",
            "high_sensitivity_troponin",
            "hemoglobin",
            "albumin",
            "glucose_blood_gas",
            "urea"
        ]
        if c in dashboard_df.columns
    ]

    try:

        # --------------------------------------------------------
        # Train models
        # --------------------------------------------------------

        q2 = _fit_binary_models(
            dashboard_df,
            q2_features,
            dashboard_df["Abnormal_Response"].astype(int)
        )

        # --------------------------------------------------------
        # Model comparison
        # --------------------------------------------------------

        q2cmp = pd.DataFrame([
            _model_metrics(
                "Logistic Regression",
                q2["y_test"],
                q2["log_pred"],
                q2["log_prob"]
            ),

            _model_metrics(
                "Random Forest",
                q2["y_test"],
                q2["rf_pred"],
                q2["rf_prob"]
            )
        ])

        st.markdown("##### Model Comparison")

        st.dataframe(
            q2cmp.round(3),
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------------
        # Prepare chart data
        # --------------------------------------------------------

        chart_q2 = q2cmp.melt(
            id_vars="Model",
            value_vars=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],
            var_name="Metric",
            value_name="Score"
        )

        # --------------------------------------------------------
        # Create Plotly chart
        # --------------------------------------------------------

        fig_q2 = px.bar(
            chart_q2,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            title="Model Performance: Abnormal Responsiveness"
        )

        # Make chart smaller
        fig_q2.update_layout(
            height=420,
            width=750
        )

        st.plotly_chart(
            fig_q2,
            use_container_width=False
        )

    except Exception as e:

        st.warning(
            f"Question 2 could not be calculated: {e}"
        )
       # ============================================================
    # QUESTION 2 — KEY INSIGHT
    # ============================================================

    st.markdown("#### Key Insight")

    st.info(
        """
        Abnormal responsiveness was very rare
        (**34 of 2,008 patients**).

        Random Forest achieved a ROC-AUC of **0.874**, but detected
        none of the abnormal cases at the evaluated threshold
        (**recall = 0.000**).

        Logistic Regression had a lower ROC-AUC (**0.788**), but
        detected **42.9%** of abnormal cases, with very low
        precision (**0.040**).
        """
    )
       # ============================================================
    # QUESTION 3 — MYOCARDIAL INFARCTION PREDICTION
    # ============================================================

    st.divider()

    st.subheader(
        "Question 3 — Myocardial Infarction Prediction"
    )

    # ------------------------------------------------------------
    # QUESTION
    # ------------------------------------------------------------

    st.markdown("#### Question")

    st.write(
        """
        Can demographic, comorbidity, laboratory, and cardiac
        characteristics predict documented myocardial infarction
        in hospitalized heart-failure patients?
        """
    )

    # ------------------------------------------------------------
    # REASON
    # ------------------------------------------------------------

    st.markdown("#### Reason")

    st.write(
        """
        This analysis evaluates whether clinically relevant
        demographic, comorbidity, laboratory, and cardiac
        characteristics can distinguish hospitalized
        heart-failure patients with documented myocardial
        infarction while emphasizing leakage prevention,
        held-out validation, uncertainty, and interpretable
        associations.
        """
    )

    # ------------------------------------------------------------
    # HYPOTHESIS
    # ------------------------------------------------------------

    st.markdown("#### Hypothesis")

    st.write(
        """
        **H₀:** The selected demographic, comorbidity, laboratory,
        and cardiac predictors do not provide useful predictive
        information for distinguishing patients with documented
        myocardial infarction.
        """
    )

    st.write(
        """
        **H₁:** The selected demographic, comorbidity, laboratory,
        and cardiac predictors provide useful predictive information
        for distinguishing patients with documented myocardial
        infarction.
        """
    )

    # ------------------------------------------------------------
    # DASHBOARD CHART
    # ------------------------------------------------------------

    st.markdown("#### Dashboard Chart")
    q3_path=Path(__file__).with_name("q3_model_comparison.png")
    if q3_path.exists(): st.image(str(q3_path),use_container_width=True)
    else: st.warning("q3_model_comparison.png must be kept beside app.py.")
       # ============================================================
    # QUESTION 3 — KEY INSIGHT
    # ============================================================

    st.markdown("#### Key Insight")

    st.info(
        """
        Among **2,008 patients**, **143 (7.1%)** had documented
        myocardial infarction.

        Extended Logistic Regression was selected using
        training-only cross-validation and achieved a held-out
        **ROC-AUC of 0.920**, **average precision of 0.469**,
        and **Brier score of 0.051**.

        At the illustrative **0.50 cutoff**, sensitivity was
        **13.8%** and specificity was **98.7%**.

        **Important:** This analysis classifies a documented
        myocardial infarction diagnosis; it does **not** predict
        a future MI event.
        """
    )
# ============================================================
# TAB 5 — CONCLUSION
# PURE STREAMLIT — NO HTML / DIV
# ============================================================

with tabs[4]:

    st.title("Conclusion")

    st.write(
        """
        This heart failure analysis combined **descriptive,
        prescriptive, and predictive approaches** to provide a
        comprehensive view of patient health and outcomes.

        The analysis examined demographic characteristics,
        comorbidity burden, laboratory biomarkers, cardiac function,
        medication patterns, hospitalization, readmission, and mortality,
        while also evaluating predictive patterns related to readmission,
        abnormal responsiveness, and myocardial infarction.
        """
    )

    st.write(
        """
        Overall, the findings help identify patient groups with
        greater clinical complexity and patterns associated with
        adverse outcomes.

        These insights can support hospitals in prioritizing closer
        monitoring, strengthening discharge and follow-up planning,
        allocating healthcare resources more effectively, and
        identifying patients who may benefit from earlier clinical review.
        """
    )

    st.divider()


    # --------------------------------------------------------
    # KEY TAKEAWAY
    # --------------------------------------------------------

    st.subheader("Key Takeaway")

    st.info(
        """
        The analysis supports **data-driven clinical decision-making**
        by helping identify clinically important patient patterns and
        higher-risk groups.

        These findings can support patient monitoring, follow-up
        planning, resource allocation, and earlier clinical review.
        """
    )


    # --------------------------------------------------------
    # ANALYSIS SUMMARY
    # --------------------------------------------------------

    st.subheader("Analysis Summary")

    col1, col2, col3 = st.columns(3)


    # DESCRIPTIVE
    with col1:

        with st.container(border=True):

            st.markdown("#### 📊 Descriptive Analysis")

            st.write(
                """
                Explored patient demographics, laboratory
                biomarkers, cardiac function, comorbidities,
                and responsiveness patterns.
                """
            )


    # PRESCRIPTIVE
    with col2:

        with st.container(border=True):

            st.markdown("#### 🎯 Prescriptive Analysis")

            st.write(
                """
                Identified patient groups that may warrant
                closer monitoring, cardiac assessment,
                follow-up, and clinical review.
                """
            )


    # PREDICTIVE
    with col3:

        with st.container(border=True):

            st.markdown("#### 📈 Predictive Analysis")

            st.write(
                """
                Evaluated predictive patterns related to
                readmission, abnormal responsiveness,
                and documented myocardial infarction.
                """
            )


    # --------------------------------------------------------
    # CLINICAL NOTE
    # --------------------------------------------------------

    st.warning(
        """
        **Clinical Note:** These analytical findings are intended
        to support data-driven healthcare decisions and should
        complement, rather than replace, professional medical
        assessment and clinical judgment.
        """
    )