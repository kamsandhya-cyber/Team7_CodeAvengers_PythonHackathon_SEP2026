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

st.set_page_config(page_title="Heart Failure Analytics Dashboard", page_icon="❤", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.1rem;padding-bottom:2rem;max-width:1500px}
.hero{background:linear-gradient(90deg,#9E1B00,#641500);border-radius:18px;padding:22px;text-align:center;color:white;margin-bottom:16px}
.hero h1{font-size:30px;margin:0}.hero p{font-size:16px;margin:4px 0 0}
[data-testid="stMetric"]{background:white;border:1px solid #E5EBF2;border-radius:12px;padding:12px}
.section-card{background:white;border:1px solid #E5EBF2;border-radius:14px;padding:24px;margin:8px 0 18px;color:#26384D}
.red{color:#C4162A}.blue{color:#16395F}
</style>""",unsafe_allow_html=True)

DATA_URL="https://raw.githubusercontent.com/kamsandhya-cyber/Team7_CodeAvengers_PythonHackathon_SEP2026/main/cleaned_data/"
@st.cache_data(ttl=3600, show_spinner="Loading clinical datasets...")
def load_data():
    demo=pd.read_csv(DATA_URL+"Team7_CodeAvengers_Category1_DataCleaning_demography.csv")
    labs=pd.read_csv(DATA_URL+"Team7_CodeAvengers_Category1_DataCleaning_labs.csv")
    cc=pd.read_csv(DATA_URL+"Team7_CodeAvengers_Category1_DataCleaning_cardiac_complications.csv")
    resp=pd.read_csv(DATA_URL+"Team7_CodeAvengers_Category1_DataCleaning_responsivenes.csv")
    ph=pd.read_csv(DATA_URL+"Team7_CodeAvengers_Category1_DataCleaning_patienthistory_cleaned.csv")
    hd=pd.read_csv(DATA_URL+"Team7_CodeAvengers_Category1_DataCleaning_hospitalization_discharge.csv")
    pp=pd.read_csv(DATA_URL+"Team7_CodeAvengers_Category1_DataCleaning_patient_precriptions.csv")
    demo_d=demo[["inpatient_number","agecat","gender"]].drop_duplicates("inpatient_number")
    resp_d=resp[["inpatient_number","consciousness"]].drop_duplicates("inpatient_number")
    ph_d=ph[["inpatient_number","cci_score"]].drop_duplicates("inpatient_number")
    cc_d=cc[["inpatient_number","nyha_cardiac_function_classification","type_of_heart_failure","lvef"]].drop_duplicates("inpatient_number")
    labs_cols=["inpatient_number","brain_natriuretic_peptide","creatinine_enzymatic_method","glomerular_filtration_rate","high_sensitivity_troponin","hemoglobin","albumin","glucose_blood_gas","urea"]
    labs_d=labs[labs_cols].drop_duplicates("inpatient_number")
    hd_d=hd[["inpatient_number","re_admission_within_6_months"]].drop_duplicates("inpatient_number")
    df=demo_d.merge(resp_d,on="inpatient_number",how="left").merge(ph_d,on="inpatient_number",how="left").merge(cc_d,on="inpatient_number",how="left").merge(labs_d,on="inpatient_number",how="left").merge(hd_d,on="inpatient_number",how="left")
    for col in ["cci_score","lvef","brain_natriuretic_peptide","creatinine_enzymatic_method","glomerular_filtration_rate","high_sensitivity_troponin","hemoglobin","albumin","glucose_blood_gas","urea","re_admission_within_6_months"]:
        if col in df: df[col]=pd.to_numeric(df[col],errors="coerce")
    df["Abnormal_Response"]=df["consciousness"].astype(str).str.strip().str.lower().ne("clear")
    return demo,labs,cc,resp,ph,hd,pp,df

demo,labs,cc,resp,ph,hd,pp,dashboard_df=load_data()
def create_egfr_chart(filtered_df):
    tmp = filtered_df[['agecat', 'gender', 'glomerular_filtration_rate']].copy()
    tmp['glomerular_filtration_rate'] = pd.to_numeric(tmp['glomerular_filtration_rate'], errors='coerce')
    tmp = tmp.dropna(subset=['agecat', 'gender', 'glomerular_filtration_rate'])
    egfr_by_group = tmp.groupby(['agecat', 'gender'], observed=True)['glomerular_filtration_rate'].agg(['mean', 'count']).reset_index()
    fig = px.line(egfr_by_group, x='agecat', y='mean', color='gender', markers=True, title='Average eGFR by Age Category and Gender', labels={'agecat': 'Age Category', 'mean': 'Average eGFR', 'gender': 'Gender'}, color_discrete_sequence=['#B576A5', '#147D50'], hover_data=['count'])
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    overall_avg = tmp['glomerular_filtration_rate'].mean()
    fig.add_hline(y=overall_avg, line_dash='dash', line_color='#5B9DB8', annotation_text=f'Overall Avg: {overall_avg:.1f}', annotation_position='top left')
    fig.update_layout(autosize=True, height=500, plot_bgcolor='#F2FBFD', paper_bgcolor='white', title_x=0.5, title_font=dict(size=16), xaxis_title='Age Category', yaxis_title='Average eGFR', legend_title='Gender', font=dict(size=11), margin=dict(l=45, r=20, t=70, b=45), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    return fig

def create_radar_chart(filtered_df):
    df = filtered_df.copy()
    df['Response_Status'] = df['consciousness'].astype(str).str.strip().str.lower().apply(lambda x: 'Normal' if x == 'clear' else 'Abnormal')
    markers = ['cci_score', 'lvef', 'brain_natriuretic_peptide', 'creatinine_enzymatic_method', 'glomerular_filtration_rate']
    for col in markers:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    normalized_df = df.copy()
    for col in markers:
        min_val = df[col].min()
        max_val = df[col].max()
        if pd.notna(min_val) and pd.notna(max_val):
            if max_val != min_val:
                normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col] = 0
    radar_data = normalized_df.groupby('Response_Status')[markers].median()
    labels = ['CCI Score', 'LVEF', 'BNP', 'Creatinine', 'GFR']
    if 'Normal' not in radar_data.index or 'Abnormal' not in radar_data.index:
        fig = go.Figure()
        fig.add_annotation(text='Normal and Abnormal groups are required for this filter selection.', x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        fig.update_layout(height=500, autosize=True)
        return fig
    normal_values = radar_data.loc['Normal', markers].tolist()
    abnormal_values = radar_data.loc['Abnormal', markers].tolist()
    radar_labels = labels + [labels[0]]
    normal_values = normal_values + [normal_values[0]]
    abnormal_values = abnormal_values + [abnormal_values[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=normal_values, theta=radar_labels, fill='toself', name='Normal', line=dict(color='#2196F3', width=3), marker=dict(color='#2196F3', size=7), fillcolor='rgba(33,150,243,0.15)', hovertemplate='<b>Normal</b><br>Marker: %{theta}<br>Normalized Median: %{r:.3f}<extra></extra>'))
    fig.add_trace(go.Scatterpolar(r=abnormal_values, theta=radar_labels, fill='toself', name='Abnormal', line=dict(color='#E53935', width=3), marker=dict(color='#E53935', size=7), fillcolor='rgba(229,57,53,0.15)', hovertemplate='<b>Abnormal</b><br>Marker: %{theta}<br>Normalized Median: %{r:.3f}<extra></extra>'))
    fig.update_layout(title=dict(text='Clinical Markers: Normal vs Abnormal Responsiveness', x=0.5, xanchor='center', font=dict(size=16)), polar=dict(bgcolor='white', radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1], gridcolor='lightgray'), angularaxis=dict(gridcolor='lightgray')), legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5), autosize=True, height=500, template='plotly_white', margin=dict(l=40, r=40, t=70, b=70))
    return fig

def create_biomarker_boxplots(filtered_df):
    df = filtered_df.copy()
    df['brain_natriuretic_peptide'] = pd.to_numeric(df['brain_natriuretic_peptide'], errors='coerce')
    df['high_sensitivity_troponin'] = pd.to_numeric(df['high_sensitivity_troponin'], errors='coerce')
    df['nyha_cardiac_function_classification'] = df['nyha_cardiac_function_classification'].astype(str).str.replace('.0', '', regex=False)
    df = df[df['nyha_cardiac_function_classification'].isin(['2', '3', '4'])].copy()
    order = ['2', '3', '4']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x='nyha_cardiac_function_classification', y='brain_natriuretic_peptide', hue='nyha_cardiac_function_classification', order=order, hue_order=order, ax=axes[0], palette='Blues_d', legend=False, showmeans=True, meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': '7'})
    axes[0].set_yscale('log')
    axes[0].set_title('BNP Distribution by NYHA Class', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('NYHA Class')
    axes[0].set_ylabel('BNP (pg/mL) - Log Scale')
    sns.boxplot(data=df, x='nyha_cardiac_function_classification', y='high_sensitivity_troponin', hue='nyha_cardiac_function_classification', order=order, hue_order=order, ax=axes[1], palette='Reds_d', legend=False, showmeans=True, meanprops={'marker': 'o', 'markerfacecolor': 'white', 'markeredgecolor': 'black', 'markersize': '7'})
    axes[1].set_yscale('log')
    axes[1].set_title('High-Sensitivity Troponin Distribution by NYHA Class', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('NYHA Class')
    axes[1].set_ylabel('Troponin (ng/mL) - Log Scale')
    fig.suptitle('Cardiac Biomarker Distributions across NYHA Classes', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def create_cardiac_priority_chart(filtered_df):
    df_cardiac = filtered_df.copy()
    df_cardiac['brain_natriuretic_peptide'] = pd.to_numeric(df_cardiac['brain_natriuretic_peptide'], errors='coerce')
    df_cardiac['lvef'] = pd.to_numeric(df_cardiac['lvef'], errors='coerce')
    df_cardiac = df_cardiac.dropna(subset=['brain_natriuretic_peptide', 'lvef', 'agecat', 'gender'])
    bnp_q75 = df_cardiac['brain_natriuretic_peptide'].quantile(0.75)
    lvef_q25 = df_cardiac['lvef'].quantile(0.25)
    df_cardiac['Higher_BNP'] = df_cardiac['brain_natriuretic_peptide'] >= bnp_q75
    df_cardiac['Lower_LVEF'] = df_cardiac['lvef'] <= lvef_q25
    df_cardiac['Cardiac_Priority'] = (df_cardiac['Higher_BNP'] & df_cardiac['Lower_LVEF']).astype(int)
    polar_data = df_cardiac.groupby(['agecat', 'gender'], observed=True).agg(Total_Patients=('inpatient_number', 'count'), Priority_Patients=('Cardiac_Priority', 'sum'), Priority_Percent=('Cardiac_Priority', 'mean'), Median_BNP=('brain_natriuretic_peptide', 'median'), Median_LVEF=('lvef', 'median')).reset_index()
    polar_data['Priority_Percent'] = polar_data['Priority_Percent'] * 100
    polar_data = polar_data.sort_values(['agecat', 'gender'])
    polar_data['Patient_Group'] = polar_data['agecat'].astype(str) + ' | ' + polar_data['gender'].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Barpolar(r=polar_data['Priority_Percent'], theta=polar_data['Patient_Group'], marker=dict(color=polar_data['Priority_Percent'], colorscale=[[0.0, '#FFFDE7'], [0.25, '#FFE082'], [0.5, '#FFB74D'], [0.75, '#F4511E'], [1.0, '#B71C1C']], colorbar=dict(title='Priority %', thickness=12), line=dict(color='white', width=2)), customdata=polar_data[['Total_Patients', 'Priority_Patients', 'Median_BNP', 'Median_LVEF']], hovertemplate='<b>%{theta}</b><br><br>Higher BNP + Lower LVEF: %{r:.1f}%<br>Total Patients: %{customdata[0]}<br>Priority Patients: %{customdata[1]}<br>Median BNP: %{customdata[2]:.2f}<br>Median LVEF: %{customdata[3]:.2f}<extra></extra>'))
    fig.update_layout(title=dict(text='Cardiac Assessment Priority by Age & Gender', x=0.5, xanchor='center', font=dict(size=16)), polar=dict(radialaxis=dict(title='Priority %', ticksuffix='%', showgrid=True, gridcolor='lightgray'), angularaxis=dict(direction='clockwise', rotation=90)), template='plotly_white', autosize=True, height=600, margin=dict(l=30, r=70, t=75, b=40))
    return fig

def create_lab_sankey(filtered_df):
    patient = filtered_df.copy()
    lab_options = {'Hemoglobin': ['hemoglobin', 'hemoglobin_blood_gas'], 'Albumin': ['albumin', 'albumin_serum'], 'Glucose': ['glucose_blood_gas', 'glucose'], 'Creatinine': ['creatinine_enzymatic_method', 'creatinine'], 'Urea': ['urea']}
    selected_labs = {}
    for label, options in lab_options.items():
        col = next((c for c in options if c in patient.columns), None)
        if col:
            selected_labs[label] = col
    if not selected_labs:
        raise ValueError('No laboratory biomarkers found.')
    ABOVE = 'Above Comparison Range'
    WITHIN = 'Within Comparison Range'
    BELOW = 'Below Comparison Range'
    result_categories = [ABOVE, WITHIN, BELOW]
    results = []
    for label, col in selected_labs.items():
        tmp = patient[['inpatient_number', 'agecat', col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors='coerce')
        tmp[col] = tmp[col].replace([np.inf, -np.inf], np.nan)
        tmp = tmp.dropna()
        if tmp.empty:
            continue
        tmp = tmp.groupby(['inpatient_number', 'agecat'], observed=True, as_index=False)[col].median()
        lower = tmp[col].quantile(1 / 3)
        upper = tmp[col].quantile(2 / 3)
        if lower >= upper:
            continue
        tmp['Lab Result'] = np.select([tmp[col] > upper, tmp[col] < lower], [ABOVE, BELOW], default=WITHIN)
        tested = tmp.groupby('agecat', observed=True).size().rename('Tested Patients').reset_index()
        summary = tmp.groupby(['agecat', 'Lab Result'], observed=True).size().reset_index(name='Patient Count')
        summary = summary.merge(tested, on='agecat', how='left')
        summary['Biomarker'] = label
        results.append(summary)
    if not results:
        raise ValueError('No laboratory results available.')
    sankey_df = pd.concat(results, ignore_index=True)
    sankey_df = sankey_df.rename(columns={'agecat': 'Age Category'})
    sankey_df['Age Category'] = sankey_df['Age Category'].astype(str)
    lab_colors = {'Hemoglobin': '#00A896', 'Albumin': '#3A86FF', 'Glucose': '#FFBE0B', 'Creatinine': '#EF476F', 'Urea': '#9B5DE5'}
    age_colors = ['#1D3557', '#2A4D69', '#355C7D', '#436F8E', '#4F81A0', '#5C93B1', '#6AA5C2', '#78B7D3']
    result_colors = {ABOVE: '#E63946', WITHIN: '#2EC4B6', BELOW: '#FFB703'}

    def age_sort_key(age):
        nums = re.findall('\\d+', str(age))
        return int(nums[0]) if nums else -1
    age_nodes = sorted(sankey_df['Age Category'].unique(), key=age_sort_key, reverse=True)
    lab_nodes = [lab for lab in lab_options if lab in sankey_df['Biomarker'].unique()]
    result_nodes = [r for r in result_categories if r in sankey_df['Lab Result'].unique()]
    age_ids = [f'age:{x}' for x in age_nodes]
    lab_ids = [f'lab:{x}' for x in lab_nodes]
    result_ids = [f'result:{x}' for x in result_nodes]
    all_ids = age_ids + lab_ids + result_ids
    all_labels = age_nodes + lab_nodes + result_nodes
    node_map = {node_id: i for i, node_id in enumerate(all_ids)}
    links_1 = sankey_df.groupby(['Age Category', 'Biomarker'], observed=True)['Patient Count'].sum().reset_index()
    links_2 = sankey_df.groupby(['Biomarker', 'Lab Result'], observed=True)['Patient Count'].sum().reset_index()
    source = [node_map[f'age:{age}'] for age in links_1['Age Category']] + [node_map[f'lab:{lab}'] for lab in links_2['Biomarker']]
    target = [node_map[f'lab:{lab}'] for lab in links_1['Biomarker']] + [node_map[f'result:{result}'] for result in links_2['Lab Result']]
    values = links_1['Patient Count'].tolist() + links_2['Patient Count'].tolist()
    node_colors = []
    for label in all_labels:
        if label in age_nodes:
            node_colors.append(age_colors[age_nodes.index(label) % len(age_colors)])
        elif label in lab_nodes:
            node_colors.append(lab_colors[label])
        else:
            node_colors.append(result_colors[label])

    def hex_to_rgba(hex_color, opacity=0.55):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f'rgba({r},{g},{b},{opacity})'
    link_colors = []
    for lab in links_1['Biomarker']:
        link_colors.append(hex_to_rgba(lab_colors[lab], 0.45))
    for result in links_2['Lab Result']:
        link_colors.append(hex_to_rgba(result_colors[result], 0.58))
    fig = go.Figure(go.Sankey(arrangement='snap', node=dict(pad=15, thickness=18, line=dict(color='white', width=1), label=all_labels, color=node_colors), link=dict(source=source, target=target, value=values, color=link_colors)))
    fig.update_layout(title=dict(text='Laboratory Result Flow Across Age Groups', x=0.5, xanchor='center', font=dict(size=16, color='#16395F')), height=600, autosize=True, paper_bgcolor='white', font=dict(size=10, color='#16395F'), margin=dict(t=75, l=15, r=15, b=30))
    return fig

def create_correlation_heatmap(filtered_df):
    df_corr = filtered_df.copy()
    analysis_cols = ['cci_score', 'lvef', 'brain_natriuretic_peptide', 'high_sensitivity_troponin', 'creatinine_enzymatic_method', 'glomerular_filtration_rate']
    for col in analysis_cols:
        df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')
    correlation_matrix = df_corr[analysis_cols].corr(method='spearman')
    readable_names = {'cci_score': 'CCI Score', 'lvef': 'LVEF', 'brain_natriuretic_peptide': 'BNP', 'high_sensitivity_troponin': 'Troponin', 'creatinine_enzymatic_method': 'Creatinine', 'glomerular_filtration_rate': 'GFR'}
    correlation_matrix = correlation_matrix.rename(index=readable_names, columns=readable_names)
    fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values, x=correlation_matrix.columns, y=correlation_matrix.index, text=np.round(correlation_matrix.values, 2), texttemplate='%{text:.2f}', textfont=dict(size=11), colorscale=[[0.0, '#2166AC'], [0.25, '#67A9CF'], [0.5, '#F7F7F7'], [0.75, '#EF8A62'], [1.0, '#B2182B']], zmin=-1, zmax=1, zmid=0, hovertemplate='<b>Marker 1:</b> %{y}<br><b>Marker 2:</b> %{x}<br><b>Spearman Correlation:</b> %{z:.3f}<extra></extra>', colorbar=dict(title='Correlation', thickness=12, len=0.75)))
    fig.update_layout(title=dict(text='Relationships Between Clinical & Cardiac Markers', x=0.5, xanchor='center', font=dict(size=16)), xaxis=dict(title='Clinical Marker', tickangle=-40), yaxis=dict(title='Clinical Marker', autorange='reversed'), autosize=True, height=520, template='plotly_white', margin=dict(l=75, r=55, t=70, b=90))
    return fig

def create_sunburst_chart(filtered_df, prescriptions):
    patient = filtered_df.copy()
    ID_COL = 'inpatient_number'
    AGE_COL = 'agecat'
    GLUCOSE_COL = 'glucose_blood_gas'
    med_options = ['drug_name', 'medication_name', 'medicine_name', 'prescription_name', 'generic_name', 'drug', 'medication', 'medicine', 'drug_class', 'drug_code']
    MED_COL = next((col for col in med_options if col in prescriptions.columns), None)
    if MED_COL is None:
        raise ValueError('Medication column not found in prescriptions table.')
    required_patient_cols = [ID_COL, AGE_COL, GLUCOSE_COL]
    missing_patient_cols = [col for col in required_patient_cols if col not in patient.columns]
    if missing_patient_cols:
        raise ValueError(f'Missing columns in filtered_df: {missing_patient_cols}')
    if ID_COL not in prescriptions.columns:
        raise ValueError(f"'{ID_COL}' not found in prescriptions table.")
    glucose_data = patient[[ID_COL, AGE_COL, GLUCOSE_COL]].copy()
    glucose_data[GLUCOSE_COL] = pd.to_numeric(glucose_data[GLUCOSE_COL], errors='coerce')
    glucose_data[GLUCOSE_COL] = glucose_data[GLUCOSE_COL].replace([np.inf, -np.inf], np.nan)
    glucose_data = glucose_data.dropna(subset=[ID_COL, AGE_COL, GLUCOSE_COL])
    glucose_data[ID_COL] = glucose_data[ID_COL].astype(str).str.strip()
    glucose_data = glucose_data.groupby(ID_COL, as_index=False).agg({AGE_COL: 'first', GLUCOSE_COL: 'median'})
    med_data = prescriptions[[ID_COL, MED_COL]].copy()
    med_data = med_data.dropna(subset=[ID_COL, MED_COL])
    med_data[ID_COL] = med_data[ID_COL].astype(str).str.strip()
    med_data[MED_COL] = med_data[MED_COL].astype(str).str.strip().str.lower()
    med_data = med_data[med_data[MED_COL] != '']
    med_counts = med_data.groupby(ID_COL)[MED_COL].nunique().reset_index(name='Medication Count')
    sunburst_data = glucose_data.merge(med_counts, on=ID_COL, how='left')
    sunburst_data['Medication Count'] = sunburst_data['Medication Count'].fillna(0).astype(int)

    def medication_group(count):
        if count == 0:
            return 'No Prescription'
        elif count <= 2:
            return '1–2 Medications'
        elif count <= 5:
            return '3–5 Medications'
        else:
            return '6+ Medications'
    sunburst_data['Medication Group'] = sunburst_data['Medication Count'].apply(medication_group)
    lower = sunburst_data[GLUCOSE_COL].quantile(1 / 3)
    upper = sunburst_data[GLUCOSE_COL].quantile(2 / 3)
    sunburst_data['Glucose Level'] = np.select([sunburst_data[GLUCOSE_COL] < lower, sunburst_data[GLUCOSE_COL] > upper], ['Lower Glucose', 'Higher Glucose'], default='Middle Glucose')
    sunburst_data['Age Category'] = sunburst_data[AGE_COL].astype(str)
    summary = sunburst_data.groupby(['Age Category', 'Medication Group', 'Glucose Level'], observed=True).size().reset_index(name='Patient Count')
    if summary.empty:
        fig = go.Figure()
        fig.add_annotation(text='No data available for the selected filters.', x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False, font=dict(size=15, color='#16395F'))
        fig.update_layout(height=520, template='plotly_white')
        return fig
    fig = px.sunburst(summary, path=['Age Category', 'Medication Group', 'Glucose Level'], values='Patient Count', color='Glucose Level', color_discrete_map={'Lower Glucose': '#FFBE0B', 'Middle Glucose': '#2EC4B6', 'Higher Glucose': '#EF476F'})
    age_color = '#DCEAF7'
    medication_color = '#5B8DB8'
    glucose_colors = {'Lower Glucose': '#FFBE0B', 'Middle Glucose': '#2EC4B6', 'Higher Glucose': '#EF476F'}
    trace = fig.data[0]
    new_colors = []
    for label, node_id in zip(trace.labels, trace.ids):
        label = str(label)
        node_id = str(node_id)
        depth = node_id.count('/')
        if depth == 0:
            new_colors.append(age_color)
        elif depth == 1:
            new_colors.append(medication_color)
        else:
            new_colors.append(glucose_colors.get(label, '#D9E2EC'))
    fig.update_traces(marker=dict(colors=new_colors, line=dict(color='white', width=1.5)), textinfo='label+percent parent', insidetextorientation='radial', hovertemplate='<b>%{label}</b><br>Patients: %{value}<br>Percentage: %{percentParent:.1%}<extra></extra>')
    fig.update_layout(title=dict(text='Prescription Medication & Glucose Patterns by Age Group', x=0.5, xanchor='center', font=dict(size=16, color='#16395F')), autosize=True, height=520, template='plotly_white', paper_bgcolor='white', margin=dict(l=20, r=20, t=70, b=20))
    return fig

def create_comorbidity_triad_chart(filtered_df):
    temp_df = filtered_df.copy()
    condition_cols = ['chronic_obstructive_pulmonary_disease', 'diabetes', 'cerebrovascular_disease']
    missing_cols = [col for col in condition_cols if col not in temp_df.columns]
    if missing_cols:
        history_lookup = ph[['inpatient_number'] + missing_cols].copy()
        history_lookup = history_lookup.drop_duplicates(subset='inpatient_number')
        temp_df['inpatient_number'] = temp_df['inpatient_number'].astype(str)
        history_lookup['inpatient_number'] = history_lookup['inpatient_number'].astype(str)
        temp_df = temp_df.merge(history_lookup, on='inpatient_number', how='left')
    for col in condition_cols:
        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0)
    q7_data = temp_df.groupby('agecat', observed=True)[condition_cols].mean().mul(100).reset_index()
    q7_data = q7_data.rename(columns={'chronic_obstructive_pulmonary_disease': 'COPD', 'diabetes': 'Diabetes', 'cerebrovascular_disease': 'Cerebrovascular'})
    temp_df['triad_count'] = temp_df['chronic_obstructive_pulmonary_disease'] + temp_df['diabetes'] + temp_df['cerebrovascular_disease']
    triad_overlap = temp_df.groupby('agecat', observed=True)['triad_count'].apply(lambda x: (x >= 2).mean() * 100).reset_index(name='Triad_Overlap')
    plot_data = q7_data.merge(triad_overlap, on='agecat', how='left')
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Bar(x=plot_data['agecat'], y=plot_data['COPD'], name='COPD', marker_color='#315B7D', hovertemplate='<b>Age Group:</b> %{x}<br><b>COPD:</b> %{y:.1f}%<extra></extra>'), secondary_y=False)
    fig.add_trace(go.Bar(x=plot_data['agecat'], y=plot_data['Diabetes'], name='Diabetes', marker_color='#80CBC4', hovertemplate='<b>Age Group:</b> %{x}<br><b>Diabetes:</b> %{y:.1f}%<extra></extra>'), secondary_y=False)
    fig.add_trace(go.Bar(x=plot_data['agecat'], y=plot_data['Cerebrovascular'], name='Cerebrovascular Disease', marker_color='#F9D44A', hovertemplate='<b>Age Group:</b> %{x}<br><b>Cerebrovascular:</b> %{y:.1f}%<extra></extra>'), secondary_y=False)
    fig.add_trace(go.Scatter(x=plot_data['agecat'], y=plot_data['Triad_Overlap'], mode='lines+markers', name='High-Risk Triad Overlap (2+ Conditions)', line=dict(width=4, color='#C4162A'), marker=dict(size=11, symbol='diamond', color='#C4162A'), hovertemplate='<b>Age Group:</b> %{x}<br><b>2+ Conditions:</b> %{y:.1f}%<extra></extra>'), secondary_y=True)
    fig.update_layout(title=dict(text='<b>Individual Comorbidities vs. Combined Triad Overlap<br>for Discharge Intervention</b>', x=0.5, xanchor='center'), template='plotly_white', barmode='group', hovermode='x unified', autosize=True, height=550, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5), margin=dict(l=70, r=70, t=120, b=70))
    fig.update_xaxes(title_text='<b>Age Category</b>')
    fig.update_yaxes(title_text='<b>Individual Condition Prevalence (%)</b>', secondary_y=False)
    fig.update_yaxes(title_text='<b>Multi-Condition Overlap Rate (%)</b>', secondary_y=True)
    return fig

def create_responsiveness_parallel_chart(filtered_df):
    temp_df = filtered_df.copy()
    if 'bmi' not in temp_df.columns:
        bmi_lookup = demo[['inpatient_number', 'bmi']].copy()
        bmi_lookup = bmi_lookup.drop_duplicates(subset='inpatient_number')
        temp_df['inpatient_number'] = temp_df['inpatient_number'].astype(str)
        bmi_lookup['inpatient_number'] = bmi_lookup['inpatient_number'].astype(str)
        temp_df = temp_df.merge(bmi_lookup, on='inpatient_number', how='left')
    temp_df['bmi'] = pd.to_numeric(temp_df['bmi'], errors='coerce')
    temp_df['Responsiveness'] = temp_df['consciousness'].astype(str).str.strip().apply(lambda x: 'Normal' if x.lower() == 'clear' else 'Abnormal')
    temp_df['BMI Category'] = pd.cut(temp_df['bmi'], bins=[0, 18.5, 25, 30, float('inf')], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    abnormal_df = temp_df[temp_df['Responsiveness'] == 'Abnormal'].copy()
    abnormal_df = abnormal_df.dropna(subset=['agecat', 'gender', 'BMI Category', 'consciousness'])
    color_map = {'Responsive To Sound': 1, 'Responsive To Pain': 2, 'Nonresponsive': 3}
    abnormal_df['Response Color'] = abnormal_df['consciousness'].map(color_map).fillna(1)
    fig = px.parallel_categories(abnormal_df, dimensions=['agecat', 'gender', 'BMI Category', 'consciousness'], color='Response Color', color_continuous_scale=[[0.0, '#2E86DE'], [0.5, '#F39C12'], [1.0, '#C0392B']], title='Demographic Profiles Associated with Abnormal Patient Responsiveness')
    fig.update_layout(autosize=True, height=600, title=dict(x=0.5, xanchor='center', font=dict(size=18, color='#16395F')), font=dict(size=12), margin=dict(l=40, r=40, t=80, b=40))
    return fig

def _binary_target(series):
    """Convert common yes/no, true/false, 1/0 values to a nullable binary target."""
    numeric = pd.to_numeric(series, errors='coerce')
    out = pd.Series(np.nan, index=series.index, dtype='float64')
    valid_numeric = numeric.isin([0, 1])
    out.loc[valid_numeric] = numeric.loc[valid_numeric]
    text = series.astype(str).str.strip().str.lower()
    yes = {'yes', 'y', 'true', '1', 'readmitted', 'abnormal'}
    no = {'no', 'n', 'false', '0', 'not readmitted', 'normal'}
    out.loc[text.isin(yes)] = 1
    out.loc[text.isin(no)] = 0
    return out

def _fit_binary_models(data, feature_cols, target, random_state=42):
    """Fit balanced Logistic Regression and Random Forest models."""
    work = data[feature_cols].copy()
    y = pd.Series(target, index=data.index, dtype='float64')
    valid = y.notna()
    X = work.loc[valid].copy()
    y = y.loc[valid].astype(int)
    if y.nunique() < 2:
        raise ValueError('The target must contain both 0 and 1 classes.')
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    transformers = []
    if numeric_features:
        transformers.append(('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features))
    if categorical_features:
        transformers.append(('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers)
    class_counts = y.value_counts()
    stratify_y = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=stratify_y)
    logistic_model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=random_state))])
    random_forest_model = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight='balanced', random_state=random_state, n_jobs=-1))])
    logistic_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)
    log_pred = logistic_model.predict(X_test)
    log_prob = logistic_model.predict_proba(X_test)[:, 1]
    rf_pred = random_forest_model.predict(X_test)
    rf_prob = random_forest_model.predict_proba(X_test)[:, 1]
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'logistic_model': logistic_model, 'random_forest_model': random_forest_model, 'log_pred': log_pred, 'log_prob': log_prob, 'rf_pred': rf_pred, 'rf_prob': rf_prob}

def _model_metrics(model_name, actual, prediction, probability):
    return {'Model': model_name, 'Accuracy': accuracy_score(actual, prediction), 'Precision': precision_score(actual, prediction, zero_division=0), 'Recall': recall_score(actual, prediction, zero_division=0), 'F1 Score': f1_score(actual, prediction, zero_division=0), 'ROC-AUC': roc_auc_score(actual, probability)}

def create_roc_curve(y_test, log_prob, rf_prob):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    RocCurveDisplay.from_predictions(y_test, log_prob, name='Logistic Regression', ax=ax)
    RocCurveDisplay.from_predictions(y_test, rf_prob, name='Random Forest', ax=ax)
    ax.plot([0, 1], [0, 1], 'k--', label='Chance')
    ax.set_title('ROC Curve: Readmission Prediction', fontsize=15, color='#16395F', pad=15)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(alpha=0.25)
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig

def pct_yes(series):
    s=series.dropna(); n=pd.to_numeric(s,errors="coerce")
    if n.notna().any():
        v=n.dropna(); return v.mean()*100 if v.between(0,1).all() else v.mean()
    return s.astype(str).str.strip().str.lower().isin({"yes","y","true","1","readmitted"}).mean()*100

def opts(s): return ["All"]+sorted(s.dropna().astype(str).unique().tolist())

st.markdown('<div class="hero"><h1>Team 7 CodeAvengers — Heart Failure Analytics Dashboard</h1><p>Descriptive + Prescriptive + Predictive Outcome Analysis</p></div>',unsafe_allow_html=True)
st.markdown("#### Dashboard Filters")
c1,c2,c3,c4=st.columns(4)
with c1: age=st.selectbox("Age Category",opts(dashboard_df["agecat"]))
with c2: gender=st.selectbox("Gender",opts(dashboard_df["gender"]))
with c3: nyha=st.selectbox("NYHA Classification",opts(dashboard_df["nyha_cardiac_function_classification"]))
with c4: hf=st.selectbox("Heart Failure Type",opts(dashboard_df["type_of_heart_failure"]))
filtered=dashboard_df.copy()
for col,val in [("agecat",age),("gender",gender),("nyha_cardiac_function_classification",nyha),("type_of_heart_failure",hf)]:
    if val!="All": filtered=filtered[filtered[col].astype(str)==val]

k1,k2,k3,k4,k5,k6=st.columns(6)
k1.metric("Total Patients",f"{filtered.inpatient_number.nunique():,}")
k2.metric("Abnormal Responsiveness",f"{filtered.loc[filtered.Abnormal_Response,'inpatient_number'].nunique():,}")
k3.metric("High Comorbidity",f"{filtered.loc[filtered.cci_score>=3,'inpatient_number'].nunique():,}")
avg_bnp=filtered.brain_natriuretic_peptide.mean(); avg_lvef=filtered.lvef.mean()
k4.metric("Average BNP","N/A" if pd.isna(avg_bnp) else f"{avg_bnp:,.1f}")
k5.metric("Average LVEF","N/A" if pd.isna(avg_lvef) else f"{avg_lvef:.1f}%")
k6.metric("6-Month Readmission",f"{pct_yes(filtered.re_admission_within_6_months):.1f}%")

t1,t2,t3,t4,t5=st.tabs(["Introduction","Descriptive Analysis","Prescriptive Analysis","Predictive Analysis","Conclusion"])
with t1:
    st.markdown("""<div class="section-card"><h2 class="blue">Heart Failure Analytics Dashboard</h2><h3 class="red">Introduction</h3><p>This dashboard combines descriptive, prescriptive, and predictive analytics to examine heart-failure patient characteristics, clinical biomarkers, cardiac function, responsiveness, readmission, and outcome patterns.</p><p>Use the filters above to focus the descriptive and prescriptive views on specific age, gender, NYHA, or heart-failure groups.</p></div>""",unsafe_allow_html=True)
with t2:
    a,b=st.columns(2)
    with a: st.plotly_chart(create_egfr_chart(filtered),use_container_width=True)
    with b: st.plotly_chart(create_radar_chart(filtered),use_container_width=True)
    biomarker_fig = create_biomarker_boxplots(filtered)
    st.pyplot(biomarker_fig, use_container_width=True)
    plt.close(biomarker_fig)
    st.plotly_chart(create_responsiveness_parallel_chart(filtered),use_container_width=True)
with t3:
    a,b=st.columns(2)
    with a: st.plotly_chart(create_cardiac_priority_chart(filtered),use_container_width=True)
    with b: st.plotly_chart(create_lab_sankey(filtered),use_container_width=True)
    a,b=st.columns(2)
    with a: st.plotly_chart(create_correlation_heatmap(filtered),use_container_width=True)
    with b: st.plotly_chart(create_sunburst_chart(filtered,pp),use_container_width=True)
    st.plotly_chart(create_comorbidity_triad_chart(filtered),use_container_width=True)
with t4:
    st.subheader("Predictive Question 1 — 6-Month Readmission")
    q1_features=[c for c in ["agecat","gender","nyha_cardiac_function_classification","type_of_heart_failure","cci_score","lvef","brain_natriuretic_peptide","creatinine_enzymatic_method","glomerular_filtration_rate","high_sensitivity_troponin","hemoglobin","albumin","glucose_blood_gas","urea","consciousness"] if c in dashboard_df.columns]
    try:
        q1=_fit_binary_models(dashboard_df,q1_features,_binary_target(dashboard_df["re_admission_within_6_months"]))
        q1cmp=pd.DataFrame([_model_metrics("Logistic Regression",q1["y_test"],q1["log_pred"],q1["log_prob"]),_model_metrics("Random Forest",q1["y_test"],q1["rf_pred"],q1["rf_prob"])])
        st.dataframe(q1cmp.round(3),use_container_width=True,hide_index=True)
        fig=create_roc_curve(q1["y_test"],q1["log_prob"],q1["rf_prob"]); st.pyplot(fig,use_container_width=True); plt.close(fig)
    except Exception as e: st.warning(f"Question 1 could not be calculated: {e}")
    st.subheader("Predictive Question 2 — Abnormal Responsiveness")
    q2_features=[c for c in ["agecat","gender","nyha_cardiac_function_classification","type_of_heart_failure","cci_score","lvef","brain_natriuretic_peptide","creatinine_enzymatic_method","glomerular_filtration_rate","high_sensitivity_troponin","hemoglobin","albumin","glucose_blood_gas","urea"] if c in dashboard_df.columns]
    try:
        q2=_fit_binary_models(dashboard_df,q2_features,dashboard_df["Abnormal_Response"].astype(int))
        q2cmp=pd.DataFrame([_model_metrics("Logistic Regression",q2["y_test"],q2["log_pred"],q2["log_prob"]),_model_metrics("Random Forest",q2["y_test"],q2["rf_pred"],q2["rf_prob"])])
        st.dataframe(q2cmp.round(3),use_container_width=True,hide_index=True)
        chart=q2cmp.melt(id_vars="Model",value_vars=["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],var_name="Metric",value_name="Score")
        st.plotly_chart(px.bar(chart,x="Metric",y="Score",color="Model",barmode="group",title="Model Performance: Abnormal Responsiveness"),use_container_width=True)
    except Exception as e: st.warning(f"Question 2 could not be calculated: {e}")
    st.info("Predictive results are analytical outputs from the project dataset and are not a substitute for clinical judgment.")
with t5:
    st.markdown("""<div class="section-card"><h2 class="blue">Conclusion</h2><p>This heart failure analysis combines descriptive, prescriptive, and predictive approaches to provide a broad view of patient health and outcomes. The dashboard examines demographic characteristics, comorbidity burden, laboratory biomarkers, cardiac function, responsiveness, and readmission patterns.</p><h3 class="red">Key Takeaway</h3><p>The analysis supports data-driven review of clinically important patient patterns and groups that may warrant closer monitoring, follow-up planning, and earlier clinical review.</p><p><b>Clinical Note:</b> These analytical findings should complement, rather than replace, professional medical assessment and clinical judgment.</p></div>""",unsafe_allow_html=True)
