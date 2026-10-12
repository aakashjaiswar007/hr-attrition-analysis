import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from data_cleaning import load_and_clean
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="HR Attrition Analysis",
    page_icon="👥",
    layout="wide"
)

df = load_and_clean()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.sidebar.title("🔍 Filters")
dept = st.sidebar.multiselect(
    "Department",
    options=df['Department'].unique(),
    default=df['Department'].unique()
)

filtered_df = df[df['Department'].isin(dept)]

st.title("👥 HR Attrition Intelligence Dashboard")
st.markdown("Analyzing **1,470 IBM employees** to predict and prevent attrition")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Employees", len(filtered_df))
with col2:
    attrition_rate = filtered_df['Attrition'].mean() * 100
    st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
with col3:
    avg_salary = filtered_df['MonthlyIncome'].mean()
    st.metric("Avg Monthly Income", f"${avg_salary:,.0f}")
with col4:
    avg_tenure = filtered_df['YearsAtCompany'].mean()
    st.metric("Avg Tenure", f"{avg_tenure:.1f} yrs")

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader(" Attrition by Department")
    dept_attrition = filtered_df.groupby('Department')['Attrition']\
                                .mean().reset_index()
    dept_attrition['Attrition'] *= 100
    fig = px.bar(dept_attrition, x='Department', y='Attrition',
                 color='Attrition',
                 color_continuous_scale='Reds',
                 labels={'Attrition': 'Attrition Rate %'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💰 Income vs Attrition")
    fig2 = px.box(filtered_df, x='Attrition', y='MonthlyIncome',
                  color='Attrition',
                  labels={'Attrition': '0=Stayed, 1=Left'},
                  color_discrete_map={0: 'green', 1: 'red'})
    st.plotly_chart(fig2, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("⏰ Overtime Impact")
    ot = filtered_df.groupby('OverTime')['Attrition']\
                    .mean().reset_index()
    ot['OverTime'] = ot['OverTime'].map({1: 'Overtime', 0: 'No Overtime'})
    ot['Attrition'] *= 100
    fig3 = px.pie(ot, values='Attrition', names='OverTime',
                  color_discrete_sequence=['#EF553B', '#00CC96'])
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("Top Attrition Drivers")
    importances = pd.Series(
        model.feature_importances_,
        index=model.feature_names_in_
    ).sort_values(ascending=True).tail(10)
    fig4 = px.bar(importances, orientation='h',
                  color=importances.values,
                  color_continuous_scale='Blues')
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.subheader("Predict Employee Attrition Risk")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", 18, 60, 30)
    monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
    overtime = st.selectbox("Works Overtime?", ["Yes", "No"])
with col2:
    years_at_company = st.slider("Years at Company", 0, 40, 5)
    job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    distance = st.slider("Distance from Home (km)", 1, 30, 10)
with col3:
    num_companies = st.slider("Num Companies Worked", 0, 10, 2)
    work_life = st.slider("Work Life Balance (1-4)", 1, 4, 3)
    job_level = st.slider("Job Level (1-5)", 1, 5, 2)

if st.button("🔮 Predict Attrition Risk", type="primary"):
    df_encoded = load_and_clean()
    le = LabelEncoder()
    cat_cols = df_encoded.select_dtypes(include='object').columns
    for col in cat_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    input_data = pd.DataFrame(
        [np.zeros(len(df_encoded.drop('Attrition', axis=1).columns))],
        columns=df_encoded.drop('Attrition', axis=1).columns
    )
    input_data['Age'] = age
    input_data['MonthlyIncome'] = monthly_income
    input_data['OverTime'] = 1 if overtime == "Yes" else 0
    input_data['YearsAtCompany'] = years_at_company
    input_data['JobSatisfaction'] = job_satisfaction
    input_data['DistanceFromHome'] = distance
    input_data['NumCompaniesWorked'] = num_companies
    input_data['WorkLifeBalance'] = work_life
    input_data['JobLevel'] = job_level

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"HIGH RISK — {probability:.1%} probability of leaving")
        st.markdown("**Recommendations:** Consider salary revision, reduce overtime")
    else:
        st.success(f"LOW RISK — {probability:.1%} probability of leaving")
        st.markdown("**Employee appears stable** — continue engagement")