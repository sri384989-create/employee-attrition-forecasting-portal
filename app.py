 import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")
st.title("📊 Employee Attrition Forecasting Portal")

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Sidebar filters
st.sidebar.header("Filters")
dept = st.sidebar.multiselect("Department", df["Department"].unique(), default=df["Department"].unique())
filtered_df = df[df["Department"].isin(dept)]

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", len(filtered_df))
attrition_count = (filtered_df["Attrition"] == "Yes").sum()
col2.metric("Employees Left", attrition_count)
col3.metric("Attrition Rate", f"{(attrition_count/len(filtered_df)*100):.1f}%")

# Attrition chart
st.subheader("Attrition Count")
fig1, ax1 = plt.subplots()
filtered_df["Attrition"].value_counts().plot(kind="bar", color=["green","red"], ax=ax1)
st.pyplot(fig1)

# Department wise
st.subheader("Attrition by Department")
fig2, ax2 = plt.subplots()
sns.countplot(data=filtered_df, x="Department", hue="Attrition", ax=ax2)
st.pyplot(fig2)

# Data table
st.subheader("Employee Data")
st.dataframe(filtered_df)
