import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Employee Attrition Forecasting Portal",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Attrition Forecasting Portal")
st.write("HR Analytics | Machine Learning | Employee Retention")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# -----------------------------
# FEATURES
# -----------------------------
features = [
    "Age",
    "BusinessTravel",
    "Department",
    "DistanceFromHome",
    "JobRole",
    "JobSatisfaction",
    "MonthlyIncome",
    "OverTime",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "JobInvolvement",
    "PerformanceRating",
    "StockOptionLevel",
    "MaritalStatus",
    "NumCompaniesWorked",
    "TrainingTimesLastYear"
]

categorical_features = [
    "BusinessTravel",
    "Department",
    "JobRole",
    "OverTime",
    "MaritalStatus"
]

numeric_features = [
    col for col in features
    if col not in categorical_features
]

# -----------------------------
# TRAIN MODEL
# -----------------------------
@st.cache_resource
def train_model(data):

    X = data[features]
    y = data["Attrition"].map({
        "Yes": 1,
        "No": 0
    })

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "num",
                "passthrough",
                numeric_features
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric="logloss"
                )
            )
        ]
    )

    model.fit(X, y)

    return model


model = train_model(df)

# -----------------------------
# SIDEBAR FILTER
# -----------------------------
st.sidebar.header("🔎 HR Dashboard Filters")

dept = st.sidebar.multiselect(
    "Department",
    df["Department"].unique(),
    default=df["Department"].unique()
)

filtered_df = df[
    df["Department"].isin(dept)
]

# -----------------------------
# PREDICTION SECTION
# -----------------------------
st.header("🔮 Employee Attrition Prediction")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        18,
        70,
        30
    )

    distance = st.number_input(
        "Distance From Home",
        1,
        30,
        5
    )

    monthly_income = st.number_input(
        "Monthly Income",
        1000,
        50000,
        5000
    )

    department = st.selectbox(
        "Department",
        df["Department"].unique()
    )

    job_role = st.selectbox(
        "Job Role",
        df["JobRole"].unique()
    )

with col2:

    business_travel = st.selectbox(
        "Business Travel",
        df["BusinessTravel"].unique()
    )

    overtime = st.selectbox(
        "OverTime",
        df["OverTime"].unique()
    )

    job_satisfaction = st.slider(
        "Job Satisfaction",
        1,
        4,
        3
    )

    environment_satisfaction = st.slider(
        "Environment Satisfaction",
        1,
        4,
        3
    )

    work_life_balance = st.slider(
        "Work Life Balance",
        1,
        4,
        3
    )

with col3:

    marital_status = st.selectbox(
        "Marital Status",
        df["MaritalStatus"].unique()
    )

    total_working_years = st.number_input(
        "Total Working Years",
        0,
        50,
        5
    )

    years_at_company = st.number_input(
        "Years At Company",
        0,
        40,
        3
    )

    years_current_role = st.number_input(
        "Years In Current Role",
        0,
        20,
        2
    )

    years_promotion = st.number_input(
        "Years Since Last Promotion",
        0,
        20,
        1
    )

years_manager = st.number_input(
    "Years With Current Manager",
    0,
    20,
    2
)

job_involvement = st.slider(
    "Job Involvement",
    1,
    4,
    3
)

performance_rating = st.slider(
    "Performance Rating",
    1,
    4,
    3
)

stock_option = st.slider(
    "Stock Option Level",
    0,
    3,
    1
)

num_companies = st.number_input(
    "Number of Companies Worked",
    0,
    10,
    1
)

training = st.number_input(
    "Training Times Last Year",
    0,
    10,
    3
)

# -----------------------------
# PREDICT
# -----------------------------
if st.button("🔮 Predict Attrition"):

    employee = pd.DataFrame([{

        "Age": age,
        "BusinessTravel": business_travel,
        "Department": department,
        "DistanceFromHome": distance,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "OverTime": overtime,
        "TotalWorkingYears": total_working_years,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_current_role,
        "YearsSinceLastPromotion": years_promotion,
        "YearsWithCurrManager": years_manager,
        "EnvironmentSatisfaction": environment_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "JobInvolvement": job_involvement,
        "PerformanceRating": performance_rating,
        "StockOptionLevel": stock_option,
        "MaritalStatus": marital_status,
        "NumCompaniesWorked": num_companies,
        "TrainingTimesLastYear": training

    }])

    probability = model.predict_proba(employee)[0][1]

    st.subheader("📈 Prediction Result")

    st.metric(
        "Attrition Probability",
        f"{probability * 100:.2f}%"
    )

    if probability >= 0.70:

        st.error("🔴 High Risk")

    elif probability >= 0.40:

        st.warning("🟠 Medium Risk")

    else:

        st.success("🟢 Low Risk")

# -----------------------------
# HR DASHBOARD
# -----------------------------
st.divider()

st.header("📊 HR Analytics Dashboard")

# KPI
col1, col2, col3 = st.columns(3)

total_employees = len(filtered_df)

employees_left = (
    filtered_df["Attrition"] == "Yes"
).sum()

if total_employees > 0:
    attrition_rate = (
        employees_left / total_employees
    ) * 100
else:
    attrition_rate = 0

col1.metric(
    "👥 Total Employees",
    total_employees
)

col2.metric(
    "🚪 Employees Left",
    employees_left
)

col3.metric(
    "📈 Attrition Rate",
    f"{attrition_rate:.1f}%"
)

# -----------------------------
# ATTRITION COUNT
# -----------------------------
st.subheader("📌 Attrition Count")

attrition_count = (
    filtered_df["Attrition"]
    .value_counts()
)

st.bar_chart(attrition_count)

# -----------------------------
# DEPARTMENT ATTRITION
# -----------------------------
st.subheader("🏢 Attrition by Department")

department_attrition = pd.crosstab(
    filtered_df["Department"],
    filtered_df["Attrition"]
)

st.bar_chart(department_attrition)

# -----------------------------
# JOB ROLE ATTRITION
# -----------------------------
st.subheader("💼 Attrition by Job Role")

jobrole_attrition = pd.crosstab(
    filtered_df["JobRole"],
    filtered_df["Attrition"]
)

st.bar_chart(jobrole_attrition)

# -----------------------------
# JOB SATISFACTION
# -----------------------------
st.subheader("😊 Job Satisfaction vs Attrition")

satisfaction_attrition = pd.crosstab(
    filtered_df["JobSatisfaction"],
    filtered_df["Attrition"]
)

st.bar_chart(satisfaction_attrition)

# -----------------------------
# YEARS AT COMPANY
# -----------------------------
st.subheader("📅 Years at Company vs Attrition")

years_attrition = pd.crosstab(
    filtered_df["YearsAtCompany"],
    filtered_df["Attrition"]
)

st.bar_chart(years_attrition)

# -----------------------------
# EMPLOYEE DATA
# -----------------------------
st.subheader("📋 Employee Data")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# -----------------------------
# FOOTER
# -----------------------------
st.divider()

st.caption(
    "Academic/demo model. Predictions should support, "
    "not replace, HR decision-making."
)
