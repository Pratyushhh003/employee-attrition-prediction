# ============================================================
# Employee Attrition Prediction - Streamlit Dashboard
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👔",
    layout="wide"
)

# ------------------------------
# Load Model & Preprocessor
# ------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    feature_names = joblib.load("data/processed/feature_names.pkl")
    return model, preprocessor, feature_names

model_loaded = False
try:
    model, preprocessor, feature_names = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.info("""Make sure these files exist in your project folder:
- models/best_model.pkl
- models/preprocessor.pkl
- data/processed/feature_names.pkl""")

# ------------------------------
# Title
# ------------------------------
st.title("👔 Employee Attrition Prediction System")
st.markdown("### Predict the risk of an employee leaving the company")
st.markdown("---")

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("About the Project")
    st.write("""
    This system predicts whether an employee is likely to leave the organization 
    based on various HR attributes.
    
    **How to use:**
    1. Fill in the employee details
    2. Click **Predict Attrition Risk**
    3. View the probability and risk level
    """)
    st.markdown("---")
    st.caption("Final Year Machine Learning Project")

# ------------------------------
# Input Form
# ------------------------------
st.subheader("Enter Employee Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=60, value=32)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000, step=100)
    total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=8)
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=4)
    years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=2)

with col2:
    department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ])
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    education = st.selectbox(
        "Education",
        [1, 2, 3, 4, 5],
        format_func=lambda x: {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}[x]
    )

with col3:
    job_satisfaction = st.selectbox(
        "Job Satisfaction", [1, 2, 3, 4],
        format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x]
    )
    environment_satisfaction = st.selectbox(
        "Environment Satisfaction", [1, 2, 3, 4],
        format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x]
    )
    work_life_balance = st.selectbox(
        "Work Life Balance", [1, 2, 3, 4],
        format_func=lambda x: {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}[x]
    )
    job_involvement = st.selectbox(
        "Job Involvement", [1, 2, 3, 4],
        format_func=lambda x: {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}[x]
    )
    distance_from_home = st.number_input("Distance From Home (km)", min_value=1, max_value=30, value=8)

# Additional Details
with st.expander("Additional Details (Optional)"):
    col4, col5 = st.columns(2)
    with col4:
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education_field = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"
        ])
    with col5:
        stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=10, value=2)
        training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2)
        years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
        years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=15, value=2)

# ------------------------------
# Predict Button
# ------------------------------
st.markdown("---")
predict_btn = st.button("🔍 Predict Attrition Risk", type="primary", use_container_width=True)

if predict_btn and model_loaded:

    # Create input dictionary
    input_data = {
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": 800,
        "Department": department,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": environment_satisfaction,
        "Gender": gender,
        "HourlyRate": 65,
        "JobInvolvement": job_involvement,
        "JobLevel": 2,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": 14000,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": overtime,
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
        "YearsAtOtherCompanies": total_working_years - years_at_company,
        "IncomePerYearExp": monthly_income / (total_working_years + 1),
        "RoleStability": years_in_current_role / (years_at_company + 1),
        "PromotionGap": years_at_company - years_since_last_promotion,
        "AgeWhenStarted": age - total_working_years
    }

    input_df = pd.DataFrame([input_data])

    try:
        # Transform input
        input_processed = preprocessor.transform(input_df)

        # Make prediction
        probability = model.predict_proba(input_processed)[0][1]
        prediction = model.predict(input_processed)[0]

        # Results Section
        st.markdown("## Prediction Result")

        if probability >= 0.65:
            risk_level = "🔴 High Risk"
            advice = "Immediate attention recommended. Consider retention strategies."
        elif probability >= 0.40:
            risk_level = "🟠 Medium Risk"
            advice = "Monitor this employee. Schedule a discussion if needed."
        else:
            risk_level = "🟢 Low Risk"
            advice = "Employee appears stable. Continue normal engagement."

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Attrition Probability", f"{probability*100:.1f}%")
        with col_b:
            st.metric("Risk Level", risk_level)
        with col_c:
            st.metric("Prediction", "Will Leave" if prediction == 1 else "Will Stay")

        st.info(advice)
        st.progress(float(probability))

        st.markdown("### Key Factors that usually drive attrition:")
        st.markdown("""
        - Working **OverTime**
        - Low **Job Satisfaction** or **Environment Satisfaction**
        - Lower **Monthly Income**
        - Fewer **Years at Company**
        - Certain **Job Roles** and **Departments** (especially Sales)
        """)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("This usually happens if the input columns don't exactly match the training data.")

elif predict_btn and not model_loaded:
    st.warning("Model not loaded. Please check the model files.")