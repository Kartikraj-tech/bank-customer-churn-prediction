import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Bank Churn Intelligence",
    page_icon="🏦",
    layout="wide"
)


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

MODEL_FILE = "bank_churn_model.pkl"
DATA_FILE = "European_Bank.csv"

# These are the exact features used by the final model.
FEATURES = [
    "Year",
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
    "Gender_Male",
    "BalanceSalaryRatio",
    "ProductDensity",
    "EngagementProduct",
    "AgeTenureInteraction"
]


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


try:
    model = load_model()
    df = load_data()

except FileNotFoundError:
    st.error(
        "Could not find the model or dataset. "
        "Make sure bank_churn_model.pkl and European_Bank.csv "
        "are in the same folder as app.py."
    )
    st.stop()


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

def create_features(data):
    """
    Apply the same preprocessing and feature engineering
    used while training the model.
    """

    data = data.copy()

    # Remove identifiers
    data = data.drop(
        columns=["CustomerId", "Surname"],
        errors="ignore"
    )

    # One-hot encode categorical variables
    data = pd.get_dummies(
        data,
        columns=["Geography", "Gender"],
        drop_first=True
    )

    # Required engineered features
    data["BalanceSalaryRatio"] = (
        data["Balance"] /
        (data["EstimatedSalary"] + 1e-6)
    )

    data["ProductDensity"] = (
        data["NumOfProducts"] /
        (data["Tenure"] + 1)
    )

    data["EngagementProduct"] = (
        data["IsActiveMember"] *
        data["NumOfProducts"]
    )

    data["AgeTenureInteraction"] = (
        data["Age"] *
        data["Tenure"]
    )

    # Make sure the columns are in exactly the
    # same order as during model training.
    data = data.reindex(
        columns=FEATURES,
        fill_value=0
    )

    return data


# ---------------------------------------------------------
# RISK FUNCTION
# ---------------------------------------------------------

def get_risk_level(probability):

    if probability <= 0.30:
        return "Low"

    elif probability <= 0.60:
        return "Medium"

    elif probability <= 0.80:
        return "High"

    else:
        return "Very High"


# ---------------------------------------------------------
# CUSTOMER INPUT
# ---------------------------------------------------------

st.sidebar.title("👤 Customer Details")

st.sidebar.markdown(
    "Enter customer information to calculate churn risk."
)

year = st.sidebar.number_input(
    "Year",
    min_value=2025,
    max_value=2025,
    value=2025,
    disabled=True
)

credit_score = st.sidebar.slider(
    "Credit Score",
    min_value=350,
    max_value=850,
    value=650
)

geography = st.sidebar.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=92,
    value=38
)

tenure = st.sidebar.slider(
    "Tenure (Years)",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.sidebar.number_input(
    "Balance",
    min_value=0.0,
    max_value=300000.0,
    value=75000.0,
    step=1000.0
)

num_products = st.sidebar.selectbox(
    "Number of Products",
    [1, 2, 3, 4]
)

has_credit_card = st.sidebar.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

is_active = st.sidebar.selectbox(
    "Active Member?",
    ["Yes", "No"]
)

estimated_salary = st.sidebar.number_input(
    "Estimated Salary",
    min_value=0.01,
    max_value=250000.0,
    value=100000.0,
    step=1000.0
)


# ---------------------------------------------------------
# CREATE CUSTOMER DATAFRAME
# ---------------------------------------------------------

def build_customer(
    year,
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_credit_card,
    is_active,
    estimated_salary
):

    customer = pd.DataFrame({
        "Year": [year],
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [
            1 if has_credit_card == "Yes" else 0
        ],
        "IsActiveMember": [
            1 if is_active == "Yes" else 0
        ],
        "EstimatedSalary": [estimated_salary]
    })

    return create_features(customer)


customer_data = build_customer(
    year,
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_credit_card,
    is_active,
    estimated_salary
)


# ---------------------------------------------------------
# CALCULATE CURRENT CUSTOMER RISK
# ---------------------------------------------------------

customer_probability = model.predict_proba(
    customer_data
)[0, 1]

customer_flag = int(customer_probability >= 0.50)

customer_risk = get_risk_level(
    customer_probability
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🏦 Bank Customer Churn Intelligence")

st.write(
    "Predictive modeling and risk scoring system "
    "for identifying customers who may churn."
)

st.markdown("---")


# ---------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Risk Calculator",
        "📊 Churn Analytics",
        "📌 Feature Importance",
        "🧪 What-If Simulator"
    ]
)


# =========================================================
# TAB 1 — RISK CALCULATOR
# =========================================================

with tab1:

    st.header("Customer Churn Risk Calculator")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{customer_probability:.2%}"
        )

    with col2:
        st.metric(
            "Risk Level",
            customer_risk
        )

    with col3:

        if customer_flag == 1:
            st.metric(
                "Churn Flag",
                "Churn"
            )
        else:
            st.metric(
                "Churn Flag",
                "Retained"
            )

    st.markdown("---")

    st.subheader("Risk Interpretation")

    if customer_risk == "Low":
        st.success(
            "This customer currently has a relatively low "
            "predicted churn probability."
        )

    elif customer_risk == "Medium":
        st.warning(
            "This customer has a moderate predicted churn risk "
            "and may benefit from additional monitoring."
        )

    elif customer_risk == "High":
        st.warning(
            "This customer has a high predicted churn risk. "
            "A targeted retention strategy may be considered."
        )

    else:
        st.error(
            "This customer has a very high predicted churn risk. "
            "The customer may require priority attention."
        )

    st.subheader("Customer Information")

    profile_col1, profile_col2 = st.columns(2)

    with profile_col1:

        st.write(f"**Age:** {age}")
        st.write(f"**Credit Score:** {credit_score}")
        st.write(f"**Geography:** {geography}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Tenure:** {tenure} years")

    with profile_col2:

        st.write(f"**Balance:** {balance:,.2f}")
        st.write(f"**Products:** {num_products}")
        st.write(
            f"**Credit Card:** {has_credit_card}"
        )
        st.write(
            f"**Active Member:** {is_active}"
        )
        st.write(
            f"**Estimated Salary:** "
            f"{estimated_salary:,.2f}"
        )

    st.caption(
        "Risk bands used in this project: "
        "Low ≤ 30%, Medium > 30–60%, "
        "High > 60–80%, Very High > 80%."
    )


# =========================================================
# TAB 2 — CHURN ANALYTICS
# =========================================================

with tab2:

    st.header("Churn Analytics")

    prepared_data = create_features(df)

    all_probabilities = model.predict_proba(
        prepared_data
    )[:, 1]

    total_customers = len(df)

    actual_churn_rate = (
        df["Exited"].mean()
    )

    high_risk_count = np.sum(
        all_probabilities >= 0.60
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "Actual Churn Rate",
            f"{actual_churn_rate:.2%}"
        )

    with col3:
        st.metric(
            "Predicted High-Risk Customers",
            f"{high_risk_count:,}"
        )

    st.markdown("---")

    # Probability distribution
    st.subheader("Churn Probability Distribution")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        all_probabilities,
        bins=20,
        edgecolor="black"
    )

    ax.set_title(
        "Distribution of Predicted Churn Probabilities"
    )

    ax.set_xlabel("Churn Probability")
    ax.set_ylabel("Number of Customers")

    st.pyplot(fig)

    st.markdown("---")

    # Actual churn distribution
    st.subheader("Actual Customer Churn")

    churn_counts = df["Exited"].value_counts().sort_index()

    churn_display = pd.DataFrame({
        "Customer Status": [
            "Stayed",
            "Churned"
        ],
        "Customers": [
            churn_counts.get(0, 0),
            churn_counts.get(1, 0)
        ]
    })

    st.bar_chart(
        churn_display.set_index(
            "Customer Status"
        )
    )


# =========================================================
# TAB 3 — FEATURE IMPORTANCE
# =========================================================

with tab3:

    st.header("Feature Importance Dashboard")

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=True)

    st.subheader(
        "Gradient Boosting Feature Importance"
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    importance.plot(
        kind="barh",
        ax=ax
    )

    ax.set_title(
        "Feature Importance - Gradient Boosting"
    )

    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Top Churn Drivers")

    top_features = (
        importance
        .sort_values(ascending=False)
        .head(10)
        .rename("Importance")
        .to_frame()
    )

    top_features["Importance"] = (
        top_features["Importance"]
        .round(4)
    )

    st.dataframe(
        top_features,
        use_container_width=True
    )


# =========================================================
# TAB 4 — WHAT-IF SIMULATOR
# =========================================================

with tab4:

    st.header("🧪 What-If Scenario Simulator")

    st.write(
        "Change engagement and product values to see "
        "how the model's predicted churn probability changes."
    )

    st.markdown("---")

    st.subheader("Scenario Settings")

    scenario_col1, scenario_col2 = st.columns(2)

    with scenario_col1:

        scenario_products = st.selectbox(
            "Scenario: Number of Products",
            [1, 2, 3, 4],
            index=[1, 2, 3, 4].index(num_products)
        )

    with scenario_col2:

        scenario_active = st.selectbox(
            "Scenario: Active Member",
            ["Yes", "No"],
            index=(
                0 if is_active == "Yes" else 1
            )
        )

    scenario_data = build_customer(
        year,
        credit_score,
        geography,
        gender,
        age,
        tenure,
        balance,
        scenario_products,
        has_credit_card,
        scenario_active,
        estimated_salary
    )

    scenario_probability = model.predict_proba(
        scenario_data
    )[0, 1]

    scenario_risk = get_risk_level(
        scenario_probability
    )

    probability_change = (
        scenario_probability -
        customer_probability
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Current Customer")

        st.metric(
            "Current Churn Probability",
            f"{customer_probability:.2%}"
        )

        st.write(
            f"Risk Level: **{customer_risk}**"
        )

    with col2:

        st.subheader("What-If Scenario")

        st.metric(
            "Scenario Churn Probability",
            f"{scenario_probability:.2%}",
            delta=f"{probability_change:+.2%}"
        )

        st.write(
            f"Risk Level: **{scenario_risk}**"
        )

    st.markdown("---")

    if probability_change < 0:

        st.success(
            f"The scenario lowers the model's predicted "
            f"churn probability by "
            f"{abs(probability_change):.2%}."
        )

    elif probability_change > 0:

        st.warning(
            f"The scenario increases the model's predicted "
            f"churn probability by "
            f"{probability_change:.2%}."
        )

    else:

        st.info(
            "The scenario produces the same predicted "
            "churn probability."
        )

    st.caption(
        "What-if results represent changes in model predictions. "
        "They should not be interpreted as proof of a causal effect."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Bank Customer Churn Prediction | "
    "Gradient Boosting Model | "
    "Predictive risk scoring dashboard"
)