import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# GLOBAL INTERFACE DESIGN & COLOR PALETTE

PRIMARY_COLOR = "#1f77b4"   # Engineering Blue
DANGER_COLOR  = "#d62728"   # Churn Red
SUCCESS_COLOR = "#2ca02c"   # Retention Green
NEUTRAL_COLOR = "#ff7f0e"   # Warning Orange

# Establish unified statistical plotting styles across all figures
sns.set_style("whitegrid")
sns.set_palette([PRIMARY_COLOR, DANGER_COLOR, SUCCESS_COLOR, NEUTRAL_COLOR])

# Initialize global configuration parameters for Streamlit layout
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Analysis Dashboard")
st.markdown("**Goal:** Identify critical behavioral drivers and segment customer cohorts by churn vulnerability.")

# PIPELINE DATA LOADING AND ENGINNERING (CACHED)

@st.cache_data
def load_data():
    """Reads raw customer relational telemetry data directly from local filesystem."""
    return pd.read_csv("C:\\Users\\Hedaya_city\\Downloads\\WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

# Data Cleaning: Coerce financial charges to numeric and handle structural nulls
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Feature Encoding: Map binary categorical markers to standard integers
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# Operational Pipeline: Homogenize redundant categorical labels
df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
replace_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
for col in replace_cols:
    df[col] = df[col].replace("No internet service", "No")

# Discretization: Bin continuous attributes into explicit risk groups
df["TenureCategory"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
df["MonthlyChargesCategory"] = pd.cut(df["MonthlyCharges"], bins=[0, 35, 70, 120], labels=["Low", "Medium", "High"])


# HIGH LEVEL TAB INTERACTION FRAMEWORK

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "👥 Customer Profile",
    "📡 Services",
    "💰 Financial",
    "🔥 Churn Drivers",
    "⚠️ Risk Analysis"
])


# --- Tab 1: Executive Summary Metrics ---
with tab1:
    st.subheader("📊 Executive Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Baseline Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
        fig, ax = plt.subplots()
        sns.countplot(x="Churn", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_xticklabels(["Stayed", "Churned"])
        ax.set_title("Overall Churn Label Distribution")
        st.pyplot(fig)
        plt.close(fig) # Optimized: Free figure memory block buffer immediately

    with col2:
        st.metric("Average Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
        st.metric("Average Lifespan Tenure", f"{df['tenure'].mean():.1f} months")


# --- Tab 2: Demographic Analysis Mapping ---
with tab2:
    st.subheader("👥 Customer Profile vs Churn Risk")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.barplot(x="SeniorCitizen", y="Churn", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Demographics Impact: Senior Citizen Ratio")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.barplot(x="Partner", y="Churn", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Social Status Impact: Registered Partners")
        st.pyplot(fig)
        plt.close(fig)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots()
        sns.barplot(x="Dependents", y="Churn", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Family Density Impact: Active Dependents")
        st.pyplot(fig)
        plt.close(fig)
    with c4:
        fig, ax = plt.subplots()
        sns.barplot(x="gender", y="Churn", data=df, palette=[PRIMARY_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Gender Identity Variance Assessment")
        st.pyplot(fig)
        plt.close(fig)


# --- Tab 3: Core Service Ecosystem Telemetry ---
with tab3:
    st.subheader("📡 Operational Product & Services Impact")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.barplot(x="InternetService", y="Churn", data=df, palette=[PRIMARY_COLOR, NEUTRAL_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Internet Telemetry Infrastructure Type Variance")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.barplot(x="TechSupport", y="Churn", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Value Added Service Layer: Premium Tech Support")
        st.pyplot(fig)
        plt.close(fig)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots()
        sns.barplot(x="OnlineSecurity", y="Churn", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Security Architecture Impact: Online Firewall Subscriptions")
        st.pyplot(fig)
        plt.close(fig)
    with c4:
        fig, ax = plt.subplots()
        sns.barplot(x="StreamingMovies", y="Churn", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Entertainment Package Layer: Streaming Movies Active Matrix")
        st.pyplot(fig)
        plt.close(fig)


# --- Tab 4: Revenue & Financial Distribution Profiles ---
with tab4:
    st.subheader("💰 Financial Risk & Revenue Distribution Profiles")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_xticklabels(["Stayed", "Churned"])
        ax.set_title("Monthly Subscription Price Points Density Distributions")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.boxplot(x="Churn", y="TotalCharges", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_xticklabels(["Stayed", "Churned"])
        ax.set_title("Cumulative Historical Lifespan Billing Values Boxplots")
        st.pyplot(fig)
        plt.close(fig)


# --- Tab 5: Contractual and Systematic Drivers ---
with tab5:
    st.subheader("🔥 Key Contractual & Lifespan Drivers")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.barplot(x="Contract", y="Churn", data=df, palette=[PRIMARY_COLOR, NEUTRAL_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Contract Structuring Typology Risk Assessment")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.barplot(x="TenureCategory", y="Churn", data=df, palette=[PRIMARY_COLOR, NEUTRAL_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Customer Lifecycle Cohort Retention Intervals")
        st.pyplot(fig)
        plt.close(fig)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots()
        sns.barplot(x="PaymentMethod", y="Churn", data=df, palette=[PRIMARY_COLOR, NEUTRAL_COLOR, DANGER_COLOR], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right") # Fixed: Clean localized axis rotation
        ax.set_title("Transactional Clearing Architecture Churn Risk")
        st.pyplot(fig)
        plt.close(fig)
    with c4:
        st.info(
            "🔍 **Key Structural Insights:**\n\n"
            "- **Month-to-Month Contracts** hold an overwhelmingly dominant statistical correlation to immediate user attrition.\n"
            "- Premium billing configurations coupled with **No Dedicated Technical Support lines** accelerate churn likelihood.\n"
            "- Financial risk is heavily skewed within the **initial 12 months** of customer activation cycles."
        )


# --- Tab 6: Advanced Cross-Feature Cohort Risk Breakdown ---
with tab6:
    st.subheader("⚠️ High Vulnerability Cluster Cross-Analysis")
    
    # Feature Interaction Engineering: Construct joint structural interaction markers
    df["Internet_Charges"] = df["InternetService"].astype(str) + "_" + df["MonthlyChargesCategory"].astype(str)
    risk_cohorts = df.groupby("Internet_Charges")["Churn"].mean().sort_values(ascending=False).head(10)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        risk_cohorts.plot(kind="bar", ax=ax, color=DANGER_COLOR)
        ax.set_title("Top 10 Most Vulnerable Interactive Feature Segments")
        ax.set_ylabel("Mean Segment Churn Proportions")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right") # Fixed: Prevent string overlap artifacts
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.warning(
            "⚠️ **Critical High Risk Archetype Profiles Summary:**\n\n"
            "- Customers using **Fiber Optic Infrastructure** combined with maximum pricing tiers.\n"
            "- Transacting via unautomated operational rails (**Electronic Checks**).\n"
            "- Retaining active profiles under **Short Term Month-to-Month Contracts** without bundled security packages."
        )

st.markdown("---")
st.markdown("📌 **Customer Churn Insights Engine – Production Evaluation Architecture Ready**")
