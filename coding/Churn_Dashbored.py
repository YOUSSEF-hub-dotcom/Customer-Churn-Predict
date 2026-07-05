import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= Colors =================
PRIMARY_COLOR = "#1f77b4"   # Blue
DANGER_COLOR  = "#d62728"   # Red
SUCCESS_COLOR = "#2ca02c"   # Green
NEUTRAL_COLOR = "#ff7f0e"   # Orange

sns.set_style("whitegrid")
sns.set_palette([PRIMARY_COLOR, DANGER_COLOR, SUCCESS_COLOR, NEUTRAL_COLOR])

# ================= Streamlit Config =================
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Analysis Dashboard")
st.markdown("**Goal:** Understand customer churn drivers and identify high-risk segments.")

# ================= Load Data =================
@st.cache_data
def load_data():
    return pd.read_csv("C:\\Users\\Hedaya_city\\Downloads\\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = load_data()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Encode binary columns
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# Replace "No service"
df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
replace_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
for col in replace_cols:
    df[col] = df[col].replace("No internet service", "No")

# Create categories
df["TenureCategory"] = pd.cut(df["tenure"], bins=[0,12,24,48,72], labels=["0-12","13-24","25-48","49-72"])
df["MonthlyChargesCategory"] = pd.cut(df["MonthlyCharges"], bins=[0,35,70,120], labels=["Low","Medium","High"])

# ================= Tabs =================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "👥 Customer Profile",
    "📡 Services",
    "💰 Financial",
    "🔥 Churn Drivers",
    "⚠️ Risk Analysis"
])

# ================= Tab 1: Overview =================
with tab1:
    st.subheader("📊 Executive Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
        fig, ax = plt.subplots()
        sns.countplot(x="Churn", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_xticklabels(["Stayed", "Churned"])
        ax.set_title("Churn Distribution")
        st.pyplot(fig)

    with col2:
        st.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
        st.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")

# ================= Tab 2: Customer Profile =================
with tab2:
    st.subheader("👥 Customer Profile vs Churn")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.barplot(x="SeniorCitizen", y="Churn", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Senior Citizen vs Churn")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.barplot(x="Partner", y="Churn", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Partner Status vs Churn")
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots()
        sns.barplot(x="Dependents", y="Churn", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Dependents vs Churn")
        st.pyplot(fig)
    with c4:
        fig, ax = plt.subplots()
        sns.barplot(x="gender", y="Churn", data=df, palette=[PRIMARY_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Gender vs Churn")
        st.pyplot(fig)

# ================= Tab 3: Services =================
with tab3:
    st.subheader("📡 Services Impact on Churn")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.barplot(x="InternetService", y="Churn", data=df, palette=[PRIMARY_COLOR, NEUTRAL_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Internet Service vs Churn")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.barplot(x="TechSupport", y="Churn", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Tech Support vs Churn")
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots()
        sns.barplot(x="OnlineSecurity", y="Churn", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Online Security vs Churn")
        st.pyplot(fig)
    with c4:
        fig, ax = plt.subplots()
        sns.barplot(x="StreamingMovies", y="Churn", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Streaming Movies vs Churn")
        st.pyplot(fig)

# ================= Tab 4: Financial =================
with tab4:
    st.subheader("💰 Financial Behavior")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_xticklabels(["Stayed", "Churned"])
        ax.set_title("Monthly Charges vs Churn")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.boxplot(x="Churn", y="TotalCharges", data=df, palette=[SUCCESS_COLOR, DANGER_COLOR], ax=ax)
        ax.set_xticklabels(["Stayed", "Churned"])
        ax.set_title("Total Charges vs Churn")
        st.pyplot(fig)

# ================= Tab 5: Churn Drivers =================
with tab5:
    st.subheader("🔥 Core Churn Drivers")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        sns.barplot(x="Contract", y="Churn", data=df, palette=[PRIMARY_COLOR, NEUTRAL_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Contract Type vs Churn")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots()
        sns.barplot(x="TenureCategory", y="Churn", data=df, palette=[PRIMARY_COLOR, NEUTRAL_COLOR, DANGER_COLOR], ax=ax)
        ax.set_title("Tenure vs Churn")
        st.pyplot(fig)

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots()
        sns.barplot(x="PaymentMethod", y="Churn", data=df, palette=[PRIMARY_COLOR, NEUTRAL_COLOR, DANGER_COLOR], ax=ax)
        plt.xticks(rotation=45)
        ax.set_title("Payment Method vs Churn")
        st.pyplot(fig)
    with c4:
        st.info(
            "🔍 **High Churn Indicators**\n\n"
            "- Month-to-Month contracts\n"
            "- High monthly charges\n"
            "- Short tenure\n"
            "- No tech support"
        )

# ================= Tab 6: Risk Analysis =================
with tab6:
    st.subheader("⚠️ High Risk Segments")
    df["Internet_Charges"] = df["InternetService"].astype(str) + "_" + df["MonthlyChargesCategory"].astype(str)
    risk = df.groupby("Internet_Charges")["Churn"].mean().sort_values(ascending=False).head(10)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6,4))
        risk.plot(kind="bar", ax=ax, color=DANGER_COLOR)
        ax.set_title("Top High Risk Segments")
        ax.set_ylabel("Churn Rate")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with c2:
        st.warning(
            "⚠️ **Most Risky Profile**\n\n"
            "- Fiber Optic Internet\n"
            "- High Monthly Charges\n"
            "- Short Tenure\n"
            "- Month-to-Month Contract"
        )

st.markdown("---")
st.markdown("📌 **Customer Churn Dashboard – Portfolio Ready**")
