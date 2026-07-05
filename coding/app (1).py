import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# إعدادات الصفحة
st.set_page_config(page_title="Customer Churn Prediction Pro", page_icon="📉", layout="wide")

API_BASE_URL = "http://127.0.0.1:8000"


# دالة مؤشر المخاطرة
def plot_risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Churn Risk Level %", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 30], 'color': "#d4edda"},  # Safe
                {'range': [30, 70], 'color': "#fff3cd"},  # Warning
                {'range': [70, 100], 'color': "#f8d7da"}  # High Risk
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# --- واجهة المستخدم ---
st.title("📉 Customer Churn Analytics Dashboard")
st.markdown("Predict customer behavior and analyze retention trends.")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208726.png", width=100)
    st.title("System Status")
    st.success("Model: Logistic Regression/XGBoost ✅")
    st.info("The model analyzes contract type, charges, and tenure to predict loyalty.")

# فورم الإدخال (نفس اللي معاك مع تنظيم Columns)
with st.form("churn_form"):
    st.subheader("📌 Customer Input Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)

    with col3:
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                                        "Credit card (automatic)"])
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0)

    TechSupport_OnlineSecurity = st.selectbox("Tech Support & Online Security",
                                              ["Yes_Yes", "Yes_No", "No_Yes", "No_No"])
    submitted = st.form_submit_button("🚀 Run Churn Analysis", type="primary", use_container_width=True)

if submitted:
    payload = {
        "Contract": Contract, "tenure": tenure, "InternetService": InternetService,
        "MonthlyCharges": MonthlyCharges, "PaymentMethod": PaymentMethod,
        "TechSupport_OnlineSecurity": TechSupport_OnlineSecurity, "TotalCharges": TotalCharges
    }

    with st.spinner("🔍 Calculating Risk..."):
        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                prob = result['churn_probability']

                st.divider()
                c1, c2 = st.columns([1, 1.5])

                with c1:
                    st.subheader("Prediction Detail")
                    status = "CHURN ❌" if result['churn_prediction'] == 1 else "STAY ✅"
                    color = "red" if result['churn_prediction'] == 1 else "green"
                    st.markdown(f"### Decision: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
                    st.metric("Probability Score", f"{prob:.2%}")
                    st.caption(f"Analysis completed in {result.get('latency_seconds', 0)}s")

                with c2:
                    st.plotly_chart(plot_risk_gauge(prob), use_container_width=True)
            else:
                st.error("API Error.")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# --- الـ History والرسوم البيانية العامة ---
st.divider()
st.subheader("📊 Global Retention Insights")

if st.button("🔄 Refresh Dashboard Data", use_container_width=True):
    try:
        history_res = requests.get(f"{API_BASE_URL}/predictions")
        if history_res.status_code == 200:
            df = pd.DataFrame(history_res.json())
            if not df.empty:
                col_pie, col_scatter = st.columns(2)

                with col_pie:
                    # Pie Chart لتوزيع الـ Churn
                    churn_counts = df['churn_prediction'].replace({1: 'Churn', 0: 'Stay'}).value_counts()
                    fig_pie = px.pie(values=churn_counts.values, names=churn_counts.index,
                                     title="Overall Churn Distribution", hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col_scatter:
                    # Scatter Plot للعلاقة بين التكلفة والمدة
                    fig_scatter = px.scatter(df, x="tenure", y="MonthlyCharges", color="churn_prediction",
                                             title="Charges vs Tenure (Colored by Churn)",
                                             labels={'churn_prediction': 'Churn (1=Yes)'})
                    st.plotly_chart(fig_scatter, use_container_width=True)

                st.write("**Recent Logs:**")
                st.dataframe(df.tail(10), use_container_width=True)
            else:
                st.info("Database is empty.")
    except:
        st.error("Make sure /predictions endpoint is active in your FastAPI.")