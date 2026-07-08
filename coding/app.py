import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Configure global Streamlit page layouts and identity tokens
st.set_page_config(page_title="Customer Churn Prediction Pro", page_icon="📉", layout="wide")

# Target backend API server routing destination URI
API_BASE_URL = "http://127.0.0.1:8000"


def plot_risk_gauge(probability, threshold):
    """
    Generates an interactive Plotly Gauge indicator chart.
    Dynamically maps color decision boundaries around the model's optimized operational threshold.
    """
    # Transform numeric ratio boundaries to absolute percentage spaces
    thresh_pct = threshold * 100
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Churn Risk Level %", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, thresh_pct - 10], 'color': "#d4edda"},              # Safe Green Zone
                {'range': [thresh_pct - 10, thresh_pct + 10], 'color': "#fff3cd"}, # Uncertain/Frontier Boundary Warning Zone
                {'range': [thresh_pct + 10, 100], 'color': "#f8d7da"}             # High Risk Critical Zone
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ==============================================================================
# USER INTERFACE GRAPHICAL ARCHITECTURE
# ==============================================================================
st.title("📉 Customer Churn Analytics Dashboard")
st.markdown("Predict customer behavior and analyze retention trends using production-grade ML pipeline.")

# --- Sidebar Control Center ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3208/3208726.png", width=100)
    st.title("System Status")
    # Updated: Aligned documentation text with the actual active CatBoost engine core
    st.success("Model: CatBoostClassifier (Production) 🚀")
    st.info("The production pipeline analyzes live contract structures, telemetry charges, and tenure bounds dynamically.")

# --- Real-Time Single Customer Evaluation Form ---
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

# --- Process Single Inbound Inference Request Payload ---
if submitted:
    payload = {
        "Contract": Contract, "tenure": tenure, "InternetService": InternetService,
        "MonthlyCharges": MonthlyCharges, "PaymentMethod": PaymentMethod,
        "TechSupport_OnlineSecurity": TechSupport_OnlineSecurity, "TotalCharges": TotalCharges
    }

    with st.spinner("🔍 Calculating Risk Profile..."):
        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                prob = result['churn_probability']
                threshold = result.get('threshold_used', 0.5) # Extract the dynamic threshold used by the PyFunc model

                st.divider()
                c1, c2 = st.columns([1, 1.5])

                with c1:
                    st.subheader("Prediction Detail")
                    status_str = "CHURN ❌" if result['churn_prediction'] == 1 else "STAY ✅"
                    color = "red" if result['churn_prediction'] == 1 else "green"
                    st.markdown(f"### Decision: <span style='color:{color}'>{status_str}</span>", unsafe_allow_html=True)
                    st.metric("Probability Score", f"{prob:.2%}")
                    st.markdown(f"**Decision Threshold Implemented:** `{threshold:.2f}`")
                    st.caption(f"Analysis completed in {result.get('latency_seconds', 0)}s")

                with c2:
                    st.plotly_chart(plot_risk_gauge(prob, threshold), use_container_width=True)
            else:
                st.error(f"API Error. Status Code: {response.status_code}")
        except Exception as e:
            st.error(f"Connection Error: {e}")


# ==============================================================================
# GLOBAL HISTORICAL TELEMETRY AUDIT LOGS & VISUALIZATIONS
# ==============================================================================
st.divider()
st.subheader("📊 Global Retention Insights")

# Fixed: Trigger automatic historical data loading on initial page render initialization
try:
    history_res = requests.get(f"{API_BASE_URL}/predictions")
    if history_res.status_code == 200:
        df = pd.DataFrame(history_res.json())
        if not df.empty:
            
            # Context reload mechanism action triggers
            if st.button("🔄 Trigger Cache Refresh", use_container_width=False):
                st.rerun()

            col_pie, col_scatter = st.columns(2)

            with col_pie:
                # Distribution analysis: Categorical distribution proportions mapping
                churn_counts = df['churn_prediction'].replace({1: 'Churn', 0: 'Stay'}).value_counts()
                fig_pie = px.pie(values=churn_counts.values, names=churn_counts.index,
                                 title="Overall Churn Distribution", hole=0.4,
                                 color=churn_counts.index,
                                 color_discrete_map={'Stay': '#28a745', 'Churn': '#dc3545'})
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_scatter:
                # Correlative analysis: Financial density vs Customer lifespan duration values mapping
                fig_scatter = px.scatter(df, x="tenure", y="MonthlyCharges", color="churn_prediction",
                                         title="Charges vs Tenure (Colored by Decision Outcome)",
                                         labels={'churn_prediction': 'Churn (1=Yes)'},
                                         color_continuous_scale=px.colors.diverging.RdYlGn_r)
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.write("**Recent Audit Logs (Top 10):**")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Database is initialized but contains empty history tables.")
    else:
        st.error(f"API Backend responded with unexpected operational status: {history_res.status_code}")
except Exception as e:
    st.warning("Unable to fetch operational metrics history. Ensure your FastAPI /predictions endpoint is online.")
