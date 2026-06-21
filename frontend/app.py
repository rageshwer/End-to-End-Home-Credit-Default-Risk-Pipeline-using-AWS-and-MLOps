import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="Home-Credit Risk Scoring Portal",
    page_icon="🏦",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

@st.cache_data
def load_batch():
    try:
        response = requests.get(f'{API_URL}/fetch-batch')
        # Check if the server actually returned a 200 OK status
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend returned an error status: {response.status_code}")
            st.code(response.text) # This will print the actual error if FastAPI crashed
            return None
    except Exception as e:
        st.error(f"Failed to connect to the API server at {API_URL}")
        st.exception(e) # This prints the full network traceback on the UI screen
        return None

@st.cache_data
def load_ids():
    app_ids=pd.read_parquet('ui_data/app_ids.parquet')
    return app_ids['SK_ID_CURR'].astype(int).unique().tolist()

existing_ids=load_ids()



# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App background — soft fintech slate */
.stApp{
    background:#f4f6fb;
}

.block-container{
    padding-top:1.6rem;
    padding-bottom:3rem;
    max-width:1500px;
}

/* ---------- Top banner ---------- */
.topbar{
    background:linear-gradient(120deg,#0b1e4a 0%, #1d3fa3 55%, #2f6fed 100%);
    border-radius:16px;
    padding:26px 32px;
    margin-bottom:22px;
    box-shadow:0 8px 24px rgba(29,63,163,0.25);
    color:#ffffff;
}

.main-title{
    font-size:32px;
    font-weight:800;
    letter-spacing:-0.5px;
    color:#ffffff;
    margin-bottom:4px;
}

.main-subtitle{
    font-size:14px;
    color:#cfe0ff;
    font-weight:500;
    letter-spacing:0.2px;
}

.badge-row{
    margin-top:14px;
}

.badge{
    display:inline-block;
    background:rgba(255,255,255,0.14);
    border:1px solid rgba(255,255,255,0.25);
    color:#ffffff;
    font-size:12px;
    font-weight:600;
    padding:5px 12px;
    border-radius:20px;
    margin-right:8px;
}

/* ---------- Section headers ---------- */
.section-header{
    display:flex;
    align-items:center;
    gap:10px;
    background:linear-gradient(150deg,#eef3ff 0%,#e3ecff 100%);
    color:#0b1e4a;
    padding:12px 18px;
    border-radius:10px;
    border-left:5px solid #2f6fed;
    border-top:1px solid #cddaf7;
    border-right:1px solid #cddaf7;
    border-bottom:1px solid #cddaf7;
    font-weight:800;
    font-size:14px;
    letter-spacing:0.8px;
    text-transform:uppercase;
    margin-top:30px;
    margin-bottom:16px;
    box-shadow:0 2px 8px rgba(16,24,64,0.06);
}

/* ---------- Generic card wrapper ---------- */
.fc-card{
    background:#ffffff;
    border-radius:14px;
    padding:18px 20px;
    border:1px solid #e7eaf3;
    box-shadow:0 2px 10px rgba(16,24,64,0.05);
    height:100%;
}

.kpi-card{
    background-color:#f8f9fa;
    padding:15px;
    border-radius:10px;
    border:1px solid #ddd;
    height:100px;
}

.kpi-title{
    font-size:16px;
    font-weight:600;
}

.kpi-value{
    font-size:28px;
    font-weight:700;
}

/* ---------- st.metric polish ---------- */
div[data-testid="stMetric"]{
    background:linear-gradient(150deg,#eef3ff 0%,#e3ecff 100%);
    border:1px solid #cddaf7;
    border-left:4px solid #2f6fed;
    border-radius:14px;
    padding:16px 18px;
    box-shadow:0 2px 10px rgba(16,24,64,0.06);
}

div[data-testid="stMetricLabel"]{
    font-weight:600;
    color:#3a4a78;
    font-size:13px;
    text-transform:uppercase;
    letter-spacing:0.5px;
}

div[data-testid="stMetricValue"]{
    color:#0b1e4a;
    font-weight:800;
}

/* ---------- Expander ---------- */
div[data-testid="stExpander"]{
    border:1px solid #cddaf7;
    border-radius:14px;
    background:linear-gradient(150deg,#f3f7ff 0%,#eaf0ff 100%);
    box-shadow:0 2px 10px rgba(16,24,64,0.05);
}

div[data-testid="stExpander"] summary{
    font-weight:700;
    color:#0b1e4a;
}

/* ---------- Buttons ---------- */
div.stButton > button{
    background:linear-gradient(120deg,#1d3fa3,#2f6fed);
    color:#ffffff;
    border-radius:8px;
    border:none;
    font-weight:700;
    padding:8px 26px;
    box-shadow:0 4px 12px rgba(47,111,237,0.3);
    transition:0.15s ease-in-out;
}
div.stButton > button:hover{
    background:linear-gradient(120deg,#2f6fed,#1d3fa3);
    box-shadow:0 6px 16px rgba(47,111,237,0.4);
}

/* ---------- Alerts ---------- */
div[data-testid="stAlert"]{
    border-radius:10px;
    border:1px solid transparent;
}

div[data-testid="stAlertContentSuccess"]{
    color:#0f5132;
}
div[data-baseweb="notification"]:has(div[data-testid="stAlertContentSuccess"]),
div[data-testid="stAlert"]:has(div[data-testid="stAlertContentSuccess"]){
    background:linear-gradient(135deg,#e7f9ee,#d7f5e1);
    border:1px solid #a9e8c1;
}

div[data-testid="stAlert"]:has(div[data-testid="stAlertContentError"]){
    background:linear-gradient(135deg,#fff0f0,#fde2e2);
    border:1px solid #f3b9b9;
}

div[data-testid="stAlert"]:has(div[data-testid="stAlertContentInfo"]){
    background:linear-gradient(135deg,#eaf1ff,#dfeaff);
    border:1px solid #b9cdf5;
}

div[data-testid="stAlert"]:has(div[data-testid="stAlertContentWarning"]){
    background:linear-gradient(135deg,#fff8e6,#fdf0cf);
    border:1px solid #f3dca1;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{
    background:#0b1e4a;
}
section[data-testid="stSidebar"] *{
    color:#e7ecff !important;
}
section[data-testid="stSidebar"] hr{
    border-color:rgba(255,255,255,0.15);
}

/* ---------- Progress bar ---------- */
div[data-testid="stProgress"] > div > div{
    background:linear-gradient(90deg,#1d3fa3,#2f6fed) !important;
}
/* ---------- Labels ---------- */
label{
    font-weight:600 !important;
    color:#374151 !important;
}

/* ---------- Misc ---------- */
hr{
    margin:20px 0;
    border-color:#e7eaf3;
}

h3, .stSubheader, div[data-testid="stMarkdownContainer"] h3{
    color:#0b1e4a !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR (branding / navigation only — no logic)
# ==========================================================

with st.sidebar:
    st.markdown("## Home Credit Datasets")
    st.markdown("**Risk Intelligence Suite**")
    st.markdown("---")
    st.markdown("##### Navigation")
    st.markdown("T-1 Executive Summary")
    st.markdown("T-2 Local Diagnostics")
    st.markdown("T-3 Portfolio Analytics")
    st.markdown("---")
    st.markdown("##### Model")
    st.markdown("LightGBM")
    st.markdown("Last retrained: Q2 2026")
    st.markdown("---")
    st.caption("Internal decisioning tool · For credit risk analysts")

# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    """
    <div class="topbar">
        <div class="main-title"> Home Credit Risk Scoring Portal</div>
        <div class="main-subtitle">AI-powered credit risk decisioning &amp; explainability dashboard</div>
        <div class="badge-row">
            <span class="badge">⚡ Real-time scoring</span>
            <span class="badge">🔍 SHAP explainability</span>
            <span class="badge">📊 Portfolio analytics</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
with st.expander("Model Metrics", expanded=True):

    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.metric("AUC","0.779")
    with col2:
        st.metric("KS","41%")
    with col3:
        st.metric("Recall","59%")
    with col4:
        st.metric("Precision","21%")
# ==========================================================
# SECTION 1
# ==========================================================
st.markdown(
    """
    This dashboard helps loan officers evaluate credit risk and understand the **"why"** behind a model's prediction using a specific **Application ID**.

    ##### Why can't you manually fill out the form?
    Our production model utilizes **Over 1000 highly engineered and aggregated features** to make an accurate prediction. Expecting a loan officer to manually input over 100 data points for a single check is highly impractical. 
    
    Therefore, for this interactive prototype, we pre-loaded a processed **Test Dataset** containing fully feature-engineered profiles. You only need to select or type an ID to see how the system operates in real time.

    ##### How to use this app:
    1. **Browse or Search:** Select an ID from the pre-loaded batch dropdown, or type an ID directly into the input box to pull an applicant's profile.
    2. **The "Master Key" Impact:** When you pick an ID, the backend instantly runs a lookup that triggers two steps:
        * **The Decision:** It extracts the raw customer metrics to calculate a fresh risk score, rendering the custom **APPROVE** or **DECLINE** badge.
        * **The Explanation:** It pulls the pre-calculated **SHAP values** to dynamically generate the charts below, showing exactly which features drove that specific individual's risk up or down.
    """
)
st.markdown(
    '<div class="section-header">🧮 Tier 1 &nbsp;·&nbsp; Executive Summary &amp; KPI Cards</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns([1.2,1.2,1.2,1.2])

with col1:

    applicant_id = st.selectbox(
        options=existing_ids,label='Select Application ID'
    )
    submit=st.button('Submit')
    if submit:
        with st.spinner("Getting prediction..."):
            response = requests.post(
                f'{API_URL}/predict',
                params={"id": int(applicant_id)}
            )

with col3:

    if submit and response.status_code == 200:
        result = response.json()
        probability = result["Probability of Default :"]
        st.success("Prediction Successfull!")
        if probability >= 0.08:
            st.error("The borrower has high default probability.")
            decision = 'DECLINE'
        else:
            st.success("Low Risk Customer.")
            decision = 'APPROVE'

with col2:

    if submit and response.status_code == 200:
        result = response.json()
        probability = result["Probability of Default :"]

        st.metric(
            label="Probability of Default",
            value=f"{probability:.2%}"
        )

        st.progress(float(probability))

with col4:
    if submit:
        if decision == "DECLINE":
            st.markdown(
                """
                <div style="background:linear-gradient(135deg,#fff1f1,#fde2e2);padding:14px;border-radius:12px;text-align:center;margin-top:18px;border:1px solid #f3b9b9;box-shadow:0 4px 14px rgba(185,28,28,0.12);">
                    <h2 style="color:#b91c1c;margin:0;">🚫 DECLINE</h2>
                </div>
                """,
                unsafe_allow_html=True  # <-- Make sure this is present and spelled exactly like this
            )
        else:
            st.markdown(
                """
                <div style="background:linear-gradient(135deg,#effdf4,#e1faea);padding:14px;border-radius:12px;text-align:center;margin-top:18px;border:1px solid #b7ebc6;box-shadow:0 4px 14px rgba(22,101,52,0.12);">
                    <h2 style="color:#166534;margin:0;">✅ APPROVE</h2>
                </div>
                """,
                unsafe_allow_html=True  # <-- Make sure this is present and spelled exactly like this
            )
# ==========================================================
# SECTION 2
# ==========================================================

st.markdown(
    '<div class="section-header">🔍 Tier 2 &nbsp;·&nbsp; Local Credit Diagnostic (Individual ID)</div>',
    unsafe_allow_html=True
)

st.subheader("Individual Risk Contribution (Waterfall)")

if submit:

    with st.spinner("Getting SHAP values for waterfall plot..."):
        response3 = requests.post(
            f'{API_URL}/waterfall',
            params={"id": int(applicant_id)}
        )

        if response3.status_code == 200:
            data = response3.json()
            
            base_value = data["base_value"]
            prediction_value = data["prediction_value"]
            features = data["features"]
            
            # 2. Prepare data lists for Plotly
            x_labels = ["Baseline (Avg)"]
            y_values = [base_value]
            measures = ["absolute"]  # Baseline starts at an absolute position
            
            # Populate features
            for f in features:
                # Format label to show name and actual value if available
                if f["feature_value"] is not None:
                    label = f"{f['name']} ({f['feature_value']})"
                else:
                    label = f["name"]
                    
                x_labels.append(label)
                y_values.append(f["shap_value"])
                measures.append("relative")  # Features shift the bar relatively
                
            # Add the final prediction bar
            x_labels.append("Final Prediction")
            y_values.append(prediction_value)
            measures.append("total")  # Final prediction drops back down to zero/total line
            
            # 3. Create the Plotly Waterfall Figure
            fig = go.Figure(go.Waterfall(
                name="SHAP Breakdown",
                orientation="v",
                measure=measures,
                x=x_labels,
                textposition="outside",
                text=[f"{v:+.4f}" for v in y_values], # Format labels with +/- signs
                y=y_values,
                connector={"line": {"color": "rgb(63, 63, 63)", "dash": "dot"}},
                decreasing={"marker": {"color": "#03d156"}}, # Green for lowering risk
                increasing={"marker": {"color": "#f40606"}}, # Red for increasing risk
                totals={"marker": {"color": "#1cc8ef"}}
            ))
            
            fig.update_layout(
                title=f"Waterfall Prediction Path for ID: {applicant_id}",
                showlegend=False,
                height=600,
                xaxis_tickangle=-45, # Tilt labels so they don't overlap
                margin=dict(l=50, r=50, t=80, b=150),
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=12, color="#374151"),
                title_font=dict(size=18, color="#0b1e4a")
            )
            
            # 4. Display in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            # Display summary metrics
            col1, col2 = st.columns(2)
            col1.metric("Baseline Score", f"{base_value:.4f}")
            col2.metric("Final Risk Prediction", f"{prediction_value:.4f}")
            
        else:
            st.error(f"Backend Error ({response3.status_code}): {response3.text}")

else:
    st.info("Select applicant and click Submit.")

# ==========================================================
# SECTION 3
# ==========================================================

st.markdown(
    '<div class="section-header">🌐 Tier 3 &nbsp;·&nbsp; Global Portfolio Analytics (Aggregate Data)</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2)

# ==========================================================
# PANEL 1
# ==========================================================

with left:

    st.markdown(
        "### Panel 1: Global Portfolio Summary (Beeswarm)"
    )

    st.subheader(
        "Global Importance (Beeswarm)"
    )

    response2=load_batch()
    # Verify that response is a dictionary and not None before indexing
    if response2 is not None:

        features_list = response2["features"]
        shap_values_matrix = np.array(response2["shap_values"])
        raw_values_matrix = np.array(response2["raw_values"])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(
            shap_values=shap_values_matrix,
            features=raw_values_matrix,
            feature_names=features_list,
            show=False
        )
        st.pyplot(fig, clear_figure=True)
    else:
        st.warning("Data payload is empty. Please verify your backend server is running.")

# ==========================================================
# PANEL 2
# ==========================================================

with right:

    st.markdown(
        "### Panel 2: Global Risk Thresholds (Dependence with Interaction)"
    )

    st.subheader(
        "New SHAP Dependence Plot"
    )

    feature = st.selectbox(
        "Feature",features_list)

    interaction_feature = st.selectbox(
        "Interaction Feature",features_list)

    # API Call for dependence plot:
    dep_plot=st.button('Generate')
    if dep_plot:
        response4=requests.post(f'{API_URL}/dependence',params={'feat':feature,'int_feat':interaction_feature})
        if response4.status_code==200:
            response4=response4.json()
            fig = px.scatter(
                x=response4['raw_feat'],
                y=response4['shap_feat'],
                color=response4['raw_int_feat'],
                color_continuous_scale="Viridis",  # Smooth color scale for numeric, distinct colors for categories
                labels={
                    "x": f"Actual {feature}",
                    "y": f"SHAP Value ({feature})",
                    "color": interaction_feature
                },
                title=f"How {feature} Affects Predictions (Colored by {interaction_feature})"
            )
            
            # Clean background template
            fig.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", size=12, color="#374151"),
                title_font=dict(size=16, color="#0b1e4a")
            )
            
            # Render in Streamlit
            st.plotly_chart(fig, use_container_width=True)


        else:
            st.error("Could not retrieve dependency data!")