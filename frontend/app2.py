import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Home Credit Risk Scoring Portal",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"

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
    app_ids=pd.read_parquet('frontend/data/app_ids.parquet')
    return app_ids['SK_ID_CURR'].astype(int).unique().tolist()

existing_ids=load_ids()



# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main-title{
    font-size:40px;
    font-weight:800;
    margin-bottom:10px;
}

.section-header{
    background-color:#e9ecef;
    padding:8px 15px;
    border-radius:5px;
    border:1px solid #d0d0d0;
    font-weight:bold;
    font-size:22px;
    margin-top:15px;
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

</style>
""", unsafe_allow_html=True)

# ==========================================================
# TITLE
# ==========================================================

st.markdown(
    '<div class="main-title">Home Credit Risk Scoring Portal</div>',
    unsafe_allow_html=True
)
with st.expander("Model Metrics"):

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
    '<div class="section-header">TIER 1: EXECUTIVE SUMMARY & KPI CARDS</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns([1.2,1.2,1.2,1.2])

with col1:

    applicant_id = st.selectbox(
        options=existing_ids,label='Write Application ID'
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
                <div style="background:#fdeaea;padding:12px;border-radius:8px;text-align:center;margin-top:18px;">
                    <h2 style="color:#b91c1c;margin:0;">🚫 DECLINE</h2>
                </div>
                """,
                unsafe_allow_html=True  # <-- Make sure this is present and spelled exactly like this
            )
        else:
            st.markdown(
                """
                <div style="background:#ecfdf3;padding:12px;border-radius:8px;text-align:center;margin-top:18px;">
                    <h2 style="color:#166534;margin:0;">✅ APPROVE</h2>
                </div>
                """,
                unsafe_allow_html=True  # <-- Make sure this is present and spelled exactly like this
            )
# ==========================================================
# SECTION 2
# ==========================================================

st.markdown(
    '<div class="section-header">TIER 2: LOCAL CREDIT DIAGNOSTIC (INDIVIDUAL ID)</div>',
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
                margin=dict(l=50, r=50, t=80, b=150)
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
    '<div class="section-header">TIER 3: GLOBAL PORTFOLIO ANALYTICS (AGGREGATE DATA)</div>',
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
        "Feature",
        [
            "EXT_SOURCE_3",
            "AMT_CREDIT",
            "DAYS_BIRTH",
            "AMT_INCOME_TOTAL"
        ]
    )

    interaction_feature = st.selectbox(
        "Interaction Feature",
        [
            "AMT_CREDIT",
            "EXT_SOURCE_3",
            "DAYS_BIRTH"
        ]
    )

    # ======================================================
    # TODO:
    # API CALL
    #
    # /dependence_plot
    #
    # send:
    # feature
    # interaction_feature
    #
    # receive:
    # x
    # y
    # color
    # ======================================================

    dep_fig = go.Figure()

    dep_fig.update_layout(
        title="Dependence Plot Placeholder",
        height=500
    )

    st.plotly_chart(
        dep_fig,
        use_container_width=True
    )
