import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Home Credit Default Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Home Credit Default Prediction")
st.write("Enter a customer ID to predict the probability of default.")

sk_id_curr = st.number_input(
    "SK_ID_CURR",step=1,format='%d'
)

if st.button("Predict Risk", use_container_width=True):

    try:
        with st.spinner("Getting prediction..."):

            response = requests.post(
                API_URL,
                params={"id": int(sk_id_curr)}
            )

        if response.status_code == 200:

            result = response.json()

            probability = result["Probability of Default :"]

            st.success("Prediction successful")

            st.metric(
                label="Probability of Default",
                value=f"{probability:.2%}"
            )

            st.progress(float(probability))

            if probability >= 0.08:
                st.error("High Risk Customer")
            else:
                st.success("Low Risk Customer")

        else:
            error = response.json().get("detail", "Unknown error")
            st.error(error)

    except requests.exceptions.ConnectionError:
        st.error("Unable to connect to prediction API.")

    except requests.exceptions.Timeout:
        st.error("API request timed out.")

    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")


