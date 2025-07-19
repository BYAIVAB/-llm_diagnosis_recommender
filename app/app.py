import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import os
from dotenv import load_dotenv
import sys
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.langchain_agent import LLMAgent

# Load environment variables (still useful for other potential env vars)
load_dotenv()

# Initialize LLM Agent
# No API key needed for Ollama, model name is default 'mistral'
llm_agent = LLMAgent()

st.set_page_config(layout="wide", page_title="LLM Diagnosis Recommender")

st.title("⚙️ LLM-Powered Anomaly Diagnosis & Recommendation")

st.markdown("""
This application uses an Isolation Forest model to detect anomalies in sensor data and then leverages a Large Language Model (LLM) to provide a diagnosis and recommend corrective actions.
""")

# --- File Upload Section ---
st.header("1. Upload Sensor Data")
uploaded_file = st.file_uploader("Upload your CSV file (e.g., sample_sensor_data.csv)", type=["csv"])

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        st.success("Data loaded successfully!")
        st.write("**First 5 rows of your data:**")
        st.dataframe(df.head())

        st.write("**Summary Statistics:**")
        st.write(df.describe())

    except Exception as e:
        st.error(f"Error loading data: {e}. Please ensure your CSV has a 'timestamp' column and other sensor data columns.")
else:
    st.info("Please upload a CSV file to proceed with anomaly detection.")

# --- Anomaly Detection Section ---
st.header("2. Anomaly Detection")
if df is not None:
    sensor_columns = st.multiselect(
        "Select sensor columns for anomaly detection (numerical data only):",
        options=df.columns.tolist(),
        default=[col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    )

    if not sensor_columns:
        st.warning("Please select at least one sensor column for anomaly detection.")
    else:
        st.write("**Selected Sensor Data Trends:**")
        fig, axes = plt.subplots(len(sensor_columns), 1, figsize=(12, 4 * len(sensor_columns)))
        if len(sensor_columns) == 1:
            axes = [axes]
        for i, col in enumerate(sensor_columns):
            axes[i].plot(df.index, df[col])
            axes[i].set_title(f'{col} Readings')
            axes[i].set_ylabel('Value')
            axes[i].grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig) # Close the figure to prevent display issues

        contamination = st.slider(
            "Contamination (expected proportion of outliers):",
            min_value=0.01, max_value=0.5, value=0.05, step=0.01
        )

        if st.button("Run Anomaly Detection"):
            try:
                model = IsolationForest(contamination=contamination, random_state=42)
                model.fit(df[sensor_columns])
                df['anomaly'] = model.predict(df[sensor_columns])

                anomalies = df[df['anomaly'] == -1]
                st.success(f"Anomaly detection complete! Found {len(anomalies)} anomalies.")

                if not anomalies.empty:
                    st.write("**Detected Anomalies:**")
                    st.dataframe(anomalies)

                    st.write("**Visualizing Anomalies:**")
                    fig_anom, axes_anom = plt.subplots(len(sensor_columns), 1, figsize=(12, 4 * len(sensor_columns)))
                    if len(sensor_columns) == 1:
                        axes_anom = [axes_anom]
                    for i, col in enumerate(sensor_columns):
                        axes_anom[i].plot(df.index, df[col], label=col)
                        anomalies_col = anomalies[anomalies[col].notna()]
                        axes_anom[i].scatter(anomalies_col.index, anomalies_col[col], color='red', label='Anomaly', s=50, zorder=5)
                        axes_anom[i].set_title(f'{col} Readings with Anomalies')
                        axes_anom[i].set_ylabel('Value')
                        axes_anom[i].legend()
                        axes_anom[i].grid(True)
                    plt.tight_layout()
                    st.pyplot(fig_anom)
                    plt.close(fig_anom) # Close the figure

                    # Prepare data for LLM
                    anomaly_data_str = anomalies.to_string()
                    anomaly_details_str = "\n".join([f"Timestamp: {idx}, Data: {row.drop('anomaly').to_dict()}" for idx, row in anomalies.iterrows()])

                    st.session_state['anomaly_data'] = anomaly_data_str
                    st.session_state['anomaly_details'] = anomaly_details_str
                    st.session_state['anomalies_found'] = True
                else:
                    st.info("No anomalies detected with the current settings.")
                    st.session_state['anomalies_found'] = False

            except Exception as e:
                st.error(f"Error during anomaly detection: {e}. Please check your selected columns.")
else:
    st.info("Please upload data first to perform anomaly detection.")

# --- LLM Diagnosis & Recommendation Section ---
st.header("3. LLM Diagnosis & Recommendation")
if llm_agent and st.session_state.get('anomalies_found', False):
    user_context = st.text_area(
        "Provide additional context for the LLM (e.g., operating conditions, recent changes):",
        height=100
    )

    if st.button("Get LLM Diagnosis & Recommendations"):
        with st.spinner("Generating diagnosis and recommendations with LLM..."):
            try:
                anomaly_dict = {
                    "anomaly_data": st.session_state['anomaly_data'],
                    "context": user_context,
                    "anomaly_details": st.session_state['anomaly_details']
                }
                report = llm_agent.run_diagnosis(anomaly_dict)
                st.session_state['llm_report'] = report
                st.success("LLM report generated!")
            except Exception as e:
                st.error(f"Error generating LLM report: {e}. Check your API key and network connection.")

    if st.session_state.get('llm_report'):
        st.subheader("Generated Report from LLM:")
        st.markdown(st.session_state['llm_report'])
elif df is not None:
    st.info("Run anomaly detection first to get LLM diagnosis and recommendations.")
else:
    st.info("Upload data and run anomaly detection to enable LLM features.")

st.markdown("---")
st.markdown("Developed by Byaivab Sarkar")
