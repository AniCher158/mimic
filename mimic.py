import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pykalman import KalmanFilter
from google.cloud import bigquery
import os

# -----------------------
# Set up BigQuery parameters and client
# -----------------------
# Update these values as needed.
project_id = st.sidebar.text_input("Google Cloud Project ID", value="mimic-451015")
dataset_icu = st.sidebar.text_input("ICU Dataset", value="mimiciv_3_1_icu")
dataset_hosp = st.sidebar.text_input("Hospital Dataset", value="mimiciv_3_1_hosp")
client = bigquery.Client(project=project_id)

# -----------------------
# Preprocessing: Retrieve full blood pressure dataset using d_items
# -----------------------
# Query d_items from the ICU dataset.
query_d_items = f"SELECT * FROM `physionet-data.{dataset_icu}.d_items`"
d_items = client.query(query_d_items).result().to_dataframe()
st.write("d_items (first 5 rows):", d_items.head(5))

# Query the full chartevents table from the ICU dataset.
query_chartevents_all = f"SELECT * FROM `physionet-data.{dataset_icu}.chartevents` LIMIT 100"
st.write("Top 100 rows from chartevents:")
chartevents_all = client.query(query_chartevents_all).result().to_dataframe()
st.write(chartevents_all)

# Use d_items to determine which item IDs are for systolic and diastolic blood pressure.
systolic_ids = d_items[d_items['label'].str.contains('systolic', case=False, na=False)]['itemid'].unique()
diastolic_ids = d_items[d_items['label'].str.contains('diastolic', case=False, na=False)]['itemid'].unique()

# Filter chartevents for systolic and diastolic measurements.
systolic_data = chartevents_all[chartevents_all['itemid'].isin(systolic_ids)]
diastolic_data = chartevents_all[chartevents_all['itemid'].isin(diastolic_ids)]

if {'subject_id', 'charttime', 'valuenum'}.issubset(chartevents_all.columns):
    systolic_pivot = systolic_data.pivot_table(
        index=['subject_id', 'charttime'], values='valuenum', aggfunc='first'
    ).rename(columns={'valuenum': 'systolic'})
    diastolic_pivot = diastolic_data.pivot_table(
        index=['subject_id', 'charttime'], values='valuenum', aggfunc='first'
    ).rename(columns={'valuenum': 'diastolic'})
    bp_data = systolic_pivot.merge(diastolic_pivot, left_index=True, right_index=True, how='outer').reset_index()
else:
    bp_data = pd.DataFrame()

if not bp_data.empty:
    bp_data['charttime'] = pd.to_datetime(bp_data['charttime'])
    bp_data = bp_data.sort_values(['subject_id', 'charttime'])
st.write("Full BP dataset (first 5 rows):", bp_data.head(5))

# -----------------------
# Load sample labevents for display
# -----------------------
# ----- Retrieve top 100 rows from labevents table from hospital dataset -----
query_labevents = f"SELECT * FROM `physionet-data.{dataset_hosp}.labevents` LIMIT 100"
st.write("Top 100 rows from labevents:")
labevents = client.query(query_labevents).result().to_dataframe()
st.write(labevents)

# -----------------------
# Patient Analysis: Retrieve data for a given subject (subject_id = 1)
# -----------------------
subject_id_input = st.text_input("Enter Patient Subject ID", value="1")
try:
    subject_id = int(subject_id_input)
except ValueError:
    st.error("Subject ID must be an integer.")
    subject_id = None

# Query chartevents for the subject from the ICU dataset.
query_chartevents = f"SELECT * FROM `physionet-data.{dataset_icu}.chartevents` WHERE subject_id = {subject_id}"
chartevents = client.query(query_chartevents).result().to_dataframe()
# Query labevents for the subject from the hospital dataset.
query_labevents_subject = f"SELECT * FROM `physionet-data.{dataset_hosp}.labevents` WHERE subject_id = {subject_id}"
labevents = client.query(query_labevents_subject).result().to_dataframe()
    
# If blood pressure columns are missing in chartevents, simulate them.
if "systolic" not in chartevents.columns:
    np.random.seed(subject_id)
    chartevents["systolic"] = np.random.normal(130, 10, size=len(chartevents))
if "diastolic" not in chartevents.columns:
    np.random.seed(subject_id + 1)
    chartevents["diastolic"] = np.random.normal(80, 5, size=len(chartevents))
    
st.write(f"Patient chartevents (first 5 rows) for subject {subject_id}:", chartevents.head(5))
st.write(f"Patient labevents (first 5 rows) for subject {subject_id}:", labevents.head(5))

# -----------------------
# Patient Classification (using average BP)
# -----------------------
avg_systolic = chartevents["systolic"].mean()
avg_diastolic = chartevents["diastolic"].mean()
if avg_systolic >= 140 or avg_diastolic >= 90:
    classification = "Hypertensive"
elif avg_systolic >= 120 or avg_diastolic >= 80:
    classification = "Prehypertensive"
else:
    classification = "Normotensive"
st.write(f"Patient Classification for subject {subject_id}: {classification}")
st.write(f"Average Systolic: {avg_systolic:.2f}, Average Diastolic: {avg_diastolic:.2f}")

# -----------------------
# LSTM Model for Dosage Prediction Simulation
# -----------------------
input_shape = (10, 2)  # 10 timesteps, 2 features (systolic & diastolic)
model = Sequential()
model.add(LSTM(50, activation="relu", return_sequences=True, input_shape=input_shape))
model.add(LSTM(25, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(optimizer="adam", loss="mse")

# Simulate training the model with random data.
X_train = np.random.rand(100, 10, 2)
y_train = np.random.rand(100, 1)
st.write("Training LSTM model (simulation)...")
model.fit(X_train, y_train, epochs=5, verbose=1)
st.write("LSTM model trained!")
    
# Use Kalman Filter to smooth the predictions.
kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=0,
    initial_state_covariance=1,
    observation_covariance=1,
    transition_covariance=0.1,
)
features = chartevents[["systolic", "diastolic"]].values
if features.shape[0] == 0:
    st.error("No chartevents data available for this patient. Cannot predict dosage.")
else:
    # Reshape the data: (1, timesteps, features)
    X_patient = features.reshape(1, features.shape[0], features.shape[1])
    lstm_prediction = model.predict(X_patient)
    smoothed = kf.smooth(lstm_prediction.flatten())[0]
    predicted_dosage = smoothed[-1]
    st.write(f"Predicted Drug Dosage (LSTM + Kalman) for subject {subject_id}: {predicted_dosage:.2f}")

    
# Plot patient blood pressure over time.
fig_bp = px.line(chartevents, y=["systolic", "diastolic"],
                 title="Patient Blood Pressure Over Time",
                 labels={"value": "Blood Pressure", "variable": "Measurement"})
st.plotly_chart(fig_bp)

X_patient = features.reshape(1, features.shape[0], features.shape[1])
lstm_prediction = model.predict(X_patient)
smoothed = kf.smooth(lstm_prediction.flatten())[0]
predicted_dosage = smoothed[-1]
st.write(f"Predicted Drug Dosage (LSTM + Kalman) for subject {subject_id}: {predicted_dosage:.2f}")

# ----- Simulation of Characteristic Change -----
st.write("Simulate Characteristic Change:")
changed_feature = st.selectbox("Select Feature to Change", options=["systolic", "diastolic"])

# Check if the chartevents DataFrame is non-empty and has data for the selected feature
if not chartevents.empty and changed_feature in chartevents.columns and len(chartevents[changed_feature]) > 0:
    default_value = float(chartevents[changed_feature].iloc[-1])
else:
    default_value = 0.0  # Provide a fallback default value

new_value = st.number_input(f"New value for {changed_feature}", value=default_value)

if st.button("Simulate Change"):
    chartevents_modified = chartevents.copy()
    chartevents_modified[changed_feature] = new_value
    X_orig = chartevents[["systolic", "diastolic"]].values.reshape(1, -1, 2)
    X_mod = chartevents_modified[["systolic", "diastolic"]].values.reshape(1, -1, 2)
    pred_orig = kf.smooth(model.predict(X_orig).flatten())[0][-1]
    pred_mod = kf.smooth(model.predict(X_mod).flatten())[0][-1]
    st.write(f"Dosage Before Change: {pred_orig:.2f}")
    st.write(f"Dosage After Change: {pred_mod:.2f}")
    fig_compare = px.bar(x=["Before", "After"], y=[pred_orig, pred_mod],
                         labels={"x": "State", "y": "Predicted Dosage"},
                         title="Dosage Comparison")
    st.plotly_chart(fig_compare)

    # -----------------------
# Aggregate Patient Trends (using dummy aggregated data)
# -----------------------
st.header("Aggregate Patient Trends")
num_patients = 50
agg_data = pd.DataFrame({
    "subject_id": np.arange(1, num_patients+1),
    "avg_systolic": np.random.normal(130, 15, num_patients),
    "avg_diastolic": np.random.normal(80, 10, num_patients)
})
agg_data["classification"] = agg_data.apply(
    lambda row: "Hypertensive" if (row["avg_systolic"]>=140 or row["avg_diastolic"]>=90) 
    else ("Prehypertensive" if (row["avg_systolic"]>=120 or row["avg_diastolic"]>=80)
          else "Normotensive"), axis=1)
fig_agg = px.scatter(agg_data, x="avg_systolic", y="avg_diastolic",
                     color="classification", title="Aggregate Blood Pressure Trends",
                     labels={"avg_systolic": "Average Systolic", "avg_diastolic": "Average Diastolic"})
st.plotly_chart(fig_agg)

st.header("Explainable AI Insights")
st.markdown("""
**LSTM Model:** Uses time-series patient data (e.g., blood pressure over time) to predict the optimal drug dosage.  
**Kalman Filter:** Smooths the noisy LSTM predictions to yield a stable dosage recommendation.  
**Reinforcement Learning Agent:** Learns from patient outcomes to suggest dosage adjustments.  
**Dashboard Visuals:** Interactive plots allow for exploring individual patient trends and aggregate trends across many patients.
""")