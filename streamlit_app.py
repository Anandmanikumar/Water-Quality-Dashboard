import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# === Load Model ===
model = joblib.load("water_quality_model.pkl")

# === Load Data ===
df = pd.read_csv("water_dataX.csv", encoding='ISO-8859-1')

# === Preprocess Data ===
df['pH'] = pd.to_numeric(df['PH'], errors='coerce')
df['Conductivity'] = pd.to_numeric(df['CONDUCTIVITY (µmhos/cm)'], errors='coerce')
df['Coliform'] = pd.to_numeric(df['TOTAL COLIFORM (MPN/100ml)Mean'], errors='coerce')
df.dropna(subset=['pH', 'Conductivity', 'Coliform'], inplace=True)

# === Rule-based Quality Label (For Visualization) ===
def classify_status(row):
    if 6.5 <= row['pH'] <= 8.5 and row['Conductivity'] <= 500 and row['Coliform'] <= 1000:
        return 'Good'
    else:
        return 'Poor'

df['Status'] = df.apply(classify_status, axis=1)

# === Streamlit Dashboard ===
st.set_page_config(page_title="Water Quality Dashboard", layout="wide")
st.title("💧 Water Quality Monitoring & Prediction Dashboard")

# --- Section 1: Data Table ---
st.subheader("📋 Water Quality Sample Data")
st.dataframe(df[['LOCATIONS', 'pH', 'Conductivity', 'Coliform', 'Status']].head(20))

# --- Section 2: Visualizations ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Average pH by Location")
    avg_pH = df.groupby('LOCATIONS')['pH'].mean().reset_index()
    fig_pH = px.bar(avg_pH, x='LOCATIONS', y='pH', title="Average pH by Location")
    st.plotly_chart(fig_pH, use_container_width=True)

with col2:
    st.subheader("🥧 Water Quality Distribution (Rule-based)")
    status_counts = df['Status'].value_counts().reset_index()
    status_counts.columns = ['Status', 'Count']
    fig_pie = px.pie(status_counts, values='Count', names='Status', title='Water Quality Status')
    st.plotly_chart(fig_pie, use_container_width=True)

# --- Section 3: Model-based Prediction ---
st.markdown("---")
st.subheader("🔮 Predict Water Quality Using Machine Learning")

ph = st.slider("pH", 0.0, 14.0, 7.0)
conductivity = st.number_input("Conductivity (µmhos/cm)", min_value=0.0, value=250.0)
coliform = st.number_input("Total Coliform (MPN/100ml)", min_value=0.0, value=500.0)

if st.button("Predict Water Quality"):
    input_data = [[ph, conductivity, coliform]]
    prediction = model.predict(input_data)[0]
    status = "Good" if prediction == 0 else "Poor"  # NOTE: confirm label encoding
    st.success(f"✅ Predicted Water Quality: **{status}**")

# --- Section 4: More Visual Insights ---
st.markdown("---")
st.subheader("📈 Data Distributions")

col3, col4, col5 = st.columns(3)

with col3:
    fig_ph, ax_ph = plt.subplots()
    sns.histplot(df['pH'], bins=30, kde=True, ax=ax_ph)
    ax_ph.set_title("pH Distribution")
    st.pyplot(fig_ph)

with col4:
    fig_conductivity, ax_conductivity = plt.subplots()
    sns.histplot(df['Conductivity'], bins=30, kde=True, ax=ax_conductivity)
    ax_conductivity.set_title("Conductivity Distribution")
    st.pyplot(fig_conductivity)

with col5:
    fig_coliform, ax_coliform = plt.subplots()
    sns.histplot(df['Coliform'], bins=30, kde=True, ax=ax_coliform)
    ax_coliform.set_title("Coliform Distribution")
    st.pyplot(fig_coliform)
