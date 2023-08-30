import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor  # Import RandomForestRegressor

st.subheader('Kibon Kiprono Solomon')
# Load the saved regression model
model = joblib.load('regression_model.pkl')

# Streamlit app title and description
st.title('Air Quality Prediction App')
st.write('Enter the input features below to predict air quality.')
# Sidebar with project information
st.sidebar.title('Project Information')
st.sidebar.subheader('Project description and Data Visualizations ')
st.sidebar.write('Air pollution is a growing concern in urban areas, with adverse effects on public health and the environment. Developing an air quality monitoring and pollution prediction system using machine learning can help raise awareness, inform policy decisions, and enable citizens to take protective measures.')
# Load the raw data
raw_data = pd.read_csv('Air_quality_data.csv')
# Checkbox for displaying raw data
display_raw_data = st.sidebar.checkbox('Display Raw Data')

# Display raw data section if checkbox is checked
if display_raw_data:
    st.header('Raw Data')
    st.write('Here is the raw data used in the app:')
    st.dataframe(raw_data)
#load clean dataset
def load_data():
    data = pd.read_csv('cleaned_data.csv')
    return data
data=load_data()
# Calculate the correlation matrix
corr_matrix = data.corr(numeric_only=True)
# Checkbox for displaying correlation matrix heatmap
display_corr_matrix = st.sidebar.checkbox('Display Correlation Matrix Heatmap')
# Display correlation matrix heatmap if checkbox is checked
if display_corr_matrix:
    st.subheader('Correlation Matrix Heatmap')
    fig = px.imshow(corr_matrix, title='Correlation Matrix Heatmap')
    st.sidebar.plotly_chart(fig)
# Checkbox for displaying histogram
display_histogram = st.sidebar.checkbox('Display Histogram of PM2.5 Values')
# Display histogram if checkbox is checked
if display_histogram:
    st.header('Histogram of PM2.5 Values')
    fig_hist = px.histogram(data, x='PM2.5', nbins=30, title='Histogram of PM2.5 Values')
    st.plotly_chart(fig_hist)
# Checkbox for displaying scatter plot
display_scatter_plot = st.sidebar.checkbox('Display Scatter Plot: PM2.5 vs AQI')
# Display scatter plot if checkbox is checked
if display_scatter_plot:
    st.header('Scatter Plot: PM2.5 vs AQI')
    fig_scatter = px.scatter(data, x='PM2.5', y='AQI',color="AQI_Bucket", title='Scatter Plot: PM2.5 vs AQI')
    st.plotly_chart(fig_scatter)
# Display a scatter plot comparing features with AQI
if st.sidebar.checkbox('Display Feature visualization using Scatter Plot'):
    st.subheader("Compare features with AQI using a scatter plot")
    feature = st.selectbox("Select the a feature to visualize:",data.columns[:-2])
    scatter_plot = px.scatter(data, x=feature, y='AQI', color="AQI_Bucket", hover_name="AQI_Bucket")
    st.plotly_chart(scatter_plot)
# Display a histogram comparing features with AQI
if st.sidebar.checkbox('Display AQI Distribution using Histogam'):
    st.subheader(" AQI distribution using histogram")
    hist_plot = px.histogram(data, x='AQI',nbins=30, color="AQI_Bucket", hover_name="AQI_Bucket")
    st.plotly_chart(hist_plot)
# Input fields for user to enter feature values
pm25 = st.number_input('PM2.5',min_value=0.0, max_value=1000.0, value=50.0,step=0.1)
pm10 = st.number_input('PM10',min_value=0.0, max_value=1000.0, value=30.0,step=0.1)
no = st.number_input('NO', min_value=0.0, max_value=1000.0,value=0.5,step=0.1)
no2 = st.number_input('NO2',min_value=0.0, max_value=1000.0, value=20.0,step=0.1)
nox = st.number_input('NOx',min_value=0.0, max_value=1000.0, value=25.0,step=0.1)
nh3 = st.number_input('NH3',min_value=0.0, max_value=1000.0, value=20.0,step=0.1)
co = st.number_input('CO', min_value=0.0, max_value=1000.0,value=1.0,step=0.1)
so2 = st.number_input('SO2',min_value=0.0, max_value=1000.0, value=10.0,step=0.1)
o3 = st.number_input('O3',min_value=0.0, max_value=1000.0, value=40.0,step=0.1)
Benzene = st.number_input('Benzene',min_value=0.0, max_value=1000.0, value=1.0,step=0.1)
Toluene = st.number_input('Toluene',min_value=0.0, max_value=1000.0, value=3.0,step=0.1)

# Create a feature array from user input
input_features = [[pm25, pm10, no, no2, nox, nh3, co, so2, o3,Benzene,Toluene]]

# Make a prediction using the model
predicted_aqi = model.predict(input_features)[0]

# Display the predicted AQI to the user
st.header('Prediction')
st.write(f'Predicted Air Quality Index (AQI): {predicted_aqi:.2f}')
#result analysis
st.subheader('Conclusion')

if predicted_aqi <= 50:
    st.success('Air quality index = Good')
elif 50 < predicted_aqi <= 110:
    st.success('Air quality index = Satisfactory')
elif 110 < predicted_aqi <= 200:
    st.success('Air quality index = Moderate')
elif 200 < predicted_aqi <= 300:
    st.error('Air quality index = Poor')
else:
    st.error('Air quality index = Very Poor')