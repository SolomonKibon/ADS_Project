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
raw_data = pd.read_csv('cleaned_data.csv')
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
    st.plotly_chart(fig)
    st.write('In this project,I conducted an in-depth analysis to explore the correlation between various environmental features and the Air Quality Index (AQI). The primary goal was to understand how different factors contribute to changes in AQI, providing valuable insights for air quality monitoring and prediction.My findings reveal several noteworthy observations: PM2.5 Concentration Dominates AQI: The concentration of fine particulate matter (PM2.5) displayed the highest positive correlation with AQI. This suggests that PM2.5 is a significant contributor to overall air quality, and reductions in PM2.5 levels are likely to lead to improvements in AQI.  While other features such as NO2, CO, and SO2 showed correlations with AQI, their impact was less pronounced compared to PM2.5. These pollutants still play roles in determining air quality but to a lesser extent. In summary, our analysis highlights the complex interplay of various environmental factors in determining AQI. PM2.5 concentration emerged as the most influential factor, underscoring the importance of targeted efforts to reduce PM2.5 emissions for improving air quality.')
# Checkbox for displaying histogram
display_histogram = st.sidebar.checkbox('Display Histogram of PM2.5 Values')
# Display histogram if checkbox is checked
if display_histogram:
    st.header('Histogram of PM2.5 Values')
    fig_hist = px.histogram(data, x='PM2.5', nbins=30, title='Histogram of PM2.5 Values')
    st.plotly_chart(fig_hist)
    st.write('In this project,I delved into the analysis of PM2.5 (Particulate Matter with a diameter of 2.5 micrometers or less) concentration levels to gain insights into air quality. PM2.5 is a crucial air pollutant known for its adverse health effects, and understanding its distribution is paramount for air quality management.My analysis revealed the following key observations:Skewed Distribution: The histogram of PM2.5 concentrations displayed a positively skewed distribution. This skewness suggests that most of the observations fall within lower concentration levels, while a relatively smaller proportion of data points exhibit higher PM2.5 concentrations. This pattern is consistent with real-world scenarios, where air quality tends to vary, and pollution events are less frequent.Peak Concentration Range: The histogram exhibited a prominent peak in the lower PM2.5 concentration range. This peak corresponds to periods when air quality is relatively better, reflecting the absence of significant pollution sources or meteorological conditions conducive to pollutant dispersion.Outlier Events: Although most data points clustered around lower PM2.5 concentrations, the histogram`s extended tail to the right indicated the presence of outlier events. These outliers represent instances of elevated PM2.5 concentrations, often associated with pollution events, adverse weather conditions, or local emission sources.')
# Checkbox for displaying scatter plot
display_scatter_plot = st.sidebar.checkbox('Display Scatter Plot: PM2.5 vs AQI')
# Display scatter plot if checkbox is checked
if display_scatter_plot:
    st.header('Scatter Plot: PM2.5 vs AQI')
    fig_scatter = px.scatter(data, x='PM2.5', y='AQI',color="AQI_Bucket", title='Scatter Plot: PM2.5 vs AQI')
    st.plotly_chart(fig_scatter)
    st.write("The scatter plot depicting the relationship between PM2.5 levels and Air Quality Index (AQI) provides valuable insights into the air quality in our dataset. Here's a conclusion based on the observed patterns:Correlation: The scatter plot shows a discernible pattern indicating a correlation between PM2.5 levels and AQI. As PM2.5 levels increase, there is a clear tendency for AQI values to rise as well. This aligns with the general understanding that higher concentrations of fine particulate matter (PM2.5) in the air often lead to poorer air quality. AQI Buckets: The plot employs color to distinguish different AQI buckets, allowing us to see how data points are distributed within each category. This visual representation aids in understanding the distribution of air quality conditions within the dataset.Outliers: Although not explicitly shown on the plot, it is worth noting that there may be outliers in the data. These outliers could represent exceptional events or measurement errors. Further data analysis and cleansing may be required to address such cases.In summary, this scatter plot serves as a valuable tool for assessing the relationship between PM2.5 levels and AQI, helping to identify trends and patterns in the dataset. It reinforces the importance of monitoring PM2.5 concentrations as a key factor in determining air quality and its impact on public health and the environment.")
# Display a scatter plot comparing features with AQI
if st.sidebar.checkbox('Display Feature visualization using Scatter Plot'):
    st.subheader("Compare features with AQI using a scatter plot")
    feature = st.selectbox("Select a feature to visualize:",data.columns[:-2])
    scatter_plot = px.scatter(data, x=feature, y='AQI', color="AQI_Bucket", hover_name="AQI_Bucket")
    st.plotly_chart(scatter_plot)
    st.write('The scatter plot feature analysis provides a visual comparison of different features with the Air Quality Index (AQI). Here is a brief conclusion for interpreting these scatter plots: Feature Selection: You can select a specific feature from the dropdown menu to visualize its relationship with AQI. Each feature represents a potential factor influencing air quality. Correlation Trends: The scatter plots reveal how each feature correlates with AQI. If points tend to cluster in a specific pattern, it suggests a correlation. For instance, if increasing values of a feature coincide with higher AQI levels, it implies a potential impact on air quality. AQI Bucket Differentiation: The color-coding of data points by AQI bucket helps distinguish different air quality conditions. This distinction allows you to observe how various features affect air quality across different AQI categories.')
# Display a histogram comparing features with AQI
if st.sidebar.checkbox('Display AQI Distribution using Histogam'):
    st.subheader(" AQI distribution using histogram")
    hist_plot = px.histogram(data, x='AQI',nbins=30, color="AQI_Bucket", hover_name="AQI_Bucket")
    st.plotly_chart(hist_plot)
    st.write('The histogram displaying the distribution of AQI (Air Quality Index) values provides essential insights into the air quality conditions within the dataset. Here is a concise conclusion for interpreting this histogram: AQI Distribution: The histogram visually represents the frequency of AQI values. Each bar on the histogram corresponds to a range of AQI values (bin), and the height of each bar indicates how many data points fall into that specific range. Central Tendency: The tallest bar in the histogram represents the mode of the AQI values. This mode signifies the most frequently occurring AQI range within the dataset, giving an insight into the typical air quality conditions. Spread and Variation: The width of the histogram provides information about the spread and variability of AQI values. A broader histogram suggests a wider range of air quality conditions present in the dataset. AQI Bucket Differentiation: The histogram is color-coded by AQI buckets, making it easy to distinguish between different air quality categories. This differentiation allows for a quick assessment of the prevalence of various air quality conditions. Outliers: Although the histogram does not explicitly display outliers, it can help identify potential extreme values if they appear as isolated bars far from the central peak.')
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