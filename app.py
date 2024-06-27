import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the pre-trained models for crop and fertilizer recommendation
model_crop_loaded = pickle.load(open('model_saved.pkl', 'rb'))
model_fert_loaded = pickle.load(open('model_saved_fert.pkl', 'rb'))

# Define the function for making crop recommendations
def recommend_crop(features):
    test = model_crop_loaded.predict([features])
    return test[0]

# Define the function for making fertilizer recommendations
def recommend_fertilizer(features):
    test = model_fert_loaded.predict([features])
    return test[0]

# Soil type label mapping
soil_label_mapping = {'Black': 0, 'Clayey': 1, 'Loamy': 2, 'Red': 3, 'Sandy': 4}

# Crop type label mapping
crop_label_mapping = {
    'Barley': 0, 'Cotton': 1, 'Ground Nuts': 2, 'Maize': 3, 'Millets': 4,
    'Oil seeds': 5, 'Paddy': 6, 'Pulses': 7, 'Sugarcane': 8, 'Tobacco': 9, 'Wheat': 10
}

# Define the Streamlit app
def main():
    st.title('Crop and Fertilizer Recommendation System')
    st.write('Enter the following details to get crop and fertilizer recommendations.')

    # Collect user input for features
    temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=100.0, step=0.1)
    humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, step=0.1)
    moisture = st.number_input('Moisture (%)', min_value=0.0, max_value=100.0, step=0.1)
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=1000.0, step=0.1)
    ph = st.number_input('PH Level', min_value=0.0, max_value=14.0, step=0.1)
    soil_type = st.selectbox('Soil Type', list(soil_label_mapping.keys()))
    crop_type = st.selectbox('Crop Type', list(crop_label_mapping.keys()))
    nitrogen = st.number_input('Nitrogen (ppm)', min_value=0.0, max_value=1000.0, step=0.1)
    phosphorus = st.number_input('Phosphorus (ppm)', min_value=0.0, max_value=1000.0, step=0.1)
    potassium = st.number_input('Potassium (ppm)', min_value=0.0, max_value=1000.0, step=0.1)

    # Convert soil type and crop type to numeric representation
    soil_type_num = soil_label_mapping[soil_type]
    crop_type_num = crop_label_mapping[crop_type]

    # Make crop and fertilizer recommendations on button click
    if st.button('Get Recommendations'):
        # Combine features for crop recommendation
        features_crop = [temperature, humidity, rainfall, ph, nitrogen, phosphorus, potassium]
        features_crop = np.array(features_crop)  # Convert to numpy array
        crop_recommendation = recommend_crop(features_crop)
        
        # Combine features for fertilizer recommendation
        features_fert = [temperature, humidity, moisture, soil_type_num, crop_type_num, nitrogen, phosphorus, potassium]
        features_fert = np.array(features_fert)  # Convert to numpy array
        fertilizer_recommendation = recommend_fertilizer(features_fert)
        
        st.write('Recommended Crop:', crop_recommendation)
        st.write('Recommended Fertilizer:', fertilizer_recommendation)

if __name__ == '__main__':
    main()
