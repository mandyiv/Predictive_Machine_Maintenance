import streamlit as st
import pandas as pd
import joblib

# Load the model, scaler, and column information
model = joblib.load('models/predictive_maintenance_model.pkl')
scaler = joblib.load('models/scaler.pkl')
column = joblib.load('models/model_columns.pkl')

def predict(data):
  # Convert data to dataframe
  input_data = pd.DataFrame([data])

  # Ensure correct data types
  input_data = input_data.astype({
    'Type': 'object',
    'Air_temperature': 'float64',
    'Process_temperature': 'float64',
    'Rotational_speed': 'int64',
    'Torque': 'float64',
    'Tool_wear': 'int64',
    'TWF': 'int64',
    'HDF': 'int64',
    'PWF': 'int64',
    'OSF': 'int64',
    'RNF': 'int64'
  })

  # One-hot encode the 'Type' feature
  input_data = pd.get_dummies(input_data, columns=['Type'])

  # Align the input data columns with the training data columns
  input_data = input_data.reindex(columns=column, fill_value=0)

  # Scale the input data
  input_data_scaled = scaler.transform(input_data)

  # Make prediction
  prediction = model.predict(input_data_scaled)

  # Map the prediction to a readable format
  prediction_label = 'Machine will fail' if prediction[0] == 1 else 'Machine will not fail'
  return prediction_label

st.title('Predictive Maintenance Detection')

# Create a form using Streamlit elements
type_input = st.selectbox('Type', ('M', 'L', 'H'))
air_temp = st.number_input('Air temperature [K]', min_value=0.0)
process_temp = st.number_input('Process temperature [K]', min_value=0.0)
rotational_speed = st.number_input('Rotational speed [rpm]')
torque = st.number_input('Torque [Nm]')
tool_wear = st.number_input('Tool wear [min]')
twf = st.number_input('TWF (0 or 1)', min_value=0, max_value=1)
hdf = st.number_input('HDF (0 or 1)', min_value=0, max_value=1)
pwf = st.number_input('PWF (0 or 1)', min_value=0, max_value=1)
osf = st.number_input('OSF (0 or 1)', min_value=0, max_value=1)
rnf = st.number_input('RNF (0 or 1)', min_value=0, max_value=1)

# Submit button
submit_button = st.button('Predict!')

# Make prediction on button click
if submit_button:
  # Prepare data dictionary from user input
  data = {
      'Type': type_input,
      'Air_temperature': air_temp,
      'Process_temperature': process_temp,
      'Rotational_speed': rotational_speed,
      'Torque': torque,
      'Tool_wear': tool_wear,
      'TWF': twf,
      'HDF': hdf,
      'PWF': pwf,
      'OSF': osf,
      'RNF': rnf
  }
  # Make prediction using the defined function
  prediction = predict(data)
  st.write(f"Prediction: {prediction}")
