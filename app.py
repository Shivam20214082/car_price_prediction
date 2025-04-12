from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import traceback

# Initialize Flask app and enable CORS
app = Flask(__name__)
cors = CORS(app)

# Load XGBoost model
xgboost_model = pickle.load(open('models/xgb_model.pkl', 'rb'))

# Load car dataset
car = pd.read_csv('data/car_data4.csv')

# Create the 'model' column by extracting from 'name'
car['model'] = car['name'].apply(lambda x: ' '.join(x.split()[1:]))  # Assuming first word is the brand

def preprocess_input_data(input_data):
    try:
        # Rename columns to match the expected ones
        input_data.rename(columns={'kilo_driven': 'kms_driven'}, inplace=True)

        # Convert 'year' and 'kms_driven' to numeric if they are not already
        input_data['year'] = pd.to_numeric(input_data['year'], errors='coerce')  # Convert to numeric, set errors to NaN if invalid
        input_data['kms_driven'] = pd.to_numeric(input_data['kms_driven'], errors='coerce')  # Ensure 'kms_driven' is numeric
        
        # Remove brand name from model
        input_data['model'] = input_data['car_models'].apply(lambda x: ' '.join(x.split()[1:]))  
        
        # Calculate car age (Ensure 'year' is numeric)
        input_data['car_age'] = 2025 - input_data['year']
        
        # Drop columns that are not required for prediction
        input_data.drop(['car_models', 'year'], axis=1, inplace=True)

        # Reorder columns to match the expected order for the model
        input_data = input_data[['fuel_type', 'kms_driven', 'company', 'model', 'car_age']]

        return input_data
    except Exception as e:
        raise ValueError(f"Error in preprocessing data: {str(e)}")

@app.route('/', methods=['GET', 'POST'])
def index():
    # Extract unique values for dropdowns
    companies = sorted(car['company'].unique())
    companies.insert(0, 'Select Company')
    
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())

    # Set predictions to empty dictionary for GET request
    predictions = {}

    return render_template('index.html', predictions=predictions, companies=companies, 
                           car_models=car_models, years=year, fuel_types=fuel_type)

@app.route('/get_car_models/<company>', methods=['GET'])
def get_car_models(company):
    # Filter car models based on the selected company
    models = sorted(car[car['company'] == company]['name'].unique())
    return jsonify(models)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = request.form
        # print("Received form data:", data)
        
        # Convert the form data into a DataFrame
        input_data = pd.DataFrame([data])
        # print("Dataframe columns:", input_data.columns)
         
        # Preprocess the input data
        input_data = preprocess_input_data(input_data)
        # print("Processed input data columns:", input_data.columns)
        
        # Ensure that 'model' is in the dataframe and car dataset
        if 'model' not in input_data.columns:
            raise ValueError("'model' column missing in the input data.")
        
        # Ensure the columns match the model's expected features
        expected_columns = ['fuel_type', 'kms_driven', 'company', 'model', 'car_age']
        if list(input_data.columns) != expected_columns:
            raise ValueError(f"Expected columns: {expected_columns}, but got: {list(input_data.columns)}")
        
        # Map categorical variables to the mean price for each category
        for col in ['company', 'fuel_type', 'model']:
            if col not in car.columns:
                raise ValueError(f"Column '{col}' missing in car dataset.")
            means = car.groupby(col)['Price'].mean()
            input_data[col] = input_data[col].map(means)
        
        # Ensure no NaN values are in the input after mapping
        if input_data.isnull().values.any():
            raise ValueError("Missing values in input data after encoding.")
        
        # Make the prediction using XGBoost model
        xgboost_prediction = xgboost_model.predict(input_data)[0]

        # Prepare the predictions dictionary (only XGBoost)
        predictions = {
            'baseline_xgboost': np.round(float(xgboost_prediction), 2)  # Convert to native float
        }

        return jsonify({"predictions": predictions})
    
    except Exception as e:
        # Log the error traceback for debugging
        error_message = traceback.format_exc()
        print("Error during prediction:", error_message)
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=10000)
