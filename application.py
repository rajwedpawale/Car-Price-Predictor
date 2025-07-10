from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)
CORS(app)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model and dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car_data = pd.read_csv('Cleaned_Car_data.csv')




def preprocess_input(company, car_model, year, fuel_type, driven):
    """
    Preprocesses the input data and returns a DataFrame for prediction.

    Args:
        company (str): The company name of the car.
        car_model (str): The model name of the car.
        year (str): The year of the car.
        fuel_type (str): The fuel type of the car.
        driven (str): The number of kilometers driven by the car.

    Returns:
        pd.DataFrame or None: Preprocessed input data as DataFrame or None if input is invalid.
    """
    try:
        year = int(year)
        driven = float(driven)
    except ValueError:
        logger.error("Invalid input format")
        return None

    if year < 1900 or year > 2100:
        logger.error("Invalid year")
        return None

    if driven < 0:
        logger.error("Invalid kilometers driven")
        return None

    if company not in car_data['company'].unique() or car_model not in car_data['name'].unique() or fuel_type not in \
            car_data['fuel_type'].unique():
        logger.error("Invalid company, car model, or fuel type")
        return None

    input_data = pd.DataFrame({
        'name': [car_model],
        'company': [company],
        'year': [year],
        'kms_driven': [driven],
        'fuel_type': [fuel_type]
    })
    return input_data


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Renders the index.html template with dropdown options.


    Returns:
        HTML template: The rendered index.html template.
    """


    companies = sorted(car_data['company'].unique())
    car_models = sorted(car_data['name'].unique())
    years = sorted(car_data['year'].unique(), reverse=True)
    fuel_types = car_data['fuel_type'].unique()
    companies.insert(0, 'Select Company')

    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests to predict the car price.

    Returns:
        JSON: Response containing the predicted price or error message.
    """
    data = request.form
    company = data.get('company')
    car_model = data.get('car_models')
    year = data.get('year')
    fuel_type = data.get('fuel_type')
    driven = data.get('kilo_driven')

    input_data = preprocess_input(company, car_model, year, fuel_type, driven)
    if input_data is None:
        return jsonify({'error': 'Invalid input data'}), 400

    try:
        prediction = model.predict(input_data)[0]
        formatted_prediction = round(prediction, 2)
        return jsonify({'prediction': formatted_prediction})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500


@app.errorhandler(404)
def page_not_found(e):
    """
    Renders a custom 404 error page.

    Args:
        e: Error object.

    Returns:
        HTML template: The rendered 404.html template.
    """
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True)
