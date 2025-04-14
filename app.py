from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model and data columns
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']
    locations = data_columns[3:]  # First 3 columns are sqft, bath, bhk

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    return jsonify({
        'locations': locations
    })

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    data = request.get_json()

    location = data['location']
    total_sqft = float(data['total_sqft'])
    bhk = int(data['bhk'])
    bath = int(data['bath'])

    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    predicted_price = model.predict([x])[0]
    return jsonify({
        'estimated_price': round(predicted_price, 2)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
