from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import numpy as np

app = Flask(__name__)
CORS(app)

# Load saved model and data columns
model = pickle.load(open('banglore_home_prices_model.pickle', 'rb'))
__locations = None
__data_columns = None

with open("columns.json", "r") as f:
    __data_columns = json.load(f)['data_columns']
    __locations = __data_columns[3:]  # Assuming first 3 are sqft, bath, bhk

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    return jsonify({
        'locations': __locations
    })

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    data = request.get_json()
    location = data['location']
    sqft = data['total_sqft']
    bhk = data['bhk']
    bath = data['bath']

    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    predicted_price = model.predict([x])[0]
    return jsonify({'estimated_price': round(predicted_price, 2)})

if __name__ == "__main__":
    app.run(debug=True)
