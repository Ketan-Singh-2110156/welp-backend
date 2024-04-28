from flask import Flask, request, jsonify # type: ignore
import numpy as np # type: ignore
import json
import joblib # type: ignore
from geopy import Nominatim

app = Flask(__name__)

geolocator=Nominatim(user_agent="ketan123@gmail.com",timeout=5)

def get_lat_long(address):
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

def lo_model(model_add):
    try:
        model = joblib.load(open(model_add, 'rb')) 
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

pre_model = lo_model('model.pkl')


@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        if 'input_data' not in data:
            return jsonify({'error': 'Input data is missing'}), 400
        input_data = np.array(data['input_data'])
        prediction = pre_model.predict(input_data) 
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/geocode', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'input_data' not in data:
            return jsonify({'error': 'Input data is missing'}), 400
        result = get_lat_long(data['input_data'])
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)