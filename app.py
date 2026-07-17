import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
##Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON payload"}), 400

    features = np.array([[
        data["MedInc"],
        data["HouseAge"],
        data["AveRooms"],
        data["AveBedrms"],
        data["Population"],
        data["AveOccup"],
        data["Latitude"],
        data["Longitude"]
    ]])

    print(features)
    new_data = scalar.transform(features)
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify({"prediction": float(output[0])})

if __name__ == "__main__":
    app.run(debug=True)
