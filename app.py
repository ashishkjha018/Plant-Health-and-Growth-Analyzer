from flask import Flask, request, render_template, jsonify
from predicttion import Apple, Cherry, Grape, Potato, Strawberry
import tensorflow as tf
from keras.models import load_model
import numpy as np

app = Flask("__name__")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/apple")
def apple():
    return render_template("apple.html")

@app.route("/cherry")
def cherry():
    return render_template("cherry.html")

@app.route("/grape")
def grape():
    return render_template("grape.html")

@app.route("/potato")
def potato():
    return render_template("potato.html")

@app.route("/strawberry")
def strawberry():
    return render_template("strawberry.html")

@app.route("/submitApple", methods=['POST'])
def get_output_apple():
    if request.method == 'POST':
        data = request.get_json()

        # Assuming the key for the image data is 'image_data'
        image_data = data.get('image_data')

        # Use the image_data for prediction
        prediction = Apple(image_data)

        # You can return JSON response if needed
        return jsonify({'prediction': prediction})
    
        

@app.route("/submitCherry", methods=['POST'])
def get_output_cherry():
    if request.method == 'POST':
        data = request.get_json()

        # Assuming the key for the image data is 'image_data'
        image_data = data.get('image_data')

        # Use the image_data for prediction
        prediction = Cherry(image_data)

        # You can return JSON response if needed
        return jsonify({'prediction': prediction})

@app.route("/submitGrape", methods=['POST'])
def get_output_grape():
    if request.method == 'POST':
        data = request.get_json()

        # Assuming the key for the image data is 'image_data'
        image_data = data.get('image_data')

        # Use the image_data for prediction
        prediction = Grape(image_data)

        # You can return JSON response if needed
        return jsonify({'prediction': prediction})

@app.route("/submitPotato", methods=['POST'])
def get_output_potato():
    if request.method == 'POST':
        data = request.get_json()

        # Assuming the key for the image data is 'image_data'
        image_data = data.get('image_data')

        # Use the image_data for prediction
        prediction = Potato(image_data)

        # You can return JSON response if needed
        return jsonify({'prediction': prediction})

@app.route("/submitStrawberry", methods=['POST'])
def get_output_strawberry():
    if request.method == 'POST':
        data = request.get_json()

        # Assuming the key for the image data is 'image_data'
        image_data = data.get('image_data')

        # Use the image_data for prediction
        prediction = Strawberry(image_data)

        # You can return JSON response if needed
        return jsonify({'prediction': prediction})



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)