import cv2
import dash
import dash_core_components as dcc
from dash import html
import numpy as np
import os
import pandas as pd
import pickle
import subprocess
from PIL import Image, ImageOps
from dash.dependencies import Input, Output, State
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from dash_canvas import DashCanvas

def make_dataset(size=28):
    # Same code as before, up until:
    # ... 
    # Save PCA model as pickle file
    with open("pca.pkl", "wb") as f:
        pickle.dump(pca, f)

def convert_json_to_image(canvas_json):
    data_uri = canvas_json['objects'][0]['data']
    img = Image.open(BytesIO(base64.b64decode(data_uri)))
    return img

def preprocess_image(img):
    # Convert to grayscale and resize to the same dimensions as during training
    img = ImageOps.grayscale(img)
    img = img.resize((28, 28))
    img_array = np.array(img)
    return img_array

def load_model():
    # Load the trained model from a file
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    # Load the PCA model
    with open('pca.pkl', 'rb') as file:
        pca = pickle.load(file)
    return model, pca

def perform_prediction(img, number):
    # Load the trained model
    model, pca = load_model()  
    
    # Preprocess the image and perform PCA
    img = img.flatten()
    img = img.reshape(1, -1)
    img_pca = pca.transform(img)
    
    # Perform the prediction using the model and the preprocessed image
    prediction = model.predict(img_pca)
    
    # Return the prediction result
    return prediction

app = dash.Dash(__name__)
app.layout = html.Div([
    DashCanvas(id='canvas', width=280, height=280, style={'border': '1px solid black'}),
    html.Button('Predict', id='predict-button', n_clicks=0),
    html.Div(id='prediction-output')
])
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('canvas', 'json_data'),
)
def predict_number(n_clicks, canvas_json):
    if canvas_json is None:
        return html.H1('Please draw a number.')
        
    # Convert the canvas_json to an image
    img = convert_json_to_image(canvas_json)
    
    # Preprocess the image
    preprocessed_img = preprocess_image(img)
    
    # Perform the prediction
    prediction = perform_prediction(preprocessed_img)
    
    # Return the prediction result as a Dash component
    return html.H1(f'Prediction: {prediction[0]}')

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)