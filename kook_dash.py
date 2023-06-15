import cv2
from PIL import Image, ImageOps
import os
import git
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
import subprocess
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import randint
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from flask import Flask, request
from PIL import Image
from io import BytesIO
import base64
import dash
from dash import dcc, html
from dash_canvas import DashCanvas
import dash_bootstrap_components as dbc
import requests
import json
from dash_canvas.utils import parse_jsonstring
from sklearn.model_selection import GridSearchCV

# In[ ]: Standardization steps are included in make_dataset()
## the last step is making a pickel file called std_params.pkl to store standardization steps.
# We will use this to standardize when receive a picture from canvas
def make_dataset(size=28):
    repo_url = "https://github.com/pongsapaks/Thai-handwrittingnumberproject.git"
    repo_dir = "Thai-handwrittingnumberproject"
    subprocess.run(["git", "clone", repo_url, repo_dir])

    image_dir = os.path.join(repo_dir, "raw")
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".png"):
                image_files.append(os.path.join(root, file))

    print("Total image files:", len(image_files))

    X = []
    Y = []

    for image_path in image_files:
        img = cv2.imread(image_path)
        img = Image.open(image_path).convert("L")
        img = ImageOps.invert(img)
        img = img.resize((size, size))
        label = os.path.basename(os.path.dirname(image_path))
        x = np.array(img)
        X.append(x)
        Y.append(label)

    X = np.asarray(X)
    Y = np.asarray(Y)

    reshaped_X = X.reshape((X.shape[0], -1))
    Ydf = pd.DataFrame(Y)
    Xdf = pd.DataFrame(reshaped_X)

    X_mean = Xdf.mean()
    X_std = Xdf.std()
    Z = (Xdf - X_mean) / X_std
    Z = Z.fillna(0)

    with open("std_params.pkl", "wb") as f:
        pickle.dump((X_mean, X_std), f)

# In[ ]: preprocessing step.
# model.pkl is to store ML model so we can use with canvas
# pca.pkl is to store PCA step so we can preprocess the picture received from canvas
    pca = PCA(n_components=0.8)
    pca.fit(Z)
    X_pca = pca.transform(Z)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.15, random_state=150)

    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}
    sv = SVC()
    clf = GridSearchCV(sv, parameters)
    clf.fit(X_train, y_train)
    
    print("Best parameters found: ", clf.best_params_)
    
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(f"Model accuracy: {accuracy}")

    # Save the trained model
    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("pca.pkl", "wb") as f:
        pickle.dump(pca, f)
make_dataset()

# In[ ]: preprocess_image() is used with a pictured received from canvas.
# This is to adjust the size to be 28*28 and covert to grayscale.
def preprocess_image(image_path, size=28):
    img = Image.open(image_path).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((size, size))
    img_arr = np.array(img)
    return img_arr

# In[ ]:
def predict_image(image, model_path="model.pkl", pca_path="pca.pkl", std_params_path="std_params.pkl", size=28):
    # Load the model, pca, and standardization parameters from files
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)
    with open(std_params_path, 'rb') as f:
        X_mean, X_std = pickle.load(f)

    # Preprocess the image
    # If image is not a path but already an image, skip this line
    if isinstance(image, str):
        image = Image.open(image).convert("L")
    image = ImageOps.invert(image)
    image = image.resize((size, size))
    img_arr = np.array(image)

    # Flatten and standardize the image
    reshaped_image = img_arr.reshape((1, -1))
    # Avoid division by zero by adding a small constant to X_std
    epsilon = 1e-8
    standardized_image = (reshaped_image - np.array(X_mean).reshape(1,-1)) / (np.array(X_std).reshape(1,-1) + epsilon)

    if np.isnan(standardized_image).any():
        print("standardized_image still contains NaN values!")

    # Apply PCA
    transformed_image = pca.transform(standardized_image)

    # Predict the label
    prediction = model.predict(transformed_image)

    return prediction

# Initialize the Flask app
server = Flask(__name__)

@server.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    image_data = data['image']
    image_data = base64.b64decode(image_data.split(',')[1])
    image = Image.open(BytesIO(image_data))

    # preprocess and predict
    prediction = predict_image(image) # Modify this line to suit the preprocessing and prediction in your case
    return str(prediction)

# Initialize the Dash app
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets) # server=server connects Dash to Flask

canvas_width = 500

app.layout = html.Div([
    DashCanvas(id='canvas',
               lineWidth=5,
               width=canvas_width,
               lineColor='Black'
               ),
    html.Button('Predict', id='button_predict', n_clicks=0),
    html.Div(id='prediction')
])

@app.callback(
    dash.dependencies.Output('prediction', 'children'),
    [dash.dependencies.Input('button_predict', 'n_clicks')],
    [dash.dependencies.State('canvas', 'json_data')]
)
def update_output(n_clicks, json_data):
    print("Button clicked")  # This line will print when the button is clicked
    if n_clicks > 0 and json_data is not None:
        print("Data received")  # This line will print if json_data is not None
        mask = parse_jsonstring(json_data)
        image = Image.fromarray(mask.astype('uint8') * 255)
        print("Image created")  # This line will print if the image object is created without any issues
        prediction = predict_image(image)
        print("Prediction made:", prediction)  # This line will print the prediction
        return 'Predicted number: {}'.format(prediction)
    else:
        print("No data received")  # This line will print if json_data is None
        return 'Please draw on the canvas before predicting'

if __name__ == '__main__':
    app.run_server(debug=True)