import base64
import cv2
import numpy as np
from PIL import ImageOps
from sklearn import joblib
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# Load the pre-trained model and PCA
sv = joblib.load("model.pkl")
pca = joblib.load("pca.pkl")

# Load the mean and standard deviation for feature scaling
X_mean = np.load("X_mean.npy")
X_std = np.load("X_std.npy")

# Define the size of the canvas
size = 200

# Create the Dash app
app = dash.Dash(__name__)

# Set up the app layout
app.layout = html.Div(children=[
    dcc.Loading(
        className='loading',
        children=[
            html.Div(className='canvas-container', children=[
                html.Canvas(id="canvas", width=size, height=size, style={"border": "1px solid black"}),
            ]),
            html.Button("Recognize", id="recognize-btn"),
            html.Div(id="prediction-output")
        ]
    )
])

# Callback function to handle canvas drawing and recognition
@app.callback(
    Output("prediction-output", "children"),
    Input("recognize-btn", "n_clicks"),
    State("canvas", "toDataURL"),
)
def recognize_handwriting(n_clicks, canvas_data):
    if n_clicks is not None:
        # Save the canvas image to a file
        img_data = base64.b64decode(canvas_data.split(",")[1])
        with open("user_input.png", "wb") as f:
            f.write(img_data)

        # Load and preprocess the user input image
        img = cv2.imread("user_input.png")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (28, 28))
        x = img_resized.flatten().reshape(1, -1)
        x_scaled = (x - X_mean) / X_std
        x_pca = pca.transform(x_scaled)
        prediction = sv.predict(x_pca)[0]

        return html.H2(f"Prediction: {prediction}")

    return ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)