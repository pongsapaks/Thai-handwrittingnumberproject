import cv2
from PIL import Image, ImageOps
import os
import git
import numpy as np
import pandas as pd
import pickle
import glob
import seaborn as sns
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
import subprocess
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tune_sklearn import TuneSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint
import numpy as np
import shutil
from sklearn.svm import SVC
from dash import Dash, dcc, html
import joblib

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
    data = {"X": X, "Y": Y}
    pickle.dump(data, open(f"thainumber_{size}.pkl", "wb"), protocol=2)

    reshaped_X = X.reshape((X.shape[0], -1))
    Ydf = pd.DataFrame(Y)
    Xdf = pd.DataFrame(reshaped_X)

    X_mean = Xdf.mean()
    X_std = Xdf.std()
    Z = (Xdf - X_mean) / X_std
    Z = Z.fillna(0)

    pca = PCA(n_components=0.75)
    pca.fit(Z)
    X_pca = pca.transform(Z)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=42)

    n_s = np.linspace(0.70, 0.85, num=16)

    for n in n_s:
        pca = PCA(n_components=n)
        pca.fit(Z)
        X_pca = pca.transform(Z)
        X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.2, random_state=42)
        sv = SVC(C=10, gamma=0.001, kernel='rbf')
        sv.fit(X_train, y_train)
        pred = sv.predict(X_test)
        correct = 0
        for i in range(len(y_test)):
            if pred[i] == y_test[i]:
                correct += 1

    sv = SVC(C=10, gamma=0.001, kernel='rbf')
    sv.fit(X_train, y_train)
    pred = sv.predict(X_test)
    correct = 0
    for i in range(len(y_test)):
        if pred[i] == y_test[i]:
            correct += 1

    y_pred_sv = sv.predict(X_test)

    # Train and save the model
    sv = SVC(C=10, gamma=0.001, kernel='rbf')
    sv.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(sv, f)
    with open("pca.pkl", "wb") as f:
        pickle.dump(pca, f)
make_dataset()