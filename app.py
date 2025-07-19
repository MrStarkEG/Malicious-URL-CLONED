# importing required libraries

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import warnings
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble._gb_losses import BinomialDeviance

warnings.filterwarnings("ignore")
from feature import FeatureExtraction

def fix_gradient_boosting_model(model):
    """Fix missing attributes in GradientBoostingClassifier loaded from older sklearn versions"""
    if not hasattr(model, '_loss'):
        # For binary classification, use BinomialDeviance
        model._loss = BinomialDeviance(model.n_classes_)
    if not hasattr(model, '_raw_predict_init'):
        # Add the method if missing
        model._raw_predict_init = lambda X: model._loss.get_init_raw_predictions(X, model.init_)
    return model

file = open("pickle/model.pkl", "rb")
gbc = pickle.load(file)
file.close()

# Fix the model
gbc = fix_gradient_boosting_model(gbc)


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        y_pred = gbc.predict(x)[0]
        # 1 is safe
        # -1 is unsafe
        y_pro_phishing = gbc.predict_proba(x)[0, 0]
        y_pro_non_phishing = gbc.predict_proba(x)[0, 1]
        # if(y_pred ==1 ):
        pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing * 100)
        return render_template("index.html", xx=round(y_pro_non_phishing, 2), url=url)
    return render_template("index.html", xx=-1)


if __name__ == "__main__":
    app.run(debug=True)
