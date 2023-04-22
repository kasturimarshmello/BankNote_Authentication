import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np

app = Flask(__name__)
xgbmodel = pickle.load(open('BankNotePrediction.pkl', 'rb'))

def predict(data):
    return xgbmodel.predict(np.array(data).reshape(1, -1))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/', methods = ['POST'])
def prediction():
    data = [float(x) for x in request.form.values()]
    output = "Real!" if predict(data) == 1 else "Fake!"
    return render_template('home.html', prediction_text = "This Note is <b>{}</b>".format(output))

if __name__ == "__main__":
    app.run(debug = True)