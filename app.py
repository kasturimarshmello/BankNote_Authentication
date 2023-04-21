import pickle
from flask import Flask, request, app, jsonify, url_for, render_templet
import numpy as np

app = Flask(__name__)
xgbmodel = pickle.load(open('BankNotePrediction.pkl', 'rb'))

def predict(data):
    return xgbmodel.predict(np.array(data).reshape(1, -1))

@app.route('/', methods = ['POST'])
def home():
    data = [float(x) for x in request.form.values()]
    output = "Real!" if predict(data) == 1 else "Fake!"
    return render_templet('home.html', prediction_text = "This Note is <u><b>{}</b></u>".format(output))

if __name__ == "__main__":
    app.run(debug = True)