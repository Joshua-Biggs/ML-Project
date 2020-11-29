# -*- coding: utf-8 -*-


import numpy as np
from flask import Flask,render_template,request
import pickle


app = Flask(__name__)
model = pickle.load(open('model1.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

classx = []
@app.route('/predict_expenses',methods=['POST'])
def predict_expenses():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]


    return render_template('index.html', prediction_text='Predicted class of sample: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)