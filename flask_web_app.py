#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 22:41:46 2021

@author: rahul
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('my_model.pkl', 'rb'))
vectorizer=pickle.load(open('vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        fmessg = request.form['fmessg']
        data = [fmessg]
        vect = vectorizer.transform(data).toarray()
        
        y_pred = model.predict(vect)
        if y_pred==1:
            p="Beaware!!!! It's a Spam"
        else:
            p="You are Safe.It's not a Spam"
        return render_template('homepage.html',prediction_text=p)



if __name__ == "__main__":
    app.run(debug=True)