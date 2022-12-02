from flask import Flask, jsonify, flash, request, redirect, url_for, session, Response
import pickle
import json
from datetime import datetime


app = Flask(__name__)
filename = (r'C:/Users/PMR LAB/Desktop/New folder/Arima_pickle/finalized_model_arima_1.pickle')
filename1 = (r'C:/Users/PMR LAB/Desktop/New folder/Sarima_pickle/finalized_model_Sarima_1.pickle')

model_arima = pickle.load(open(filename, 'rb'))
model_sarima = pickle.load(open(filename1, 'rb'))


@app.route('/predict',methods=['POST'])
def arima_predict():
    print(request.json)
    start = request.json['start']
    end = request.json['end']
    preds_arima = model_arima.predict(start = start, end = end, dynamic= True)
    preds_sarima = model_sarima.predict(start = start, end = end, dynamic= True)

    return{
        'resultStatus': 'SUCCESS',
        'model_arima': str(preds_arima.to_dict()),
        'model_sarima': str(preds_sarima.to_dict())
    }
