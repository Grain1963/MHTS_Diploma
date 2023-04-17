# Импортируем необходимые библиотеки для нашего приложения
import numpy as np
import pandas as pd
from sklearn.preprocessing  import MinMaxScaler

import os

import pickle

import tensorflow as tf
from tensorflow import keras

import flask
from flask import Flask, request, render_template 

app =  flask.Flask(__name__,  template_folder = 'templates')

@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['GET', 'POST'])
def main():
        
    if flask.request.method == 'GET':
        return render_template('main.html')

    if request.method == 'POST':
        X_pred = []
        result = 1.0
        # загрузка нашей модели model_RFM_flask
        model = keras.models.load_model(r'./data_block5_flask/model_RFM_flask/')

        # загрузка наших нормализаторов scaler_y.pkl  и scaler_X.pkl
        with open(r'./data_block5_flask/model_RFM_flask/scaler_X.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        with open(r'./data_block5_flask/model_RFM_flask/scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f) 

        # получим данные из наших форм main.html и заполняем список X_pred из 12 параметров
        for i in range(1,13,1):
            exp = float(flask.request.form(f'param{i}')) # убрать .get как у преподавателя???
            X_pred.append(exp)
        #  нормализуем полученные от пользователя данные с помощью загруженног scaler_X
        X_pred_norm = scaler_X.transform(np.array(X_pred))     # .reshape(-1, 11)) # !!!!!!! вопрос  по reshape
        
        # Отправляем нормализованные даные в модель и получаем нормализованную y_pred_norm
        y_pred_norm = model.predict([[X_pred_norm]])
       
        # Инвертируем расчитанныю y_pred_norm  в выходное значение реального result для передачи в форму ответа
        y_pred = scaler_y.inverse_transform(y_pred_norm)
        #result = y_pred[0][0]
        
        # указываем шаблон и прототип сайта для вывода    
        return render_template('main.html', result = y_pred) 
    
# Запускаем приложен
if __name__ == '__main__':
    app.run() # с отладкой debug = True
    
PS C:\Users\grain\Work_folder\MGTU_Diplom\Diploma_block5_flask> & C:/Users/grain/anaconda3/envs/MyENV_JupitePS C:\Users\grain\Work_folder\MGTU_Diplom\Diploma_block5_flask> & C:/Users/grain/anaconda3/envs/MyENV_JupiterLab/python.exe c:/Users/grain/Work_folder/MGTU_Diplom/Diploma_block5_flask/app.py
 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
127.0.0.1 - - [17/Apr/2023 14:41:32] "GET / HTTP/1.1" 200 -