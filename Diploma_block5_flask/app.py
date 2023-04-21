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
            exp = float(flask.request.form.get(f'param{i}')) 
            X_pred.append(exp)
        #  нормализуем полученные от пользователя данные с помощью загруженног scaler_X
        X_pred_norm = scaler_X.transform(np.array(X_pred).reshape(1, -1))    
        # Отправляем нормализованные даные в модель и получаем нормализованную y_pred_norm
        y_pred_norm = model.predict([[X_pred_norm]])
        # Инвертируем расчитанныю y_pred_norm  в выходное значение
        y_pred = scaler_y.inverse_transform(y_pred_norm)
        result = y_pred
        # указываем шаблон и прототип сайта для вывода    
        return render_template('main.html', result = result) 
# Запускаем приложен
if __name__ == '__main__':
    app_zzz.run(debug = True) # с отладкой debug = True