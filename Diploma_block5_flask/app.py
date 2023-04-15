# Импортируем необходимые библиотеки для нашего приложения
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import tensorflow as tf
from tensorflow import keras
import flask
from flask import Flask, request, render_template #, send_from_directory 

app =  flask.Flask(__name__) #,  template_folder = 'templates')

# загрузка модели model_RFM_flask
model = keras.models.load_model(r'./data_block5_flask/model_RFM_flask/')

# загрузка нормализаторов scaler_y.pkl  и scaler_X.pkl
with open(r'./data_block5_flask/model_RFM_flask/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open(r'./data_block5_flask/model_RFM_flask/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

@app.route('/', methods=['POST', 'GET'])

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

def app_calc_predict():
    X_pred = []
    message = ''
    if request.method == 'POST':
        
        # получим данные из наших форм из index.html и заполняем список X_pred
        for i in range(1,13,1):
            X_pred.append(float(flask.request.form.get(f'param{i}')))
        
        #  нормализуем полученные от пользователя данные 
        X_pred_flask = scaler_X.transform(np.array(X_pred))     # .reshape(-1, 11)) # !!!!!!! вопрос  по reshape
        
        # Отправляем нормализованные даные в модель и получаем нормализованную y_pred_flask
        y_pred_flask = model.predict(X_pred_flask)
       
        # Инвертируем расчитанные y_pred_flask  в выходное значение реального y_pred_fin для передачи в 'message'
        y_pred = scaler_y.inverse_transform(y_pred_flask)
        y_pred_fin = y_pred[0,0]
   
        message = y_pred_fin
    
    # указываем шаблон и прототип сайта для вывода    
    return render_template(r'./Diploma_block5_flask/templates/index.html', message=message) 
# Запускаем приложен
if __name__ == '__main__':
    app.run()
'''
* Restarting with watchdog (windowsapi)
2023-04-15 18:52:21.074534: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performanPS C:\Users\grain\Work_folder\MGTU_Diplom\Diploma5_flask> & C:/Users/grain/anaconda3/envs/MyENV_JupiterLab/python.exe c:/Users/grain/Work_folder/MGTU_Diplom/Diploma5_flask/app.py
2023-04-15 18:55:54.930232: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-04-15 18:55:54.931137: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
'''