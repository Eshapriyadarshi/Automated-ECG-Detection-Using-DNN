from sys import version
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from runner import predict_result
import tensorflow as tf
import pandas as pd

#Initialize the flask App
app = Flask(__name__)
#model = pickle.load(open('./model files/mymodel_cnn+lstm.pkl', 'rb'))
model_dir= "./cnn+lstm"
localhost_save_option = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")

model = tf.keras.models.load_model(model_dir, options=localhost_save_option)

#default page of our web-app
@app.route('/')
def home():
    return render_template('front.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    patient_record = request.files['csvfile']
    file_path = './static/files/' + patient_record.filename 
    patient_record.save(file_path)
    result = predict_result(file_path, model)
    form_details = [x for x in request.form.values()]
    record_no= form_details[0]
    Patient_name= form_details[1]
    patient_id= form_details[2]

    print(form_details)
    classes = ['N', 'L', 'R', 'A', 'V']
    normal = len([x for x in result[0] if x==True])
    left_bundle = len([x for x in result[1] if x==True])
    right_bundle = len([x for x in result[2] if x==True])
    Artrial_filtration = len([x for x in result[3] if x==True])
    Ventricular = len([x for x in result[4] if x==True])
    
    return render_template('res.html', record_no=record_no, Patient_name=Patient_name, patient_id=patient_id, prediction_text=f'Abnormalities Found are :Normal-{normal}, left Bundle branch block-{left_bundle}, right bundle branch block-{right_bundle}, artrial premature-{Artrial_filtration}, Premature Ventricular-{Ventricular}')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0',port=5000)


