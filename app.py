from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

Log_final = pickle.load(open('titanic_log_reg_flask.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contactme', methods=['GET'])
def contactme():
    return render_template('contactme.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    
    if request.method == 'POST':
    
        Age = int(request.form['Age'])
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = float(request.form['Fare'])
        
        temp_array = temp_array + [Age,SibSp,Parch,Fare]
        
        Gender = request.form['Gender']
        if Gender == 'Male':
            temp_array = temp_array + [1]
        else:
            temp_array = temp_array + [0]
          
        Embarked = request.form['Embarked']
        if Embarked == 'Queenstown':
            temp_array = temp_array + [1,0]
        elif Embarked == 'Southampton':
            temp_array = temp_array + [0,1]    
        else:
            temp_array = temp_array + [0,0] 
            
        Passenger_Class = request.form['Passenger Class']
        if Passenger_Class == '1':
            temp_array = temp_array + [0,0]
        elif Passenger_Class == '2':
            temp_array = temp_array + [1,0]    
        elif Passenger_Class == '3':
            temp_array = temp_array + [0,1]    
            
        
        data = np.array([temp_array])
        my_prediction = Log_final.predict_proba(data)[:,1]
        survival = np.round(my_prediction *100)[0]
        
       
    return render_template('result.html', prediction = survival)    
        
   
if __name__ == '__main__':
    app.run(debug = True)
