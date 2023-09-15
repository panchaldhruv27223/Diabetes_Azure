from flask import Flask, redirect, request, url_for ,Request, render_template
import pickle
import pandas as pd
import numpy as np
import seaborn as sns 


app = Flask(__name__)

scaler_model = pickle.load(open("C:\\Users\\dhruv\\Documents\\ML\\ML_pw\\project\\Diabetes- azur\\model\\scaler_model.pkl","rb"))
gaussion_model = pickle.load(open("C:\\Users\\dhruv\\Documents\\ML\\ML_pw\\project\\Diabetes- azur\\model\\gaussion_model.pkl","rb"))

@app.route("/")
def first():
    return render_template("index.html")

@app.route("/my_data", methods=['GET','POST'])
def my_data():
    if request.method == 'POST':
        Pregnancies	= float(request.form.get("Pregnancies"))
        Glucose	= float(request.form.get("Glucose"))
        BloodPressure= float(request.form.get("BloodPressure"))
        SkinThickness	= float(request.form.get("SkinThickness"))
        Insulin	= float(request.form.get("Insulin"))
        BMI	= float(request.form.get("BMI"))
        DiabetesPedigreeFunction	= float(request.form.get("DiabetesPedigreeFunction"))
        Age	= float(request.form.get("Age"))
        scale_value  = scaler_model.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predicted_value = gaussion_model.predict(scale_value)[0]

        return render_template("result.html",result=predicted_value)


if __name__== "__main__":
    app.run(host="0.0.0.0",debug=True)
