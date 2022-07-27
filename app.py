import os
import pickle
import numpy as np
import pandas as pd
import csv
import shutil

from os import path
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from flask import Flask,request, render_template, redirect, url_for, send_from_directory

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if path.exists('static/predicted_slaes.csv'):
        os.remove('static/predicted_sales.csv')
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            new_filename = 'test.csv'
            file.save(os.path.join('input',new_filename))
        shopdata = pd.read_csv("input/test.csv")
        shopdata["Item_Weight"].fillna(shopdata["Item_Weight"].mean(), inplace = True)
        modeOutletSize = shopdata.pivot_table(values = "Outlet_Size",columns = "Outlet_Type", aggfunc =(lambda x: x.mode()[0]))
        shopdata.replace({"Item_Fat_Content" : {"low fat":"Low Fat",'LF':'Low Fat','reg':'Regular'}}, inplace=True)
        encoder = LabelEncoder()
        shopdata['Item_Identifier']= encoder.fit_transform(shopdata['Item_Identifier'])
        shopdata['Item_Fat_Content']= encoder.fit_transform(shopdata['Item_Fat_Content'])
        shopdata['Item_Type']= encoder.fit_transform(shopdata['Item_Type'])
        shopdata['Outlet_Identifier']= encoder.fit_transform(shopdata['Outlet_Identifier'])
        shopdata['Outlet_Size']= encoder.fit_transform(shopdata['Outlet_Size'])
        shopdata['Outlet_Location_Type']= encoder.fit_transform(shopdata['Outlet_Location_Type'])
        shopdata['Outlet_Type']= encoder.fit_transform(shopdata['Outlet_Type'])
        prediction = model.predict(shopdata)
        header = ['Item_Outlet_Sales']
        sale_rows=[]
        sales=[]
        for i in prediction:
          sale_rows.append(i)
          sales.append(sale_rows)
          sale_rows=[]
        with open('predicted_sales.csv','w',newline='') as fil:
          writer = csv.writer(fil)
          writer.writerow(header)
          writer.writerows(sales)
        a=r'predicted_sales.csv'
        b=r'static'
        shutil.move(a,b)
        return send_from_directory('static','predicted_sales.csv')
    

if __name__=="__main__":
    app.run(debug=True)