

from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
import pickle
import json


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


file ="KNNimputer.pkl"
with open(file,'rb') as file:
    imputer = pickle.load(file)
    
file ="SVD.pkl"
with open(file,'rb') as file:
    SVD = pickle.load(file)
    
file ="PCA.pkl"
with open(file,'rb') as file:
    PCA = pickle.load(file)
    
file ="scaler.pkl"
with open(file,'rb') as file:
    Minmax = pickle.load(file)

file ="Tree_sale.pkl"
with open(file,'rb') as file:
    tree_sale = pickle.load(file)

file ="Tree_fore.pkl"
with open(file,'rb') as file:
    tree_forecast = pickle.load(file)
    
    
file = "bestmodel.pkl"
with open(file,'rb') as file:
    bestmodel = pickle.load(file)





def final_function_1(X):
    # replacing -99 by Nan in performance column
    X.perf_6_month_avg.replace({-99.0 : np.nan},inplace=True)
    X.perf_12_month_avg.replace({-99.0 : np.nan},inplace=True)
 
    # Converting categories like Yes and No to 0s and 1s
    categorical_columns = ['rev_stop','stop_auto_buy','ppap_risk','oe_constraint','deck_risk','potential_issue']
    for col in categorical_columns:
        X[col].replace({'Yes':1,'No':0},inplace=True)
        X[col]=X[col].astype(int)
 
    # Removing outliers points by taking only values below 99 percentile
    X = X[(X.national_inv >= 0.000) & (X.national_inv <= 5487.000) & (X.in_transit_qty <= 5510.000 ) &        (X.forecast_3_month <= 2280.000) & (X.forecast_6_month <= 4335.659999999916) &        (X.forecast_9_month <= 6316.000) & (X.sales_1_month <= 693.000) & (X.sales_3_month <= 2229.000) &        (X.sales_6_month <= 4410.000) & (X.sales_9_month <= 6698.000) & (X.min_bank <= 679.6599999999162)]
    
  
    # KNN Imputation
    cols = X.columns
    X = pd.DataFrame(imputer.transform(X),columns = cols)

    # Getting PCA Features
    X_pca = PCA.transform(X)
    #Adding PCA  features in the main dataframe
    for i in range(2):
        X['PCA'+str(i)] = X_pca[:,i]
    # Getting SVD Features
    X_svd = SVD.transform(X)

    # Adding SVD  features in the main dataframe
    for i in range(2):
        X['SVD'+str(i)] = X_svd[:,i]
    
    # Dicretisation using Decision Tree
    X['sales_9_tree'] = tree_sale.predict_proba(X.sales_9_month.to_frame())[:,1]
    
    # For forecast columns
    X['forecast_9_tree'] = tree_forecast.predict_proba(X.forecast_9_month.to_frame())[:,1]

    
    # Performing MinMaxScaler on Data
    cols = X.columns
    X = pd.DataFrame(Minmax.transform(X),columns = cols)

    opt = bestmodel.predict(X)

    return opt
def json_validate(data):
    try:
        # giving json data to a variable
        load_value = json.loads(data)
        return "Valid Data"
    except Exception as e:
        print(e)
        return "Invalid"
    else:
        return "Valid"

def dataframe_validate(data):
    try:
        data_value = pd.DataFrame.from_dict(data)
        return "Valid Data"
    except Exception as e:
        print(e)
        return "Invalid"
    else:
        return "Valid"

def feature_validate(df):
    print("dataframe shape : ",df.shape)
    if(df.shape[0]>0) and (df.shape[1] == 21) :
        return "Valid"
    else:
        return "Invalid"

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/data')
def fetch():
    f = open("input_text.txt", "r")
    data = f.read()
    return data

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    data = data['text_data']
    data = data.replace("\'","\"")
    validity = json_validate(data)
    if validity == "Invalid":
        return jsonify({'Predicted Output':'Provided string is Invalid'})

    data = json.loads(data)

    df_validity = dataframe_validate(data)
    if df_validity == "Invalid":
        return jsonify({'Predicted Output':'Provided dataframe is invalid'})

    data = pd.DataFrame.from_dict(data)

    feature_validity = feature_validate(data)
    if feature_validity=="Invalid":
        return jsonify({'Predicted Output':'Please check if features are more or less than expected.'})

    predict = final_function_1(data)
    if predict==0:
        output = "is not on Backorder"
    else:
        output = "is on Backorder"
        
    return render_template('index.html', prediction_text='The Product {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)

