from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import numpy as np
from sklearn import linear_model
import joblib
from bs4 import BeautifulSoup
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import os



# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


def preprocess(query):
    preprocessed_query = []
    
    def process(x,pattern):
        r = re.compile(pattern)
        l = r.findall(x)
        return len(l)
    
    def combined_keywords(x):
        r = re.compile(r'null')
        m = re.compile(r'chr')
        n = re.compile(r'char')
        l = r.findall(x)
        k = m.findall(x)
        j = n.findall(x)
        return len(l) + len(k) + len(j)
    
    def genuine(x):
        count = 0
        genuine_keys = ['select','top','order','fetch','join','avg','count','sum','rows']
        for i in x.split():
            if(i in genuine_keys):
                count = count + 1
        return count
    
    preprocessed_query.append(process(query,"'"))
    preprocessed_query.append(process(query,'"'))
    preprocessed_query.append(process(query,"[!\"#$%&\'()*+,-.\/:;<=>?@[\\]^_`{|}~]"))
    preprocessed_query.append(process(query,'(--)'))
    preprocessed_query.append(process(query,'(\/\*)'))
    preprocessed_query.append(process(query,'\s+'))
    preprocessed_query.append(process(query,"%"))
    preprocessed_query.append(process(query,'\snot\s|\sand\s|\sor\s|\sxor\s|&&|\|\||!'))
    preprocessed_query.append(process(query,"'\+|-|[^\/]\*|\/[^\*]'"))
    preprocessed_query.append(process(query,"null"))
    preprocessed_query.append(process(query,'0[xX][0-9a-fA-F]+\s'))
    preprocessed_query.append(process(query,'[a-zA-Z]'))
    preprocessed_query.append(process(query,'[0-9]'))
    preprocessed_query.append(combined_keywords(query))
    preprocessed_query.append(genuine(query))
    
    return preprocessed_query
    

@app.route('/predict', methods=['POST'])
def predict():
    '''
    query parameter is a string given to predict that it is Sql injection or not.
    
    '''
    li = []
    to_predict_list = request.form.to_dict()
    query = to_predict_list['query']
    query = query.lower()#converting query to lowercase
    arr = preprocess(query)#preprocessing the query
    li.append(query)
    
    with open('train_bow','rb') as f:
         train_bow = pickle.load(f)
    unigram_bow = train_bow.transform(li)
    
    combine = hstack((unigram_bow,arr))
    
    #loading the model
    xgboost_model = joblib.load('saved_model.pkl')
    
    #predicting the output from the loaded model
    predict = xgboost_model.predict(combine)
    
    for i in predict:
        if(i == 1):
              prediction = "Positive"
        else:
              prediction = "Negative"
    
    
    
    return flask.render_template('predict.html', prediction = prediction)
    

  

    
        


if __name__ == '__main__':
    app.run(debug=False)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
