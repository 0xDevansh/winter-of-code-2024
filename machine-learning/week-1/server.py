from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel
import pandas as pd
from math import log10
# most code adapted from https://www.geeksforgeeks.org/deploying-ml-models-as-api-using-fastapi/

class request_body(BaseModel):
    step : int
    amount : float
    oldbalanceOrig : float
    newbalanceOrig : float
    oldbalanceDest : float
    newbalanceDest : float
    trans_type: str
    scaled: bool | None = None

app = FastAPI()
model = joblib.load('xgb_model.joblib')

@app.get('/')
def main():
    return {'message': 'Fraud detection API, send POST requests to /predict'}

@app.post('/predict')
def predict(data : request_body):
    type_CASH_IN = 1 if data.trans_type == 'CASH_IN' else 0
    type_CASH_OUT = 1 if data.trans_type == 'CASH_OUT' else 0
    type_TRANSFER = 1 if data.trans_type == 'TRANSFER' else 0
    type_PAYMENT = 1 if data.trans_type == 'PAYMENT' else 0
    type_DEBIT = 1 if data.trans_type == 'DEBIT' else 0


    test_data = pd.DataFrame([[
            data.step, 
            data.amount if data.scaled else log10(data.amount), 
            data.oldbalanceOrig if data.scaled else log10(data.oldbalanceOrig), 
            data.newbalanceOrig if data.scaled else log10(data.newbalanceOrig),
            data.oldbalanceDest if data.scaled else log10(data.oldbalanceDest),
            data.newbalanceDest if data.scaled else log10(data.newbalanceDest),
            type_CASH_IN,
            type_CASH_OUT,
            type_DEBIT,
            type_PAYMENT,
            type_TRANSFER
    ]], columns=['step', 'amount', 'oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'])
    print(test_data)
    result = model.predict(test_data)
    print(result)
    print(result[0])
    print(result[0] == 1)

    return { 'is_fraud' : int(result[0]) == 1}