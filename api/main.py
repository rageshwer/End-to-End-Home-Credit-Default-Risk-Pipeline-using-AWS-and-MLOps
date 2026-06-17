from fastapi import FastAPI,Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,field_validator
from typing import Annotated
import json
import joblib
import pickle
import pandas as pd
from config.config import top_features_750

# Loading the model
model=joblib.load('artifacts/lgbm_model.pkl')

# Loading the application test dataset features
app_test=pd.read_parquet('data/processed/final_test.parquet')
app_test['SK_ID_CURR']=app_test['SK_ID_CURR'].astype(int)
existing_ids=set(app_test['SK_ID_CURR'].to_list())

# Pydantic validation of input application id
class validate_id(BaseModel):
    app_id:int
    @field_validator('app_id')
    @classmethod
    def validate_app_id(cls,app_id:int)->int:
        if app_id not in existing_ids:
            raise ValueError('ID not in test set.')
        return app_id

# Creating the Endpoint
app=FastAPI()

@app.post('/predict')
def predict(id:int)->JSONResponse:
    validate_id_obj=validate_id(app_id=id)
    row=app_test[app_test['SK_ID_CURR']==validate_id_obj.app_id]

    # LGBM requires the columns same in order and type as it was trained on.
    str_cols = row[top_features_750].select_dtypes(
    include=['object', 'string']
    ).columns
    for col in str_cols:
        row[col] = row[col].astype('category')
    row=row[top_features_750]

    prediction=model.predict_proba(row)[0]
    prob_def=prediction[1]
    return JSONResponse(status_code=200,content={
        'Probability of Default :':prob_def,
        'Result :':"The borrower has high default probability." if prob_def>0.08 else "Loan can be approved."
    })

