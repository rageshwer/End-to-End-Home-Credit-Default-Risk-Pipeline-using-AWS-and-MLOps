from fastapi import FastAPI,Path
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel,field_validator
from functools import lru_cache
import mlflow
from mlflow.lightgbm import load_model
import pandas as pd
import numpy as np

def load_model(RUN_ID):
    # Loading the model
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    # Update this string whenever we train a better model run! 
    return mlflow.lightgbm.load_model(f"runs:/{RUN_ID}/lgbm_750feat")

# Loading the application test dataset features
@lru_cache
def load_test_data():
    # Load the full dataframe so we can query rows later
    df = pd.read_parquet('api/data/final_test.parquet')
    df['SK_ID_CURR'] = df['SK_ID_CURR'].astype(int)
    valid_ids = set(df['SK_ID_CURR'].tolist())
    return df, valid_ids

@lru_cache
def load_lookup():
    df=pd.read_parquet('api/data/lookup_df.parquet')
    return df

#================================================================================================

# Load the model using run_id
model=load_model("01fa115e85d34f5da1de226caca345eb")
features=model.feature_name_
# Loading the dataframes
app_test, existing_ids = load_test_data()
lookup_df=load_lookup()

# Pydantic validation of input application id
class validate_id(BaseModel):
    app_id:int
    @field_validator('app_id')
    @classmethod
    def validate_app_id(cls,app_id:int)->int:
        if app_id not in existing_ids:
            raise ValueError('ID not in test set.')
        return app_id

# Creating endpoint for predicting probability of default when button clicked
app=FastAPI()
@app.post('/predict')
def predict(id:int)->JSONResponse:
    validate_id_obj=validate_id(app_id=id)
    row=app_test[app_test['SK_ID_CURR']==validate_id_obj.app_id]

    # LGBM requires the columns same in order and type as it was trained on.
    str_cols = row[features].select_dtypes(
    include=['object', 'string']
    ).columns
    for col in str_cols:
        row[col] = row[col].astype('category')
    row=row[features]

    prediction=model.predict_proba(row)[0]
    prob_def=prediction[1]
    return JSONResponse(status_code=200,content={
        'Probability of Default :':prob_def,
        'Result :':"The borrower has high default probability." if prob_def>0.08 else "Loan can be approved."
    })

# Creating endpoint for sending 1000 rows data for beeswarm
@app.get('/fetch-batch')
def beeswarm()->JSONResponse:
    shap_slice = lookup_df.head(1000)
    ids = shap_slice['SK_ID_CURR'].astype(int).values
    
    raw_slice = app_test[app_test['SK_ID_CURR'].isin(ids)].copy()
    raw_slice['SK_ID_CURR'] = pd.Categorical(raw_slice['SK_ID_CURR'], categories=ids, ordered=True)
    raw_slice = raw_slice.sort_values('SK_ID_CURR')
    
    shap_cols = [f'shap_{col}' for col in features]
    
    # 1. Handle SHAP cleanly
    shap_matrix = shap_slice[shap_cols].fillna(0).values.tolist()
    
    # 2. Extract raw features as a list of lists directly
    # This keeps numeric types intact (int/float) and objects as strings
    raw_matrix_df = raw_slice[features].copy()
    
    # Replace infinities before converting
    raw_matrix_df = raw_matrix_df.replace([np.inf, -np.inf], np.nan)
    
    # Convert entire DF to a nested Python list
    raw_matrix = raw_matrix_df.values.tolist()
    
    # 3. Deep-clean the nested lists to replace np.nan with None (JSON null)
    # This avoids converting valid numbers or None into strings like "nan"
    cleaned_raw_matrix = [
        [None if (isinstance(val, float) and np.isnan(val)) else val for val in row]
        for row in raw_matrix
    ]
    
    content = {
        "features": features,
        "shap_values": shap_matrix,
        "raw_values": cleaned_raw_matrix
    }
    
    # FastAPI's JSONResponse automatically converts Python's None to JSON null
    return JSONResponse(status_code=200, content=content)

# Endpoint for waterfall plot of individual application id
@app.post('/waterfall')
def waterfall(id:int)->JSONResponse:
    validate_id_obj=validate_id(app_id=id)
    sk_id=validate_id_obj.app_id

    # 1. Fetch the SHAP row and Raw row for this specific ID
    shap_row = lookup_df[lookup_df['SK_ID_CURR'] == sk_id].iloc[0]
    raw_row = app_test[app_test['SK_ID_CURR'] == sk_id].iloc[0]

    # 2. Extract or define the base value (expected value)
    # If expected_value isn't a column, change this to your known model baseline (e.g., 0.35)
    base_value = float(shap_row.get('base_value')) 

    # 3. Pair SHAP values with Raw values across all 750 features
    all_features = []
    for col in features:
        shap_val = float(shap_row.get(f'shap_{col}', 0.0))
        raw_val = raw_row.get(col, None)
        
        # Handle JSON serialization safe replacements for NaN/Inf
        if isinstance(raw_val, float) and (np.isnan(raw_val) or np.isinf(raw_val)):
            raw_val = None
        elif isinstance(raw_val, (np.integer, np.floating)):
            raw_val = float(raw_val)

        all_features.append({
            "name": col,
            "feature_value": raw_val,
            "shap_value": shap_val
        })

    # 4. Sort all features by absolute SHAP value (descending impact)
    all_features.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    # 5. Segment into Top 10 and the rest
    max_display = 10
    top_features = all_features[:max_display]
    rest_features = all_features[max_display:]

    # 6. Aggregate the remaining 740 features into a single element
    if rest_features:
        rest_shap_sum = sum(f["shap_value"] for f in rest_features)
        top_features.append({
            "name": f"{len(rest_features)} Other Features",
            "feature_value": None,
            "shap_value": round(rest_shap_sum, 6)
        })

    # 7. Calculate total prediction value from the sliced data matrix
    total_shap_sum = sum(float(shap_row.get(f'shap_{col}', 0.0)) for col in features)
    prediction_value = base_value + total_shap_sum

    content = {
        "base_value": base_value,
        "prediction_value": prediction_value,
        "features": top_features
    }

    return JSONResponse(status_code=200,content=content)