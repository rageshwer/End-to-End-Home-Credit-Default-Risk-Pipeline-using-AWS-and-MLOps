import pandas as pd
from src.features.bureau_features import create_bureau_features

# Sample dataset that will be fed to the function that creates the bureau features
sample= pd.read_csv('tests/mock_data/raw/bureau_sample.csv')
result=create_bureau_features(sample.copy())
result.to_parquet('tests/mock_data/processed/bureau_fe.parquet')

def sample_bureau():
    return sample.copy()

def fe_result():
    return result
    
#==========================================================================================
# Testing Methods
#==========================================================================================

def test_output_shape():

    assert len(fe_result())==len(sample_bureau())

def test_required_columns_exist():
    required_cols = [
        "b_exp_annuity",
        "b_debt_to_credit",
        "b_credit_age",
        "b_has_overdue"
    ]

    for col in required_cols:
        assert col in fe_result().columns
