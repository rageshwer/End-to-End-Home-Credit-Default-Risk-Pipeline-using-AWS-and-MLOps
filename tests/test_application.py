'''This is the test file for the application features.'''
import pandas as pd
from src.features.application_features import create_application_features

sample= pd.read_csv('tests/mock_data/raw/application_train_sample.csv')
target=sample['TARGET']
sample=sample.drop('TARGET',axis=1)

result=create_application_features(sample.copy())
result_=result.copy()
result_['TARGET']=target
result_.to_parquet('tests/mock_data/processed/app_train_fe.parquet')

sample_t=pd.read_csv('tests/mock_data/raw/application_test_sample.csv')
result_t=create_application_features(sample_t)
result_t.to_parquet('tests/mock_data/processed/app_test_fe.parquet')

def application_sample():
    return sample.copy()

def fe_result():
    return result

def test_application_feature_engineering_runs():
    """
    Verify feature engineering runs successfully
    on a realistic sample dataset.
    """
    assert isinstance(fe_result(), pd.DataFrame)
    assert len(fe_result()) == len(application_sample())


def test_expected_feature_columns_created():
    """
    Verify important engineered columns exist.
    """
    expected_cols = [
        "age_years",
        "employment_years",
        "credit_to_income",
        "income_per_person",
        "ext_mean",
        "contact_count",
        "ORG_IS_RARE",
        "EMPLOYMENT_SEGMENT"
    ]
    for col in expected_cols:
        assert col in fe_result().columns


def test_days_employed_placeholder_replaced():
    """
    Verify Home Credit sentinel value
    365243 is converted to NaN.
    """
    df = application_sample()
    df.loc[df.index[0], "DAYS_EMPLOYED"] = 365243
    result=create_application_features(df)
    assert pd.isna(
        result.loc[df.index[0], "DAYS_EMPLOYED"]
    )


def test_row_count_preserved():
    """
    Feature engineering should not
    add or remove rows.
    """
    n_rows_before = len(application_sample())
    assert len(fe_result()) == n_rows_before

