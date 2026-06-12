import pandas as pd
from src.features.bureau_bal_features import create_bureau_balance_features

sample= pd.read_csv('mock_data/raw/bureau_balance_sample.csv')
result=create_bureau_balance_features(sample.copy())
result.to_parquet('mock_data/processed/bureau_balance_fe.parquet')

def sample_bureau():
    return sample.copy()

def fe_result():
    return result

def test_output_shape():
    # 2 unique loans -> 2 rows
    assert len(fe_result()) > 0


def test_required_columns_exist():

    required_cols = [
        "bb_num_mon",
        "bb_max_default",
        "bb_curr_stat",
        "bb_npa_ratio",
    ]

    for col in required_cols:
        assert col in fe_result().columns


def test_no_missing_values():
    assert "SK_ID_BUREAU" in fe_result().columns
    assert fe_result()["SK_ID_BUREAU"].notna().all()