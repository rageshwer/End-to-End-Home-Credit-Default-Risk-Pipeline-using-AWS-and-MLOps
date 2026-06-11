import pandas as pd
from src.features.bureau_bal_features import create_bureau_balance_features

# Sample dataset fed to the feature engineering file to check for outputs
def sample_bureau_balance():

    return pd.DataFrame(
        {
            "SK_ID_BUREAU": [1, 1, 1, 2, 2],
            "MONTHS_BALANCE": [-2, -1, 0, -1, 0],
            "STATUS": ["0", "1", "2", "C", "5"]
        }
    )


def test_output_shape():

    result = create_bureau_balance_features(
        sample_bureau_balance()
    )

    # 2 unique loans -> 2 rows
    assert result.shape[0] == 2


def test_required_columns_exist():

    result = create_bureau_balance_features(
        sample_bureau_balance()
    )

    required_cols = [
        "bb_num_mon",
        "bb_max_default",
        "bb_curr_stat",
        "bb_npa_ratio",
    ]

    for col in required_cols:
        assert col in result.columns


def test_feature_calculations():

    result = create_bureau_balance_features(
        sample_bureau_balance()
    )

    loan_1 = result[
        result["SK_ID_BUREAU"] == 1
    ].iloc[0]

    assert loan_1["bb_num_mon"] == 3
    assert loan_1["bb_max_default"] == 2
    assert loan_1["bb_curr_stat"] == 2


def test_no_missing_values():

    result = create_bureau_balance_features(
        sample_bureau_balance()
    )

    assert result.isnull().sum().sum() == 0