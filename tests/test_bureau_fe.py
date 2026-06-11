import pandas as pd
from src.features.bureau_features import create_bureau_features

# Sample dataset that will be fed to the function that creates the bureau features
def sample_bureau():

    return pd.DataFrame(
        {
            "CREDIT_TYPE": ["Consumer credit"],
            "AMT_ANNUITY": [100],
            "AMT_CREDIT_MAX_OVERDUE": [50],
            "DAYS_ENDDATE_FACT": [-50],
            "AMT_CREDIT_SUM_LIMIT": [0],
            "AMT_CREDIT_SUM_DEBT": [250],
            "DAYS_CREDIT_ENDDATE": [100],
            "AMT_CREDIT_SUM": [1000],
            "CREDIT_ACTIVE": ["Active"],
            "AMT_CREDIT_SUM_OVERDUE": [100],
            "CREDIT_DAY_OVERDUE": [10],
            "CNT_CREDIT_PROLONG": [1],
            "DAYS_CREDIT": [-200],
            "DAYS_CREDIT_UPDATE": [-20]
        }
    )
    
#==========================================================================================
# Testing Methods
#==========================================================================================

def test_output_shape():
    result=create_bureau_features(sample_bureau())
    assert len(result)==len(sample_bureau())

def test_required_columns_exist(sample_bureau):

    result = create_bureau_features(
        sample_bureau
    )

    required_cols = [
        "b_exp_annuity",
        "b_debt_to_credit",
        "b_credit_age",
        "b_has_overdue"
    ]

    for col in required_cols:
        assert col in result.columns

def test_debt_to_credit_ratio(sample_bureau):

    result = create_bureau_features(
        sample_bureau
    )

    assert result.loc[0, "b_debt_to_credit"] == 0.25

def test_has_overdue_flag(sample_bureau):

    result = create_bureau_features(
        sample_bureau
    )

    assert result.loc[0, "b_has_overdue"] == 1