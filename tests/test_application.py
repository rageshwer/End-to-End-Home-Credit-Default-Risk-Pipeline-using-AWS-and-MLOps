'''This is the test file for the application features.'''
import pandas as pd
from src.features.application_features import create_application_features

sample= pd.read_csv('tests/mock_data/sample_application.csv')
def application_sample():
    return sample.copy()

def test_application_feature_engineering_runs():
    """
    Verify feature engineering runs successfully
    on a realistic sample dataset.
    """

    result = create_application_features(
        application_sample().copy()
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(application_sample())


def test_expected_feature_columns_created():
    """
    Verify important engineered columns exist.
    """

    result = create_application_features(
        application_sample().copy()
    )

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
        assert col in result.columns


def test_days_employed_placeholder_replaced():
    """
    Verify Home Credit sentinel value
    365243 is converted to NaN.
    """

    df = application_sample().copy()

    df.loc[df.index[0], "DAYS_EMPLOYED"] = 365243

    result = create_application_features(df)

    assert pd.isna(
        result.loc[df.index[0], "DAYS_EMPLOYED"]
    )


def test_row_count_preserved():
    """
    Feature engineering should not
    add or remove rows.
    """

    n_rows_before = len(application_sample())

    result = create_application_features(
        application_sample().copy()
    )

    assert len(result) == n_rows_before

