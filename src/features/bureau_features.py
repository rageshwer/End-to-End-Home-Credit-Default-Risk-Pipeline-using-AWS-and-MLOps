''' This script takes input a bureau dataset and feature engineers it to return a dataframe with all
the new required features'''

import numpy as np
import pandas as pd

def create_bureau_features(bureau:pd.DataFrame)->pd.DataFrame:
    #==================================================================================================
    # Missing Imputing
    #==================================================================================================
    missing_col=['AMT_ANNUITY','AMT_CREDIT_MAX_OVERDUE','DAYS_ENDDATE_FACT','AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_DEBT','DAYS_CREDIT_ENDDATE','AMT_CREDIT_SUM']
    # Imputing credit category wise median into annuity 
    category_medians = bureau.groupby('CREDIT_TYPE')['AMT_ANNUITY'].transform('median')
    bureau['b_annuity_imp'] = bureau['AMT_ANNUITY'].fillna(category_medians).fillna(0)

    #==================================================================================================
    # Feature Engineering
    #==================================================================================================
    # Annuity Features
    #============================
    # Creating expected annuity feature for loans 
    debt_paid = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    closed_days = bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT']
    active_days = np.abs(bureau['DAYS_CREDIT']) + 1
    days_elapsed = np.where(
        bureau['CREDIT_ACTIVE'] == 'Closed', 
        closed_days, 
        active_days
    )
    days_elapsed = np.nan_to_num(days_elapsed, nan=1.0)
    days_elapsed = np.maximum(days_elapsed, 1)
    bureau['b_exp_annuity']=(debt_paid/days_elapsed)*30.4     # avg of all 12 months
    bureau['b_exp_annuity']=bureau['b_exp_annuity'].fillna(0)

    # Is missing binary feature for annuity
    bureau['b_annuity_missing'] = bureau['AMT_ANNUITY'].isna().astype('int8')

    # Repayment intensity feature
    bureau['b_annuity_loan_ratio'] = bureau['b_annuity_imp'] / (bureau['AMT_CREDIT_SUM'] + 1).fillna(0)

    # Overdue Features
    #============================
    # Max overdue missing or not
    bureau['b_max_overdue_missing']=bureau['AMT_CREDIT_MAX_OVERDUE'].isna().astype(np.int8)

    # Max overdue relative to total credit amount
    bureau['b_max_overdue_to_credit'] =    bureau['AMT_CREDIT_MAX_OVERDUE']/bureau['AMT_CREDIT_SUM'].replace(0, np.nan)

    # Max overdue relative to outstanding debt
    bureau['b_max_overdue_to_debt'] = (
        bureau['AMT_CREDIT_MAX_OVERDUE'] /
        bureau['AMT_CREDIT_SUM_DEBT'].replace(0, np.nan)
    )

    # Current overdue relative to historical maximum overdue
    bureau['b_current_to_max_overdue'] = (
        bureau['AMT_CREDIT_SUM_OVERDUE'] /
        bureau['AMT_CREDIT_MAX_OVERDUE'].replace(0, np.nan)
    )

    # Historical severity × current delinquency
    bureau['b_max_overdue_x_curr_dpd'] = (
        bureau['AMT_CREDIT_MAX_OVERDUE'] *
        bureau['CREDIT_DAY_OVERDUE']
    )

    # Amount Credit Sum
    #============================
    # Helper variables
    credit_age = np.abs(bureau['DAYS_CREDIT']).replace(0, np.nan)

    total_tenure = np.abs(
        bureau['DAYS_CREDIT_ENDDATE'] -
        bureau['DAYS_CREDIT']
    ).replace(0, np.nan)

    amount_paid = (
        bureau['AMT_CREDIT_SUM'] -
        bureau['AMT_CREDIT_SUM_DEBT']
    )

    # 1. Amount paid per day since loan started
    bureau['b_paid_to_days'] = (
        amount_paid /
        credit_age
    )

    # 2. Fraction of loan still unpaid
    bureau['b_left_to_full'] = (
        bureau['AMT_CREDIT_SUM_DEBT'] /
        bureau['AMT_CREDIT_SUM'].replace(0, np.nan)
    )

    # 3. Credit amount per day of loan tenure
    bureau['b_loan_to_tenure'] = (
        bureau['AMT_CREDIT_SUM'] /
        total_tenure
    )

    # 4. Current overdue relative to total credit
    bureau['b_curr_overdue_to_loan'] = (
        bureau['AMT_CREDIT_SUM_OVERDUE'] /
        bureau['AMT_CREDIT_SUM'].replace(0, np.nan)
    )

    # 5. Credit exposure × current overdue days
    bureau['b_credit_x_curr_dpd'] = (
        bureau['AMT_CREDIT_SUM'] *
        bureau['CREDIT_DAY_OVERDUE']
    )

    # 6. Credit exposure × number of prolongations
    bureau['b_credit_x_prolonged'] = (
        bureau['AMT_CREDIT_SUM'] *
        bureau['CNT_CREDIT_PROLONG']
    )

    # Amount Credit Sum Debt Features
    #=================================
    # Missing flag
    bureau['b_debt_missing'] = (
        bureau['AMT_CREDIT_SUM_DEBT']
        .isna()
        .astype(np.int8)
    )

    # Debt burden relative to total credit
    bureau['b_debt_to_credit'] =bureau['AMT_CREDIT_SUM_DEBT'] /bureau['AMT_CREDIT_SUM'].replace(0, np.nan)


    # Debt burden relative to loan age
    bureau['b_debt_to_days'] = bureau['AMT_CREDIT_SUM_DEBT'] /np.abs(bureau['DAYS_CREDIT']).replace(0, np.nan)

    # Debt burden relative to total tenure
    bureau['b_debt_to_tenure'] = (
        bureau['AMT_CREDIT_SUM_DEBT'] /
        np.abs(
            bureau['DAYS_CREDIT_ENDDATE'] -
            bureau['DAYS_CREDIT']
        ).replace(0, np.nan)
    )

    # Current overdue relative to debt
    bureau['b_curr_overdue_to_debt'] = (
        bureau['AMT_CREDIT_SUM_OVERDUE'] /
        bureau['AMT_CREDIT_SUM_DEBT'].replace(0, np.nan)
    )

    # Historical max overdue relative to debt
    bureau['b_max_overdue_to_debt'] = (
        bureau['AMT_CREDIT_MAX_OVERDUE'] /
        bureau['AMT_CREDIT_SUM_DEBT'].replace(0, np.nan)
    )

    # Debt exposure × current DPD
    bureau['b_debt_x_curr_dpd'] = (
        bureau['AMT_CREDIT_SUM_DEBT'] *
        bureau['CREDIT_DAY_OVERDUE']
    )

    # Debt exposure × prolongations
    bureau['b_debt_x_prolonged'] = (
        bureau['AMT_CREDIT_SUM_DEBT'] *
        bureau['CNT_CREDIT_PROLONG']
    )

    # Status Interaction Features

    bureau['b_active_debt'] = np.where(
        bureau['CREDIT_ACTIVE'] == 'Active',
        bureau['AMT_CREDIT_SUM_DEBT'],
        0
    )

    bureau['b_closed_debt'] = np.where(
        bureau['CREDIT_ACTIVE'] == 'Closed',
        bureau['AMT_CREDIT_SUM_DEBT'],
        0
    )

    bureau['b_sold_debt'] = np.where(
        bureau['CREDIT_ACTIVE'] == 'Sold',
        bureau['AMT_CREDIT_SUM_DEBT'],
        0
    )

    bureau['b_bad_debt_debt'] = np.where(
        bureau['CREDIT_ACTIVE'] == 'Bad debt',
        bureau['AMT_CREDIT_SUM_DEBT'],
        0
    )

    # -----------------------------------
    # AMT_CREDIT_SUM_OVERDUE Features
    # -----------------------------------

    # Missing flag
    bureau['b_overdue_missing'] = (
        bureau['AMT_CREDIT_SUM_OVERDUE']
        .isna()
        .astype(np.int8)
    )

    # Has current overdue
    bureau['b_has_overdue'] = (
        bureau['AMT_CREDIT_SUM_OVERDUE'] > 0
    ).astype(np.int8)

    # Relative Overdue Features
    # Current overdue relative to total credit
    bureau['b_overdue_to_credit'] = (
        bureau['AMT_CREDIT_SUM_OVERDUE'] /
        bureau['AMT_CREDIT_SUM'].replace(0, np.nan)
    )

    # Current overdue relative to outstanding debt
    bureau['b_overdue_to_debt'] = (
        bureau['AMT_CREDIT_SUM_OVERDUE'] /
        bureau['AMT_CREDIT_SUM_DEBT'].replace(0, np.nan)
    )

    # Current overdue relative to historical max overdue
    bureau['b_overdue_to_max_overdue'] = (
        bureau['AMT_CREDIT_SUM_OVERDUE'] /
        bureau['AMT_CREDIT_MAX_OVERDUE'].replace(0, np.nan)
    )

    # Time-Based Features
    # Overdue amount per day of loan age
    bureau['b_overdue_to_days'] = (
        bureau['AMT_CREDIT_SUM_OVERDUE'] /
        np.abs(bureau['DAYS_CREDIT']).replace(0, np.nan)
    )

    # Overdue amount per day of loan tenure
    bureau['b_overdue_to_tenure'] = (
        bureau['AMT_CREDIT_SUM_OVERDUE'] /
        np.abs(
            bureau['DAYS_CREDIT_ENDDATE']
            - bureau['DAYS_CREDIT']
        ).replace(0, np.nan)
    )

    # Interaction Features
    # Overdue amount × overdue days
    bureau['b_overdue_x_curr_dpd'] = (
        bureau['AMT_CREDIT_SUM_OVERDUE'] *
        bureau['CREDIT_DAY_OVERDUE']
    )

    # Overdue amount × prolongations
    bureau['b_overdue_x_prolonged'] = (
        bureau['AMT_CREDIT_SUM_OVERDUE'] *
        bureau['CNT_CREDIT_PROLONG']
    )

    # Status Interaction Features
    bureau['b_active_overdue'] = np.where(
        bureau['CREDIT_ACTIVE'] == 'Active',
        bureau['AMT_CREDIT_SUM_OVERDUE'],
        0
    )

    bureau['b_closed_overdue'] = np.where(
        bureau['CREDIT_ACTIVE'] == 'Closed',
        bureau['AMT_CREDIT_SUM_OVERDUE'],
        0
    )

    bureau['b_sold_overdue'] = np.where(
        bureau['CREDIT_ACTIVE'] == 'Sold',
        bureau['AMT_CREDIT_SUM_OVERDUE'],
        0
    )

    bureau['b_bad_debt_overdue'] = np.where(
        bureau['CREDIT_ACTIVE'] == 'Bad debt',
        bureau['AMT_CREDIT_SUM_OVERDUE'],
        0
    )

# -----------------------------------
# CREDIT_DAY_OVERDUE Features
# -----------------------------------

    # Missing flag
    bureau['b_curr_dpd_missing'] = (
        bureau['CREDIT_DAY_OVERDUE']
        .isna()
        .astype(np.int8)
    )

    # Any current delinquency
    bureau['b_has_curr_dpd'] = (
        bureau['CREDIT_DAY_OVERDUE'] > 0
    ).astype(np.int8)

    # Delinquency severity flags
    bureau['b_dpd30_flag'] = (
        bureau['CREDIT_DAY_OVERDUE'] >= 30
    ).astype(np.int8)

    bureau['b_dpd60_flag'] = (
        bureau['CREDIT_DAY_OVERDUE'] >= 60
    ).astype(np.int8)

    bureau['b_dpd90_flag'] = (
        bureau['CREDIT_DAY_OVERDUE'] >= 90
    ).astype(np.int8)

    # Log transformed DPD
    bureau['b_log_curr_dpd'] = np.log1p(
        bureau['CREDIT_DAY_OVERDUE']
    )

    # -----------------------------------
    # CNT_CREDIT_PROLONG Features
    # -----------------------------------

    # Missing flag
    bureau['b_prolong_missing'] = (
        bureau['CNT_CREDIT_PROLONG']
        .isna()
        .astype(np.int8)
    )

    # Has ever been prolonged
    bureau['b_has_prolong'] = (
        bureau['CNT_CREDIT_PROLONG'] > 0
    ).astype(np.int8)

    # Multiple prolongations
    bureau['b_multi_prolong'] = (
        bureau['CNT_CREDIT_PROLONG'] >= 2
    ).astype(np.int8)

    # Prolongations × current delinquency
    bureau['b_prolong_x_curr_dpd'] = (
        bureau['CNT_CREDIT_PROLONG'] *
        bureau['CREDIT_DAY_OVERDUE']
    )

    # Prolongations × current overdue amount
    bureau['b_prolong_x_overdue'] = (
        bureau['CNT_CREDIT_PROLONG'] *
        bureau['AMT_CREDIT_SUM_OVERDUE']
    )


    # -----------------------------------
    # Time-Based Bureau Features
    # -----------------------------------

    # Loan age (days since loan originated)
    bureau['b_credit_age'] = np.abs(
        bureau['DAYS_CREDIT']
    )

    # Remaining time until expected closure
    bureau['b_remaining_term'] = (
        bureau['DAYS_CREDIT_ENDDATE']
    )

    # Original loan tenure
    bureau['b_total_tenure'] = np.abs(
        bureau['DAYS_CREDIT_ENDDATE']
        - bureau['DAYS_CREDIT']
    )

    # Actual completed tenure
    bureau['b_actual_tenure'] = np.abs(
        bureau['DAYS_ENDDATE_FACT']
        - bureau['DAYS_CREDIT']
    )

    # Difference between expected and actual closure
    bureau['b_tenure_gap'] = (
        bureau['DAYS_ENDDATE_FACT']
        - bureau['DAYS_CREDIT_ENDDATE']
    )

    # Bureau recency
    bureau['b_update_recency'] = np.abs(
        bureau['DAYS_CREDIT_UPDATE']
    )

    # Time since last bureau update relative to age
    bureau['b_update_to_age'] = (
        np.abs(bureau['DAYS_CREDIT_UPDATE']) /
        np.abs(bureau['DAYS_CREDIT']).replace(0, np.nan)
)

    # -----------------------------------
    # Timing Flags
    # -----------------------------------

    # Expected end date already passed
    bureau['b_expired_loan'] = (
        bureau['DAYS_CREDIT_ENDDATE'] < 0
    ).astype(np.int8)

    # Loan closed
    bureau['b_has_actual_end'] = (
        bureau['DAYS_ENDDATE_FACT'].notna()
    ).astype(np.int8)

    # Closed later than expected
    bureau['b_closed_late'] = (
        bureau['DAYS_ENDDATE_FACT']
        >
        bureau['DAYS_CREDIT_ENDDATE']
    ).astype(np.int8)

    # Closed earlier than expected
    bureau['b_closed_early'] = (
        bureau['DAYS_ENDDATE_FACT']
        <
        bureau['DAYS_CREDIT_ENDDATE']
    ).astype(np.int8)

    # Recently opened loans
    bureau['b_recent_30d'] = (
        bureau['DAYS_CREDIT'] >= -30
    ).astype(np.int8)

    bureau['b_recent_90d'] = (
        bureau['DAYS_CREDIT'] >= -90
    ).astype(np.int8)

    bureau['b_recent_180d'] = (
        bureau['DAYS_CREDIT'] >= -180
    ).astype(np.int8)

    bureau['b_recent_365d'] = (
        bureau['DAYS_CREDIT'] >= -365
    ).astype(np.int8)

    # -----------------------------------
    # Time × Exposure Features
    # -----------------------------------

    # Credit intensity per day
    bureau['b_credit_per_day'] = (
        bureau['AMT_CREDIT_SUM'] /
        np.abs(bureau['DAYS_CREDIT']).replace(0, np.nan)
    )

    # Debt intensity per day
    bureau['b_debt_per_day'] = (
        bureau['AMT_CREDIT_SUM_DEBT'] /
        np.abs(bureau['DAYS_CREDIT']).replace(0, np.nan)
    )

    # Overdue intensity per day
    bureau['b_overdue_per_day'] = (
        bureau['AMT_CREDIT_SUM_OVERDUE'] /
        np.abs(bureau['DAYS_CREDIT']).replace(0, np.nan))

    
    return bureau
