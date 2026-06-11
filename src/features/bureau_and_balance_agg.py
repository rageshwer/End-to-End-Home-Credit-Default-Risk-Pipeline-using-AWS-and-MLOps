'''This script aggregates the bureau and bureau_balance datasets on different parameters. The resultant are 4 datasets of bureau and balance aggregation on different parameters stored as parquet in interim folder : 
Bureau_Bal_Agg_Full, Bureau_Bal_Agg_Active, Bureau_Bal_Agg_Closed and Bureau_Bal_Agg_Debt Type.'''
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_bureau_balance_agg(bureau_fe,bb_fe):
    # Merging on SK_ID_BUREAU to get the bureau_balance features in the bureau dataset
    bureau_master=bureau_fe.merge(bb_fe,on='SK_ID_BUREAU',how='left')

    # The ratios might have large values
    bureau_master.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True)
    
    # Dropping SK_ID_BUREAU as it is not needed for the model and it is not a feature.
    bureau_master=bureau_master.drop('SK_ID_BUREAU',axis=1)

    # Creating binary missing columns for the features which have missing values. 
    # This will help the model to understand that the value is missing and not just a zero or some other value.
    bureau_master['bb_missing'] = (
    bureau_master['bb_oldest_mon']
    .isna()
    .astype('int8'))

    bureau_master['is_sold'] = (
    bureau_master['CREDIT_ACTIVE'] == 'Sold'
    ).astype('int8')

    bureau_master['is_bad_debt'] = (
        bureau_master['CREDIT_ACTIVE'] == 'Bad debt'
    ).astype('int8')

    bureau_master['loan_count']=1       # adding this will give number of loans for each SK_ID_CURR when we do the aggregation. This is a very important feature as it will help the model to understand the number of loans a person has taken. It will also help to understand the credit history of the person.
    #====================================================================================
    # Creating columns for aggregation :
    #====================================================================================

    money_cols = [
    'AMT_CREDIT_SUM',
    'AMT_CREDIT_SUM_DEBT',                          # sum, mean, max, std
    'AMT_CREDIT_SUM_LIMIT',
    'AMT_CREDIT_SUM_OVERDUE',
    'AMT_CREDIT_MAX_OVERDUE',
    'b_annuity_imp',
    'b_exp_annuity']
    max_overdue=['AMT_CREDIT_MAX_OVERDUE']          # mean, max (no point of sum)
    time_cols = [
    'DAYS_CREDIT',
    'DAYS_CREDIT_ENDDATE',
    'DAYS_ENDDATE_FACT',
    'DAYS_CREDIT_UPDATE',           # min, max, mean, std
    'b_credit_age',
    'b_remaining_term',
    'b_total_tenure',
    'b_actual_tenure',
    'b_tenure_gap',
    'b_update_recency',
    'b_log_curr_dpd',
    'CREDIT_DAY_OVERDUE',
    'CNT_CREDIT_PROLONG']
    missing_cols = [
    'bb_missing',
    'b_annuity_missing',
    'b_max_overdue_missing',
    'b_debt_missing',               # sum, mean
    'b_overdue_missing',
    'b_curr_dpd_missing',
    'b_prolong_missing']
    flag_cols = [
    'is_sold',
    'is_bad_debt',
    'b_has_overdue',
    'b_has_curr_dpd',

    'b_dpd30_flag',
    'b_dpd60_flag',
    'b_dpd90_flag',

    'b_has_prolong',
    'b_multi_prolong',
                                        # sum, mean
    'b_expired_loan',
    'b_has_actual_end',

    'b_closed_late',
    'b_closed_early',

    'b_recent_30d',
    'b_recent_90d',
    'b_recent_180d',
    'b_recent_365d']
    interaction_cols = [
    'b_credit_x_curr_dpd',
    'b_credit_x_prolonged',

    'b_debt_x_curr_dpd',
    'b_debt_x_prolonged',
                                        # sum, mean, max
    'b_overdue_x_curr_dpd',
    'b_overdue_x_prolonged',

    'b_prolong_x_curr_dpd',
    'b_prolong_x_overdue',

    'b_max_overdue_x_curr_dpd']
    debt_state_cols = [
    'b_active_debt',
    'b_closed_debt',
    'b_sold_debt',
    'b_bad_debt_debt',
                                        # sum, mean, max
    'b_active_overdue',
    'b_closed_overdue',
    'b_sold_overdue',
    'b_bad_debt_overdue']
    ratio_cols = [
    'b_annuity_loan_ratio',

    'b_max_overdue_to_credit',
    'b_max_overdue_to_debt',
    'b_current_to_max_overdue',

    'b_paid_to_days',
    'b_left_to_full',
    'b_loan_to_tenure',

    'b_curr_overdue_to_loan',
                                        # mean, max, std
    'b_debt_to_credit',
    'b_debt_to_days',
    'b_debt_to_tenure',

    'b_curr_overdue_to_debt',

    'b_overdue_to_credit',
    'b_overdue_to_debt',
    'b_overdue_to_max_overdue',
    'b_overdue_to_days',
    'b_overdue_to_tenure',

    'b_update_to_age',

    'b_credit_per_day',
    'b_debt_per_day',
    'b_overdue_per_day']
    bur_bal_history=['bb_oldest_mon',
    'bb_latest_mon',
    'bb_history_age',                   # mean, max, std
    'bb_num_mon']
    delinquency=[
    'bb_max_default',
    'bb_curr_stat',
    'bb_mean_dpd']               # mean, max
    npa_counts=[            # sum, mean, max
    'bb_sma0',
    'bb_sma1',
    'bb_sma2',
    'bb_npa']
    npa_ratios=[            # mean, max, std
    'bb_sma0_ratio',
    'bb_sma1_ratio',
    'bb_sma2_ratio',
    'bb_npa_ratio']

    #====================================================================================
    # Creating the aggregation dictionary :
    #====================================================================================
    agg_dict={}

    # loan count
    agg_dict['loan_count'] = ['sum']

    # monetary
    for col in money_cols:
        agg_dict[col] = ['sum', 'mean', 'max', 'std']

    # overdue
    for col in max_overdue:
        agg_dict[col] = ['mean', 'max']

    # time
    for col in time_cols:
        agg_dict[col] = ['min', 'max', 'mean', 'std']

    # missing flags
    for col in missing_cols:
        agg_dict[col] = ['sum', 'mean']

    # risk flags
    for col in flag_cols:
        agg_dict[col] = ['sum', 'mean']

    # interaction
    for col in interaction_cols:
        agg_dict[col] = ['sum', 'mean', 'max']

    # debt state
    for col in debt_state_cols:
        agg_dict[col] = ['sum', 'mean', 'max']

    # ratios
    for col in ratio_cols:
        agg_dict[col] = ['mean', 'max', 'std']

    # bureau balance severity
    for col in delinquency:
        agg_dict[col] = ['max', 'mean']

    # bureau balance history
    for col in bur_bal_history:
        agg_dict[col] = ['max', 'mean', 'std']

    # bureau balance counts
    for col in npa_counts:
        agg_dict[col] = ['sum', 'mean', 'max']

    # bureau balance ratios
    for col in npa_ratios:
        agg_dict[col] = ['mean', 'max', 'std']

    #====================================================================================
    # Creating the bureau_bal_agg_full:
    #====================================================================================
    bureau_bal_agg_full = (
    bureau_master
    .groupby('SK_ID_CURR')
    .agg(agg_dict)
    )
    # flatten the aggregated columns
    bureau_bal_agg_full.columns = [
        f'BUREAU_{col}_{agg}'.upper()
        for col, agg in bureau_bal_agg_full.columns
    ]

    bureau_bal_agg_full = bureau_bal_agg_full.reset_index()

    #====================================================================================
    # Creating bureau_bal_agg_active and bureau_bal_agg_closed:
    #====================================================================================
    bureau_active=bureau_master[bureau_master['CREDIT_ACTIVE']=='Active'].copy()
    bureau_closed=bureau_master[bureau_master['CREDIT_ACTIVE']=='Closed'].copy()

    bureau_bal_agg_active=bureau_active.groupby(
        ['SK_ID_CURR']
    ).agg(agg_dict)

    bureau_bal_agg_closed=bureau_closed.groupby(
        ['SK_ID_CURR']
    ).agg(agg_dict)

    bureau_bal_agg_active.columns = [
    f'ACTIVE_BUREAU_{col}_{agg}'.upper()
    for col, agg in bureau_bal_agg_active.columns
    ]

    bureau_bal_agg_active = bureau_bal_agg_active.reset_index()

    bureau_bal_agg_closed.columns = [
        f'CLOSED_BUREAU_{col}_{agg}'.upper()
        for col, agg in bureau_bal_agg_closed.columns
    ]

    bureau_bal_agg_closed = bureau_bal_agg_closed.reset_index()

    #====================================================================================
    # Creating bureau_bal_agg_debt_type:
    #====================================================================================   
    bureau_type_cols=bureau_master['CREDIT_TYPE'].value_counts(ascending=False)
    bureau_type_cols[:5].index
    bureau_type_debt=pd.DataFrame()
    bureau_type_debt.index=bureau_master['SK_ID_CURR'].unique()
    for col_ in bureau_type_cols[:5].index:
        col = (col_.lower().replace(' ', '_').replace('-', '_'))
        bureau_type_debt[f'total_{col}_debt']=bureau_master[bureau_master['CREDIT_TYPE']==col_].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum()
        bureau_type_debt[f'total_{col}_count']=bureau_master[bureau_master['CREDIT_TYPE']==col_].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].count()
        bureau_type_debt[f'total_{col}_overdue']=bureau_master[bureau_master['CREDIT_TYPE']==col_].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].sum()
        bureau_type_debt[f'total_{col}_max_overdue']=bureau_master[bureau_master['CREDIT_TYPE']==col_].groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].sum()

        bureau_type_debt[f'{col}_overdue_to_typedebt']=bureau_type_debt[f'total_{col}_overdue']/bureau_type_debt[f'total_{col}_debt']
        bureau_type_debt[f'{col}_max_overdue_to_typedebt']=bureau_type_debt[f'total_{col}_max_overdue']/bureau_type_debt[f'total_{col}_debt']
        bureau_type_debt[f'{col}_overdue_to_total']=bureau_type_debt[f'total_{col}_overdue']/(bureau_master[bureau_master['CREDIT_TYPE']==col_].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum())
        bureau_type_debt[f'{col}_max_overdue_to_total']=bureau_type_debt[f'total_{col}_max_overdue']/(bureau_master[bureau_master['CREDIT_TYPE']==col_].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum())
        bureau_type_debt[f'{col}_debt_share']=bureau_type_debt[f'total_{col}_debt']/(bureau_master[bureau_master['CREDIT_TYPE']==col_].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum())

    bureau_type_debt.index.name='SK_ID_CURR'


    return bureau_bal_agg_full,bureau_bal_agg_active,bureau_bal_agg_closed,bureau_type_debt