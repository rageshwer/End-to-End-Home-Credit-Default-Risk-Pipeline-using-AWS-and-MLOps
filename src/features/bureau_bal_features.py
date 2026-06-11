''' This python script creates the bureau balance dataset features'''

import numpy as np
import pandas as pd

def create_bureau_bal_features(bureau_bal: pd.DataFrame)->pd.DataFrame:
    # Firstly ordinal encoding of the status column :
    categories={
    'C':0,
    'X':0,
    '0':0,
    '1':1,
    '2':2,
    '3':3,
    '4':4,
    '5':5
    }
    bureau_bal['STATUS']=bureau_bal['STATUS'].map(categories)

    # New df with unique loan id that we will return with all the features :

    bb_fe=pd.DataFrame({
        'SK_ID_BUREAU': bureau_bal['SK_ID_BUREAU'].unique()
    })
    bb_fe

    #================================================================================
    # Creating new features in the bb_fe data frame :
    #================================================================================

    # Related to days and age of the loan (Months Balance) :

    bb_fe['bb_oldest_mon']=bb_fe['SK_ID_BUREAU'].map(bureau_bal.groupby(
    'SK_ID_BUREAU'
    )['MONTHS_BALANCE'].min())

    bb_fe['bb_latest_mon']=bb_fe['SK_ID_BUREAU'].map(bureau_bal.groupby(
        'SK_ID_BUREAU'
    )['MONTHS_BALANCE'].max())

    bb_fe['bb_history_age']=bb_fe['SK_ID_BUREAU'].map(abs(bureau_bal.groupby(
        'SK_ID_BUREAU'
    )['MONTHS_BALANCE'].min()))

    bb_fe['bb_num_mon']=-bb_fe['bb_oldest_mon']+bb_fe['bb_latest_mon']+1

    # Features related to the status of the loan. This includes the SMA and NPA ratios :

    bb_fe['bb_max_default']=bb_fe['SK_ID_BUREAU'].map(bureau_bal.groupby(
    'SK_ID_BUREAU'
    )['STATUS'].max())

    bb_fe['bb_curr_stat'] = bb_fe['SK_ID_BUREAU'].map(
        bureau_bal.sort_values(
            ['SK_ID_BUREAU','MONTHS_BALANCE'],
            ascending=[True,False]
        )
        .groupby('SK_ID_BUREAU')['STATUS']
        .first()
    )

    bb_fe['bb_mean_dpd']=bb_fe['SK_ID_BUREAU'].map(bureau_bal.groupby(
    'SK_ID_BUREAU'
    )['STATUS'].mean())

    bb_fe['bb_sma0']=bb_fe['SK_ID_BUREAU'].map(bureau_bal[bureau_bal['STATUS']==1].groupby(
    'SK_ID_BUREAU'
    ).size())

    bb_fe['bb_sma1']=bb_fe['SK_ID_BUREAU'].map(bureau_bal[bureau_bal['STATUS']==2].groupby(
        'SK_ID_BUREAU'
    )['STATUS'].size())

    bb_fe['bb_sma2']=bb_fe['SK_ID_BUREAU'].map(bureau_bal[bureau_bal['STATUS']==3].groupby(
        'SK_ID_BUREAU'
    )['STATUS'].size())

    bb_fe['bb_npa']=bb_fe['SK_ID_BUREAU'].map(bureau_bal[bureau_bal['STATUS']>=4].groupby(
        'SK_ID_BUREAU'
    )['STATUS'].size())

    cols = [
        'bb_sma0',
        'bb_sma1',
        'bb_sma2',
        'bb_npa'
    ]

    bb_fe[cols] = bb_fe[cols].fillna(0)

    bb_fe['bb_sma0_ratio']=bb_fe['bb_sma0']/bb_fe['bb_num_mon']
    bb_fe['bb_sma1_ratio']=bb_fe['bb_sma1']/bb_fe['bb_num_mon']
    bb_fe['bb_sma2_ratio']=bb_fe['bb_sma2']/bb_fe['bb_num_mon']
    bb_fe['bb_npa_ratio']=bb_fe['bb_npa']/bb_fe['bb_num_mon']

    return bb_fe