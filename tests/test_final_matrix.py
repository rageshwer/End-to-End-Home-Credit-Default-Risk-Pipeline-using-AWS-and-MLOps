import pandas as pd
from src.features.bureau_and_balance_agg import create_bureau_balance_agg

def test_final_agg():
    a_trn=pd.read_parquet('mock_data/processed/app_train_fe.parquet')
    a_tst=pd.read_parquet('mock_data/processed/app_test_fe.parquet')
    bur=pd.read_parquet('mock_data/processed/bureau_fe.parquet')
    bb=pd.read_parquet('mock_data/processed/bureau_balance_fe.parquet')
    bb_full,bb_active,bb_closed,bb_debt_type=create_bureau_balance_agg(bur,bb)

    final_train_matrix=a_trn.merge(bb_full,on='SK_ID_CURR',how='left')
    final_train_matrix=final_train_matrix.merge(bb_active,on='SK_ID_CURR',how='left')
    final_train_matrix=final_train_matrix.merge(bb_closed,on='SK_ID_CURR',how='left')
    final_train_matrix=final_train_matrix.merge(bb_debt_type,on='SK_ID_CURR',how='left')

    final_test_matrix=a_tst.merge(bb_full,on='SK_ID_CURR',how='left')
    final_test_matrix=final_test_matrix.merge(bb_active,on='SK_ID_CURR',how='left')
    final_test_matrix=final_test_matrix.merge(bb_closed,on='SK_ID_CURR',how='left')
    final_test_matrix=final_test_matrix.merge(bb_debt_type,on='SK_ID_CURR',how='left')

    # dropping the columns with > 99.5% missing values
    drop_columns=[
    'ACTIVE_BUREAU_B_TENURE_GAP_STD'          ,  
    'ACTIVE_BUREAU_B_ACTUAL_TENURE_STD'        ,    
    'ACTIVE_BUREAU_DAYS_ENDDATE_FACT_STD'       ,
    'CLOSED_BUREAU_B_MAX_OVERDUE_TO_DEBT_STD'    , 
    'CLOSED_BUREAU_B_CURR_OVERDUE_TO_DEBT_STD'    , 
    'CLOSED_BUREAU_B_OVERDUE_TO_DEBT_STD'    ,
    'ACTIVE_BUREAU_B_TENURE_GAP_MAX'        ,
    'ACTIVE_BUREAU_B_TENURE_GAP_MIN'              ,
    'ACTIVE_BUREAU_B_TENURE_GAP_MEAN'             ,
    'ACTIVE_BUREAU_B_ACTUAL_TENURE_MEAN'          ,
    'ACTIVE_BUREAU_B_ACTUAL_TENURE_MIN'           ,
    'ACTIVE_BUREAU_DAYS_ENDDATE_FACT_MEAN'        ,
    'ACTIVE_BUREAU_DAYS_ENDDATE_FACT_MAX'         ,
    'ACTIVE_BUREAU_DAYS_ENDDATE_FACT_MIN'         ,
    'ACTIVE_BUREAU_B_ACTUAL_TENURE_MAX'
    ]
    final_train_matrix=final_train_matrix.drop(drop_columns,axis=1,errors='ignore')
    final_test_matrix=final_test_matrix.drop(drop_columns,axis=1,errors='ignore')

    # Saving the final datasets
    final_train_matrix.to_parquet('mock_data/processed/final_train.parquet')
    final_test_matrix.to_parquet('mock_data/processed/final_test.parquet')