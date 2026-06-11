'''This script is responsible for building the final train and test datasets. It takes the preprocessed data, 
applies any necessary aggregations, and saves the final datasets for model training and evaluation.'''

import pandas as pd

def main():
    app_train_fe=pd.read_parquet('data/interim/application_train_fe.parquet')
    app_test_fe=pd.read_parquet('data/interim/application_test_fe.parquet')
    bureau_agg=pd.read_parquet('data/interim/Bureau_Bal_Agg_Full.parquet')
    bureau_active=pd.read_parquet('data/interim/Bureau_Bal_Agg_Active.parquet')
    bureau_closed=pd.read_parquet('data/interim/Bureau_Bal_Agg_Closed.parquet')
    bureau_type_debt=pd.read_parquet('data/interim/Bureau_Bal_Agg_Debt_Type.parquet')
    bureau_type_debt.reset_index(inplace=True)

    # Merging the datasets
    final_train_matrix=app_train_fe.merge(bureau_agg,on='SK_ID_CURR',how='left')
    final_train_matrix=final_train_matrix.merge(bureau_active,on='SK_ID_CURR',how='left')
    final_train_matrix=final_train_matrix.merge(bureau_closed,on='SK_ID_CURR',how='left')
    final_train_matrix=final_train_matrix.merge(bureau_type_debt,on='SK_ID_CURR',how='left')

    final_test_matrix=app_test_fe.merge(bureau_agg,on='SK_ID_CURR',how='left')
    final_test_matrix=final_test_matrix.merge(bureau_active,on='SK_ID_CURR',how='left')
    final_test_matrix=final_test_matrix.merge(bureau_closed,on='SK_ID_CURR',how='left')
    final_test_matrix=final_test_matrix.merge(bureau_type_debt,on='SK_ID_CURR',how='left')

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
    final_train_matrix.to_parquet('data/processed/final_train.parquet')
    final_test_matrix.to_parquet('data/processed/final_test.parquet')


if __name__ == "__main__":
    main()