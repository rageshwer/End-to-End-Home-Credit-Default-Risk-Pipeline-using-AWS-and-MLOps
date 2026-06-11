'''This script contains the function to build the aggregation of the bureau and bureau_balance datasets. The resultant are
4 datasets of bureau and balance aggregation on different parameters stored as parquet in interim folder : 
Bureau_Bal_Agg_Full, Bureau_Bal_Agg_Active, Bureau_Bal_Agg_Closed and Bureau_Bal_Agg_Debt Type.'''
# These datasets will be merged with application train and test datasets in the next step of the pipeline : Final train test datasets building.
import pandas as pd
from src.features.bureau_and_balance_agg import create_bureau_balance_agg

def main():
    bureau_fe=pd.read_parquet('data/interim/bureau_fe.parquet')
    bb_fe=pd.read_parquet('data/interim/bb_fe.parquet')
    # Merging the bureau and balance datasets on SK_ID_BUREAU
    bureau_bal_agg_full,bureau_bal_agg_active,bureau_bal_agg_closed,bureau_bal_agg_debt_type=create_bureau_balance_agg(bureau_fe,bb_fe)
    # Saving the resultant datasets as parquet in interim folder
    bureau_bal_agg_full.to_parquet('data/interim/Bureau_Bal_Agg_Full.parquet')
    bureau_bal_agg_active.to_parquet('data/interim/Bureau_Bal_Agg_Active.parquet')
    bureau_bal_agg_closed.to_parquet('data/interim/Bureau_Bal_Agg_Closed.parquet')
    bureau_bal_agg_debt_type.to_parquet('data/interim/Bureau_Bal_Agg_Debt_Type.parquet')

if __name__ == "__main__":
    main()

    
