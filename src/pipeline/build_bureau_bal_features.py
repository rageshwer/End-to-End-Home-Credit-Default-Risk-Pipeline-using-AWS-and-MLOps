''' This script utilizes the create bureau balance features script to create features from the dataset
given and saves the feature engineered dataset csv into interim data folder.'''
import pandas as pd
from src.features.bureau_bal_features import create_bureau_bal_features

#=====================================================
# Creating pipeline for loading bureau balance dataset, perform feature engineering 
# and save feature engineered dataset parquet.
#=====================================================

# Loading the dataset


# generating features
def main():
    bureau_bal=pd.read_csv('data/raw/bureau_balance.csv')
    bb_features=create_bureau_bal_features(bureau_bal=bureau_bal)

    # Saving the parquet
    bb_features.to_parquet('data/interim/bb_fe.parquet')

if __name__=='__main__':
    main()