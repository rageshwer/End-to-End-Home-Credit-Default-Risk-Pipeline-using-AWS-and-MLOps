'''This script loads raw bureau data, feature engineers it and stores parquet in interim data folder.'''
import pandas as pd
from src.features.bureau_features import create_bureau_features

def main():
    bureau=pd.read_csv('data/raw/bureau.csv')
    bureau_fe=create_bureau_features(bureau)
    bureau_fe.to_parquet('data/interim/bureau_fe.parquet')

if __name__=='__main__':
    main()
