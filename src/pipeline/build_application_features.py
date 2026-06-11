'''This script loads the application datasets, applies feature engineering using create application features function 
and saves the processed datasets in interim data folder.'''
import pandas as pd
from src.features.application_features import create_application_features

def main():
    train=pd.read_csv('data/raw/application_train.csv')
    test=pd.read_csv('data/raw/application_test.csv')
    train_processed=create_application_features(train.drop(columns=['TARGET']))
    test_processed=create_application_features(test)
    train_processed['TARGET']=train['TARGET']
    train_processed.to_parquet('data/interim/application_train_fe.parquet',index=False)
    test_processed.to_parquet('data/interim/application_test_fe.parquet',index=False)
    
if __name__ == "__main__":
    main()