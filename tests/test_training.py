import pandas as pd
import os,shutil
from src.training.train_save_model import prepare_train_data,train_cv

def test_train_save():
    train=pd.read_parquet('tests/mock_data/processed/final_train.parquet')
    _,_,cat_cols=prepare_train_data(train,'TARGET')
    model,feature_importance_mean,_,trees = train_cv(train,cat_cols,target='TARGET',ci_mode=True)

    assert model is not None, "Model failed to instantiate and train."
    assert isinstance(feature_importance_mean, pd.DataFrame), "Feature importance object must be a pandas DataFrame."
    assert not feature_importance_mean.empty, "Feature importance data cannot be empty."
    assert "importance" in feature_importance_mean.columns, "Missing 'importance' metric column."
    
