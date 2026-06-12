import pandas as pd
import os
from src.training.train_save_model import prepare_train_data,train_cv
from src.training.utils import save_artifacts

def test_train_save():
    train=pd.read_parquet('mock_data/processed/final_train.parquet')
    _,_,cat_cols=prepare_train_data(train,'TARGET')
    model,feature_importance_mean,cv_scores,overall_auc,params,feature_names = train_cv(train,cat_cols,target='TARGET')
    save_artifacts(
    model=model,
    feature_names=feature_names,
    feature_importance=feature_importance_mean,
    params=params,
    oof_auc=overall_auc,
    cv_scores=cv_scores,
    )

    assert model is not None

    assert len(feature_names) > 0

    assert len(cv_scores) > 0

    assert os.path.exists("artifacts")