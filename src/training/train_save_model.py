import warnings
warnings.filterwarnings('ignore')
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

#============================================================
# Methods
#============================================================
def load_data(path)->pd.DataFrame:
    return pd.read_parquet(path)

def train_cv(df:pd.DataFrame,cat_cols:list,target:str,ci_mode:bool=False):

    N_SPLITS = 10
    RANDOM_STATE = 42

    from config.config import top_features_750

    X=df[top_features_750]
    y = df[target]

    # ============================================================
    # LIGHTGBM PARAMETERS
    # ============================================================
    from config.config import params
    if ci_mode:
        N_SPLITS = 2
        params= {
            "n_estimators": 5,
            "max_depth": 2,
            "verbose": -1
        }

        callbacks = []
    else:
        callbacks = [
        lgb.early_stopping(
            stopping_rounds=300,
            verbose=True
        ),
        lgb.log_evaluation(period=200)
        ]

    # ============================================================
    skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
    )

    oof_preds = np.zeros(len(df))
    cv_scores = []
    feature_importance = pd.DataFrame()
    #============================================================
    # Training loop
    #============================================================
    for fold,(train_idx,valid_idx) in enumerate(skf.split(X,y),1):
        print("="*60)
        print(f"FOLD {fold}")
        print("="*60)

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model = lgb.LGBMClassifier(
            **params
        )

        model.fit(
            X_train,
            y_train,

            eval_set=[
                (X_valid, y_valid)
            ],

            eval_metric='auc',

            categorical_feature=cat_cols,

            callbacks=callbacks
        )
        valid_preds = model.predict_proba(
        X_valid,
        num_iteration=model.best_iteration_
        )[:, 1]
        oof_preds[valid_idx] = valid_preds

        fold_auc = roc_auc_score(
        y_valid,
        valid_preds)
        cv_scores.append(fold_auc)
        print(f"Fold {fold} AUC = {fold_auc:.6f}")

        fold_importance = pd.DataFrame({
        'feature': top_features_750,
        'importance': model.feature_importances_,
        'fold': fold})
        feature_importance = pd.concat(
            [feature_importance, fold_importance],
            axis=0,
            ignore_index=True)
        del(X_train,X_valid,y_train,y_valid,valid_preds)
        gc.collect()

    # Printing overall CV score after the last fold
    overall_auc = roc_auc_score(y,oof_preds)
    print(f"Mean CV AUC : {np.mean(cv_scores):.6f}")
    print(f"Std CV AUC  : {np.std(cv_scores):.6f}")
    print(f"OOF AUC      : {overall_auc:.6f}")

    feature_importance_mean = (
    feature_importance
    .groupby('feature')['importance']
    .mean()
    .reset_index()
    .sort_values(
        by='importance',
        ascending=False
    ))

    return model,feature_importance_mean,cv_scores, overall_auc,params,top_features_750


def prepare_train_data(train_df:pd.DataFrame,target:str):
    from config.config import top_features_750

    str_cols = train_df[top_features_750].select_dtypes(
    include=['object', 'string']
    ).columns
    for col in str_cols:
        train_df[col] = train_df[col].astype('category')

    X=train_df[top_features_750]
    y = train_df[target]

    cat_cols = X.select_dtypes(
        include=['category']
    ).columns.tolist()

    return X,y,cat_cols

#============================================================
# Executing training and saving artifacts
#============================================================

def main():
    # Loading data
    train_df=load_data('data/interim/final_train.parquet')
    X,y,cat_cols = prepare_train_data(train_df,'TARGET')
    
    # Training
    model,feature_importance_mean,cv_scores,overall_auc,params,feature_names = train_cv(train_df,cat_cols,target='TARGET')
    # Keeping model so that in testing, we dont have to train separately. 
    # We can directly load the model and save in artifacts.
    # Saving artifacts

    final_model=lgb.LGBMClassifier(
        **params
    ).fit(
        X,
        y,
        categorical_feature=cat_cols
    )

    from src.training.utils import save_artifacts 
    save_artifacts(
    model=final_model,
    feature_names=feature_names,
    feature_importance=feature_importance_mean,
    params=params,
    oof_auc=overall_auc,
    cv_scores=cv_scores,
    )


if __name__ == "__main__":
    main()
