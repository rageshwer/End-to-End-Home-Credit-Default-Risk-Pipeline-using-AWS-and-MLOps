import warnings
warnings.filterwarnings('ignore')
import gc
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import mlflow
from mlflow.lightgbm import log_model

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

    #=============================================================
    # Logging parameters if active experimentation run
    if mlflow.active_run():
        mlflow.log_params(params)
        mlflow.log_param('Folds_Straified',N_SPLITS)

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
        valid_preds = model.predict_proba(X_valid,num_iteration=model.best_iteration_)[:, 1]
        oof_preds[valid_idx] = valid_preds

        fold_auc = roc_auc_score(y_valid,valid_preds)
        cv_scores.append(fold_auc)
        print(f"Fold {fold} AUC = {fold_auc:.6f}")

        # Save this fold's AUC in MLFLOW :
        if mlflow.active_run():
            mlflow.log_metric(f'Fold {fold} AUC',fold_auc)

        # Feature importances in this fold
        fold_importance = pd.DataFrame({
        'feature': top_features_750,
        'importance': model.feature_importances_,
        'fold': fold})

        # Feature importances appended of all folds
        feature_importance = pd.concat(
            [feature_importance, fold_importance],
            axis=0,
            ignore_index=True)
        del(X_train,X_valid,y_train,y_valid,valid_preds)
        gc.collect()

    # Printing overall CV score after the last fold
    overall_auc = roc_auc_score(y,oof_preds)
    print(f"Mean CV AUC : {np.mean(cv_scores):.6f}")
    print(f"OOF AUC      : {overall_auc:.6f}")

    # The mean of feature importances given in each fold
    feature_importance_mean = (
    feature_importance
    .groupby('feature')['importance']
    .mean()
    .reset_index()
    .sort_values(
        by='importance',
        ascending=False
    ))

    # Logging the mean AUC of all folds, the global AUC 
    if mlflow.active_run():
        mlflow.log_metric('Mean AUC of Folds',np.mean(cv_scores))
        mlflow.log_metric('Overall AUC',overall_auc)
    

    return model,feature_importance_mean,params


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
    

    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    exp=mlflow.set_experiment("Home-Credit")
    exp_id=exp.experiment_id

    with mlflow.start_run(experiment_id=exp_id):
        # Training
        _,feature_importance_mean,params = train_cv(train_df,cat_cols,target='TARGET')
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

        # Logging the final model and mean feature importances of all folds
        requirements = ['docker==7.1.0', 'fastapi==0.137.1', 'joblib==1.5.3', 'lightgbm==4.6.0', 'matplotlib==3.11.0','mlflow==3.14.0',
                        'numpy==2.4.6', 'pandas==2.3.3', 'pydantic==2.13.4', 'pytest==9.1.0', 'requests==2.34.2','scikit-learn==1.9.0',
                        'scipy==1.17.1', 'seaborn==0.13.2', 'skops==0.14.0', 'starlette==1.3.1', 'streamlit==1.58.0', 'uvicorn==0.49.0']        
        
        mlflow.lightgbm.log_model(final_model,name='lgbm_750feat',pip_requirements=requirements)

        fi_filename = "feature_importance.csv"
        feature_importance_mean.to_csv(fi_filename, index=False)
        mlflow.log_artifact(fi_filename)
        if os.path.exists(fi_filename):
            os.remove(fi_filename)
        print('The final model trained on full train dataset and Feature importance csv logged.')


if __name__ == "__main__":
    main()
