import pandas as pd
import numpy as np
import shap
import mlflow

# Loading the model based on run id
mlflow.set_tracking_uri('sqlite:///mlflow.db')
run_id="01fa115e85d34f5da1de226caca345eb"
model=mlflow.lightgbm.load_model(f'runs:/{run_id}/lgbm_750feat')
features=model.feature_name_

# Loading test dataset for generating SHAP values
X_test=pd.read_parquet('data/processed/final_test.parquet')
X_test['SK_ID_CURR']=X_test['SK_ID_CURR'].astype(int)
X_test_feats=X_test[features].copy()
str_cols=X_test_feats.select_dtypes(include=['object','string']).columns
for col in str_cols:
    X_test_feats[col]=X_test_feats[col].astype('category')

# The tree explainer
tree_explainer=shap.TreeExplainer(model)
shap_obj=tree_explainer(X_test_feats)


shap_val_matrix=shap_obj.values
base_values=shap_obj.base_values

columns=[f'shap_{col}' for col in features]
shap_val_df=pd.DataFrame(
    columns=columns,
    data=shap_val_matrix,
    index=X_test.index
)

# The dataframe with base and shap values to be exported
temp_df=pd.DataFrame({'SK_ID_CURR':X_test['SK_ID_CURR'].values,
                        'base_value':base_values   
                        },index=X_test.index)
lookup_df=pd.concat([temp_df,shap_val_df],axis=1)

lookup_df.to_parquet('api/data/lookup_df.parquet')