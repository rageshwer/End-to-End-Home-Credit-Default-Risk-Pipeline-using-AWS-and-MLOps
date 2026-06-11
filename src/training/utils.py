import json
import joblib
import pandas as pd


def save_artifacts(
    model,
    feature_names,
    feature_importance,
    params,
    oof_auc,
    cv_scores,
    artifact_dir="artifacts"
):
    """
    Save model artifacts for reproducibility and deployment.
    """

    # Model
    joblib.dump(
        model,
        f"{artifact_dir}/lgbm_model.pkl"
    )

    # Feature names
    joblib.dump(
        feature_names,
        f"{artifact_dir}/feature_names.pkl"
    )

    # Feature importance
    feature_importance.to_csv(
        f"{artifact_dir}/feature_importance.csv",
        index=False
    )

    # Parameters
    with open(
        f"{artifact_dir}/params.json",
        "w"
    ) as f:
        json.dump(
            params,
            f,
            indent=4
        )

    # Metrics
    metrics = {
        "oof_auc": float(oof_auc),
        "cv_mean_auc": float(sum(cv_scores) / len(cv_scores)),
        "cv_scores": [float(x) for x in cv_scores],
        "n_features": len(feature_names)
    }

    with open(
        f"{artifact_dir}/metrics.json",
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    print("Artifacts saved successfully.")