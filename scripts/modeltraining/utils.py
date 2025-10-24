def evaluate_classification(model, X_test, y_test, threshold: float = 0.5):
    import numpy as np, pandas as pd
    from IPython.display import display
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score, recall_score,
        roc_auc_score, average_precision_score, log_loss, jaccard_score,
        hamming_loss, f1_score, fbeta_score, matthews_corrcoef, cohen_kappa_score,
        zero_one_loss, confusion_matrix, brier_score_loss
    )

    y_true = np.asarray(y_test)

    # scores/proba
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X_test)
        y_score = (raw - raw.min()) / (raw.ptp() + 1e-12)
    else:
        y_score = model.predict(X_test).astype(float)

    y_pred = (y_score >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    # metrics dict
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision (pos=1)": precision_score(y_true, y_pred, zero_division=0),
        "Recall / Sensitivity (TPR)": recall_score(y_true, y_pred, zero_division=0),
        "Specificity (TNR)": specificity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F0.5": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
        "PR-AUC (Average Precision)": average_precision_score(y_true, y_score),
        "Log loss": log_loss(y_true, np.c_[1 - y_score, y_score], labels=[0, 1]),
        "Jaccard": jaccard_score(y_true, y_pred, zero_division=0),
        "Hamming loss": hamming_loss(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen's kappa": cohen_kappa_score(y_true, y_pred),
        "Zero-One loss": zero_one_loss(y_true, y_pred),
        "Brier score": brier_score_loss(y_true, y_score),
        "Threshold": float(threshold),
        "Positives (TP+FN)": int(tp + fn),
        "Negatives (TN+FP)": int(tn + fp),
    }

    # >>> build a 2-column DataFrame and DISPLAY it (no manual styling)
    df_metrics = pd.DataFrame(
        {"Metric": list(metrics.keys()), "Value": np.round(list(metrics.values()), 4)}
    )
    display(df_metrics)

    # confusion matrix as a small DF
    cm_df = pd.DataFrame([[tn, fp], [fn, tp]],
                         index=["Actual 0", "Actual 1"],
                         columns=["Pred 0", "Pred 1"])
    print("\nConfusion matrix:")
    display(cm_df)

    return df_metrics, {"cm": cm_df, "y_pred": y_pred, "y_score": y_score}
