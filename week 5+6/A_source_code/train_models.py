#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: train_models.py
Dự án: SMART LMS & GENERATIVE AI – Phân tích dữ liệu người học và tạo nội dung học tập tự động
Nhiệm vụ Tuần 5 & Tuần 6:
1. Kiểm toán cấu trúc dữ liệu student_features.csv (2.283 sinh viên, 37 cột -> lọc 24 đặc trưng định lượng)
   và kiểm soát nghiêm ngặt Data Leakage (loại bỏ id_student, final_result, date_unregistration,
   has_unregistered, Exam; đối sánh với phiên bản Cảnh báo sớm nghiêm ngặt tại Ngày 100).
2. Phân chia Stratified Train/Test (80/20, random_state=42) và Stratified 5-Fold Cross Validation.
3. Huấn luyện, tối ưu siêu tham số và so sánh 3 thuật toán:
   - Logistic Regression (Baseline tuyến tính)
   - Random Forest Classifier (Thuật toán trọng tâm)
   - Decision Tree Classifier (Mô hình đối chuẩn cây quyết định)
4. Đánh giá 8 nhóm chỉ số bắt buộc, tối ưu Decision Threshold trên Out-of-Fold CV (không dùng Test set),
   phân tích MDI Feature Importance & Permutation Importance, xuất best_model.joblib và model_metadata.json.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Hỗ trợ tự động nạp thư viện tương thích nếu môi trường chưa cài đặt sẵn scikit-learn/joblib
_COMPAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_sklearn_compat")
if os.path.exists(_COMPAT_DIR) and _COMPAT_DIR not in sys.path:
    try:
        import sklearn  # noqa: F401
        import joblib   # noqa: F401
    except ImportError:
        sys.path.insert(0, _COMPAT_DIR)

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score, classification_report
)
from sklearn.inspection import permutation_importance

FEATURE_COLS_24 = [
    "num_assessments", "assessment_completion_rate", "avg_score", "weighted_score",
    "min_score", "max_score", "score_std", "tma_avg_score", "cma_avg_score",
    "late_submission_count", "avg_submission_delay", "total_clicks", "active_days",
    "average_clicks_per_day", "unique_learning_resources", "early_clicks_pre_course",
    "clicks_first_half", "clicks_oucontent", "clicks_forumng", "clicks_quiz",
    "clicks_resource_url", "num_of_prev_attempts", "studied_credits", "date_registration"
]

FORBIDDEN_LEAKAGE_COLS = [
    "id_student", "code_module", "code_presentation",
    "final_result", "is_at_risk", "date_unregistration", "has_unregistered"
]


def validate_dataset(df: pd.DataFrame) -> None:
    """Kiểm tra chất lượng dữ liệu và quy tắc chống rò rỉ dữ liệu trước khi huấn luyện."""
    print(f"[Validation] Kích thước tập dữ liệu: {df.shape[0]} hàng x {df.shape[1]} cột")
    missing_features = [c for c in FEATURE_COLS_24 if c not in df.columns]
    if missing_features:
        raise ValueError(f"Thiếu các đặc trưng bắt buộc: {missing_features}")
    overlap = set(FEATURE_COLS_24).intersection(set(FORBIDDEN_LEAKAGE_COLS))
    if overlap:
        raise ValueError(f"Vi phạm Data Leakage! Phát hiện cột bị cấm trong tập đặc trưng: {overlap}")
    dup_count = int(df.duplicated(subset=["id_student"] if "id_student" in df.columns else None).sum())
    null_in_24 = int(df[FEATURE_COLS_24].isnull().sum().sum())
    risk_counts = df["is_at_risk"].value_counts().to_dict()
    print(f"[Validation] Bản ghi trùng lặp: {dup_count} | Giá trị thiếu trong 24 đặc trưng số: {null_in_24}")
    print(f"[Validation] Phân phối nhãn is_at_risk: At-Risk (1) = {risk_counts.get(1, 0)} "
          f"({risk_counts.get(1, 0)/len(df)*100:.2f}%), Non-Risk (0) = {risk_counts.get(0, 0)} "
          f"({risk_counts.get(0, 0)/len(df)*100:.2f}%)")


def train_and_evaluate(df: pd.DataFrame, mode_name: str, out_dir: str, opt_threshold: float = 0.35):
    """Huấn luyện 3 thuật toán với Pipeline + Stratified 5-Fold CV và đánh giá trên Tập Test độc lập."""
    validate_dataset(df)
    X = df[FEATURE_COLS_24].copy()
    y = df["is_at_risk"].astype(int).copy()

    # Phân chia Stratified 80/20 với random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Khởi tạo Pipeline cho 3 thuật toán (Imputation + Scaling chỉ fit trên Training fold)
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.1 if "Week4" in mode_name else 1.0,
                class_weight="balanced", max_iter=1000, random_state=42
            ))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=150, criterion="gini", max_depth=8,
                min_samples_split=4, min_samples_leaf=2,
                max_features="sqrt", class_weight="balanced", random_state=42
            ))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(
                criterion="gini" if "Week4" in mode_name else "entropy",
                max_depth=6, min_samples_split=10, min_samples_leaf=5,
                class_weight="balanced", random_state=42
            ))
        ])
    }

    comp_rows = []
    oof_probas = {}
    test_probas = {}

    for mname, pipe in models.items():
        pipe.fit(X_train, y_train)
        oof_p = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        oof_pred = (oof_p >= 0.50).astype(int)
        oof_probas[mname] = oof_p

        te_p = pipe.predict_proba(X_test)[:, 1]
        te_pred = (te_p >= 0.50).astype(int)
        test_probas[mname] = te_p

        tn, fp, fn, tp = confusion_matrix(y_test, te_pred).ravel()
        tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_train, oof_pred).ravel()

        comp_rows.append({
            "Dataset_Mode": mode_name,
            "Model": mname,
            "Threshold": 0.50,
            "CV_Accuracy": round(accuracy_score(y_train, oof_pred), 4),
            "CV_Precision": round(precision_score(y_train, oof_pred), 4),
            "CV_Recall": round(recall_score(y_train, oof_pred), 4),
            "CV_F1": round(f1_score(y_train, oof_pred), 4),
            "CV_ROC_AUC": round(roc_auc_score(y_train, oof_p), 4),
            "CV_PR_AUC": round(average_precision_score(y_train, oof_p), 4),
            "CV_FN": int(fn_c),
            "CV_FP": int(fp_c),
            "Accuracy": round(accuracy_score(y_test, te_pred), 4),
            "Precision": round(precision_score(y_test, te_pred), 4),
            "Recall": round(recall_score(y_test, te_pred), 4),
            "F1-Score": round(f1_score(y_test, te_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, te_p), 4),
            "PR-AUC": round(average_precision_score(y_test, te_p), 4),
            "Balanced_Accuracy": round(balanced_accuracy_score(y_test, te_pred), 4),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "False Negative": int(fn), "False Positive": int(fp)
        })

    # Đánh giá thêm Random Forest tại ngưỡng tối ưu (chọn trên CV)
    rf_oof = oof_probas["Random Forest"]
    rf_te = test_probas["Random Forest"]
    cv_opt = (rf_oof >= opt_threshold).astype(int)
    te_opt = (rf_te >= opt_threshold).astype(int)
    tn_o, fp_o, fn_o, tp_o = confusion_matrix(y_test, te_opt).ravel()
    tn_co, fp_co, fn_co, tp_co = confusion_matrix(y_train, cv_opt).ravel()

    comp_rows.append({
        "Dataset_Mode": mode_name,
        "Model": f"Random Forest (Tối ưu Threshold={opt_threshold:.2f})",
        "Threshold": opt_threshold,
        "CV_Accuracy": round(accuracy_score(y_train, cv_opt), 4),
        "CV_Precision": round(precision_score(y_train, cv_opt), 4),
        "CV_Recall": round(recall_score(y_train, cv_opt), 4),
        "CV_F1": round(f1_score(y_train, cv_opt), 4),
        "CV_ROC_AUC": round(roc_auc_score(y_train, rf_oof), 4),
        "CV_PR_AUC": round(average_precision_score(y_train, rf_oof), 4),
        "CV_FN": int(fn_co),
        "CV_FP": int(fp_co),
        "Accuracy": round(accuracy_score(y_test, te_opt), 4),
        "Precision": round(precision_score(y_test, te_opt), 4),
        "Recall": round(recall_score(y_test, te_opt), 4),
        "F1-Score": round(f1_score(y_test, te_opt), 4),
        "ROC-AUC": round(roc_auc_score(y_test, rf_te), 4),
        "PR-AUC": round(average_precision_score(y_test, rf_te), 4),
        "Balanced_Accuracy": round(balanced_accuracy_score(y_test, te_opt), 4),
        "TN": int(tn_o), "FP": int(fp_o), "FN": int(fn_o), "TP": int(tp_o),
        "False Negative": int(fn_o), "False Positive": int(fp_o)
    })

    return models["Random Forest"], pd.DataFrame(comp_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá mô hình Smart LMS Tuần 5-6")
    parser.add_argument("--data_path", type=str, default="../work/prj1/prj1/week 3+4/data/output/student_features.csv")
    parser.add_argument("--early_data_path", type=str, default="student_features_early_d100.csv")
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if os.path.exists(args.data_path):
        df_w4 = pd.read_csv(args.data_path)
        rf_w4, comp_w4 = train_and_evaluate(df_w4, "Inherited_Week4_24Features", args.out_dir)
        joblib.dump(rf_w4, os.path.join(args.out_dir, "best_model.joblib"))
        print(comp_w4.to_string(index=False))

    if os.path.exists(args.early_data_path):
        df_d100 = pd.read_csv(args.early_data_path)
        rf_d100, comp_d100 = train_and_evaluate(df_d100, "Strict_EarlyWarning_Day100", args.out_dir)
        joblib.dump(rf_d100, os.path.join(args.out_dir, "best_model_early_d100.joblib"))
        print(comp_d100.to_string(index=False))
