# -*- coding: utf-8 -*-
"""
Module: train_models.py
Dự án: SMART LMS & GENERATIVE AI
Sinh viên thực hiện: Bế Quốc Lập - MSSV: 202416542
Chức năng: Huấn luyện, đánh giá và đóng gói mô hình dự đoán sinh viên có nguy cơ trượt môn (Tuần 5 & 6).
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
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

# Danh sách 24 đặc trưng định lượng đầu vào (đã loại bỏ id_student, final_result, has_unregistered và Exam)
FEATURE_COLS = [
    "num_assessments", "assessment_completion_rate", "avg_score", "weighted_score",
    "min_score", "max_score", "score_std", "tma_avg_score", "cma_avg_score",
    "late_submission_count", "avg_submission_delay", "total_clicks", "active_days",
    "average_clicks_per_day", "unique_learning_resources", "early_clicks_pre_course",
    "clicks_first_half", "clicks_oucontent", "clicks_forumng", "clicks_quiz",
    "clicks_resource_url", "num_of_prev_attempts", "studied_credits", "date_registration"
]

LEAKAGE_COLS = [
    "id_student", "code_module", "code_presentation",
    "final_result", "is_at_risk", "date_unregistration", "has_unregistered"
]


def check_data_quality(df: pd.DataFrame):
    """Kiểm tra nhanh cấu trúc file student_features.csv trước khi train."""
    print(f"[Check] Kích thước dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")
    print(f"[Check] Số dòng trùng lặp: {df.duplicated().sum()}")
    print(f"[Check] Số giá trị thiếu trong 24 đặc trưng: {df[FEATURE_COLS].isnull().sum().sum()}")
    dist = df["is_at_risk"].value_counts()
    print(f"[Check] Phân phối nhãn is_at_risk: At-Risk (1) = {dist.get(1, 0)} ({dist.get(1, 0)/len(df)*100:.2f}%), "
          f"Non-Risk (0) = {dist.get(0, 0)} ({dist.get(0, 0)/len(df)*100:.2f}%)")


def run_training(data_path: str, out_dir: str = "../B_models_and_results", opt_threshold: float = 0.35):
    df = pd.read_csv(data_path)
    check_data_quality(df)

    X = df[FEATURE_COLS].copy()
    y = df["is_at_risk"].astype(int).copy()

    # Chia tập Train/Test theo tỷ lệ 80/20 có phân tầng (Stratified Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Khởi tạo Pipeline cho 3 thuật toán
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42))
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
                criterion="gini", max_depth=6,
                min_samples_split=10, min_samples_leaf=5,
                class_weight="balanced", random_state=42
            ))
        ])
    }

    os.makedirs(out_dir, exist_ok=True)
    comparison_rows = []

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        oof_pred = (oof_proba >= 0.50).astype(int)

        test_proba = pipe.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= 0.50).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
        comparison_rows.append({
            "Model": name,
            "Threshold": 0.50,
            "CV_Recall": round(recall_score(y_train, oof_pred), 4),
            "CV_F1": round(f1_score(y_train, oof_pred), 4),
            "Accuracy": round(accuracy_score(y_test, test_pred), 4),
            "Precision": round(precision_score(y_test, test_pred), 4),
            "Recall": round(recall_score(y_test, test_pred), 4),
            "F1-Score": round(f1_score(y_test, test_pred), 4),
            "ROC-AUC": round(roc_auc_score(y_test, test_proba), 4),
            "PR-AUC": round(average_precision_score(y_test, test_proba), 4),
            "Balanced_Accuracy": round(balanced_accuracy_score(y_test, test_pred), 4),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        })

    # Lưu mô hình Random Forest tốt nhất
    best_rf = models["Random Forest"]
    joblib.dump(best_rf, os.path.join(out_dir, "best_model.joblib"))
    print("[Done] Đã lưu mô hình tốt nhất tại best_model.joblib")
    return pd.DataFrame(comparison_rows)


if __name__ == "__main__":
    default_csv = "../../week 3+4/data/output/student_features.csv"
    if os.path.exists(default_csv):
        res_df = run_training(default_csv)
        print(res_df.to_string(index=False))
