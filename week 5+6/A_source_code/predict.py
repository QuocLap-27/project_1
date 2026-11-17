#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module: predict.py
Dự án: SMART LMS & GENERATIVE AI
Nhiệm vụ (Mục 4.3):
- Load mô hình Pipeline đã đóng gói (best_model.joblib) và tệp cấu hình (model_metadata.json).
- Kiểm tra nghiêm ngặt tên, kiểu dữ liệu và thứ tự của đúng 24 đặc trưng đầu vào.
- Không chấp nhận hoặc tự động loại bỏ các trường định danh/rò rỉ (id_student, final_result, v.v.).
- Tính xác suất thuộc nhóm rủi ro (risk_probability).
- Áp dụng ngưỡng quyết định đã tối ưu trên CV (decision_threshold = 0.35) để trả về nhãn dự đoán is_at_risk.
- Phân tầng mức độ cảnh báo LOW / MEDIUM / HIGH theo quy tắc định lượng rõ ràng:
    + LOW    : risk_probability < 0.35 (Dưới ngưỡng cảnh báo sớm -> An toàn)
    + MEDIUM : 0.35 <= risk_probability < 0.65 (Vượt ngưỡng cảnh báo sớm mức vừa -> Cần nhắc nhở & ôn tập)
    + HIGH   : risk_probability >= 0.65 (Nguy cơ trượt/bỏ học rất cao -> Cần can thiệp sư phạm khẩn cấp)
- Hỗ trợ tích hợp trực tiếp vào FastAPI / Flask Backend.
"""

import os
import sys
import json
from typing import Dict, Any, Union, List

# Hỗ trợ nạp thư viện tương thích nếu môi trường chưa cài đặt sẵn scikit-learn/joblib
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_COMPAT_DIR = os.path.join(_BASE_DIR, "_sklearn_compat")
if os.path.exists(_COMPAT_DIR) and _COMPAT_DIR not in sys.path:
    try:
        import sklearn  # noqa: F401
        import joblib   # noqa: F401
    except ImportError:
        sys.path.insert(0, _COMPAT_DIR)

import numpy as np
import pandas as pd
import joblib

EXPECTED_FEATURES_24: List[str] = [
    "num_assessments", "assessment_completion_rate", "avg_score", "weighted_score",
    "min_score", "max_score", "score_std", "tma_avg_score", "cma_avg_score",
    "late_submission_count", "avg_submission_delay", "total_clicks", "active_days",
    "average_clicks_per_day", "unique_learning_resources", "early_clicks_pre_course",
    "clicks_first_half", "clicks_oucontent", "clicks_forumng", "clicks_quiz",
    "clicks_resource_url", "num_of_prev_attempts", "studied_credits", "date_registration"
]


class SmartLMSRiskPredictor:
    """Bộ dự đoán nguy cơ học tập Smart LMS phục vụ tích hợp FastAPI/Flask."""

    def __init__(
        self,
        model_path: str = os.path.join(_BASE_DIR, "../B_models_and_results/best_model.joblib") if os.path.exists(os.path.join(_BASE_DIR, "../B_models_and_results/best_model.joblib")) else os.path.join(_BASE_DIR, "best_model.joblib"),
        metadata_path: str = os.path.join(_BASE_DIR, "../B_models_and_results/model_metadata.json") if os.path.exists(os.path.join(_BASE_DIR, "../B_models_and_results/model_metadata.json")) else os.path.join(_BASE_DIR, "model_metadata.json")
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình tại: {model_path}")
        self.model = joblib.load(model_path)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "feature_names": EXPECTED_FEATURES_24,
                "decision_threshold": 0.35,
                "model_version": "1.0.0"
            }

        self.feature_names: List[str] = self.metadata.get("feature_names", EXPECTED_FEATURES_24)
        self.decision_threshold: float = float(self.metadata.get("decision_threshold", 0.35))
        self.model_version: str = str(self.metadata.get("model_version", "1.0.0"))

    def validate_and_order_features(self, student_data: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Kiểm tra tính hợp lệ, kiểu dữ liệu số, giá trị âm bất hợp lý và sắp xếp đúng thứ tự 24 cột."""
        if isinstance(student_data, pd.Series):
            data_dict = student_data.to_dict()
        elif isinstance(student_data, pd.DataFrame):
            if len(student_data) != 1:
                raise ValueError(f"Đầu vào DataFrame phải chứa đúng 1 sinh viên, nhận được {len(student_data)} dòng.")
            data_dict = student_data.iloc[0].to_dict()
        elif isinstance(student_data, dict):
            data_dict = dict(student_data)
        else:
            raise TypeError("Đầu vào phải là dict, pandas.Series hoặc pandas.DataFrame 1 dòng.")

        missing_cols = [col for col in self.feature_names if col not in data_dict]
        if missing_cols:
            raise ValueError(f"Dữ liệu đầu vào thiếu {len(missing_cols)} đặc trưng bắt buộc: {missing_cols}")

        ordered_row = {}
        for col in self.feature_names:
            val = data_dict[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                ordered_row[col] = np.nan
                continue
            try:
                fval = float(val)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Đặc trưng '{col}' có giá trị không phải kiểu số hợp lệ: {val!r}") from exc

            # Kiểm tra miền giá trị nghiệp vụ cơ bản
            if col in ("total_clicks", "active_days", "num_assessments", "assessment_completion_rate") and fval < 0:
                raise ValueError(f"Đặc trưng '{col}' không được phép nhận giá trị âm ({fval}).")
            if col == "assessment_completion_rate" and fval > 1.0001:
                raise ValueError(f"Đặc trưng 'assessment_completion_rate' phải nằm trong [0, 1], nhận được {fval}.")
            ordered_row[col] = fval

        return pd.DataFrame([ordered_row], columns=self.feature_names)

    @staticmethod
    def assign_risk_level(probability: float, threshold: float = 0.35) -> str:
        """
        Quy tắc phân tầng mức độ cảnh báo:
        - LOW    : risk_probability < threshold (0.35) -> Sinh viên an toàn.
        - MEDIUM : threshold (0.35) <= risk_probability < 0.65 -> Có nguy cơ mức trung bình, cần cảnh báo sớm.
        - HIGH   : risk_probability >= 0.65 -> Nguy cơ trượt/bỏ học cao, cần can thiệp khẩn cấp.
        """
        if probability < threshold:
            return "LOW"
        elif probability < 0.65:
            return "MEDIUM"
        return "HIGH"

    def predict_student_risk(self, student_data: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> Dict[str, Any]:
        """Thực hiện suy luận rủi ro cho 1 sinh viên và trả về JSON chuẩn cho Backend."""
        try:
            X_input = self.validate_and_order_features(student_data)
            proba = float(self.model.predict_proba(X_input)[0, 1])
            pred = int(proba >= self.decision_threshold)
            level = self.assign_risk_level(proba, self.decision_threshold)
            return {
                "status": "success",
                "prediction": pred,
                "risk_probability": round(proba, 4),
                "risk_level": level,
                "decision_threshold": self.decision_threshold,
                "model_version": self.model_version
            }
        except Exception as err:
            return {
                "status": "error",
                "error_type": type(err).__name__,
                "message": str(err),
                "prediction": None,
                "risk_probability": None,
                "risk_level": None,
                "decision_threshold": self.decision_threshold,
                "model_version": self.model_version
            }


def predict_single_student(student_features: Dict[str, Any], model_path: str = None) -> Dict[str, Any]:
    """Hàm wrapper tiện lợi cho FastAPI/Flask endpoint."""
    predictor = SmartLMSRiskPredictor(model_path=model_path) if model_path else SmartLMSRiskPredictor()
    return predictor.predict_student_risk(student_features)


if __name__ == "__main__":
    print("=== KIỂM THỬ THỰC TẾ MODULE INFERENCE predict.py ===")
    predictor = SmartLMSRiskPredictor()

    w4_csv = os.path.join(_BASE_DIR, "../work/prj1/prj1/week 3+4/data/output/student_features.csv")
    if not os.path.exists(w4_csv):
        w4_csv = os.path.join(_BASE_DIR, "../B_models_and_results/student_features_early_d100.csv")
    if os.path.exists(w4_csv):
        df_sample = pd.read_csv(w4_csv)
        s_risk = df_sample[df_sample["is_at_risk"] == 1].iloc[0]
        res_risk = predictor.predict_student_risk(s_risk)
        print(f"\n1. Sinh viên thực tế nhóm Rủi ro cao (final_result={s_risk['final_result']}):")
        print(json.dumps(res_risk, indent=2, ensure_ascii=False))

        s_safe = df_sample[df_sample["is_at_risk"] == 0].iloc[0]
        res_safe = predictor.predict_student_risk(s_safe)
        print(f"\n2. Sinh viên thực tế nhóm An toàn (final_result={s_safe['final_result']}):")
        print(json.dumps(res_safe, indent=2, ensure_ascii=False))

    bad_input = {"num_assessments": -2, "avg_score": "invalid_number"}
    res_err = predictor.predict_student_risk(bad_input)
    print("\n3. Kiểm thử bắt lỗi dữ liệu đầu vào không hợp lệ:")
    print(json.dumps(res_err, indent=2, ensure_ascii=False))
