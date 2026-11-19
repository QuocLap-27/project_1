# -*- coding: utf-8 -*-
"""
Module: predict.py
Dự án: SMART LMS & GENERATIVE AI
Sinh viên thực hiện: Bế Quốc Lập - MSSV: 202416542
Chức năng:
- Load mô hình best_model.joblib và thực hiện dự đoán nguy cơ trượt môn cho sinh viên.
- Phân biệt rõ trường hợp sinh viên "Không nộp bài" với "Có nộp bài nhưng điểm kém".
- Tạo 3 kịch bản đặt Prompt (Chung chung, Thêm mức rủi ro, Thêm hồ sơ chi tiết) chuẩn bị cho Gemini API ở Tuần 7-8.
"""

import os
import json
from typing import Dict, Any, Union, List, Optional
import numpy as np
import pandas as pd
import joblib

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXPECTED_FEATURES = [
    "num_assessments", "assessment_completion_rate", "avg_score", "weighted_score",
    "min_score", "max_score", "score_std", "tma_avg_score", "cma_avg_score",
    "late_submission_count", "avg_submission_delay", "total_clicks", "active_days",
    "average_clicks_per_day", "unique_learning_resources", "early_clicks_pre_course",
    "clicks_first_half", "clicks_oucontent", "clicks_forumng", "clicks_quiz",
    "clicks_resource_url", "num_of_prev_attempts", "studied_credits", "date_registration"
]


def _find_path(filename: str) -> str:
    for rel in [filename, os.path.join("../B_models_and_results", filename), os.path.join("B_models_and_results", filename)]:
        p = os.path.abspath(os.path.join(_BASE_DIR, rel))
        if os.path.exists(p):
            return p
    return os.path.join(_BASE_DIR, filename)


class StudentRiskPredictor:
    def __init__(self, model_path: Optional[str] = None, metadata_path: Optional[str] = None):
        model_path = model_path or _find_path("best_model.joblib")
        metadata_path = metadata_path or _find_path("model_metadata.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy file mô hình: {model_path}")
        self.model = joblib.load(model_path)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"feature_names": EXPECTED_FEATURES, "decision_threshold": 0.35, "model_version": "1.0"}

        self.feature_names = self.metadata.get("feature_names", EXPECTED_FEATURES)
        self.decision_threshold = float(self.metadata.get("decision_threshold", 0.35))
        self.model_version = str(self.metadata.get("model_version", "1.0"))

    def _validate_input(self, student_data: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(student_data, pd.Series):
            data_dict = student_data.to_dict()
        elif isinstance(student_data, pd.DataFrame):
            if len(student_data) != 1:
                raise ValueError("DataFrame đầu vào chỉ được chứa 1 dòng tương ứng với 1 sinh viên.")
            data_dict = student_data.iloc[0].to_dict()
        elif isinstance(student_data, dict):
            data_dict = dict(student_data)
        else:
            raise TypeError("Đầu vào phải là dict, Series hoặc DataFrame.")

        missing = [c for c in self.feature_names if c not in data_dict]
        if missing:
            raise ValueError(f"Thiếu các đặc trưng bắt buộc: {missing}")

        row = {}
        for col in self.feature_names:
            val = data_dict[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                row[col] = np.nan
                continue
            fval = float(val)
            if col in ("total_clicks", "active_days", "num_assessments", "assessment_completion_rate") and fval < 0:
                raise ValueError(f"Giá trị của '{col}' không hợp lệ ({fval} < 0).")
            if col == "assessment_completion_rate" and fval > 1.0001:
                raise ValueError(f"Tỷ lệ hoàn thành bài tập phải nằm trong [0, 1], nhận được {fval}.")
            row[col] = fval
        return pd.DataFrame([row], columns=self.feature_names)

    @staticmethod
    def get_risk_level(prob: float, threshold: float = 0.35) -> str:
        if prob < threshold:
            return "LOW"
        elif prob < 0.65:
            return "MEDIUM"
        return "HIGH"

    @staticmethod
    def classify_submission_behavior(row: pd.Series) -> Dict[str, str]:
        """Phân biệt giữa không nộp bài và có làm bài nhưng điểm thấp."""
        comp = float(row.get("assessment_completion_rate", 0.0))
        n_sub = int(row.get("num_assessments", 0))
        avg_sc = float(row.get("avg_score", 0.0))

        if n_sub == 0 or comp < 0.25:
            return {
                "status_code": "NO_SUBMISSION",
                "status_label": "Không nộp bài / Ngừng học sớm",
                "note": f"Mới nộp {n_sub} bài ({comp*100:.0f}%). Cần nhắc nhở tiến độ và giao bài tập khởi động ngắn, dễ tiếp cận."
            }
        elif comp < 0.75 and avg_sc < 55.0:
            return {
                "status_code": "PARTIAL_SUBMISSION_LOW_SCORE",
                "status_label": "Nộp thiếu bài và điểm thấp",
                "note": f"Hoàn thành {comp*100:.0f}% số bài với điểm TB {avg_sc:.1f}/100. Cần củng cố lại kiến thức cơ bản và nhắc hạn nộp bài."
            }
        elif comp >= 0.75 and avg_sc < 55.0:
            return {
                "status_code": "SUBMITTED_LOW_SCORE",
                "status_label": "Điểm thấp dù đã nộp bài đầy đủ",
                "note": f"Đã nộp {comp*100:.0f}% số bài nhưng điểm TB chỉ đạt {avg_sc:.1f}/100. Cần tập trung giải thích kỹ lỗi sai và bản chất kiến thức."
            }
        elif comp < 0.75 and avg_sc >= 55.0:
            return {
                "status_code": "PARTIAL_SUBMISSION_OK_SCORE",
                "status_label": "Làm bài đạt yêu cầu nhưng còn bỏ sót bài kiểm tra",
                "note": f"Điểm các bài đã nộp đạt {avg_sc:.1f}/100 nhưng mới nộp {comp*100:.0f}% số bài."
            }
        return {
            "status_code": "NORMAL",
            "status_label": "Nộp bài đầy đủ, điểm đạt yêu cầu",
            "note": f"Hoàn thành {comp*100:.0f}% số bài, điểm TB {avg_sc:.1f}/100."
        }

    @staticmethod
    def build_prompt_scenarios(
        risk_prob: float,
        risk_level: str,
        weak_blocks: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """
        Tạo 3 kịch bản đặt Prompt cho học phần IT1108 - Nhập môn lập trình:
        1. baseline: Chung chung, chỉ có thông tin môn học và chủ đề.
        2. risk_level: Thêm mức rủi ro và xác suất trượt môn.
        3. detailed_profile: Thêm hồ sơ chi tiết theo từng khối kiến thức (kèm % so với lớp và trạng thái nộp bài/điểm thấp).
        """
        if not weak_blocks:
            weak_blocks = [
                {
                    "block_name": "Khối 3 – Cấu trúc điều khiển và vòng lặp",
                    "detail": "thấp hơn 85% lớp, chưa nộp bài kiểm tra định kỳ",
                    "type": "NO_SUBMISSION"
                },
                {
                    "block_name": "Khối 4 – Hàm và mảng",
                    "detail": "điểm thấp (42/100) dù đã nộp bài",
                    "type": "SUBMITTED_LOW_SCORE"
                }
            ]

        topics_text = ", ".join([b["block_name"] for b in weak_blocks])
        risk_text = "cao" if risk_level == "HIGH" else ("trung bình" if risk_level == "MEDIUM" else "thấp")

        # 1. Kịch bản chung chung (Baseline)
        p_baseline = (
            f"Hãy tạo 5 câu hỏi trắc nghiệm ôn tập cho học phần IT1108 – Nhập môn lập trình (ngôn ngữ C) "
            f"thuộc nội dung: {topics_text}. Mỗi câu có 4 đáp án A, B, C, D kèm giải thích ngắn gọn và trả về định dạng JSON."
        )

        # 2. Kịch bản thêm mức rủi ro
        p_risk = (
            f"Sinh viên này có nguy cơ trượt môn {risk_text} (xác suất {risk_prob:.2f}). "
            f"Hãy điều chỉnh độ khó và cách giải thích cho phù hợp khi tạo 5 câu hỏi trắc nghiệm ôn tập học phần "
            f"IT1108 – Nhập môn lập trình về: {topics_text}. Trả về kết quả ở định dạng JSON."
        )

        # 3. Kịch bản thêm hồ sơ chi tiết
        profile_parts = [f"{b['block_name']} ({b['detail']})" for b in weak_blocks]
        p_detailed = (
            f"Sinh viên này có nguy cơ trượt môn {risk_text} (xác suất {risk_prob:.2f}). "
            f"Hãy điều chỉnh độ khó và cách giải thích cho phù hợp. "
            f"Tình trạng cụ thể: {'; '.join(profile_parts)}. "
            f"Hãy giảm độ khó xuống mức cơ bản kèm gợi ý từng bước đối với phần sinh viên chưa nộp bài/thấp hơn 85% lớp, "
            f"và tăng câu hỏi phân tích lỗi sai thường gặp kèm giải thích chi tiết cách chạy chương trình đối với phần sinh viên "
            f"đã nộp bài nhưng điểm thấp. Tạo 5 câu hỏi trắc nghiệm và trả về định dạng JSON."
        )

        return {
            "baseline": p_baseline,
            "risk_level": p_risk,
            "detailed_profile": p_detailed
        }

    def predict(self, student_data: Union[Dict[str, Any], pd.Series, pd.DataFrame], weak_blocks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        try:
            X_df = self._validate_input(student_data)
            prob = float(self.model.predict_proba(X_df)[0, 1])
            pred = int(prob >= self.decision_threshold)
            level = self.get_risk_level(prob, self.decision_threshold)
            behavior = self.classify_submission_behavior(X_df.iloc[0])
            prompts = self.build_prompt_scenarios(prob, level, weak_blocks)
            return {
                "prediction": pred,
                "risk_probability": round(prob, 4),
                "risk_level": level,
                "decision_threshold": self.decision_threshold,
                "submission_behavior": behavior,
                "prompt_scenarios": prompts,
                "model_version": self.model_version
            }
        except Exception as e:
            return {
                "error": str(e),
                "prediction": None,
                "risk_probability": None,
                "risk_level": None,
                "decision_threshold": self.decision_threshold,
                "model_version": self.model_version
            }


if __name__ == "__main__":
    predictor = StudentRiskPredictor()
    sample_student = {
        "num_assessments": 5, "assessment_completion_rate": 0.42, "avg_score": 48.0,
        "weighted_score": 22.5, "min_score": 35.0, "max_score": 60.0, "score_std": 9.5,
        "tma_avg_score": 45.0, "cma_avg_score": 52.0, "late_submission_count": 2,
        "avg_submission_delay": 3.5, "total_clicks": 620, "active_days": 24,
        "average_clicks_per_day": 25.8, "unique_learning_resources": 45,
        "early_clicks_pre_course": 15, "clicks_first_half": 480, "clicks_oucontent": 210,
        "clicks_forumng": 40, "clicks_quiz": 120, "clicks_resource_url": 95,
        "num_of_prev_attempts": 0, "studied_credits": 60, "date_registration": -35.0
    }
    res = predictor.predict(sample_student)
    print(json.dumps(res, indent=2, ensure_ascii=False))
