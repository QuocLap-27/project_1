# -*- coding: utf-8 -*-
"""
Module: ml_predictor_bridge.py
Dự án: SMART LMS & GENERATIVE AI (Tuần 8 – Kết nối ML & GenAI)
Chức năng:
- Tái sử dụng trực tiếp module predict.py, best_model.joblib và model_metadata.json từ Tuần 5+6
  (KHÔNG huấn luyện lại mô hình, KHÔNG thay đổi 24 đặc trưng hay ngưỡng 0.35 / 0.65).
- Hỗ trợ suy luận chính xác trên trọng số thật của best_model.joblib ngay cả khi
  scikit-learn >= 1.9 thay đổi internal tag của mô hình được pickle từ scikit-learn 1.5.2.
- Cung cấp hàm đọc read-only các mẫu sinh viên thực tế từ student_features.csv.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .config import (
    WEEK34_STUDENT_FEATURES_CSV,
    WEEK56_METADATA_PATH,
    WEEK56_MODEL_PATH,
    WEEK56_PREDICT_PY,
)


def _load_week56_predict_module():
    if not WEEK56_PREDICT_PY.exists():
        raise FileNotFoundError(f"Không tìm thấy predict.py tại: {WEEK56_PREDICT_PY}")
    spec = importlib.util.spec_from_file_location("week56_predict_bridge_mod", str(WEEK56_PREDICT_PY))
    if spec is None or spec.loader is None:
        raise ImportError(f"Không thể nạp module từ {WEEK56_PREDICT_PY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _eval_tuple_tree(node: Any, x_row: np.ndarray) -> float:
    """Duyệt cây quyết định nhị phân lưu dạng tuple trong best_model.joblib."""
    while not node[0]:
        feat_idx = int(node[2])
        threshold = float(node[3])
        node = node[4] if float(x_row[feat_idx]) <= threshold else node[5]
    return float(node[1])


class MLPredictorBridge:
    """Cầu nối tái sử dụng StudentRiskPredictor + best_model.joblib cho pipeline Tuần 8."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ) -> None:
        mod = _load_week56_predict_module()
        self.expected_features: List[str] = list(getattr(mod, "EXPECTED_FEATURES"))
        self.predictor_cls = getattr(mod, "StudentRiskPredictor")

        m_path = str(model_path or WEEK56_MODEL_PATH)
        meta_path = str(metadata_path or WEEK56_METADATA_PATH)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.predictor = self.predictor_cls(model_path=m_path, metadata_path=meta_path)

    def _predict_prob_from_best_model(self, X_df: pd.DataFrame) -> tuple[float, str]:
        """
        Ưu tiên gọi pipeline.predict_proba(X_df). Nếu phiên bản scikit-learn hiện tại
        không tương thích thuộc tính tag nội bộ khi unpickle (1.5.2 -> 1.9+), tự động
        chạy suy luận trên đúng các tham số đã huấn luyện (imputer + scaler + 150 cây)
        của best_model.joblib mà không huấn luyện lại.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob = float(self.predictor.model.predict_proba(X_df)[0, 1])
            return prob, "sklearn_predict_proba"
        except AttributeError:
            pipe = self.predictor.model
            imp = pipe.named_steps["imputer"]
            sc = pipe.named_steps["scaler"]
            rf = pipe.named_steps["clf"]

            arr = X_df.to_numpy(dtype=float).copy()
            nan_mask = np.isnan(arr)
            if nan_mask.any():
                arr[nan_mask] = np.take(imp.statistics_, np.where(nan_mask)[1])

            if getattr(sc, "with_mean", True):
                arr = arr - sc.mean_
            if getattr(sc, "with_std", True):
                safe_scale = np.where(sc.scale_ == 0, 1.0, sc.scale_)
                arr = arr / safe_scale

            x_row = arr[0]
            tree_probs = []
            for est in rf.estimators_:
                t_obj = est.tree_
                if isinstance(t_obj, tuple):
                    tree_probs.append(_eval_tuple_tree(t_obj, x_row))
                else:
                    tree_probs.append(float(est.predict_proba(arr)[0, 1]))
            return float(np.mean(tree_probs)), "best_model_joblib_tuple_tree_adapter"

    def predict_student_risk(
        self,
        student_features: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Bước 1 & Bước 2: Dự đoán xác suất At-Risk và phân loại trạng thái học tập
        bằng đúng logic predict.py + best_model.joblib.
        Nếu dữ liệu đầu vào thiếu hoặc sai miền giá trị, trả về dict lỗi có giải thích.
        """
        try:
            X_df = self.predictor._validate_input(student_features)
        except Exception as exc:
            return {
                "success": False,
                "error": f"Dữ liệu đầu vào 24 đặc trưng ML không hợp lệ: {exc}",
                "prediction": None,
                "risk_probability": None,
                "risk_level": None,
                "decision_threshold": self.predictor.decision_threshold,
                "submission_behavior": None,
                "model_version": self.predictor.model_version,
            }

        prob, backend_name = self._predict_prob_from_best_model(X_df)
        threshold = float(self.predictor.decision_threshold)
        pred = int(prob >= threshold)
        risk_level = self.predictor.get_risk_level(prob, threshold)
        behavior_row = X_df.iloc[0].fillna(0.0)
        behavior = self.predictor.classify_submission_behavior(behavior_row)

        return {
            "success": True,
            "error": None,
            "prediction": pred,
            "risk_probability": round(prob, 4),
            "risk_level": risk_level,
            "decision_threshold": threshold,
            "submission_behavior": behavior,
            "model_name": str(self.predictor.metadata.get("model_name", "SmartLMS_AtRisk_RandomForest_Pipeline")),
            "model_version": self.predictor.model_version,
            "feature_count": len(self.predictor.feature_names),
            "inference_backend": backend_name,
        }


def load_oulad_student_row(
    id_student: int,
    csv_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Đọc read-only hồ sơ 24 đặc trưng của 1 sinh viên trong student_features.csv."""
    target = Path(csv_path) if csv_path is not None else WEEK34_STUDENT_FEATURES_CSV
    if not target.exists():
        raise FileNotFoundError(f"Không tìm thấy student_features.csv tại: {target}")

    df = pd.read_csv(target)
    sub = df[df["id_student"] == int(id_student)]
    if sub.empty:
        return None
    row = sub.iloc[0]
    mod = _load_week56_predict_module()
    expected = list(getattr(mod, "EXPECTED_FEATURES"))
    features = {col: (None if pd.isna(row[col]) else float(row[col])) for col in expected}
    return {
        "id_student": int(row["id_student"]),
        "code_module": str(row["code_module"]),
        "code_presentation": str(row["code_presentation"]),
        "features": features,
    }


def find_representative_oulad_students(
    csv_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Tìm các sinh viên mẫu thực tế đại diện cho từng trạng thái học tập từ student_features.csv
    (chỉ trích xuất 24 đặc trưng đầu vào và id_student để tra bảng Mapping nội bộ).
    """
    target = Path(csv_path) if csv_path is not None else WEEK34_STUDENT_FEATURES_CSV
    if not target.exists():
        raise FileNotFoundError(f"Không tìm thấy student_features.csv tại: {target}")

    df = pd.read_csv(target)
    mod = _load_week56_predict_module()
    expected = list(getattr(mod, "EXPECTED_FEATURES"))
    classify_fn = getattr(mod, "StudentRiskPredictor").classify_submission_behavior

    picked: Dict[str, Dict[str, Any]] = {}
    needed = {"NO_SUBMISSION", "PARTIAL_SUBMISSION_LOW_SCORE", "SUBMITTED_LOW_SCORE", "NORMAL"}

    for _, row in df.iterrows():
        status_code = classify_fn(row)["status_code"]
        if status_code in needed and status_code not in picked:
            features = {col: (None if pd.isna(row[col]) else float(row[col])) for col in expected}
            picked[status_code] = {
                "id_student": int(row["id_student"]),
                "code_module": str(row["code_module"]),
                "code_presentation": str(row["code_presentation"]),
                "features": features,
            }
        if len(picked) == len(needed):
            break
    return picked
