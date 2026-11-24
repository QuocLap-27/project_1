# -*- coding: utf-8 -*-
"""
Module: topic_mapper.py
Dự án: SMART LMS & GENERATIVE AI (Tuần 8 – Kết nối ML & GenAI)
Chức năng:
- Tái sử dụng 3 bảng Mapping (curriculum_topics, assessment_topic_mapping, vle_topic_mapping)
  ở chế độ read-only, tuyệt đối không sửa tên bảng, schema, khóa hay cột.
- Tính toán điểm số từng khối kiến thức IT1108, tỷ lệ hoàn thành và phân vị lớp thật
  (lower_than_class_pct) từ dữ liệu OULAD FFF-2013J khi có oulad_student_id.
- Xây dựng hồ sơ cá nhân hóa (Step 4) bằng mã sinh viên giả lập; tuyệt đối KHÔNG tự bịa
  điểm số hoặc phân vị nếu dữ liệu thực không tồn tại.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import OULAD_ZIP_PATH, VLE_MAPPING_CSV
from .curriculum_loader import load_assessment_mappings, load_curriculum_blocks
from .prompt_builder import normalize_block_status
from .schemas import VALID_IT1108_BLOCKS


def load_vle_block_summary(csv_path: Optional[Path] = None) -> Dict[str, int]:
    """Đọc read-only vle_topic_mapping.csv để thống kê số lượng học liệu VLE theo khối IT1108."""
    target = Path(csv_path) if csv_path is not None else VLE_MAPPING_CSV
    if not target.exists():
        return {}
    counts: Dict[str, int] = {}
    with target.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            code = (row.get("it1108_block_code") or "").strip()
            if code in VALID_IT1108_BLOCKS:
                counts[code] = counts.get(code, 0) + 1
    return counts


def _classify_block_submission_status(
    submitted: int,
    total_mapped: int,
    avg_score: Optional[float],
    weak_score_threshold: float = 55.0,
) -> str:
    """Áp dụng quy tắc tương đồng predict.py để phân loại trạng thái từng khối kiến thức."""
    if total_mapped <= 0:
        return "NORMAL"
    comp = submitted / total_mapped
    score = 0.0 if avg_score is None else float(avg_score)

    if submitted == 0 or comp < 0.25:
        return "NO_SUBMISSION"
    if comp < 0.75 and score < weak_score_threshold:
        return "PARTIAL_SUBMISSION_LOW_SCORE"
    if comp >= 0.75 and score < weak_score_threshold:
        return "SUBMITTED_LOW_SCORE"
    if comp < 0.75 and score >= weak_score_threshold:
        return "PARTIAL_SUBMISSION_OK_SCORE"
    return "NORMAL"


class OULADBlockAnalyzer:
    """
    Phân tích kết quả học tập theo 6 khối kiến thức IT1108 từ bộ dữ liệu OULAD FFF-2013J
    (mô phỏng tương đương logic dbo.usp_GetStudentWeakTopicsForGemini bằng Python/pandas).
    """

    def __init__(
        self,
        oulad_zip_path: Optional[Path] = None,
        code_module: str = "FFF",
        code_presentation: str = "2013J",
    ) -> None:
        self.oulad_zip_path = Path(oulad_zip_path) if oulad_zip_path is not None else OULAD_ZIP_PATH
        self.code_module = code_module
        self.code_presentation = code_presentation
        self._class_block_df_cache: Dict[Optional[int], pd.DataFrame] = {}

    def _compute_class_block_table(self, cutoff_day: Optional[int] = None) -> pd.DataFrame:
        if cutoff_day in self._class_block_df_cache:
            return self._class_block_df_cache[cutoff_day]

        if not self.oulad_zip_path.exists():
            raise FileNotFoundError(f"Không tìm thấy dữ liệu OULAD zip tại: {self.oulad_zip_path}")

        curriculum_blocks = load_curriculum_blocks(include_general=False)
        mappings = load_assessment_mappings(exclude_exam=True)
        map_df = pd.DataFrame(mappings)
        map_df = map_df[map_df["it1108_block_code"].isin(VALID_IT1108_BLOCKS)].copy()

        with zipfile.ZipFile(self.oulad_zip_path, "r") as zf:
            assess_df = pd.read_csv(io.BytesIO(zf.read("assessments.csv")))
            sa_df = pd.read_csv(io.BytesIO(zf.read("studentAssessment.csv")), na_values=["?"])
            si_df = pd.read_csv(io.BytesIO(zf.read("studentInfo.csv")))

        si_sub = si_df[
            (si_df["code_module"] == self.code_module)
            & (si_df["code_presentation"] == self.code_presentation)
        ][["id_student"]].drop_duplicates()

        assess_sub = assess_df[
            (assess_df["code_module"] == self.code_module)
            & (assess_df["code_presentation"] == self.code_presentation)
            & (assess_df["assessment_type"] != "Exam")
        ].copy()

        if cutoff_day is not None:
            assess_sub = assess_sub[
                (assess_sub["date"].fillna(9999) <= cutoff_day)
                | (assess_sub["assessment_type"] == "CMA")
            ]

        eligible = assess_sub.merge(
            map_df[["id_assessment", "topic_id", "it1108_block_code", "mapping_weight"]],
            on="id_assessment",
            how="inner",
        )
        eligible["effective_weight"] = (
            eligible["weight"].apply(lambda w: float(w) if w and float(w) > 0 else 1.0)
            * eligible["mapping_weight"].astype(float)
        )

        sa_sub = sa_df[sa_df["id_assessment"].isin(eligible["id_assessment"])].copy()
        sa_sub["score"] = pd.to_numeric(sa_sub["score"], errors="coerce")
        if cutoff_day is not None:
            sa_sub = sa_sub[sa_sub["date_submitted"] <= cutoff_day]

        # Tổng số bài và tổng trọng số từng khối
        block_meta = (
            eligible.groupby("it1108_block_code")
            .agg(
                total_mapped_assessments=("id_assessment", "nunique"),
                total_block_weight=("effective_weight", "sum"),
            )
            .reset_index()
        )

        # Điểm đã nộp của từng sinh viên trên từng bài thuộc từng khối
        merged_sa = sa_sub.merge(
            eligible[["id_assessment", "it1108_block_code", "effective_weight"]],
            on="id_assessment",
            how="inner",
        )
        merged_sa["weighted_part"] = merged_sa["score"].fillna(0.0) * merged_sa["effective_weight"]

        sv_agg = (
            merged_sa.groupby(["id_student", "it1108_block_code"])
            .agg(
                submitted_assessments=("id_assessment", "nunique"),
                avg_submitted_score=("score", "mean"),
                weighted_sum=("weighted_part", "sum"),
            )
            .reset_index()
        )

        # Cross join toàn bộ sinh viên trong lớp với các khối hợp lệ để tính đúng phân vị lớp
        si_sub = si_sub.assign(_k=1)
        block_meta = block_meta.assign(_k=1)
        full_grid = si_sub.merge(block_meta, on="_k").drop(columns=["_k"])
        full_grid = full_grid.merge(sv_agg, on=["id_student", "it1108_block_code"], how="left")

        full_grid["submitted_assessments"] = full_grid["submitted_assessments"].fillna(0).astype(int)
        full_grid["weighted_sum"] = full_grid["weighted_sum"].fillna(0.0)
        full_grid["block_weighted_score"] = (
            full_grid["weighted_sum"] / full_grid["total_block_weight"].replace(0, 1.0)
        ).round(2)
        full_grid["completion_rate"] = (
            full_grid["submitted_assessments"] / full_grid["total_mapped_assessments"].replace(0, 1)
        ).round(4)

        # PERCENT_RANK() OVER (PARTITION BY it1108_block_code ORDER BY block_weighted_score ASC)
        n_students = full_grid["id_student"].nunique()
        if n_students > 1:
            rank_min = full_grid.groupby("it1108_block_code")["block_weighted_score"].rank(method="min", ascending=True)
            pct_rank = (rank_min - 1.0) / float(n_students - 1)
            full_grid["lower_than_class_pct"] = ((1.0 - pct_rank) * 100.0).round(1)
        else:
            full_grid["lower_than_class_pct"] = None

        full_grid["it1108_block_name"] = full_grid["it1108_block_code"].map(
            lambda c: (curriculum_blocks.get(c) or {}).get("it1108_block_name", c)
        )
        self._class_block_df_cache[cutoff_day] = full_grid
        return full_grid

    def get_student_block_records(
        self,
        id_student: int,
        cutoff_day: Optional[int] = None,
        weak_score_threshold: float = 55.0,
        weak_only: bool = True,
        max_blocks: int = 3,
    ) -> List[Dict[str, Any]]:
        """Trả về hồ sơ chi tiết từng khối kiến thức IT1108 từ dữ liệu OULAD thật."""
        class_df = self._compute_class_block_table(cutoff_day=cutoff_day)
        stu_df = class_df[class_df["id_student"] == int(id_student)].copy()
        if stu_df.empty:
            return []

        curriculum_blocks = load_curriculum_blocks(include_general=False)
        records: List[Dict[str, Any]] = []
        for _, row in stu_df.iterrows():
            code = str(row["it1108_block_code"])
            ctx = curriculum_blocks.get(code, {})
            submitted = int(row["submitted_assessments"])
            total_mapped = int(row["total_mapped_assessments"])
            avg_sub = None if pd.isna(row["avg_submitted_score"]) else round(float(row["avg_submitted_score"]), 2)
            weighted_sc = round(float(row["block_weighted_score"]), 2)
            comp_rate = round(float(row["completion_rate"]), 4)
            lower_pct = None if pd.isna(row["lower_than_class_pct"]) else float(row["lower_than_class_pct"])

            status = _classify_block_submission_status(
                submitted=submitted,
                total_mapped=total_mapped,
                avg_score=avg_sub,
                weak_score_threshold=weak_score_threshold,
            )
            records.append({
                "it1108_block_code": code,
                "it1108_block_name": str(ctx.get("it1108_block_name") or row["it1108_block_name"]),
                "it1108_lessons": str(ctx.get("it1108_lessons") or ""),
                "it1108_weeks": str(ctx.get("it1108_weeks") or ""),
                "it1108_cdr": str(ctx.get("it1108_cdr") or ""),
                "total_mapped_assessments": total_mapped,
                "submitted_assessments": submitted,
                "completion_rate": comp_rate,
                "avg_submitted_score": avg_sub,
                "block_weighted_score": weighted_sc,
                "lower_than_class_pct": lower_pct,
                "submission_vs_score_status": status,
                "data_origin": "OULAD_FFF_2013J",
            })

        records.sort(key=lambda r: (r["block_weighted_score"], r["completion_rate"]))
        if weak_only:
            weak = [
                r for r in records
                if r["submission_vs_score_status"] != "NORMAL"
            ]
            if weak:
                return weak[:max_blocks]
        return records[:max_blocks]


def _has_real_block_detail(block: Dict[str, Any]) -> bool:
    """Khối chỉ được coi là có dữ liệu chi tiết khi có trạng thái khối VÀ ít nhất 1 chỉ số thực."""
    has_status = bool(str(block.get("submission_vs_score_status") or block.get("type") or "").strip())
    has_metric = (
        block.get("avg_submitted_score") is not None
        or block.get("completion_rate") is not None
        or block.get("submitted_assessments") is not None
        or block.get("block_weighted_score") is not None
    )
    return has_status and has_metric


def build_personalized_profile(
    simulated_student_code: str,
    ml_result: Dict[str, Any],
    block_records: Optional[List[Dict[str, Any]]] = None,
    oulad_student_id: Optional[int] = None,
    target_block_codes: Optional[List[str]] = None,
    cutoff_day: Optional[int] = None,
    analyzer: Optional[OULADBlockAnalyzer] = None,
) -> Dict[str, Any]:
    """
    Bước 3 & Bước 4: Ánh xạ chủ đề IT1108 và xây dựng hồ sơ cá nhân hóa (ẩn danh).
    Tuyệt đối KHÔNG tự tạo điểm số hoặc phân vị nếu dữ liệu thực không tồn tại.
    """
    if not ml_result.get("success"):
        raise ValueError(f"Không thể xây dựng hồ sơ do bước ML Prediction lỗi: {ml_result.get('error')}")

    sim_code = (simulated_student_code or "").strip()
    if not sim_code:
        raise ValueError("simulated_student_code không được rỗng.")

    curriculum_blocks = load_curriculum_blocks(include_general=False)
    vle_counts = load_vle_block_summary()
    behavior = ml_result.get("submission_behavior") or {}
    coarse_status = str(behavior.get("status_code") or "NORMAL")

    resolved_blocks: List[Dict[str, Any]] = []
    block_data_source = "none"

    if block_records is not None and len(block_records) > 0:
        block_data_source = "provided_block_records"
        for item in block_records:
            code = str(item.get("it1108_block_code") or "").strip()
            if code not in curriculum_blocks:
                raise ValueError(f"Mã khối không hợp lệ trong block_records: '{code}'.")
            ctx = curriculum_blocks[code]
            resolved_blocks.append({
                "it1108_block_code": code,
                "it1108_block_name": str(item.get("it1108_block_name") or ctx["it1108_block_name"]),
                "it1108_lessons": str(item.get("it1108_lessons") or ctx["it1108_lessons"]),
                "it1108_weeks": str(item.get("it1108_weeks") or ctx["it1108_weeks"]),
                "it1108_cdr": str(item.get("it1108_cdr") or ctx["it1108_cdr"]),
                "vle_resources_count": vle_counts.get(code, 0),
                "total_mapped_assessments": item.get("total_mapped_assessments"),
                "submitted_assessments": item.get("submitted_assessments"),
                "completion_rate": item.get("completion_rate"),
                "avg_submitted_score": item.get("avg_submitted_score"),
                "block_weighted_score": item.get("block_weighted_score"),
                "lower_than_class_pct": item.get("lower_than_class_pct"),
                "submission_vs_score_status": str(item.get("submission_vs_score_status") or coarse_status),
                "normalized_status": normalize_block_status(item.get("submission_vs_score_status") or coarse_status),
            })
    elif oulad_student_id is not None:
        block_data_source = "oulad_zip_assessments"
        engine = analyzer or OULADBlockAnalyzer()
        oulad_blocks = engine.get_student_block_records(
            id_student=int(oulad_student_id),
            cutoff_day=cutoff_day,
            weak_only=True,
        )
        for item in oulad_blocks:
            code = item["it1108_block_code"]
            item_copy = dict(item)
            item_copy["vle_resources_count"] = vle_counts.get(code, 0)
            item_copy["normalized_status"] = normalize_block_status(item["submission_vs_score_status"])
            resolved_blocks.append(item_copy)
    elif target_block_codes:
        block_data_source = "target_block_codes_only_no_metrics"
        for raw_code in target_block_codes:
            code = str(raw_code).strip()
            if code not in curriculum_blocks:
                raise ValueError(f"Mã khối không hợp lệ trong target_block_codes: '{code}'.")
            ctx = curriculum_blocks[code]
            resolved_blocks.append({
                "it1108_block_code": code,
                "it1108_block_name": ctx["it1108_block_name"],
                "it1108_lessons": ctx["it1108_lessons"],
                "it1108_weeks": ctx["it1108_weeks"],
                "it1108_cdr": ctx["it1108_cdr"],
                "vle_resources_count": vle_counts.get(code, 0),
                "total_mapped_assessments": None,
                "submitted_assessments": None,
                "completion_rate": None,
                "avg_submitted_score": None,
                "block_weighted_score": None,
                "lower_than_class_pct": None,
                "submission_vs_score_status": None,
                "normalized_status": None,
            })

    has_detailed_block_data = bool(resolved_blocks) and all(_has_real_block_detail(b) for b in resolved_blocks)
    has_class_percentile = any(b.get("lower_than_class_pct") is not None for b in resolved_blocks)

    return {
        "simulated_student_code": sim_code,
        "risk_probability": ml_result["risk_probability"],
        "risk_level": ml_result["risk_level"],
        "decision_threshold": ml_result["decision_threshold"],
        "submission_behavior": behavior,
        "weak_blocks": resolved_blocks,
        "has_detailed_block_data": has_detailed_block_data,
        "has_class_percentile": has_class_percentile,
        "block_data_source": block_data_source,
    }
