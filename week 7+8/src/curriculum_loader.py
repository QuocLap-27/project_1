# -*- coding: utf-8 -*-
"""
Module: curriculum_loader.py
Dự án: SMART LMS & GENERATIVE AI (Tuần 7+8 - Kiểm thử GenAI)
Chức năng:
- Đọc dữ liệu Mapping từ Tuần 5+6 (chế độ read-only, tuyệt đối không sửa cấu trúc hay tên 3 bảng Mapping):
  + curriculum_topics.csv
  + assessment_topic_mapping.csv
- Cung cấp ngữ cảnh Khối kiến thức IT1108 (IT1108_KB01..IT1108_KB06) và bài kiểm tra mô phỏng
  để làm giàu Prompt sinh câu hỏi cho Gemini API.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ASSESSMENT_MAPPING_CSV, CURRICULUM_TOPICS_CSV


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_float(value: Any, default: float = 1.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def load_curriculum_blocks(
    include_general: bool = False,
    csv_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Đọc read-only bảng curriculum_topics.csv.
    Trả về dictionary ánh xạ theo mã khối IT1108 (ví dụ: 'IT1108_KB01' .. 'IT1108_KB06').
    """
    target_path = Path(csv_path) if csv_path is not None else CURRICULUM_TOPICS_CSV
    if not target_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file curriculum_topics.csv tại: {target_path}")

    blocks: Dict[str, Dict[str, Any]] = {}
    with target_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            block_code = (row.get("it1108_block_code") or "").strip()
            if not block_code:
                continue
            if not include_general and block_code == "IT1108_KB_GEN":
                continue

            blocks[block_code] = {
                "topic_id": _parse_int(row.get("topic_id")),
                "code_module": (row.get("code_module") or "").strip(),
                "code_presentation": (row.get("code_presentation") or "").strip(),
                "topic_code": (row.get("topic_code") or "").strip(),
                "topic_name": (row.get("topic_name") or "").strip(),
                "topic_description": (row.get("topic_description") or "").strip(),
                "learning_outcomes": (row.get("learning_outcomes") or "").strip(),
                "week_from": _parse_int(row.get("week_from")),
                "week_to": _parse_int(row.get("week_to")),
                "verification_status": (row.get("verification_status") or "").strip(),
                "it1108_course_code": (row.get("it1108_course_code") or "IT1108").strip(),
                "it1108_block_code": block_code,
                "it1108_block_name": (row.get("it1108_block_name") or "").strip(),
                "it1108_lessons": (row.get("it1108_lessons") or "").strip(),
                "it1108_weeks": (row.get("it1108_weeks") or "").strip(),
                "it1108_cdr": (row.get("it1108_cdr") or "").strip(),
                "mapping_rule": (row.get("mapping_rule") or "").strip(),
            }
    return blocks


def load_assessment_mappings(
    exclude_exam: bool = True,
    csv_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Đọc read-only bảng assessment_topic_mapping.csv.
    Mặc định loại bỏ bài thi cuối kỳ (mapping_type == 'Excluded_Exam') để chống Data Leakage.
    """
    target_path = Path(csv_path) if csv_path is not None else ASSESSMENT_MAPPING_CSV
    if not target_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file assessment_topic_mapping.csv tại: {target_path}")

    mappings: List[Dict[str, Any]] = []
    with target_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping_type = (row.get("mapping_type") or "").strip()
            block_code = (row.get("it1108_block_code") or "").strip()
            if exclude_exam and (mapping_type == "Excluded_Exam" or block_code == "EXCLUDED_EXAM"):
                continue

            mappings.append({
                "mapping_id": _parse_int(row.get("mapping_id")),
                "id_assessment": _parse_int(row.get("id_assessment")),
                "topic_id": _parse_int(row.get("topic_id")),
                "mapping_type": mapping_type,
                "mapping_evidence": (row.get("mapping_evidence") or "").strip(),
                "confidence": (row.get("confidence") or "").strip(),
                "mapping_status": (row.get("mapping_status") or "").strip(),
                "it1108_block_code": block_code,
                "it1108_mapped_block_name": (row.get("it1108_mapped_block_name") or "").strip(),
                "it1108_simulated_assessment_title": (row.get("it1108_simulated_assessment_title") or "").strip(),
                "mapping_weight": _parse_float(row.get("mapping_weight"), default=1.0),
                "oulad_to_it1108_mapping_rule": (row.get("oulad_to_it1108_mapping_rule") or "").strip(),
            })
    return mappings


def get_block_context(
    block_code: str,
    curriculum_csv_path: Optional[Path] = None,
    assessment_csv_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    Gộp thông tin khối kiến thức IT1108 và danh sách bài kiểm tra mô phỏng tương ứng.
    Trả về None nếu mã khối không tồn tại.
    """
    code = (block_code or "").strip()
    if not code:
        return None

    blocks = load_curriculum_blocks(include_general=True, csv_path=curriculum_csv_path)
    block = blocks.get(code)
    if block is None:
        return None

    assessments = [
        item for item in load_assessment_mappings(exclude_exam=True, csv_path=assessment_csv_path)
        if item.get("it1108_block_code") == code
    ]
    result = dict(block)
    result["simulated_assessments"] = assessments
    return result
