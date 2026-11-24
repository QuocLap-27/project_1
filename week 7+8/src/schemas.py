# -*- coding: utf-8 -*-
"""
Module: schemas.py
Dự án: SMART LMS & GENERATIVE AI (Tuần 7+8 - Kiểm thử GenAI)
Chức năng:
- Định nghĩa JSON Output Schema chuẩn cho Gemini API (responseSchema) với 6 trường cốt lõi
  trên mỗi câu hỏi: question, options, correct_answer, explanation, topic, difficulty
  (cùng các trường sư phạm mở rộng: question_number, it1108_block_code, question_type, code_snippet, hint).
- Kiểm tra và chuẩn hóa JSON trả về, đảm bảo tương thích CHECK (ISJSON(quiz_content) > 0) của SQL Server.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ALLOWED_SCENARIOS = ("baseline", "risk_level", "detailed_profile")
ALLOWED_DIFFICULTIES = ("easy", "medium", "hard")
ALLOWED_QUESTION_TYPES = ("concept", "code_trace", "bug_analysis", "guided_basic")
ALLOWED_OPTION_KEYS = ("A", "B", "C", "D")
VALID_IT1108_BLOCKS = (
    "IT1108_KB01",
    "IT1108_KB02",
    "IT1108_KB03",
    "IT1108_KB04",
    "IT1108_KB05",
    "IT1108_KB06",
)

REQUIRED_TOP_LEVEL_KEYS = (
    "course_code",
    "course_name",
    "scenario_type",
    "target_blocks",
    "difficulty_strategy",
    "study_advice",
    "questions",
)

CORE_QUESTION_KEYS = (
    "question",
    "options",
    "correct_answer",
    "explanation",
    "topic",
    "difficulty",
)

EXTENDED_QUESTION_KEYS = (
    "question_number",
    "it1108_block_code",
    "question_type",
    "code_snippet",
    "hint",
)


def get_gemini_response_schema() -> Dict[str, Any]:
    """
    Trả về JSON Schema rút gọn theo quy ước Google Gemini API / Vertex AI (bỏ mô tả dài để
    giảm thời gian giải mã ràng buộc, đồng thời bảo đảm đủ cấu trúc 6 trường bắt buộc).
    """
    return {
        "type": "OBJECT",
        "required": list(REQUIRED_TOP_LEVEL_KEYS),
        "properties": {
            "course_code": {"type": "STRING"},
            "course_name": {"type": "STRING"},
            "scenario_type": {"type": "STRING", "enum": list(ALLOWED_SCENARIOS)},
            "target_blocks": {
                "type": "ARRAY",
                "items": {"type": "STRING", "enum": list(VALID_IT1108_BLOCKS)},
            },
            "review_topics": {"type": "ARRAY", "items": {"type": "STRING"}},
            "difficulty_strategy": {"type": "STRING"},
            "study_advice": {"type": "STRING"},
            "questions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "required": list(CORE_QUESTION_KEYS) + list(EXTENDED_QUESTION_KEYS),
                    "properties": {
                        "question_number": {"type": "INTEGER"},
                        "it1108_block_code": {"type": "STRING", "enum": list(VALID_IT1108_BLOCKS)},
                        "topic": {"type": "STRING"},
                        "difficulty": {"type": "STRING", "enum": list(ALLOWED_DIFFICULTIES)},
                        "question_type": {"type": "STRING", "enum": list(ALLOWED_QUESTION_TYPES)},
                        "question": {"type": "STRING"},
                        "code_snippet": {"type": "STRING"},
                        "options": {
                            "type": "OBJECT",
                            "required": list(ALLOWED_OPTION_KEYS),
                            "properties": {
                                "A": {"type": "STRING"},
                                "B": {"type": "STRING"},
                                "C": {"type": "STRING"},
                                "D": {"type": "STRING"},
                            },
                        },
                        "correct_answer": {"type": "STRING", "enum": list(ALLOWED_OPTION_KEYS)},
                        "hint": {"type": "STRING"},
                        "explanation": {"type": "STRING"},
                    },
                },
            },
        },
    }


def extract_json_text(raw_text: str) -> str:
    """Bóc tách sạch nội dung JSON nếu model trả về kèm khối markdown ```json ... ```."""
    if not isinstance(raw_text, str):
        return ""
    cleaned = raw_text.strip()
    if not cleaned:
        return ""

    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    if not cleaned.startswith("{"):
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]
    return cleaned


@dataclass
class QuizValidationResult:
    """Kết quả kiểm tra định dạng JSON và cấu trúc Schema của bộ câu hỏi AI."""

    is_valid: bool
    json_valid: bool
    schema_valid: bool
    sql_isjson_compatible: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    normalized_data: Optional[Dict[str, Any]] = None
    valid_questions: List[Dict[str, Any]] = field(default_factory=list)
    invalid_question_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "json_valid": self.json_valid,
            "schema_valid": self.schema_valid,
            "sql_isjson_compatible": self.sql_isjson_compatible,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "normalized_data": self.normalized_data,
            "valid_questions": list(self.valid_questions),
            "invalid_question_indices": list(self.invalid_question_indices),
        }


def _normalize_options_field(
    raw_options: Any,
    prefix: str,
    errors: List[str],
) -> Dict[str, str]:
    normalized_options: Dict[str, str] = {}
    if isinstance(raw_options, list):
        if len(raw_options) != 4:
            errors.append(f"{prefix}.options dạng mảng phải có đúng 4 phương án lựa chọn (hiện có {len(raw_options)}).")
            return {}
        raw_dict = {key: raw_options[i] for i, key in enumerate(ALLOWED_OPTION_KEYS)}
    elif isinstance(raw_options, dict):
        raw_dict = {str(k).strip().upper(): v for k, v in raw_options.items()}
    else:
        errors.append(f"{prefix}.options phải là object chứa 4 khóa A, B, C, D hoặc mảng 4 phần tử.")
        return {}

    missing_opts = [k for k in ALLOWED_OPTION_KEYS if k not in raw_dict]
    if missing_opts:
        errors.append(f"{prefix}.options thiếu phương án: {missing_opts}.")
        return {}

    seen_texts = set()
    for opt_key in ALLOWED_OPTION_KEYS:
        opt_val = str(raw_dict.get(opt_key, "")).strip()
        if not opt_val:
            errors.append(f"{prefix}.options.{opt_key} không được rỗng.")
        else:
            norm_opt = re.sub(r"\s+", " ", opt_val.lower())
            if norm_opt in seen_texts:
                errors.append(f"{prefix}.options có nội dung đáp án bị trùng lặp tại '{opt_key}'.")
            seen_texts.add(norm_opt)
        normalized_options[opt_key] = opt_val
    return normalized_options


def _validate_single_question(
    q: Any,
    idx: int,
    fallback_block_code: str,
    errors: List[str],
    warnings: List[str],
) -> Optional[Dict[str, Any]]:
    prefix = f"questions[{idx}]"
    if not isinstance(q, dict):
        errors.append(f"{prefix} phải là object JSON.")
        return None

    # Chuẩn hóa bí danh nhẹ (nếu có) về 6 trường chuẩn Prompt 2: question, options, correct_answer, explanation, topic, difficulty
    question_raw = q.get("question")
    if question_raw is None and "question_text" in q:
        question_raw = q.get("question_text")
        warnings.append(f"{prefix}: đã chuẩn hóa khóa 'question_text' -> 'question'.")

    topic_raw = q.get("topic")
    if topic_raw is None and "it1108_block_name" in q:
        topic_raw = q.get("it1108_block_name")
        warnings.append(f"{prefix}: đã chuẩn hóa khóa 'it1108_block_name' -> 'topic'.")

    for core_key in CORE_QUESTION_KEYS:
        if core_key == "question" and question_raw is not None:
            continue
        if core_key == "topic" and topic_raw is not None:
            continue
        if core_key not in q:
            errors.append(f"{prefix} thiếu trường bắt buộc: '{core_key}'.")

    q_num = q.get("question_number", idx + 1)
    if not isinstance(q_num, int) or isinstance(q_num, bool):
        errors.append(f"{prefix}.question_number phải là số nguyên.")
    elif q_num != idx + 1:
        errors.append(f"{prefix}.question_number={q_num} không khớp thứ tự kỳ vọng {idx + 1}.")

    block_code = str(q.get("it1108_block_code") or fallback_block_code).strip()
    if block_code not in VALID_IT1108_BLOCKS:
        errors.append(f"{prefix}.it1108_block_code='{block_code}' không nằm trong {VALID_IT1108_BLOCKS}.")

    topic = str(topic_raw or "").strip()
    if not topic:
        errors.append(f"{prefix}.topic không được rỗng.")

    difficulty = str(q.get("difficulty", "")).strip().lower()
    if difficulty not in ALLOWED_DIFFICULTIES:
        errors.append(f"{prefix}.difficulty='{difficulty}' không hợp lệ (chỉ nhận {ALLOWED_DIFFICULTIES}).")

    q_type = str(q.get("question_type") or "concept").strip().lower()
    if q_type not in ALLOWED_QUESTION_TYPES:
        errors.append(f"{prefix}.question_type='{q_type}' không hợp lệ (chỉ nhận {ALLOWED_QUESTION_TYPES}).")

    q_text = str(question_raw or "").strip()
    if len(q_text) < 10:
        errors.append(f"{prefix}.question quá ngắn hoặc bị rỗng.")

    raw_snippet = q.get("code_snippet", "")
    code_snippet = "" if raw_snippet is None else str(raw_snippet).strip()
    if q_type in ("code_trace", "bug_analysis") and not code_snippet:
        warnings.append(f"{prefix} có question_type='{q_type}' nhưng code_snippet đang rỗng.")

    normalized_options = _normalize_options_field(q.get("options"), prefix, errors)

    correct_answer = str(q.get("correct_answer", "")).strip().upper()
    if len(correct_answer) == 2 and correct_answer.endswith(".") and correct_answer[0] in ALLOWED_OPTION_KEYS:
        correct_answer = correct_answer[0]
    if correct_answer not in ALLOWED_OPTION_KEYS:
        errors.append(f"{prefix}.correct_answer='{correct_answer}' phải là một trong {ALLOWED_OPTION_KEYS}.")

    raw_hint = q.get("hint", "")
    hint = "" if raw_hint is None else str(raw_hint).strip()
    if not hint:
        warnings.append(f"{prefix}.hint đang rỗng.")

    explanation = str(q.get("explanation", "")).strip()
    if len(explanation) < 15:
        errors.append(f"{prefix}.explanation quá ngắn (< 15 ký tự) hoặc bị rỗng.")

    if errors:
        return None

    return {
        "question_number": q_num,
        "it1108_block_code": block_code,
        "topic": topic,
        "difficulty": difficulty,
        "question_type": q_type,
        "question": q_text,
        "code_snippet": code_snippet,
        "options": normalized_options,
        "correct_answer": correct_answer,
        "hint": hint,
        "explanation": explanation,
    }


def parse_and_validate_quiz_json(
    raw_text: str,
    expected_num_questions: int = 5,
    expected_scenario: Optional[str] = None,
    expected_course_code: str = "IT1108",
) -> QuizValidationResult:
    """
    Phân tích chuỗi trả về từ LLM và kiểm tra chặt toàn bộ cấu trúc JSON Schema.
    Mỗi câu hỏi chuẩn hóa có đầy đủ 6 khóa bắt buộc:
    question, options (4 lựa chọn A-D), correct_answer, explanation, topic, difficulty.
    """
    errors: List[str] = []
    warnings: List[str] = []

    extracted = extract_json_text(raw_text)
    if not extracted:
        return QuizValidationResult(
            is_valid=False,
            json_valid=False,
            schema_valid=False,
            sql_isjson_compatible=False,
            errors=["Phản hồi từ AI rỗng, không chứa dữ liệu JSON."],
        )

    try:
        data = json.loads(extracted)
    except json.JSONDecodeError as exc:
        return QuizValidationResult(
            is_valid=False,
            json_valid=False,
            schema_valid=False,
            sql_isjson_compatible=False,
            errors=[f"Lỗi cú pháp JSON (DecodeError tại dòng {exc.lineno}, cột {exc.colno}): {exc.msg}"],
        )

    if not isinstance(data, dict):
        return QuizValidationResult(
            is_valid=False,
            json_valid=True,
            schema_valid=False,
            sql_isjson_compatible=False,
            errors=["JSON gốc phải là một object ({...}), không phải array hoặc giá trị đơn."],
        )

    for req_key in ("course_code", "scenario_type", "target_blocks", "difficulty_strategy", "study_advice", "questions"):
        if req_key not in data:
            errors.append(f"Thiếu trường cấp cao bắt buộc: '{req_key}'.")

    course_code = str(data.get("course_code", "")).strip().upper()
    target_course = (expected_course_code or "IT1108").strip().upper()
    if course_code != target_course:
        errors.append(f"course_code='{course_code}' không hợp lệ (kỳ vọng '{target_course}').")

    course_name = str(data.get("course_name") or "Nhập môn lập trình").strip()

    scenario_type = str(data.get("scenario_type", "")).strip()
    if scenario_type not in ALLOWED_SCENARIOS:
        errors.append(f"scenario_type='{scenario_type}' không hợp lệ (chỉ nhận {ALLOWED_SCENARIOS}).")
    elif expected_scenario and scenario_type != expected_scenario:
        warnings.append(f"scenario_type='{scenario_type}' khác với kịch bản yêu cầu '{expected_scenario}'.")

    target_blocks = data.get("target_blocks")
    normalized_blocks: List[str] = []
    if not isinstance(target_blocks, list) or len(target_blocks) == 0:
        errors.append("target_blocks phải là danh sách không rỗng các mã khối IT1108.")
    else:
        for blk in target_blocks:
            blk_str = str(blk).strip()
            if blk_str not in VALID_IT1108_BLOCKS:
                errors.append(f"Mã khối '{blk_str}' trong target_blocks không hợp lệ.")
            elif blk_str not in normalized_blocks:
                normalized_blocks.append(blk_str)

    raw_review_topics = data.get("review_topics")
    review_topics: List[str] = []
    if isinstance(raw_review_topics, list):
        review_topics = [str(t).strip() for t in raw_review_topics if str(t).strip()]

    difficulty_strategy = str(data.get("difficulty_strategy", "")).strip()
    if not difficulty_strategy:
        errors.append("difficulty_strategy không được rỗng.")

    study_advice = str(data.get("study_advice", "")).strip()
    if not study_advice:
        errors.append("study_advice không được rỗng.")

    fallback_blk = normalized_blocks[0] if normalized_blocks else VALID_IT1108_BLOCKS[0]
    questions_raw = data.get("questions")
    normalized_questions: List[Dict[str, Any]] = []
    invalid_question_indices: List[int] = []
    if not isinstance(questions_raw, list):
        errors.append("questions phải là một mảng (array) chứa các câu hỏi.")
    else:
        if len(questions_raw) != expected_num_questions:
            errors.append(f"Số lượng câu hỏi là {len(questions_raw)}, kỳ vọng đúng {expected_num_questions} câu.")
        for idx, q in enumerate(questions_raw):
            q_errors: List[str] = []
            norm_q = _validate_single_question(q, idx, fallback_blk, q_errors, warnings)
            errors.extend(q_errors)
            if norm_q is not None:
                normalized_questions.append(norm_q)
            else:
                invalid_question_indices.append(idx)

    schema_valid = len(errors) == 0
    normalized_data: Optional[Dict[str, Any]] = None
    sql_isjson_compatible = False

    if schema_valid:
        if not review_topics:
            review_topics = list(dict.fromkeys(q["topic"] for q in normalized_questions))
        normalized_data = {
            "course_code": course_code,
            "course_name": course_name,
            "scenario_type": scenario_type,
            "target_blocks": normalized_blocks,
            "review_topics": review_topics,
            "difficulty_strategy": difficulty_strategy,
            "study_advice": study_advice,
            "questions": normalized_questions,
        }
        try:
            serialized = json.dumps(normalized_data, ensure_ascii=False)
            round_trip = json.loads(serialized)
            sql_isjson_compatible = isinstance(round_trip, dict) and len(serialized) > 0
        except (TypeError, ValueError) as exc:
            errors.append(f"Không thể tuần tự hóa JSON cho SQL Server ISJSON: {exc}")
            schema_valid = False
            normalized_data = None

    return QuizValidationResult(
        is_valid=schema_valid and sql_isjson_compatible,
        json_valid=True,
        schema_valid=schema_valid,
        sql_isjson_compatible=sql_isjson_compatible,
        errors=errors,
        warnings=warnings,
        normalized_data=normalized_data,
        valid_questions=normalized_questions,
        invalid_question_indices=invalid_question_indices,
    )
