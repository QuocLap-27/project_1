# -*- coding: utf-8 -*-
"""
Module: prompt_builder.py
Dự án: SMART LMS & GENERATIVE AI (Tuần 7+8 - Kiểm thử GenAI)
Chức năng:
- Tái sử dụng hàm build_prompt_scenarios() từ week 5+6/A_source_code/predict.py (không sửa file Tuần 5+6).
- Làm giàu ngữ cảnh học phần IT1108 – Nhập môn lập trình từ curriculum_topics.csv (read-only).
- Hỗ trợ sinh Prompt với đầy đủ 7 tham số: mã học phần, tên học phần, khối kiến thức IT1108,
  chủ đề cần ôn tập, số lượng câu hỏi, mức độ khó, và kịch bản Prompt.
- Tuân thủ nguyên tắc: không chứa dữ liệu cá nhân/nhạy cảm (NFR-02) và KHÔNG để AI tự chẩn đoán
  hay kết luận tuyệt đối về năng lực của người học.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .config import WEEK56_PREDICT_PY
from .curriculum_loader import get_block_context, load_curriculum_blocks
from .pii_guard import sanitize_profile_for_prompt, scan_prompt_for_pii
from .schemas import ALLOWED_DIFFICULTIES, ALLOWED_SCENARIOS, VALID_IT1108_BLOCKS

REUSED_WEEK56_DIRECTLY: bool = False
_week56_build_fn = None

if WEEK56_PREDICT_PY.exists():
    try:
        _spec = importlib.util.spec_from_file_location("week56_predict_module", str(WEEK56_PREDICT_PY))
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _predictor_cls = getattr(_mod, "StudentRiskPredictor", None)
            if _predictor_cls is not None and hasattr(_predictor_cls, "build_prompt_scenarios"):
                _week56_build_fn = _predictor_cls.build_prompt_scenarios
                REUSED_WEEK56_DIRECTLY = True
    except Exception:
        REUSED_WEEK56_DIRECTLY = False
        _week56_build_fn = None


SYSTEM_INSTRUCTION_IT1108: str = """Bạn là Cố vấn Học tập AI và Giảng viên hỗ trợ môn IT1108 – Nhập môn lập trình (ngôn ngữ lập trình C).
Nhiệm vụ của bạn là tạo bộ câu hỏi trắc nghiệm ôn tập bằng tiếng Việt giúp người học củng cố kiến thức đúng trọng tâm.

QUY TẮC BẮT BUỘC:
1. Chỉ trả về DUY NHẤT một JSON object hợp lệ (không bọc markdown ```json, không thêm văn bản ngoài JSON).
2. Toàn bộ đoạn mã minh họa (code_snippet) phải viết bằng ngôn ngữ C chuẩn (ANSI C / C99), biên dịch hợp lệ, không dùng thư viện không chuẩn như <conio.h> hay void main().
3. Mỗi câu hỏi bắt buộc có đủ 6 khóa: question, options (đúng 4 phương án A, B, C, D phân biệt rõ ràng, chỉ 1 đáp án đúng), correct_answer, explanation, topic, difficulty (easy/medium/hard), cùng các khóa bổ trợ trong Schema.
4. Mỗi trường văn bản viết súc tích, đúng trọng tâm để bảo đảm phản hồi nhanh dưới 20 giây: `question` 1–2 câu rõ ràng, `code_snippet` ngắn gọn tối đa 6 dòng C (hoặc chuỗi rỗng "" nếu không cần), `hint` đúng 1 câu ngắn, `explanation` 1–2 câu giải thích rõ đáp án đúng và lỗi sai chính, `difficulty_strategy` và `study_advice` mỗi trường 1–2 câu.
5. Tuyệt đối KHÔNG đề cập hoặc tự suy đoán thông tin cá nhân của sinh viên (họ tên, MSSV, email, ID, dữ liệu nhạy cảm).
6. KHÔNG tự chẩn đoán, phán xét hay đưa ra kết luận tuyệt đối về năng lực hoặc khả năng qua môn của sinh viên; chỉ dùng các chỉ số mô phỏng như ngữ cảnh tham khảo để điều chỉnh độ khó câu hỏi và đưa ra lời khuyên ôn tập mang tính hỗ trợ."""


_STATUS_ALIASES = {
    "NO_SUBMISSION": "MISSING_SUBMISSION",
    "MISSING_SUBMISSION": "MISSING_SUBMISSION",
    "SUBMITTED_LOW_SCORE": "LOW_SCORE_DESPITE_SUBMISSION",
    "LOW_SCORE_DESPITE_SUBMISSION": "LOW_SCORE_DESPITE_SUBMISSION",
    "PARTIAL_SUBMISSION_LOW_SCORE": "PARTIAL_SUBMISSION_AND_LOW_SCORE",
    "PARTIAL_SUBMISSION_AND_LOW_SCORE": "PARTIAL_SUBMISSION_AND_LOW_SCORE",
    "PARTIAL_SUBMISSION_OK_SCORE": "PARTIAL_SUBMISSION_ONLY",
    "PARTIAL_SUBMISSION_ONLY": "PARTIAL_SUBMISSION_ONLY",
    "NORMAL": "NORMAL",
}


@dataclass
class QuizGenerationRequest:
    """Cấu trúc 7 tham số sinh câu hỏi + ngữ cảnh học tập mô phỏng (ẩn danh)."""

    course_code: str = "IT1108"
    course_name: str = "Nhập môn lập trình"
    knowledge_block: Optional[str] = None
    review_topics: List[str] = field(default_factory=list)
    num_questions: int = 5
    difficulty: str = "auto"
    scenario_type: str = "detailed_profile"
    risk_level: str = "MEDIUM"
    risk_probability: float = 0.50
    weak_blocks: List[Dict[str, Any]] = field(default_factory=list)


def normalize_block_status(raw_status: Optional[str]) -> str:
    """Chuẩn hóa mã trạng thái nộp bài/điểm số (hỗ trợ cả nhãn predict.py và Stored Procedure Tuần 6)."""
    text = (raw_status or "").strip().upper()
    for key, canonical in _STATUS_ALIASES.items():
        if key in text:
            return canonical
    return "LOW_SCORE_DESPITE_SUBMISSION"


def normalize_review_topics(review_topics: Optional[Union[str, List[str]]]) -> List[str]:
    """Chuẩn hóa tham số chủ đề cần ôn tập về danh sách chuỗi sạch."""
    if review_topics is None:
        return []
    if isinstance(review_topics, str):
        parts = [p.strip() for p in review_topics.replace(";", ",").split(",")]
        return [p for p in parts if p]
    if isinstance(review_topics, list):
        return [str(p).strip() for p in review_topics if str(p).strip()]
    return []


def _legacy_build_prompt_scenarios(
    risk_prob: float,
    risk_level: str,
    weak_blocks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """Bản sao tương thích chính xác hàm StudentRiskPredictor.build_prompt_scenarios() từ week 5+6/predict.py."""
    if not weak_blocks:
        weak_blocks = [
            {
                "block_name": "Khối 3 – Cấu trúc điều khiển và vòng lặp",
                "detail": "thấp hơn 85% lớp, chưa nộp bài kiểm tra định kỳ",
                "type": "NO_SUBMISSION",
            },
            {
                "block_name": "Khối 4 – Hàm và mảng",
                "detail": "điểm thấp (42/100) dù đã nộp bài",
                "type": "SUBMITTED_LOW_SCORE",
            },
        ]

    topics_text = ", ".join([b["block_name"] for b in weak_blocks])
    risk_text = "cao" if risk_level == "HIGH" else ("trung bình" if risk_level == "MEDIUM" else "thấp")

    p_baseline = (
        f"Hãy tạo 5 câu hỏi trắc nghiệm ôn tập cho học phần IT1108 – Nhập môn lập trình (ngôn ngữ C) "
        f"thuộc nội dung: {topics_text}. Mỗi câu có 4 đáp án A, B, C, D kèm giải thích ngắn gọn và trả về định dạng JSON."
    )
    p_risk = (
        f"Sinh viên này có nguy cơ trượt môn {risk_text} (xác suất {risk_prob:.2f}). "
        f"Hãy điều chỉnh độ khó và cách giải thích cho phù hợp khi tạo 5 câu hỏi trắc nghiệm ôn tập học phần "
        f"IT1108 – Nhập môn lập trình về: {topics_text}. Trả về kết quả ở định dạng JSON."
    )
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
        "detailed_profile": p_detailed,
    }


def invoke_week56_build_prompt_scenarios(
    risk_prob: float,
    risk_level: str,
    weak_blocks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """Gọi hàm build_prompt_scenarios() của Tuần 5+6 (ưu tiên import trực tiếp từ predict.py)."""
    if _week56_build_fn is not None:
        return _week56_build_fn(risk_prob, risk_level, weak_blocks)
    return _legacy_build_prompt_scenarios(risk_prob, risk_level, weak_blocks)


def _enrich_weak_blocks(
    raw_blocks: Optional[List[Dict[str, Any]]],
    knowledge_block: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Chuẩn hóa và ghép ngữ cảnh từ curriculum_topics.csv cho từng khối kiến thức IT1108."""
    curriculum = load_curriculum_blocks(include_general=False)
    target_kb = (knowledge_block or "").strip()

    if target_kb:
        matched = [b for b in (raw_blocks or []) if str(b.get("it1108_block_code") or "").strip() == target_kb]
        if matched:
            raw_blocks = matched
        else:
            raw_blocks = [
                {
                    "it1108_block_code": target_kb,
                    "submission_vs_score_status": "LOW_SCORE_DESPITE_SUBMISSION",
                }
            ]
    elif not raw_blocks:
        raw_blocks = [
            {
                "it1108_block_code": "IT1108_KB03",
                "lower_than_class_pct": 85.0,
                "avg_submitted_score": 0.0,
                "submission_vs_score_status": "MISSING_SUBMISSION",
            },
            {
                "it1108_block_code": "IT1108_KB04",
                "lower_than_class_pct": 68.0,
                "avg_submitted_score": 42.0,
                "submission_vs_score_status": "LOW_SCORE_DESPITE_SUBMISSION",
            },
        ]

    enriched: List[Dict[str, Any]] = []
    for item in raw_blocks:
        block_code = str(item.get("it1108_block_code") or "").strip()
        if block_code not in VALID_IT1108_BLOCKS:
            raise ValueError(f"Mã khối IT1108 không hợp lệ: '{block_code}'. Chỉ chấp nhận {VALID_IT1108_BLOCKS}.")

        ctx = get_block_context(block_code) or curriculum.get(block_code, {})
        block_name = str(item.get("it1108_block_name") or ctx.get("it1108_block_name") or block_code).strip()
        lessons = str(item.get("it1108_lessons") or ctx.get("it1108_lessons") or "").strip()
        cdr = str(item.get("it1108_cdr") or ctx.get("it1108_cdr") or "").strip()
        weeks = str(item.get("it1108_weeks") or ctx.get("it1108_weeks") or "").strip()

        status = normalize_block_status(item.get("submission_vs_score_status") or item.get("type"))
        lower_pct = item.get("lower_than_class_pct")
        avg_score = item.get("avg_submitted_score")

        detail_text = str(item.get("detail") or "").strip()
        if not detail_text:
            parts: List[str] = []
            if lower_pct is not None:
                parts.append(f"thấp hơn {float(lower_pct):.1f}% lớp")
            if status == "MISSING_SUBMISSION":
                parts.append("chưa nộp bài kiểm tra ở mốc đánh giá")
            elif status == "LOW_SCORE_DESPITE_SUBMISSION":
                if avg_score is not None:
                    parts.append(f"điểm bài tập {float(avg_score):.1f}/100 dù đã nộp bài")
                else:
                    parts.append("điểm bài tập thấp dù đã nộp bài")
            elif status == "PARTIAL_SUBMISSION_AND_LOW_SCORE":
                if avg_score is not None:
                    parts.append(f"nộp thiếu bài và điểm trung bình {float(avg_score):.1f}/100")
                else:
                    parts.append("nộp thiếu bài và điểm trung bình chưa đạt ngưỡng")
            elif status == "PARTIAL_SUBMISSION_ONLY":
                if avg_score is not None:
                    parts.append(f"điểm bài đã nộp đạt {float(avg_score):.1f}/100 nhưng còn bỏ sót bài")
                else:
                    parts.append("điểm bài đã nộp đạt yêu cầu nhưng còn bỏ sót bài")
            else:
                if avg_score is not None:
                    parts.append(f"nộp bài đầy đủ, điểm trung bình {float(avg_score):.1f}/100")
                else:
                    parts.append("nộp bài đầy đủ, điểm đạt yêu cầu")
            detail_text = ", ".join(parts)

        sim_titles = [
            a.get("it1108_simulated_assessment_title", "")
            for a in ctx.get("simulated_assessments", [])
            if a.get("it1108_simulated_assessment_title")
        ]

        enriched.append({
            "it1108_block_code": block_code,
            "it1108_block_name": block_name,
            "it1108_lessons": lessons,
            "it1108_weeks": weeks,
            "it1108_cdr": cdr,
            "simulated_assessments": sim_titles,
            "lower_than_class_pct": float(lower_pct) if lower_pct is not None else None,
            "avg_submitted_score": float(avg_score) if avg_score is not None else None,
            "submission_vs_score_status": status,
            "detail": detail_text,
        })
    return enriched


def format_weak_block_for_week56(block: Dict[str, Any]) -> Dict[str, Any]:
    """Chuyển đổi block đã chuẩn hóa sang dict đầu vào của StudentRiskPredictor.build_prompt_scenarios()."""
    status = normalize_block_status(block.get("submission_vs_score_status"))
    legacy_type = "NO_SUBMISSION" if status == "MISSING_SUBMISSION" else (
        "SUBMITTED_LOW_SCORE" if status == "LOW_SCORE_DESPITE_SUBMISSION" else status
    )
    return {
        "block_name": f"{block['it1108_block_code']} – {block['it1108_block_name']}",
        "detail": block.get("detail", ""),
        "type": legacy_type,
    }


def _build_json_contract_block(
    course_code: str,
    course_name: str,
    scenario_type: str,
    target_blocks: List[Dict[str, Any]],
    review_topics: List[str],
    num_questions: int,
) -> str:
    codes_json = json.dumps([b["it1108_block_code"] for b in target_blocks], ensure_ascii=False)
    topics_json = json.dumps(review_topics, ensure_ascii=False)
    block_ref_lines = ", ".join(
        [f"{b['it1108_block_code']}='{b['it1108_block_name']}'" for b in target_blocks]
    )
    sample_q = {
        "question_number": 1,
        "it1108_block_code": target_blocks[0]["it1108_block_code"],
        "topic": review_topics[0] if review_topics else target_blocks[0]["it1108_block_name"],
        "difficulty": "easy | medium | hard",
        "question_type": "concept | code_trace | bug_analysis | guided_basic",
        "question": "1-2 câu ngắn gọn",
        "code_snippet": "tối đa 6 dòng C hoặc chuỗi rỗng",
        "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
        "correct_answer": "A | B | C | D",
        "hint": "1 câu gợi ý ngắn",
        "explanation": "1-2 câu giải thích trọng tâm",
    }
    sample_payload = {
        "course_code": course_code,
        "course_name": course_name,
        "scenario_type": scenario_type,
        "target_blocks": [b["it1108_block_code"] for b in target_blocks],
        "review_topics": review_topics,
        "difficulty_strategy": "1-2 câu chiến lược độ khó",
        "study_advice": "1-2 câu lời khuyên ôn tập hỗ trợ",
        "questions": [sample_q],
    }
    return (
        f"ĐỊNH DẠNG JSON BẮT BUỘC (đúng {num_questions} câu, đủ 6 khóa question/options/correct_answer/explanation/topic/difficulty):\n"
        f"- Mã khối hợp lệ {codes_json}: {block_ref_lines}.\n"
        f"- Chủ đề mục tiêu: {topics_json}.\n"
        f"- Schema mẫu: {json.dumps(sample_payload, ensure_ascii=False)}"
    )


def _build_pedagogical_directives(
    blocks: List[Dict[str, Any]],
    risk_level: str,
    requested_difficulty: str,
) -> str:
    lines: List[str] = []
    for b in blocks:
        code = b["it1108_block_code"]
        name = b["it1108_block_name"]
        status = b["submission_vs_score_status"]
        detail = b["detail"]
        if status == "MISSING_SUBMISSION":
            lines.append(
                f"  * {code} ({name} | {detail}): Ưu tiên câu hỏi nền tảng ('easy', question_type='guided_basic') "
                f"kèm gợi ý từng bước trong trường 'hint' để người học dễ khởi động lại."
            )
        elif status == "LOW_SCORE_DESPITE_SUBMISSION":
            lines.append(
                f"  * {code} ({name} | {detail}): Ưu tiên question_type='code_trace' hoặc 'bug_analysis', "
                f"kèm 'code_snippet' ngôn ngữ C ngắn gọn và giải thích lần vết từng dòng lệnh trong 'explanation'."
            )
        elif status == "PARTIAL_SUBMISSION_AND_LOW_SCORE":
            lines.append(
                f"  * {code} ({name} | {detail}): Kết hợp câu 'guided_basic' (độ khó 'easy', có 'hint' từng bước) "
                f"và câu 'bug_analysis'/'code_trace' để củng cố phần kiến thức còn hổng."
            )
        elif status == "PARTIAL_SUBMISSION_ONLY":
            lines.append(
                f"  * {code} ({name} | {detail}): Ôn tập kiến thức cốt lõi mức 'medium' kèm gợi ý hoàn thiện phần bài tập còn thiếu."
            )
        else:
            lines.append(
                f"  * {code} ({name} | {detail}): Tăng cường câu hỏi vận dụng mức 'medium'/'hard' (dạng 'code_trace' hoặc 'concept')."
            )

    if requested_difficulty != "auto":
        lines.append(f"  * Lưu ý mức độ khó được chọn từ yêu cầu kiểm thử: ưu tiên mức '{requested_difficulty}'.")
    elif risk_level == "HIGH":
        lines.append("  * Lưu ý tham khảo: Nhóm có chỉ số rủi ro cao, tránh câu hỏi đánh đố xa chương trình, tập trung củng cố nền tảng.")
    lines.append("  * Lưu ý trung lập: Không tự chẩn đoán hay khẳng định tuyệt đối về năng lực/kết quả học tập của người học.")
    return "\n".join(lines)


def build_prompt_for_scenario(
    profile: Optional[Dict[str, Any]] = None,
    scenario_type: str = "detailed_profile",
    num_questions: int = 5,
    course_code: str = "IT1108",
    course_name: str = "Nhập môn lập trình",
    knowledge_block: Optional[str] = None,
    review_topics: Optional[Union[str, List[str]]] = None,
    difficulty: str = "auto",
) -> Dict[str, Any]:
    """
    Tạo trọn bộ thông tin Prompt (system_instruction + user_prompt + metadata) theo 7 tham số
    và 3 kịch bản (baseline / risk_level / detailed_profile).
    Thực hiện sanitize PII trước khi tạo prompt và quét lại trước khi trả kết quả (NFR-02).
    """
    scenario = (scenario_type or "").strip()
    if scenario not in ALLOWED_SCENARIOS:
        raise ValueError(f"Kịch bản '{scenario_type}' không hợp lệ. Chỉ chấp nhận {ALLOWED_SCENARIOS}.")

    if not isinstance(num_questions, int) or num_questions < 1 or num_questions > 20:
        raise ValueError(f"Số lượng câu hỏi không hợp lệ: {num_questions} (chấp nhận từ 1 đến 20).")

    req_diff = (difficulty or "auto").strip().lower()
    if req_diff != "auto" and req_diff not in ALLOWED_DIFFICULTIES:
        raise ValueError(f"Mức độ khó '{difficulty}' không hợp lệ. Chỉ nhận 'auto', {ALLOWED_DIFFICULTIES}.")

    norm_course_code = (course_code or "IT1108").strip().upper()
    norm_course_name = (course_name or "Nhập môn lập trình").strip()

    raw_profile = profile if isinstance(profile, dict) else {}
    clean_profile, stripped_keys, forbidden_values = sanitize_profile_for_prompt(raw_profile)

    risk_level = str(clean_profile.get("risk_level", "MEDIUM")).strip().upper()
    if risk_level not in ("LOW", "MEDIUM", "HIGH"):
        risk_level = "MEDIUM"
    risk_prob = float(clean_profile.get("risk_probability", 0.50))

    raw_blocks = clean_profile.get("weak_blocks") or clean_profile.get("target_blocks")
    enriched_blocks = _enrich_weak_blocks(raw_blocks, knowledge_block=knowledge_block)

    topics_list = normalize_review_topics(review_topics)
    if not topics_list:
        topics_list = [b["it1108_block_name"] for b in enriched_blocks]

    w56_blocks = [format_weak_block_for_week56(b) for b in enriched_blocks]
    w56_prompts = invoke_week56_build_prompt_scenarios(risk_prob, risk_level, w56_blocks)
    week56_base_prompt = w56_prompts[scenario]

    block_names_list = ", ".join([f"{b['it1108_block_code']} – {b['it1108_block_name']}" for b in enriched_blocks])
    topics_joined = ", ".join(topics_list)
    lessons_context = "\n".join(
        [
            f"- {b['it1108_block_code']} ({b['it1108_weeks']} | CĐR {b['it1108_cdr']}): {b['it1108_lessons']}"
            for b in enriched_blocks
            if b.get("it1108_lessons")
        ]
    )
    risk_vn = "cao" if risk_level == "HIGH" else ("trung bình" if risk_level == "MEDIUM" else "thấp")
    difficulty_instruction = (
        f"Mức độ khó yêu cầu: {req_diff}."
        if req_diff != "auto"
        else "Mức độ khó: tự điều chỉnh hợp lý theo ngữ cảnh ôn tập."
    )
    json_contract = _build_json_contract_block(
        course_code=norm_course_code,
        course_name=norm_course_name,
        scenario_type=scenario,
        target_blocks=enriched_blocks,
        review_topics=topics_list,
        num_questions=num_questions,
    )

    if scenario == "baseline":
        user_prompt = (
            f"KỊCH BẢN 1 (BASELINE - CHUNG CHUNG):\n"
            f"Hãy tạo {num_questions} câu hỏi trắc nghiệm ôn tập cho học phần {norm_course_code} – {norm_course_name} (ngôn ngữ C).\n"
            f"- Khối kiến thức IT1108: {block_names_list}\n"
            f"- Chủ đề cần ôn tập: {topics_joined}\n"
            f"- {difficulty_instruction}\n"
            f"Không sử dụng bất kỳ thông tin cá nhân người học nào. Mỗi câu gồm 4 phương án A, B, C, D, 1 đáp án đúng và giải thích.\n\n"
            f"{json_contract}"
        )
    elif scenario == "risk_level":
        diff_note = (
            f"ưu tiên mức độ khó '{req_diff}'"
            if req_diff != "auto"
            else "điều chỉnh độ khó và cách giải thích phù hợp để hỗ trợ củng cố kiến thức"
        )
        user_prompt = (
            f"KỊCH BẢN 2 (RISK-AWARE - KÈM MỨC RỦI RO TỪ ML):\n"
            f"Thông tin học phần: {norm_course_code} – {norm_course_name} (ngôn ngữ C).\n"
            f"Khối kiến thức IT1108: {block_names_list}\n"
            f"Chủ đề cần ôn tập: {topics_joined}\n"
            f"Ngữ cảnh tham khảo từ mô hình ML: mức rủi ro dự đoán {risk_vn.upper()} (xác suất tham khảo: {risk_prob:.2f}).\n"
            f"Hãy {diff_note} khi tạo {num_questions} câu hỏi trắc nghiệm ôn tập (không đưa ra kết luận tuyệt đối về năng lực).\n\n"
            f"PHẠM VI BÀI HỌC IT1108:\n{lessons_context}\n\n"
            f"{json_contract}"
        )
    else:
        profile_summary = "\n".join(
            [f"- {b['it1108_block_code']} – {b['it1108_block_name']}: {b['detail']}" for b in enriched_blocks]
        )
        directives = _build_pedagogical_directives(enriched_blocks, risk_level, req_diff)
        user_prompt = (
            f"KỊCH BẢN 3 (DETAILED PROFILE - HỒ SƠ HỌC TẬP MÔ PHỎNG CHI TIẾT):\n"
            f"Thông tin học phần: {norm_course_code} – {norm_course_name} (ngôn ngữ C).\n"
            f"Khối kiến thức IT1108: {block_names_list}\n"
            f"Chủ đề cần ôn tập: {topics_joined}\n"
            f"Ngữ cảnh tham khảo từ ML: mức rủi ro {risk_vn.upper()} (xác suất tham khảo: {risk_prob:.2f}).\n\n"
            f"PHẠM VI BÀI HỌC IT1108:\n{lessons_context}\n\n"
            f"TRẠNG THÁI HỌC TẬP MÔ PHỎNG TỪNG KHỐI KIẾN THỨC:\n{profile_summary}\n\n"
            f"CHỈ DẪN SƯ PHẠM HỖ TRỢ THEO TỪNG CHỦ ĐỀ:\n{directives}\n\n"
            f"Hãy tạo {num_questions} câu hỏi trắc nghiệm phân bổ hợp lý theo các chủ đề trên và trả về đúng JSON Schema.\n\n"
            f"{json_contract}"
        )

    pii_report = scan_prompt_for_pii(user_prompt, forbidden_values=forbidden_values, strict=True)

    return {
        "scenario_type": scenario,
        "system_instruction": SYSTEM_INSTRUCTION_IT1108,
        "user_prompt": user_prompt,
        "week56_base_prompt": week56_base_prompt,
        "reused_week56_directly": REUSED_WEEK56_DIRECTLY,
        "request_params": {
            "course_code": norm_course_code,
            "course_name": norm_course_name,
            "knowledge_blocks": [b["it1108_block_code"] for b in enriched_blocks],
            "review_topics": topics_list,
            "num_questions": num_questions,
            "difficulty": req_diff,
            "scenario_type": scenario,
        },
        "sanitized_profile": {
            "risk_level": risk_level,
            "risk_probability": risk_prob,
            "weak_blocks": enriched_blocks,
        },
        "pii_report": {
            "pii_safe": pii_report["pii_safe"],
            "stripped_pii_keys": stripped_keys,
            "findings": pii_report["findings"],
        },
    }


def build_all_prompt_scenarios(
    profile: Optional[Dict[str, Any]] = None,
    num_questions: int = 5,
    course_code: str = "IT1108",
    course_name: str = "Nhập môn lập trình",
    knowledge_block: Optional[str] = None,
    review_topics: Optional[Union[str, List[str]]] = None,
    difficulty: str = "auto",
) -> Dict[str, Dict[str, Any]]:
    """Tạo đồng thời cả 3 kịch bản Prompt ('baseline', 'risk_level', 'detailed_profile')."""
    return {
        sc: build_prompt_for_scenario(
            profile=profile,
            scenario_type=sc,
            num_questions=num_questions,
            course_code=course_code,
            course_name=course_name,
            knowledge_block=knowledge_block,
            review_topics=review_topics,
            difficulty=difficulty,
        )
        for sc in ALLOWED_SCENARIOS
    }
