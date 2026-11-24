# -*- coding: utf-8 -*-
"""
Module: evaluation/compare_prompt_scenarios.py
Dự án: SMART LMS & GENERATIVE AI (Tuần 7 – Kiểm thử & So sánh 3 kịch bản Prompt Engineering)
Chức năng:
- Script kiểm thử độc lập (không chỉnh sửa mã nguồn hệ thống chính) sử dụng bộ dữ liệu mô phỏng
  it1108_benchmark_dataset.json phủ đủ 6 khối kiến thức IT1108 (IT1108_KB01..IT1108_KB06).
- So sánh công bằng 3 kịch bản (baseline / risk_level / detailed_profile) trên cùng chủ đề và số câu hỏi.
- Kiểm tra 6 tiêu chí chất lượng bắt buộc của Tuần 7:
  1. Tính hợp lệ JSON & Schema (6 trường bắt buộc + tương thích SQL ISJSON).
  2. Đúng chủ đề / khối kiến thức IT1108.
  3. Đủ đúng 4 phương án lựa chọn A, B, C, D (không rỗng, không trùng nội dung).
  4. Có đúng 1 đáp án được đánh dấu hợp lệ.
  5. Phần giải thích phù hợp với đáp án đúng (heuristic không mâu thuẫn + liên kết nội dung).
  6. Độ khó phù hợp với thông tin đầu vào của người học.
  7. Kiểm tra trùng lặp nội dung giữa các câu hỏi trong cùng bộ đề.
- Gắn nhãn tường minh data_source ('LIVE_GEMINI_API' vs 'MOCK_SIMULATED') và is_simulated,
  xuất kết quả JSON và CSV vào week 7+8/outputs/ phục vụ báo cáo Tuần 7.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

WEEK78_DIR = Path(__file__).resolve().parent.parent
if str(WEEK78_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK78_DIR))

from src.config import OUTPUTS_DIR, resolve_mode  # noqa: E402
from src.curriculum_loader import load_curriculum_blocks  # noqa: E402
from src.gemini_client import GeminiQuizClient  # noqa: E402
from src.prompt_builder import normalize_block_status  # noqa: E402
from src.schemas import ALLOWED_OPTION_KEYS, ALLOWED_SCENARIOS  # noqa: E402

DATASET_PATH: Path = Path(__file__).resolve().parent / "it1108_benchmark_dataset.json"

_BLOCK_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "IT1108_KB01": (
        "thuật toán", "sơ đồ khối", "flowchart", "mã giả", "giải quyết bài toán",
        "tính dừng", "tính xác định", "tổng quan", "biên dịch", "input", "output",
        "lập trình", "bước", "khối 1", "kb01",
    ),
    "IT1108_KB02": (
        "printf", "scanf", "kiểu dữ liệu", "int", "float", "char", "double",
        "nhị phân", "thập phân", "hệ đếm", "toán tử", "biểu thức", "ép kiểu",
        "%d", "%f", "%c", "biến", "hằng", "vào/ra", "vào ra", "khối 2", "kb02",
    ),
    "IT1108_KB03": (
        "if", "else", "switch", "case", "while", "do...while", "do while",
        "for", "vòng lặp", "rẽ nhánh", "break", "continue", "điều kiện",
        "default", "khối 3", "kb03",
    ),
    "IT1108_KB04": (
        "hàm", "mảng", "đệ quy", "tham trị", "tham chiếu", "tham số", "đối số",
        "phạm vi biến", "cục bộ", "toàn cục", "return", "chỉ số", "chiều",
        "a[", "stack", "void", "khối 4", "kb04",
    ),
    "IT1108_KB05": (
        "con trỏ", "pointer", "malloc", "calloc", "realloc", "free", "bộ nhớ động",
        "xâu", "chuỗi", "string", "strlen", "strcpy", "strcmp", "strcat",
        "\\0", "null", "địa chỉ", "heap", "*", "&", "khối 5", "kb05",
    ),
    "IT1108_KB06": (
        "struct", "cấu trúc", "typedef", "tệp", "file", "fopen", "fclose",
        "fread", "fwrite", "fprintf", "fscanf", "fgets", "fputs",
        "nhị phân", "văn bản", "->", "eof", "khối 6", "kb06",
    ),
}

_STOPWORDS_VI: Set[str] = {
    "là", "và", "của", "cho", "trong", "khi", "các", "một", "những", "được",
    "với", "từ", "đến", "về", "này", "đó", "nên", "vì", "do", "bằng", "sau",
    "trước", "không", "có", "nào", "gì", "tại", "làm", "ra", "vào", "theo",
    "để", "nếu", "thì", "như", "hoặc", "hay", "cần", "phải", "chỉ", "dùng",
}


def _tokenize(text: str) -> Set[str]:
    raw_tokens = re.findall(r"[a-zA-Z0-9_àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+", (text or "").lower())
    return {t for t in raw_tokens if len(t) >= 2 and t not in _STOPWORDS_VI}


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _check_topic_relevance(
    question: Dict[str, Any],
    expected_blocks: Set[str],
    curriculum_blocks: Dict[str, Dict[str, Any]],
) -> bool:
    b_code = str(question.get("it1108_block_code") or "").strip()
    if b_code not in expected_blocks:
        return False

    combined = " ".join([
        str(question.get("topic") or ""),
        str(question.get("question") or ""),
        str(question.get("code_snippet") or ""),
        str(question.get("explanation") or ""),
    ]).lower()

    keywords = _BLOCK_KEYWORDS.get(b_code, ())
    if any(kw in combined for kw in keywords):
        return True

    block_name = str((curriculum_blocks.get(b_code) or {}).get("it1108_block_name") or "").lower()
    name_tokens = _tokenize(block_name)
    return bool(name_tokens and (name_tokens & _tokenize(combined)))


def _check_four_options_and_single_answer(question: Dict[str, Any]) -> Tuple[bool, bool]:
    opts = question.get("options")
    four_ok = False
    if isinstance(opts, dict) and set(opts.keys()) == set(ALLOWED_OPTION_KEYS):
        vals = [_norm_text(str(opts[k])) for k in ALLOWED_OPTION_KEYS]
        if all(vals) and len(set(vals)) == 4:
            four_ok = True

    ans_raw = str(question.get("correct_answer") or "").strip().upper()
    single_ok = ans_raw in ALLOWED_OPTION_KEYS and isinstance(opts, dict) and ans_raw in opts and bool(str(opts[ans_raw]).strip())
    return four_ok, single_ok


def _check_explanation_fit(question: Dict[str, Any]) -> bool:
    exp = str(question.get("explanation") or "").strip()
    ans = str(question.get("correct_answer") or "").strip().upper()
    opts = question.get("options") if isinstance(question.get("options"), dict) else {}
    ans_text = str(opts.get(ans) or "").strip()

    if len(exp) < 25 or ans not in ALLOWED_OPTION_KEYS or not ans_text:
        return False

    # Kiểm tra mâu thuẫn rõ ràng kiểu "đáp án đúng là B" khi correct_answer là A
    contradiction = re.search(
        r"(?:đáp án|phương án|câu trả lời)\s*(?:đúng|chính xác)\s*(?:là|:)\s*([ABCD])\b",
        exp,
        flags=re.IGNORECASE,
    )
    if contradiction and contradiction.group(1).upper() != ans:
        return False

    exp_lower = exp.lower()
    # Trực tiếp nhắc đáp án đúng hoặc trùng token kỹ thuật / số liệu với nội dung đáp án đúng
    if re.search(rf"(?:đáp án|phương án|lựa chọn)\s*{ans.lower()}\b", exp_lower):
        return True

    ans_tokens = _tokenize(ans_text)
    exp_tokens = _tokenize(exp)
    if ans_tokens & exp_tokens:
        return True

    # Hỗ trợ đáp án ngắn là số hoặc ký hiệu C (ví dụ "10", "\"r\"", "&n")
    ans_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", ans_text))
    exp_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", exp))
    if ans_nums and (ans_nums & exp_nums):
        return True

    clean_ans = ans_text.strip().lower()
    if 1 <= len(clean_ans) <= 12 and clean_ans in exp_lower:
        return True

    return False


def _score_difficulty_fit(
    question: Dict[str, Any],
    status_by_block: Dict[str, str],
    risk_level: str,
) -> float:
    diff = str(question.get("difficulty") or "").strip().lower()
    if diff not in ("easy", "medium", "hard"):
        return 0.0

    b_code = str(question.get("it1108_block_code") or "").strip()
    status = status_by_block.get(b_code, "NORMAL")

    if status == "MISSING_SUBMISSION":
        if diff == "easy":
            return 1.0
        return 0.4 if diff == "medium" else 0.0

    if status in ("LOW_SCORE_DESPITE_SUBMISSION", "PARTIAL_SUBMISSION_AND_LOW_SCORE"):
        if diff in ("easy", "medium"):
            return 1.0
        return 0.2

    if status == "PARTIAL_SUBMISSION_ONLY":
        return 1.0 if diff == "medium" else 0.6

    # NORMAL: nhóm đạt yêu cầu, phù hợp medium/hard
    if risk_level == "LOW":
        return 1.0 if diff in ("medium", "hard") else 0.5
    return 1.0 if diff in ("easy", "medium") else 0.4


def _compute_duplication_metrics(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(questions) < 2:
        return {
            "max_pairwise_similarity": 0.0,
            "avg_pairwise_similarity": 0.0,
            "duplicate_pairs": 0,
            "no_excessive_duplication": True,
        }

    sims: List[float] = []
    dup_pairs = 0
    for i in range(len(questions)):
        text_i = f"{questions[i].get('question', '')} {questions[i].get('code_snippet', '')}"
        tok_i = _tokenize(text_i)
        norm_i = _norm_text(text_i)
        for j in range(i + 1, len(questions)):
            text_j = f"{questions[j].get('question', '')} {questions[j].get('code_snippet', '')}"
            tok_j = _tokenize(text_j)
            norm_j = _norm_text(text_j)

            union = tok_i | tok_j
            jaccard = (len(tok_i & tok_j) / len(union)) if union else 0.0
            seq = SequenceMatcher(None, norm_i, norm_j).ratio()
            combined_sim = 0.5 * jaccard + 0.5 * seq
            sims.append(combined_sim)
            if combined_sim >= 0.65:
                dup_pairs += 1

    max_sim = round(max(sims), 4) if sims else 0.0
    avg_sim = round(sum(sims) / len(sims), 4) if sims else 0.0
    return {
        "max_pairwise_similarity": max_sim,
        "avg_pairwise_similarity": avg_sim,
        "duplicate_pairs": dup_pairs,
        "no_excessive_duplication": dup_pairs == 0 and max_sim < 0.65,
    }


def evaluate_case_scenario(
    case: Dict[str, Any],
    scenario_type: str,
    res: Dict[str, Any],
    curriculum_blocks: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Đánh giá đầy đủ 6 tiêu chí Prompt 3 cho 1 lần sinh đề."""
    mode_used = res.get("mode_used", "mock")
    is_simulated = mode_used != "live"
    data_source = "MOCK_SIMULATED" if is_simulated else "LIVE_GEMINI_API"

    profile = case["profile"]
    weak_blocks = profile.get("weak_blocks") or []
    expected_blocks = {b["it1108_block_code"] for b in weak_blocks}
    status_by_block = {
        b["it1108_block_code"]: normalize_block_status(b.get("submission_vs_score_status"))
        for b in weak_blocks
    }
    risk_level = str(profile.get("risk_level") or "MEDIUM").upper()

    validation = res.get("validation") or {}
    pii_report = res.get("pii_report") or {}
    quiz = res.get("quiz_data")

    record: Dict[str, Any] = {
        "case_id": case["case_id"],
        "scenario_type": scenario_type,
        "data_source": data_source,
        "is_simulated": is_simulated,
        "mode_used": mode_used,
        "api_provider": res.get("api_provider"),
        "model_used": res.get("model_used"),
        "success": bool(res.get("success")),
        "http_status": res.get("http_status"),
        "error_type": res.get("error_type"),
        "error_message": res.get("error_message"),
        "latency_ms": float(res.get("latency_ms") or 0.0),
        "within_timeout_nfr01": bool(res.get("within_timeout_nfr01")),
        "pii_safe": bool(pii_report.get("pii_safe")),
        "json_valid": bool(validation.get("json_valid")),
        "schema_valid": bool(validation.get("schema_valid")),
        "sql_isjson_compatible": bool(validation.get("sql_isjson_compatible")),
        "expected_num_questions": int(case.get("num_questions", 5)),
        "actual_num_questions": 0,
        "topic_relevance_pct": 0.0,
        "four_options_pct": 0.0,
        "single_answer_pct": 0.0,
        "explanation_fit_pct": 0.0,
        "difficulty_fit_pct": 0.0,
        "max_pairwise_similarity": 0.0,
        "avg_pairwise_similarity": 0.0,
        "duplicate_pairs": 0,
        "no_excessive_duplication": False,
        "difficulty_distribution": {"easy": 0, "medium": 0, "hard": 0},
        "avg_explanation_chars": 0.0,
        "overall_quality_score": 0.0,
    }

    if not quiz or not isinstance(quiz.get("questions"), list) or not quiz["questions"]:
        return record

    questions: List[Dict[str, Any]] = quiz["questions"]
    n_q = len(questions)
    record["actual_num_questions"] = n_q

    topic_hits = 0
    four_opts_hits = 0
    single_ans_hits = 0
    exp_fit_hits = 0
    diff_fit_scores: List[float] = []
    diff_dist = {"easy": 0, "medium": 0, "hard": 0}
    exp_lens: List[int] = []

    for q in questions:
        if _check_topic_relevance(q, expected_blocks, curriculum_blocks):
            topic_hits += 1
        four_ok, single_ok = _check_four_options_and_single_answer(q)
        if four_ok:
            four_opts_hits += 1
        if single_ok:
            single_ans_hits += 1
        if _check_explanation_fit(q):
            exp_fit_hits += 1

        diff_fit_scores.append(_score_difficulty_fit(q, status_by_block, risk_level))
        d = str(q.get("difficulty") or "").strip().lower()
        if d in diff_dist:
            diff_dist[d] += 1
        exp_lens.append(len(str(q.get("explanation") or "").strip()))

    dup_metrics = _compute_duplication_metrics(questions)
    topic_pct = round(topic_hits / n_q * 100.0, 2)
    four_pct = round(four_opts_hits / n_q * 100.0, 2)
    single_pct = round(single_ans_hits / n_q * 100.0, 2)
    exp_pct = round(exp_fit_hits / n_q * 100.0, 2)
    diff_pct = round((sum(diff_fit_scores) / n_q) * 100.0, 2)
    dup_pct = 100.0 if dup_metrics["no_excessive_duplication"] else max(0.0, 100.0 - dup_metrics["duplicate_pairs"] * 25.0)

    # Điểm tổng hợp 6 tiêu chí (thang 100):
    # JSON/Schema (20) + Đúng chủ đề (20) + 4 đáp án & 1 đáp án đúng (15) + Giải thích phù hợp (15) + Độ khó phù hợp (20) + Không trùng lặp (10)
    json_pts = 20.0 if (record["json_valid"] and record["schema_valid"]) else 0.0
    overall = round(
        json_pts
        + (topic_pct / 100.0) * 20.0
        + (((four_pct + single_pct) / 2.0) / 100.0) * 15.0
        + (exp_pct / 100.0) * 15.0
        + (diff_pct / 100.0) * 20.0
        + (dup_pct / 100.0) * 10.0,
        2,
    )

    record.update({
        "topic_relevance_pct": topic_pct,
        "four_options_pct": four_pct,
        "single_answer_pct": single_pct,
        "explanation_fit_pct": exp_pct,
        "difficulty_fit_pct": diff_pct,
        "max_pairwise_similarity": dup_metrics["max_pairwise_similarity"],
        "avg_pairwise_similarity": dup_metrics["avg_pairwise_similarity"],
        "duplicate_pairs": dup_metrics["duplicate_pairs"],
        "no_excessive_duplication": dup_metrics["no_excessive_duplication"],
        "difficulty_distribution": diff_dist,
        "avg_explanation_chars": round(sum(exp_lens) / n_q, 1),
        "overall_quality_score": overall,
    })
    return record


def summarize_by_scenario(records: List[Dict[str, Any]], effective_mode: str) -> List[Dict[str, Any]]:
    is_sim = effective_mode != "live"
    data_source = "MOCK_SIMULATED" if is_sim else "LIVE_GEMINI_API"
    summary: List[Dict[str, Any]] = []
    for sc in ALLOWED_SCENARIOS:
        rows = [r for r in records if r["scenario_type"] == sc]
        if not rows:
            continue
        n = len(rows)
        ok_rows = [r for r in rows if r["success"]]
        n_ok = len(ok_rows)

        def _avg_ok(field: str) -> float:
            if n_ok == 0:
                return 0.0
            return round(sum(float(r[field]) for r in ok_rows) / n_ok, 2)

        summary.append({
            "scenario_type": sc,
            "data_source": data_source,
            "is_simulated": is_sim,
            "total_cases": n,
            "successful_cases": n_ok,
            "json_valid_pct": round(sum(1 for r in rows if r["json_valid"]) / n * 100.0, 2),
            "schema_valid_pct": round(sum(1 for r in rows if r["schema_valid"]) / n * 100.0, 2),
            "nfr01_within_20s_pct": round(sum(1 for r in rows if r["within_timeout_nfr01"]) / n * 100.0, 2),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 2),
            "avg_topic_relevance_pct": _avg_ok("topic_relevance_pct"),
            "avg_four_options_pct": _avg_ok("four_options_pct"),
            "avg_single_answer_pct": _avg_ok("single_answer_pct"),
            "avg_explanation_fit_pct": _avg_ok("explanation_fit_pct"),
            "avg_difficulty_fit_pct": _avg_ok("difficulty_fit_pct"),
            "no_duplication_rate_pct": round(
                (sum(1 for r in ok_rows if r["no_excessive_duplication"]) / n_ok * 100.0) if n_ok else 0.0,
                2,
            ),
            "avg_max_similarity": _avg_ok("max_pairwise_similarity"),
            "avg_explanation_chars": _avg_ok("avg_explanation_chars"),
            "avg_overall_quality_score": round(sum(r["overall_quality_score"] for r in rows) / n, 2),
        })
    return summary


def run_benchmark(
    mode: str = "auto",
    case_filter: Optional[List[str]] = None,
    timeout_seconds: Optional[float] = None,
    delay_seconds: float = 1.5,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    effective_mode = resolve_mode(mode)
    is_simulated = effective_mode != "live"
    data_source = "MOCK_SIMULATED" if is_simulated else "LIVE_GEMINI_API"

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cases = list(dataset.get("cases", []))
    if case_filter:
        cases = [c for c in cases if c.get("case_id") in case_filter]
    if not cases:
        raise ValueError("Không có test case phù hợp trong it1108_benchmark_dataset.json.")

    curriculum_blocks = load_curriculum_blocks(include_general=False)
    client = GeminiQuizClient(mode=effective_mode, timeout_seconds=timeout_seconds)

    detailed_records: List[Dict[str, Any]] = []
    raw_runs: List[Dict[str, Any]] = []

    for c_idx, case in enumerate(cases):
        for s_idx, scenario in enumerate(ALLOWED_SCENARIOS):
            if effective_mode == "live" and (c_idx > 0 or s_idx > 0) and delay_seconds > 0:
                time.sleep(delay_seconds)

            res = client.generate_questions(
                course_code=case["course_code"],
                course_name=case["course_name"],
                knowledge_block=None,
                review_topics=case["review_topics"],
                num_questions=int(case["num_questions"]),
                difficulty="auto",
                scenario_type=scenario,
                profile=case["profile"],
                mode=effective_mode,
            )
            eval_rec = evaluate_case_scenario(case, scenario, res, curriculum_blocks)
            detailed_records.append(eval_rec)
            raw_runs.append({
                "case_id": case["case_id"],
                "scenario_type": scenario,
                "data_source": data_source,
                "is_simulated": is_simulated,
                "evaluation": eval_rec,
                "quiz_data": res.get("quiz_data"),
                "error_type": res.get("error_type"),
                "error_message": res.get("error_message"),
            })

    summary = summarize_by_scenario(detailed_records, effective_mode)
    output_payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": dataset.get("dataset_name"),
        "effective_mode": effective_mode,
        "data_source": data_source,
        "is_simulated": is_simulated,
        "model": client.model,
        "fallback_model": client.fallback_model,
        "socket_timeout_seconds": client.timeout_seconds,
        "total_runs": len(detailed_records),
        "scenario_summary": summary,
        "detailed_records": detailed_records,
        "runs": raw_runs,
    }

    if save_outputs:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "live" if effective_mode == "live" else "mock"
        json_path = OUTPUTS_DIR / f"prompt3_comparison_{suffix}.json"
        detail_csv_path = OUTPUTS_DIR / f"prompt3_comparison_{suffix}.csv"
        summary_csv_path = OUTPUTS_DIR / f"prompt3_summary_{suffix}.csv"

        json_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if detailed_records:
            csv_rows = []
            for r in detailed_records:
                row_copy = dict(r)
                row_copy["difficulty_distribution"] = json.dumps(r["difficulty_distribution"], ensure_ascii=False)
                csv_rows.append(row_copy)
            with detail_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)

        if summary:
            with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
                writer.writeheader()
                writer.writerows(summary)

        output_payload["output_files"] = {
            "json": str(json_path),
            "detail_csv": str(detail_csv_path),
            "summary_csv": str(summary_csv_path),
        }

    return output_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="So sánh chất lượng đầu ra 3 kịch bản Prompt Engineering (Tuần 7).")
    parser.add_argument("--mode", choices=["auto", "live", "mock"], default="auto", help="Chế độ thực thi (mặc định auto).")
    parser.add_argument("--cases", type=str, default="all", help="'all' hoặc danh sách case_id cách nhau dấu phẩy.")
    parser.add_argument("--timeout", type=float, default=None, help="Ghi đè socket timeout (giây) khi test mạng chậm.")
    parser.add_argument("--delay", type=float, default=1.5, help="Thời gian nghỉ giữa các lượt gọi Live (giây).")
    args = parser.parse_args()

    case_filter = None if args.cases.strip().lower() == "all" else [
        c.strip() for c in args.cases.split(",") if c.strip()
    ]
    payload = run_benchmark(
        mode=args.mode,
        case_filter=case_filter,
        timeout_seconds=args.timeout,
        delay_seconds=args.delay,
        save_outputs=True,
    )

    print(
        f"=== KẾT QUẢ SO SÁNH 3 KỊCH BẢN PROMPT ({payload['data_source']} | is_simulated={payload['is_simulated']}) ==="
    )
    for row in payload["scenario_summary"]:
        print(
            f"- {row['scenario_type']:16s} | Ok={row['successful_cases']}/{row['total_cases']} | "
            f"JSON={row['json_valid_pct']:5.1f}% | Topic={row['avg_topic_relevance_pct']:5.1f}% | "
            f"4Opts={row['avg_four_options_pct']:5.1f}% | 1Ans={row['avg_single_answer_pct']:5.1f}% | "
            f"ExpFit={row['avg_explanation_fit_pct']:5.1f}% | DiffFit={row['avg_difficulty_fit_pct']:5.1f}% | "
            f"NoDup={row['no_duplication_rate_pct']:5.1f}% | Score={row['avg_overall_quality_score']:5.2f}/100 | "
            f"Latency={row['avg_latency_ms']:.1f}ms"
        )
    if "output_files" in payload:
        print("Files xuất ra:")
        for k, v in payload["output_files"].items():
            print(f"  + {k}: {v}")


if __name__ == "__main__":
    main()
