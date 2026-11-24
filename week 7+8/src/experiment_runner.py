# -*- coding: utf-8 -*-
"""
Module: experiment_runner.py
Dự án: SMART LMS & GENERATIVE AI (Tuần 7+8 - Kiểm thử GenAI)
Chức năng:
- Thực nghiệm đối chứng 3 kịch bản Prompt (baseline / risk_level / detailed_profile) trên bộ hồ sơ mô phỏng IT1108.
- Đo lường tỷ lệ JSON hợp lệ, Schema valid (đủ 6 trường Prompt 2: question, options, correct_answer,
  explanation, topic, difficulty), SQL ISJSON, Latency (NFR-01 <= 20s), PII Safe (NFR-02),
  và Điểm phù hợp sư phạm (Pedagogical Alignment Score).
- Xuất kết quả ra 3 tệp trong week 7+8/outputs/ phục vụ báo cáo Tuần 7.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import OUTPUTS_DIR, SAMPLE_PROFILES_JSON, resolve_mode
    from src.gemini_client import GeminiQuizClient
    from src.prompt_builder import normalize_block_status
    from src.schemas import ALLOWED_SCENARIOS
else:
    from .config import OUTPUTS_DIR, SAMPLE_PROFILES_JSON, resolve_mode
    from .gemini_client import GeminiQuizClient
    from .prompt_builder import normalize_block_status
    from .schemas import ALLOWED_SCENARIOS


_GENERIC_HINT_MARKERS = (
    "xem lại giáo trình",
    "đọc lại slide",
    "tự xem lại",
)


def load_sample_profiles(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    target = Path(path) if path is not None else SAMPLE_PROFILES_JSON
    if not target.exists():
        raise FileNotFoundError(f"Không tìm thấy sample_profiles.json tại: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    return list(data.get("profiles", []))


def compute_quiz_metrics(
    profile: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Tính các chỉ số chất lượng và Điểm phù hợp sư phạm (0..100) cho một bộ đề AI trả về."""
    validation = result.get("validation") or {}
    pii_report = result.get("pii_report") or {}
    quiz = result.get("quiz_data")

    base_metrics = {
        "profile_id": profile.get("profile_id", "custom"),
        "risk_level": profile.get("risk_level", "MEDIUM"),
        "risk_probability": profile.get("risk_probability", 0.5),
        "scenario_type": result.get("scenario_type"),
        "mode_used": result.get("mode_used"),
        "api_provider": result.get("api_provider"),
        "model_used": result.get("model_used"),
        "success": bool(result.get("success")),
        "http_status": result.get("http_status"),
        "json_valid": bool(validation.get("json_valid")),
        "schema_valid": bool(validation.get("schema_valid")),
        "sql_isjson_compatible": bool(validation.get("sql_isjson_compatible")),
        "within_timeout_nfr01": bool(result.get("within_timeout_nfr01")),
        "pii_safe": bool(pii_report.get("pii_safe")),
        "latency_ms": float(result.get("latency_ms") or 0.0),
        "prompt_tokens": int((result.get("usage") or {}).get("prompt_tokens") or 0),
        "output_tokens": int((result.get("usage") or {}).get("output_tokens") or 0),
        "avg_explanation_chars": 0.0,
        "pct_with_meaningful_hint": 0.0,
        "pct_with_code_snippet": 0.0,
        "difficulty_counts": {"easy": 0, "medium": 0, "hard": 0},
        "question_type_counts": {"guided_basic": 0, "code_trace": 0, "bug_analysis": 0, "concept": 0},
        "pedagogical_alignment_score": 0.0,
    }

    if not quiz or not isinstance(quiz.get("questions"), list) or not quiz["questions"]:
        return base_metrics

    questions: List[Dict[str, Any]] = quiz["questions"]
    weak_blocks = profile.get("weak_blocks") or []
    status_by_block = {
        b["it1108_block_code"]: normalize_block_status(b.get("submission_vs_score_status"))
        for b in weak_blocks
        if b.get("it1108_block_code")
    }
    target_block_set = set(status_by_block.keys())

    diff_counts = {"easy": 0, "medium": 0, "hard": 0}
    type_counts = {"guided_basic": 0, "code_trace": 0, "bug_analysis": 0, "concept": 0}
    exp_lengths: List[int] = []
    meaningful_hints = 0
    with_code = 0
    alignment_scores: List[float] = []
    covered_blocks = set()

    for q in questions:
        b_code = q.get("it1108_block_code", "")
        if b_code in target_block_set:
            covered_blocks.add(b_code)

        diff = str(q.get("difficulty", "medium")).lower()
        q_type = str(q.get("question_type", "concept")).lower()
        code = str(q.get("code_snippet") or "").strip()
        hint = str(q.get("hint") or "").strip()
        exp = str(q.get("explanation") or "").strip()

        if diff in diff_counts:
            diff_counts[diff] += 1
        if q_type in type_counts:
            type_counts[q_type] += 1

        exp_lengths.append(len(exp))
        if code:
            with_code += 1

        is_generic_hint = any(marker in hint.lower() for marker in _GENERIC_HINT_MARKERS)
        if len(hint) >= 30 and not is_generic_hint:
            meaningful_hints += 1

        block_status = status_by_block.get(b_code, "LOW_SCORE_DESPITE_SUBMISSION")
        if block_status == "MISSING_SUBMISSION":
            if diff == "easy" and q_type == "guided_basic":
                alignment_scores.append(1.0)
            elif diff == "easy" or q_type == "guided_basic":
                alignment_scores.append(0.6)
            else:
                alignment_scores.append(0.2)
        elif block_status == "LOW_SCORE_DESPITE_SUBMISSION":
            if q_type in ("code_trace", "bug_analysis") and code:
                alignment_scores.append(1.0)
            elif q_type in ("code_trace", "bug_analysis"):
                alignment_scores.append(0.6)
            else:
                alignment_scores.append(0.25)
        elif block_status == "PARTIAL_SUBMISSION_AND_LOW_SCORE":
            alignment_scores.append(1.0 if q_type in ("guided_basic", "bug_analysis", "code_trace") else 0.3)
        elif block_status == "PARTIAL_SUBMISSION_ONLY":
            alignment_scores.append(1.0 if diff == "medium" and q_type in ("concept", "code_trace") else 0.5)
        else:
            alignment_scores.append(
                1.0 if diff in ("medium", "hard") and q_type in ("concept", "code_trace", "bug_analysis") else 0.4
            )

    n_q = len(questions)
    coverage_pts = (len(covered_blocks) / max(1, len(target_block_set))) * 20.0
    pedagogical_pts = (sum(alignment_scores) / n_q) * 50.0
    hint_pts = (meaningful_hints / n_q) * 15.0

    avg_exp = sum(exp_lengths) / n_q
    exp_depth_pts = min(1.0, avg_exp / 120.0) * 10.0

    strategy_text = str(quiz.get("difficulty_strategy") or "").lower()
    scenario = result.get("scenario_type")
    if scenario == "detailed_profile" and any(
        k in strategy_text
        for k in (
            "chưa nộp",
            "chưa có bài nộp",
            "điểm thấp",
            "điểm còn thấp",
            "cá nhân hóa",
            "guided_basic",
            "code_trace",
            "bug_analysis",
        )
    ):
        strategy_pts = 5.0
    elif scenario == "risk_level" and any(k in strategy_text for k in ("rủi ro", "high", "medium", "low", "xác suất")):
        strategy_pts = 2.5
    else:
        strategy_pts = 0.0

    total_alignment = round(coverage_pts + pedagogical_pts + hint_pts + exp_depth_pts + strategy_pts, 2)

    base_metrics.update({
        "avg_explanation_chars": round(avg_exp, 1),
        "pct_with_meaningful_hint": round(meaningful_hints / n_q * 100.0, 1),
        "pct_with_code_snippet": round(with_code / n_q * 100.0, 1),
        "difficulty_counts": diff_counts,
        "question_type_counts": type_counts,
        "pedagogical_alignment_score": total_alignment,
    })
    return base_metrics


def aggregate_by_scenario(run_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tổng hợp kết quả trung bình theo từng kịch bản (baseline / risk_level / detailed_profile)."""
    summaries: List[Dict[str, Any]] = []
    for scenario in ALLOWED_SCENARIOS:
        rows = [r["metrics"] for r in run_records if r["metrics"].get("scenario_type") == scenario]
        if not rows:
            continue
        n = len(rows)
        summaries.append({
            "scenario_type": scenario,
            "total_runs": n,
            "success_rate_pct": round(sum(1 for r in rows if r["success"]) / n * 100.0, 1),
            "json_valid_pct": round(sum(1 for r in rows if r["json_valid"]) / n * 100.0, 1),
            "schema_valid_pct": round(sum(1 for r in rows if r["schema_valid"]) / n * 100.0, 1),
            "sql_isjson_pct": round(sum(1 for r in rows if r["sql_isjson_compatible"]) / n * 100.0, 1),
            "nfr01_within_20s_pct": round(sum(1 for r in rows if r["within_timeout_nfr01"]) / n * 100.0, 1),
            "nfr02_pii_safe_pct": round(sum(1 for r in rows if r["pii_safe"]) / n * 100.0, 1),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 2),
            "avg_explanation_chars": round(sum(r["avg_explanation_chars"] for r in rows) / n, 1),
            "avg_meaningful_hint_pct": round(sum(r["pct_with_meaningful_hint"] for r in rows) / n, 1),
            "avg_code_snippet_pct": round(sum(r["pct_with_code_snippet"] for r in rows) / n, 1),
            "avg_pedagogical_alignment_score": round(sum(r["pedagogical_alignment_score"] for r in rows) / n, 2),
        })
    return summaries


def run_experiments(
    mode: str = "auto",
    repeats: int = 1,
    profile_filter: Optional[List[str]] = None,
    model: Optional[str] = None,
    save_outputs: bool = True,
) -> Dict[str, Any]:
    """Chạy toàn bộ ma trận thực nghiệm (Hồ sơ x 3 Kịch bản x Repeats) và xuất báo cáo."""
    effective_mode = resolve_mode(mode)
    client = GeminiQuizClient(mode=effective_mode, model=model)
    all_profiles = load_sample_profiles()
    if profile_filter:
        profiles = [p for p in all_profiles if p.get("profile_id") in profile_filter]
    else:
        profiles = all_profiles

    if not profiles:
        raise ValueError("Không tìm thấy hồ sơ mẫu phù hợp để chạy thực nghiệm.")

    run_records: List[Dict[str, Any]] = []
    for rep in range(1, repeats + 1):
        for profile in profiles:
            for scenario in ALLOWED_SCENARIOS:
                res = client.generate_quiz(profile=profile, scenario_type=scenario, mode=effective_mode)
                metrics = compute_quiz_metrics(profile, res)
                metrics["repeat_index"] = rep
                run_records.append({
                    "profile_id": profile["profile_id"],
                    "profile_title": profile.get("title", ""),
                    "repeat_index": rep,
                    "scenario_type": scenario,
                    "metrics": metrics,
                    "result": res,
                })

    scenario_summary = aggregate_by_scenario(run_records)
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "effective_mode": effective_mode,
        "model": client.model,
        "repeats": repeats,
        "total_runs": len(run_records),
        "scenario_summary": scenario_summary,
        "runs": run_records,
    }

    if save_outputs:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        json_path = OUTPUTS_DIR / f"experiment_results_{effective_mode}.json"
        csv_path = OUTPUTS_DIR / f"scenario_comparison_{effective_mode}.csv"
        md_path = OUTPUTS_DIR / f"week7_experiment_summary_{effective_mode}.md"

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if scenario_summary:
            with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(scenario_summary[0].keys()))
                writer.writeheader()
                writer.writerows(scenario_summary)

        md_lines = _build_markdown_summary(payload)
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        payload["output_files"] = {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
        }

    return payload


def _build_markdown_summary(payload: Dict[str, Any]) -> List[str]:
    lines = [
        f"# Báo cáo Thực nghiệm Kiểm thử GenAI – Tuần 7 ({payload['effective_mode'].upper()})",
        "",
        f"- **Thời điểm chạy:** {payload['generated_at']}",
        f"- **Chế độ thực thi:** `{payload['effective_mode']}`",
        f"- **Model cấu hình:** `{payload['model']}`",
        f"- **Tổng số lượt kiểm thử:** {payload['total_runs']} ({payload['repeats']} lượt lặp x 3 kịch bản)",
        "",
        "## 1. Bảng So sánh Định lượng 3 Kịch bản Prompt",
        "",
        "| Kịch bản | JSON Valid (%) | Schema Valid (%) | SQL ISJSON (%) | NFR-01 <=20s (%) | NFR-02 PII Safe (%) | Độ trễ TB (ms) | Độ dài giải thích TB (ký tự) | Có gợi ý chi tiết (%) | Có code C (%) | Điểm phù hợp sư phạm (/100) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["scenario_summary"]:
        lines.append(
            f"| `{row['scenario_type']}` | {row['json_valid_pct']} | {row['schema_valid_pct']} | "
            f"{row['sql_isjson_pct']} | {row['nfr01_within_20s_pct']} | {row['nfr02_pii_safe_pct']} | "
            f"{row['avg_latency_ms']} | {row['avg_explanation_chars']} | {row['avg_meaningful_hint_pct']} | "
            f"{row['avg_code_snippet_pct']} | **{row['avg_pedagogical_alignment_score']}** |"
        )

    lines.extend(["", "## 2. Đối chiếu Chất lượng Trên Hồ sơ Tiêu biểu (`profile_01_high_mixed`)", ""])
    sample_runs = [r for r in payload["runs"] if r["profile_id"] == "profile_01_high_mixed" and r["repeat_index"] == 1]
    for r in sample_runs:
        sc = r["scenario_type"]
        res = r["result"]
        quiz = res.get("quiz_data") or {}
        lines.append(f"### Kịch bản: `{sc}` (Alignment Score: {r['metrics']['pedagogical_alignment_score']}/100)")
        lines.append(f"- **Chiến lược độ khó:** {quiz.get('difficulty_strategy', 'N/A')}")
        lines.append(f"- **Lời khuyên học tập:** {quiz.get('study_advice', 'N/A')}")
        q1 = (quiz.get("questions") or [{}])[0]
        if q1:
            lines.append(
                f"- **Ví dụ Câu 1** (`{q1.get('it1108_block_code')}` | `topic`: {q1.get('topic')} | `difficulty`: `{q1.get('difficulty')}` | `correct_answer`: `{q1.get('correct_answer')}`): {q1.get('question')}"
            )
            lines.append(f"  - *Gợi ý:* {q1.get('hint')}")
            lines.append(f"  - *Giải thích:* {q1.get('explanation')}")
        lines.append("")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm so sánh 3 kịch bản Prompt GenAI (Tuần 7+8).")
    parser.add_argument("--mode", choices=["auto", "mock", "live"], default="auto", help="Chế độ chạy (mặc định auto).")
    parser.add_argument("--repeats", type=int, default=1, help="Số lần lặp trên mỗi cặp hồ sơ-kịch bản.")
    parser.add_argument("--profiles", type=str, default="all", help="'all' hoặc danh sách profile_id cách nhau bởi dấu phẩy.")
    parser.add_argument("--model", type=str, default=None, help="Ghi đè tên model Gemini (mặc định gemini-2.5-flash).")
    args = parser.parse_args()

    profile_filter = None if args.profiles.strip().lower() == "all" else [
        p.strip() for p in args.profiles.split(",") if p.strip()
    ]
    payload = run_experiments(
        mode=args.mode,
        repeats=args.repeats,
        profile_filter=profile_filter,
        model=args.model,
        save_outputs=True,
    )

    print(f"=== HOÀN THÀNH THỰC NGHIỆM GENAI ({payload['effective_mode'].upper()}) ===")
    for row in payload["scenario_summary"]:
        print(
            f"- {row['scenario_type']:16s} | JSON={row['json_valid_pct']:5.1f}% | "
            f"Schema={row['schema_valid_pct']:5.1f}% | PII_Safe={row['nfr02_pii_safe_pct']:5.1f}% | "
            f"Alignment={row['avg_pedagogical_alignment_score']:5.2f}/100 | "
            f"Latency={row['avg_latency_ms']:.2f}ms"
        )
    if "output_files" in payload:
        print("Files xuất ra:")
        for k, v in payload["output_files"].items():
            print(f"  + {k}: {v}")


if __name__ == "__main__":
    main()
