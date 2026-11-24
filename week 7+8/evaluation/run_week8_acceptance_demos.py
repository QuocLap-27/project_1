# -*- coding: utf-8 -*-
"""
Module: evaluation/run_week8_acceptance_demos.py
Dự án: SMART LMS & GENERATIVE AI (Phần D.4 – Chạy 3 Kịch bản Demo Bắt buộc qua Web UI API)
Chức năng:
Thực thi đúng 3 kịch bản nghiệm thu qua endpoint POST /api/pipeline/run của Flask Web UI:
1. Kịch bản 1 – Hồ sơ OULAD đầy đủ -> detailed_profile -> sinh câu hỏi.
2. Kịch bản 2 – Hồ sơ thiếu dữ liệu chi tiết -> fallback risk_level, không bịa điểm/phân vị.
3. Kịch bản 3 – API gặp lỗi HTTP 503 UNAVAILABLE -> retry có giới hạn, không crash,
   không tự chuyển sang Mock, giữ nguyên data_source='LIVE_GEMINI_API' & is_simulated=False.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

WEEK78_DIR = Path(__file__).resolve().parent.parent
if str(WEEK78_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK78_DIR))

import playground.app as playground_app_mod  # noqa: E402
from src.config import OUTPUTS_DIR, has_valid_api_key, redact_secret  # noqa: E402
from src.gemini_client import GeminiQuizClient, _ServiceUnavailableAPIError  # noqa: E402
from src.ml_genai_pipeline import SmartLMSPipeline  # noqa: E402
from src.ml_predictor_bridge import find_representative_oulad_students  # noqa: E402


def run_acceptance_demos(mode: str = "mock", timeout_seconds: float = 45.0, save_output: bool = True) -> Dict[str, Any]:
    effective = mode.strip().lower()
    if effective not in ("mock", "live"):
        raise ValueError("mode chỉ nhận 'mock' hoặc 'live'.")
    if effective == "live" and not has_valid_api_key():
        raise RuntimeError("Chế độ --mode live yêu cầu GEMINI_API_KEY hợp lệ trong week 7+8/.env.")

    pipeline = SmartLMSPipeline(timeout_seconds=timeout_seconds)
    app = playground_app_mod.create_app(pipeline=pipeline)
    client = app.test_client()

    reps = find_representative_oulad_students()
    stu = reps["PARTIAL_SUBMISSION_LOW_SCORE"]

    # Kịch bản 1: Hồ sơ OULAD đầy đủ -> detailed_profile
    resp1 = client.post(
        "/api/pipeline/run",
        json={
            "oulad_student_id": stu["id_student"],
            "simulated_student_code": "SIM_DEMO_KB1_FULL",
            "include_block_details": True,
            "requested_scenario": "auto",
            "num_questions": 5,
            "difficulty": "auto",
            "mode": effective,
        },
    )
    res1 = resp1.get_json()

    # Kịch bản 2: Hồ sơ thiếu dữ liệu chi tiết -> fallback risk_level
    resp2 = client.post(
        "/api/pipeline/run",
        json={
            "oulad_student_id": stu["id_student"],
            "simulated_student_code": "SIM_DEMO_KB2_FALLBACK",
            "include_block_details": False,
            "target_block_codes": ["IT1108_KB03", "IT1108_KB04"],
            "requested_scenario": "auto",
            "num_questions": 5,
            "difficulty": "auto",
            "mode": effective,
        },
    )
    res2 = resp2.get_json()

    # Kịch bản 3: API gặp lỗi 503 UNAVAILABLE
    # - Trong --mode mock: dùng mock_provider simulate_error='service_unavailable' (hiển thị rõ MOCK_SIMULATED)
    # - Trong --mode live: mô phỏng phản hồi 503 ở tầng Live REST transport để kiểm chứng bất biến retry/no-fallback
    if effective == "mock":
        resp3 = client.post(
            "/api/pipeline/run",
            json={
                "oulad_student_id": stu["id_student"],
                "simulated_student_code": "SIM_DEMO_KB3_ERR503",
                "include_block_details": True,
                "requested_scenario": "auto",
                "num_questions": 5,
                "mode": "mock",
                "simulate_error": "service_unavailable",
            },
        )
        res3 = resp3.get_json()
        kb3_note = "Chạy simulate_error='service_unavailable' trong chế độ Mock (HTTP 503 được xử lý nhất quán)."
    else:
        with patch.dict(
            os.environ,
            {"GEMINI_MAX_RETRIES": "2", "GEMINI_TIMEOUT_SECONDS": "30"},
        ), patch("src.gemini_client.time.sleep", return_value=None), patch.object(
            GeminiQuizClient, "_call_live_via_sdk", side_effect=ImportError("skip sdk to test rest 503")
        ), patch.object(
            GeminiQuizClient,
            "_call_live_via_rest",
            side_effect=_ServiceUnavailableAPIError(
                'HTTP 503 UNAVAILABLE từ Gemini API (ai_studio): {"error":{"code":503,"message":"High demand (controlled test).","status":"UNAVAILABLE"}}'
            ),
        ):
            resp3 = client.post(
                "/api/pipeline/run",
                json={
                    "oulad_student_id": stu["id_student"],
                    "simulated_student_code": "SIM_DEMO_KB3_LIVE_503",
                    "include_block_details": True,
                    "requested_scenario": "auto",
                    "num_questions": 5,
                    "mode": "live",
                },
            )
        res3 = resp3.get_json()
        kb3_note = "Mô phỏng có kiểm soát phản hồi HTTP 503 từ transport Live để kiểm chứng retry 3 lần và không tự chuyển sang Mock."

    summary_rows = []
    for name, note, r in (
        ("KB1_FULL_OULAD_DETAILED", "Hồ sơ OULAD đầy đủ -> detailed_profile", res1),
        ("KB2_MISSING_DETAIL_FALLBACK", "Thiếu chi tiết khối -> fallback risk_level", res2),
        ("KB3_API_503_UNAVAILABLE", kb3_note, res3),
    ):
        ga = r.get("genai_result") or {}
        ps = r.get("prompt_selection") or {}
        summary_rows.append({
            "scenario_case": name,
            "note": note,
            "http_status": r.get("http_status"),
            "success": r.get("success"),
            "data_source": r.get("data_source"),
            "is_simulated": r.get("is_simulated"),
            "selected_scenario": ps.get("selected_scenario"),
            "has_detailed_block_data": (r.get("topic_mapping_summary") or {}).get("has_detailed_block_data"),
            "questions_count": len(r.get("questions") or []),
            "attempts_used": ga.get("attempts_used"),
            "models_tried": ga.get("models_tried"),
            "error_type": r.get("error_type"),
            "error_category": r.get("error_category"),
        })

    bundle = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "requested_mode": effective,
        "summary": summary_rows,
        "cases": [
            {"scenario_case": "KB1_FULL_OULAD_DETAILED", "result": res1},
            {"scenario_case": "KB2_MISSING_DETAIL_FALLBACK", "result": res2},
            {"scenario_case": "KB3_API_503_UNAVAILABLE", "result": res3},
        ],
    }

    if save_output:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUTS_DIR / f"week8_acceptance_demos_{effective}.json"
        out_path.write_text(redact_secret(json.dumps(bundle, ensure_ascii=False, indent=2)), encoding="utf-8")

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy 3 kịch bản demo nghiệm thu Tuần 8 qua Web UI API.")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    bundle = run_acceptance_demos(mode=args.mode, timeout_seconds=args.timeout, save_output=True)
    print(f"=== 3 KỊCH BẢN DEMO NGHIỆM THU TUẦN 8 ({args.mode.upper()}) ===")
    for row in bundle["summary"]:
        print(
            f"- {row['scenario_case']} | http={row['http_status']} | ok={row['success']} | "
            f"src={row['data_source']} (sim={row['is_simulated']}) | sc={row['selected_scenario']} | "
            f"q={row['questions_count']} | tries={row['attempts_used']} | err={row['error_type']}"
        )
    print(f"Saved: {OUTPUTS_DIR / f'week8_acceptance_demos_{args.mode}.json'}")


if __name__ == "__main__":
    main()
