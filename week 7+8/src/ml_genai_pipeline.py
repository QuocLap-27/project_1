# -*- coding: utf-8 -*-
"""
Module: ml_genai_pipeline.py
Dự án: SMART LMS & GENERATIVE AI (Tuần 8 – Kết nối ML & GenAI)
Chức năng:
Điều phối luồng nghiệp vụ hoàn chỉnh từ đầu đến cuối:
  Dữ liệu sinh viên (24 đặc trưng)
  -> ML Prediction (predict.py + best_model.joblib)
  -> Xác định trạng thái học tập (NO_SUBMISSION / PARTIAL_SUBMISSION_LOW_SCORE / SUBMITTED_LOW_SCORE / ...)
  -> Curriculum/Topic Mapping (curriculum_topics, assessment_topic_mapping, vle_topic_mapping)
  -> Xây dựng hồ sơ cá nhân hóa (không bịa điểm số/phân vị)
  -> Chọn Prompt Scenario (detailed_profile mặc định -> fallback risk_level nếu thiếu chi tiết; baseline chỉ khi chủ động chọn)
  -> Gemini API (responseSchema rút gọn + retry có giới hạn + 1 JSON retry)
  -> Validate JSON -> Trả về câu hỏi ôn tập hoặc thông báo lỗi rõ ràng.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import OUTPUTS_DIR
    from src.gemini_client import GeminiQuizClient
    from src.ml_predictor_bridge import MLPredictorBridge, find_representative_oulad_students
    from src.schemas import ALLOWED_SCENARIOS
    from src.topic_mapper import OULADBlockAnalyzer, build_personalized_profile
else:
    from .config import OUTPUTS_DIR
    from .gemini_client import GeminiQuizClient
    from .ml_predictor_bridge import MLPredictorBridge, find_representative_oulad_students
    from .schemas import ALLOWED_SCENARIOS
    from .topic_mapper import OULADBlockAnalyzer, build_personalized_profile


def select_prompt_scenario(
    profile: Dict[str, Any],
    requested_scenario: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Bước 5: Chọn kịch bản Prompt theo quy tắc Tuần 8:
    - 'baseline': CHỈ sử dụng khi người dùng chủ động chỉ định để kiểm thử/so sánh.
    - 'risk_level': sử dụng khi được chỉ định hoặc khi hồ sơ thiếu dữ liệu chi tiết từng khối.
    - Mặc định ('auto' / None / 'detailed_profile'): chọn 'detailed_profile' nếu đủ dữ liệu
      chi tiết từng khối; ngược lại tự động fallback sang 'risk_level'.
    """
    req = (requested_scenario or "auto").strip().lower()
    if req == "baseline":
        return "baseline", "Người dùng chủ động chọn kịch bản baseline để kiểm thử/so sánh."

    if req == "risk_level":
        return "risk_level", "Kịch bản risk_level được chỉ định trực tiếp."

    if req not in ("auto", "detailed_profile"):
        raise ValueError(
            f"requested_scenario='{requested_scenario}' không hợp lệ. Chỉ nhận None, 'auto', {ALLOWED_SCENARIOS}."
        )

    if profile.get("has_detailed_block_data"):
        return (
            "detailed_profile",
            "Hồ sơ có đủ dữ liệu chi tiết theo từng khối kiến thức; sử dụng kịch bản chính detailed_profile.",
        )

    return (
        "risk_level",
        "Hồ sơ thiếu dữ liệu chi tiết theo từng khối (chỉ có chỉ số tổng hợp ML); tự động fallback sang kịch bản risk_level.",
    )


class SmartLMSPipeline:
    """Pipeline tích hợp end-to-end ML -> Mapping -> GenAI cho Tuần 8."""

    def __init__(
        self,
        ml_bridge: Optional[MLPredictorBridge] = None,
        oulad_analyzer: Optional[OULADBlockAnalyzer] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        max_json_retries: Optional[int] = None,
    ) -> None:
        self.ml_bridge = ml_bridge or MLPredictorBridge()
        self.oulad_analyzer = oulad_analyzer or OULADBlockAnalyzer()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_json_retries = max_json_retries

    def run(
        self,
        student_features: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        simulated_student_code: str = "SIM_STUDENT_01",
        oulad_student_id: Optional[int] = None,
        block_records: Optional[List[Dict[str, Any]]] = None,
        target_block_codes: Optional[List[str]] = None,
        cutoff_day: Optional[int] = None,
        requested_scenario: Optional[str] = None,
        num_questions: int = 5,
        difficulty: str = "auto",
        use_mock: bool = False,
        simulate_error: Optional[str] = None,
        genai_client: Optional[GeminiQuizClient] = None,
    ) -> Dict[str, Any]:
        """
        Thực thi trọn vẹn luồng ML -> Mapping -> GenAI.
        Lưu ý: Chỉ chạy Mock Mode khi use_mock=True rõ ràng; tuyệt đối không tự động
        rơi về Mock khi Live API gặp lỗi.
        """
        # Bước 1 & Bước 2: Dự đoán ML & xác định trạng thái học tập
        ml_result = self.ml_bridge.predict_student_risk(student_features)
        if not ml_result.get("success"):
            return {
                "success": False,
                "http_status": 400,
                "step_failed": "ml_prediction",
                "simulated_student_code": simulated_student_code,
                "data_source": "MOCK_SIMULATED" if use_mock else "LIVE_GEMINI_API",
                "is_simulated": bool(use_mock),
                "can_retry": False,
                "error_type": "INVALID_ML_INPUT",
                "error_message": ml_result.get("error"),
                "ml_prediction": ml_result,
                "submission_behavior": None,
                "personalized_profile": None,
                "prompt_selection": None,
                "genai_result": None,
                "questions": [],
            }

        # Bước 3 & Bước 4: Topic Mapping & xây dựng hồ sơ cá nhân hóa (không bịa dữ liệu)
        try:
            personalized_profile = build_personalized_profile(
                simulated_student_code=simulated_student_code,
                ml_result=ml_result,
                block_records=block_records,
                oulad_student_id=oulad_student_id,
                target_block_codes=target_block_codes,
                cutoff_day=cutoff_day,
                analyzer=self.oulad_analyzer,
            )
        except Exception as exc:
            return {
                "success": False,
                "http_status": 400,
                "step_failed": "topic_mapping",
                "simulated_student_code": simulated_student_code,
                "data_source": "MOCK_SIMULATED" if use_mock else "LIVE_GEMINI_API",
                "is_simulated": bool(use_mock),
                "can_retry": False,
                "error_type": "TOPIC_MAPPING_ERROR",
                "error_message": str(exc),
                "ml_prediction": ml_result,
                "submission_behavior": ml_result.get("submission_behavior"),
                "personalized_profile": None,
                "prompt_selection": None,
                "genai_result": None,
                "questions": [],
            }

        # Bước 5: Chọn kịch bản Prompt (detailed_profile -> fallback risk_level)
        selected_scenario, selection_reason = select_prompt_scenario(
            personalized_profile, requested_scenario=requested_scenario
        )

        # Chuẩn bị profile đưa vào Prompt Builder (loại bỏ hoàn toàn thông tin định danh)
        prompt_profile = {
            "simulated_student_code": simulated_student_code,
            "risk_level": personalized_profile["risk_level"],
            "risk_probability": personalized_profile["risk_probability"],
            "weak_blocks": personalized_profile["weak_blocks"],
        }

        weak_blocks = personalized_profile["weak_blocks"]
        review_topics = [b["it1108_block_name"] for b in weak_blocks] if weak_blocks else None
        effective_mode = "mock" if use_mock else "live"

        # Bước 6: Sinh nội dung & Validate JSON qua GeminiQuizClient (Tuần 7 đã gia cố)
        client = genai_client or GeminiQuizClient(
            mode=effective_mode,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            max_json_retries=self.max_json_retries,
        )
        genai_result = client.generate_questions(
            course_code="IT1108",
            course_name="Nhập môn lập trình",
            knowledge_block=None,
            review_topics=review_topics,
            num_questions=num_questions,
            difficulty=difficulty,
            scenario_type=selected_scenario,
            profile=prompt_profile,
            mode=effective_mode,
            simulate_error=simulate_error if use_mock else None,
        )

        quiz_data = genai_result.get("quiz_data") or {}
        questions = list(quiz_data.get("questions") or [])

        return {
            "success": bool(genai_result.get("success")),
            "http_status": genai_result.get("http_status"),
            "step_failed": None if genai_result.get("success") else "genai_generation",
            "simulated_student_code": simulated_student_code,
            "data_source": genai_result.get("data_source"),
            "is_simulated": genai_result.get("is_simulated"),
            "mode_notice": genai_result.get("mode_notice"),
            "can_retry": genai_result.get("can_retry", True),
            "ml_prediction": {
                "prediction": ml_result["prediction"],
                "risk_probability": ml_result["risk_probability"],
                "risk_level": ml_result["risk_level"],
                "decision_threshold": ml_result["decision_threshold"],
                "model_name": ml_result["model_name"],
                "model_version": ml_result["model_version"],
                "inference_backend": ml_result["inference_backend"],
            },
            "submission_behavior": ml_result["submission_behavior"],
            "topic_mapping_summary": {
                "block_data_source": personalized_profile["block_data_source"],
                "has_detailed_block_data": personalized_profile["has_detailed_block_data"],
                "has_class_percentile": personalized_profile["has_class_percentile"],
                "mapped_blocks_count": len(weak_blocks),
                "mapped_block_codes": [b["it1108_block_code"] for b in weak_blocks],
            },
            "personalized_profile": personalized_profile,
            "prompt_selection": {
                "requested_scenario": requested_scenario or "auto",
                "selected_scenario": selected_scenario,
                "reason": selection_reason,
            },
            "genai_result": {
                "model_used": genai_result.get("model_used"),
                "models_tried": genai_result.get("models_tried"),
                "attempts_used": genai_result.get("attempts_used"),
                "api_provider": genai_result.get("api_provider"),
                "transport": genai_result.get("transport"),
                "latency_ms": genai_result.get("latency_ms"),
                "within_timeout_nfr01": genai_result.get("within_timeout_nfr01"),
                "json_retries_used": genai_result.get("json_retries_used", 0),
                "pii_report": genai_result.get("pii_report"),
                "prompt_bundle": genai_result.get("prompt_bundle"),
                "validation": genai_result.get("validation"),
            },
            "questions": questions,
            "error_type": genai_result.get("error_type"),
            "error_category": genai_result.get("error_category"),
            "error_message": genai_result.get("error_message"),
            "user_guidance": genai_result.get("user_guidance"),
        }


def run_demo_cases(
    use_mock: bool = False,
    timeout_seconds: Optional[float] = None,
    save_output: bool = True,
) -> Dict[str, Any]:
    """Chạy trình diễn 2 ca thực tế từ dữ liệu OULAD: (1) Đủ dữ liệu khối, (2) Thiếu dữ liệu khối (fallback)."""
    pipeline = SmartLMSPipeline(timeout_seconds=timeout_seconds)
    reps = find_representative_oulad_students()
    sample_stu = reps.get("PARTIAL_SUBMISSION_LOW_SCORE") or next(iter(reps.values()))

    # Ca 1: Có đủ dữ liệu chi tiết theo từng khối từ OULAD -> tự chọn detailed_profile
    case_full = pipeline.run(
        student_features=sample_stu["features"],
        simulated_student_code="SIM_WEEK8_FULL_01",
        oulad_student_id=sample_stu["id_student"],
        requested_scenario="auto",
        num_questions=5,
        use_mock=use_mock,
    )

    # Ca 2: Chỉ có 24 đặc trưng ML + danh sách mã khối cần ôn (thiếu điểm/trạng thái từng khối) -> fallback risk_level
    case_missing_detail = pipeline.run(
        student_features=sample_stu["features"],
        simulated_student_code="SIM_WEEK8_PARTIAL_02",
        oulad_student_id=None,
        block_records=None,
        target_block_codes=["IT1108_KB03", "IT1108_KB04"],
        requested_scenario="auto",
        num_questions=5,
        use_mock=use_mock,
    )

    mode_label = "mock" if use_mock else "live"
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode_label,
        "data_source": "MOCK_SIMULATED" if use_mock else "LIVE_GEMINI_API",
        "is_simulated": bool(use_mock),
        "cases": [
            {"case_name": "full_oulad_profile_detailed_scenario", "result": case_full},
            {"case_name": "missing_block_detail_fallback_risk_level", "result": case_missing_detail},
        ],
    }

    if save_output:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUTS_DIR / f"week8_pipeline_demo_{mode_label}.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["output_file"] = str(out_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy pipeline ML -> Mapping -> GenAI (Tuần 8).")
    parser.add_argument("--mode", choices=["live", "mock"], default="live", help="Chỉ chạy mock khi chọn rõ --mode mock.")
    parser.add_argument("--timeout", type=float, default=None, help="Socket timeout (giây), mặc định theo .env/config.")
    args = parser.parse_args()

    res = run_demo_cases(use_mock=(args.mode == "mock"), timeout_seconds=args.timeout, save_output=True)
    print(f"=== PIPELINE TUẦN 8 ({res['data_source']} | is_simulated={res['is_simulated']}) ===")
    for item in res["cases"]:
        r = item["result"]
        print(
            f"- {item['case_name']} | success={r['success']} | http={r['http_status']} | "
            f"ML={r['ml_prediction']['risk_level']}({r['ml_prediction']['risk_probability']}) | "
            f"behavior={r['submission_behavior']['status_code']} | "
            f"scenario={r['prompt_selection']['selected_scenario']} | "
            f"questions={len(r['questions'])} | lat={r['genai_result']['latency_ms']}ms | "
            f"err={r['error_type']}"
        )
    if "output_file" in res:
        print(f"Saved: {res['output_file']}")


if __name__ == "__main__":
    main()
