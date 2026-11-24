# -*- coding: utf-8 -*-
"""
Module: playground/app.py
Dự án: SMART LMS & GENERATIVE AI (Tuần 7+8 - Kiểm thử GenAI)
Chức năng:
- Giao diện Web Playground nhẹ bằng Flask giúp kiểm thử trực quan với đầy đủ 7 tham số:
  mã học phần, tên học phần, khối kiến thức IT1108, chủ đề cần ôn tập, số lượng câu hỏi,
  mức độ khó, kịch bản Prompt; kèm kiểm tra NFR-02 (PII Guard), JSON Output chuẩn ISJSON,
  và mô phỏng lỗi NFR-03 (503/500).
- Tuyệt đối KHÔNG gửi API Key lên frontend.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template, request

WEEK78_DIR = Path(__file__).resolve().parent.parent
if str(WEEK78_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK78_DIR))

from src.config import get_runtime_summary, redact_secret, resolve_mode  # noqa: E402
from src.curriculum_loader import load_curriculum_blocks  # noqa: E402
from src.experiment_runner import compute_quiz_metrics, load_sample_profiles  # noqa: E402
from src.gemini_client import GeminiQuizClient  # noqa: E402
from src.ml_genai_pipeline import SmartLMSPipeline  # noqa: E402
from src.ml_predictor_bridge import find_representative_oulad_students, load_oulad_student_row  # noqa: E402
from src.schemas import ALLOWED_DIFFICULTIES, ALLOWED_SCENARIOS, VALID_IT1108_BLOCKS  # noqa: E402


def _find_first_medium_student(pipeline: SmartLMSPipeline) -> Optional[Dict[str, Any]]:
    """Tìm 1 sinh viên OULAD có mức rủi ro MEDIUM thực tế từ student_features.csv để phục vụ demo UI."""
    import pandas as pd
    from src.config import WEEK34_STUDENT_FEATURES_CSV

    if not WEEK34_STUDENT_FEATURES_CSV.exists():
        return None
    df = pd.read_csv(WEEK34_STUDENT_FEATURES_CSV)
    expected = pipeline.ml_bridge.expected_features
    for _, row in df.iterrows():
        feats = {c: (None if pd.isna(row[c]) else float(row[c])) for c in expected}
        pred = pipeline.ml_bridge.predict_student_risk(feats)
        if pred.get("success") and pred.get("risk_level") == "MEDIUM":
            return {
                "id_student": int(row["id_student"]),
                "code_module": str(row["code_module"]),
                "code_presentation": str(row["code_presentation"]),
                "features": feats,
                "ml_preview": pred,
            }
    return None


def create_app(pipeline: Optional[SmartLMSPipeline] = None) -> Flask:
    app = Flask(__name__, template_folder="templates")
    app.config["JSON_AS_ASCII"] = False
    state: Dict[str, Any] = {
        "pipeline": pipeline,
        "students_catalog": None,
    }

    def _get_pipeline() -> SmartLMSPipeline:
        if state["pipeline"] is None:
            state["pipeline"] = SmartLMSPipeline()
        return state["pipeline"]

    def _get_students_catalog() -> Dict[str, Any]:
        if state["students_catalog"] is not None:
            return state["students_catalog"]

        pl = _get_pipeline()
        reps = find_representative_oulad_students()
        code_aliases = {
            "PARTIAL_SUBMISSION_LOW_SCORE": ("SIM_WEEK8_FULL_01", "HIGH – Nộp thiếu & điểm thấp"),
            "SUBMITTED_LOW_SCORE": ("SIM_OULAD_SUB_LOW_02", "HIGH – Đã nộp đủ nhưng điểm thấp"),
            "NO_SUBMISSION": ("SIM_OULAD_NOSUB_03", "HIGH – Chưa nộp bài / Ngừng sớm"),
            "NORMAL": ("SIM_OULAD_LOW_04", "LOW – Đạt yêu cầu (An toàn)"),
        }
        students: list[Dict[str, Any]] = []
        ordered_keys = ["PARTIAL_SUBMISSION_LOW_SCORE", "SUBMITTED_LOW_SCORE", "NO_SUBMISSION", "NORMAL"]
        for key in ordered_keys:
            if key not in reps:
                continue
            item = reps[key]
            pred = pl.ml_bridge.predict_student_risk(item["features"])
            sim_code, role_desc = code_aliases[key]
            students.append({
                "simulated_student_code": sim_code,
                "oulad_student_id": item["id_student"],
                "risk_level": pred["risk_level"],
                "risk_probability": pred["risk_probability"],
                "submission_behavior": pred["submission_behavior"],
                "label": f"{sim_code} (OULAD #{item['id_student']}) | {role_desc} | p={pred['risk_probability']:.2f}",
            })

        med = _find_first_medium_student(pl)
        if med is not None:
            pred_med = med["ml_preview"]
            students.insert(
                3 if len(students) >= 3 else len(students),
                {
                    "simulated_student_code": "SIM_OULAD_MEDIUM_05",
                    "oulad_student_id": med["id_student"],
                    "risk_level": pred_med["risk_level"],
                    "risk_probability": pred_med["risk_probability"],
                    "submission_behavior": pred_med["submission_behavior"],
                    "label": (
                        f"SIM_OULAD_MEDIUM_05 (OULAD #{med['id_student']}) | "
                        f"MEDIUM – Cảnh báo trung bình | p={pred_med['risk_probability']:.2f}"
                    ),
                },
            )

        blocks = list(load_curriculum_blocks(include_general=False).values())
        primary_id = students[0]["oulad_student_id"] if students else 54882
        demo_presets = [
            {
                "preset_id": "kb1_full_detailed",
                "title": "Kịch bản 1: Hồ sơ OULAD đầy đủ (detailed_profile)",
                "description": "Đủ 24 đặc trưng ML + điểm/phân vị thật từng khối IT1108 -> chọn detailed_profile.",
                "oulad_student_id": primary_id,
                "simulated_student_code": "SIM_WEEK8_FULL_01",
                "include_block_details": True,
                "target_block_codes": [],
                "requested_scenario": "auto",
                "num_questions": 5,
                "difficulty": "auto",
                "recommended_mode": "live",
                "simulate_error": "",
            },
            {
                "preset_id": "kb2_missing_detail_fallback",
                "title": "Kịch bản 2: Thiếu dữ liệu chi tiết khối (fallback risk_level)",
                "description": "Chỉ có dự đoán ML và mã khối mục tiêu, không bịa điểm/phân vị -> tự động fallback risk_level.",
                "oulad_student_id": primary_id,
                "simulated_student_code": "SIM_WEEK8_PARTIAL_02",
                "include_block_details": False,
                "target_block_codes": ["IT1108_KB03", "IT1108_KB04"],
                "requested_scenario": "auto",
                "num_questions": 5,
                "difficulty": "auto",
                "recommended_mode": "live",
                "simulate_error": "",
            },
            {
                "preset_id": "kb3_api_503_unavailable",
                "title": "Kịch bản 3: Xử lý lỗi HTTP 503 UNAVAILABLE (không fallback sang Mock)",
                "description": "Kiểm chứng retry có giới hạn và giữ nguyên LIVE_GEMINI_API hoặc mô phỏng 503 ở chế độ Mock.",
                "oulad_student_id": primary_id,
                "simulated_student_code": "SIM_WEEK8_ERR503_03",
                "include_block_details": True,
                "target_block_codes": [],
                "requested_scenario": "auto",
                "num_questions": 5,
                "difficulty": "auto",
                "recommended_mode": "mock",
                "simulate_error": "service_unavailable",
            },
        ]
        state["students_catalog"] = {
            "students": students,
            "demo_presets": demo_presets,
            "blocks": blocks,
        }
        return state["students_catalog"]

    def _resolve_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(payload.get("profile"), dict):
            return payload["profile"]
        profile_id = str(payload.get("profile_id") or "").strip()
        profiles = load_sample_profiles()
        if profile_id:
            for item in profiles:
                if item.get("profile_id") == profile_id:
                    return item
            raise ValueError(f"Không tìm thấy hồ sơ mẫu với profile_id='{profile_id}'.")
        return profiles[0]

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            runtime=get_runtime_summary(),
            profiles=load_sample_profiles(),
            blocks=list(load_curriculum_blocks(include_general=False).values()),
            scenarios=list(ALLOWED_SCENARIOS),
        )

    @app.get("/api/config")
    def api_config():
        return jsonify({
            "runtime": get_runtime_summary(),
            "allowed_scenarios": list(ALLOWED_SCENARIOS),
        })

    @app.get("/api/profiles")
    def api_profiles():
        return jsonify({"profiles": load_sample_profiles()})

    @app.get("/api/blocks")
    def api_blocks():
        blocks = list(load_curriculum_blocks(include_general=False).values())
        return jsonify({"blocks": blocks})

    @app.get("/api/pipeline/students")
    def api_pipeline_students():
        return jsonify(_get_students_catalog())

    @app.post("/api/pipeline/run")
    def api_pipeline_run():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({
                "success": False,
                "http_status": 400,
                "step_failed": "input_validation",
                "error_type": "INVALID_REQUEST",
                "error_message": "Payload JSON không hợp lệ hoặc bị rỗng.",
            }), 400

        req_mode = str(payload.get("mode") or "live").strip().lower()
        if req_mode not in ("live", "mock"):
            return jsonify({
                "success": False,
                "http_status": 400,
                "step_failed": "input_validation",
                "error_type": "INVALID_MODE",
                "error_message": "Tham số 'mode' chỉ chấp nhận 'live' hoặc 'mock' (không tự động chuyển sang Mock).",
            }), 400
        use_mock = req_mode == "mock"

        simulate_error: Optional[str] = payload.get("simulate_error")
        if simulate_error == "":
            simulate_error = None
        if simulate_error is not None and not use_mock:
            return jsonify({
                "success": False,
                "http_status": 400,
                "step_failed": "input_validation",
                "error_type": "INVALID_SIMULATION_OPTION",
                "error_message": "Không thể bật simulate_error trong chế độ Live Gemini API; hãy chọn mode='mock' nếu muốn mô phỏng lỗi.",
            }), 400

        try:
            num_questions = int(payload.get("num_questions", 5))
            if not (1 <= num_questions <= 10):
                raise ValueError("ngoài khoảng [1, 10]")
        except (TypeError, ValueError):
            return jsonify({
                "success": False,
                "http_status": 400,
                "step_failed": "input_validation",
                "error_type": "INVALID_NUM_QUESTIONS",
                "error_message": "num_questions phải là số nguyên từ 1 đến 10.",
            }), 400

        difficulty = str(payload.get("difficulty") or "auto").strip().lower()
        if difficulty != "auto" and difficulty not in ALLOWED_DIFFICULTIES:
            return jsonify({
                "success": False,
                "http_status": 400,
                "step_failed": "input_validation",
                "error_type": "INVALID_DIFFICULTY",
                "error_message": f"difficulty='{difficulty}' không hợp lệ (nhận 'auto' hoặc {ALLOWED_DIFFICULTIES}).",
            }), 400

        req_scenario_raw = payload.get("requested_scenario")
        requested_scenario = str(req_scenario_raw).strip().lower() if req_scenario_raw is not None else "auto"
        if requested_scenario not in ("auto",) + ALLOWED_SCENARIOS:
            return jsonify({
                "success": False,
                "http_status": 400,
                "step_failed": "input_validation",
                "error_type": "INVALID_SCENARIO",
                "error_message": f"requested_scenario='{requested_scenario}' không hợp lệ.",
            }), 400

        raw_stu_id = payload.get("oulad_student_id")
        parsed_stu_id: Optional[int] = None
        student_row: Optional[Dict[str, Any]] = None
        if raw_stu_id is not None and str(raw_stu_id).strip() != "":
            try:
                parsed_stu_id = int(str(raw_stu_id).strip())
            except (TypeError, ValueError):
                return jsonify({
                    "success": False,
                    "http_status": 400,
                    "step_failed": "input_validation",
                    "error_type": "INVALID_STUDENT_ID",
                    "error_message": f"Mã sinh viên OULAD '{raw_stu_id}' phải là số nguyên.",
                }), 400
            student_row = load_oulad_student_row(parsed_stu_id)
            if student_row is None and not isinstance(payload.get("student_features"), dict):
                return jsonify({
                    "success": False,
                    "http_status": 404,
                    "step_failed": "input_validation",
                    "error_type": "STUDENT_NOT_FOUND",
                    "error_message": f"Không tìm thấy mã sinh viên OULAD #{parsed_stu_id} trong dữ liệu FFF-2013J.",
                }), 404

        student_features = payload.get("student_features")
        if student_features is None and student_row is not None:
            student_features = student_row["features"]

        if student_features is None:
            return jsonify({
                "success": False,
                "http_status": 400,
                "step_failed": "input_validation",
                "error_type": "MISSING_STUDENT_INPUT",
                "error_message": "Vui lòng chọn sinh viên mẫu, nhập 'oulad_student_id' hoặc truyền 'student_features' (24 đặc trưng).",
            }), 400

        include_block_details = bool(payload.get("include_block_details", True))
        block_records = payload.get("block_records") if include_block_details else None
        pipeline_oulad_id = parsed_stu_id if include_block_details else None

        target_block_codes = payload.get("target_block_codes")
        if not include_block_details:
            if not target_block_codes:
                target_block_codes = ["IT1108_KB03", "IT1108_KB04"]
            elif not isinstance(target_block_codes, list):
                return jsonify({
                    "success": False,
                    "http_status": 400,
                    "step_failed": "input_validation",
                    "error_type": "INVALID_TARGET_BLOCKS",
                    "error_message": "target_block_codes phải là danh sách mã khối IT1108.",
                }), 400
            else:
                invalid_codes = [c for c in target_block_codes if str(c).strip() not in VALID_IT1108_BLOCKS]
                if invalid_codes:
                    return jsonify({
                        "success": False,
                        "http_status": 400,
                        "step_failed": "input_validation",
                        "error_type": "INVALID_TARGET_BLOCKS",
                        "error_message": f"Mã khối IT1108 không hợp lệ: {invalid_codes}.",
                    }), 400

        sim_code = str(
            payload.get("simulated_student_code")
            or (f"SIM_OULAD_{parsed_stu_id}" if parsed_stu_id is not None else "SIM_WEBUI_01")
        ).strip() or "SIM_WEBUI_01"

        try:
            pl = _get_pipeline()
            res = pl.run(
                student_features=student_features,
                simulated_student_code=sim_code,
                oulad_student_id=pipeline_oulad_id,
                block_records=block_records,
                target_block_codes=target_block_codes,
                requested_scenario=requested_scenario,
                num_questions=num_questions,
                difficulty=difficulty,
                use_mock=use_mock,
                simulate_error=simulate_error,
            )
        except Exception as exc:
            return jsonify({
                "success": False,
                "http_status": 500,
                "step_failed": "pipeline_execution",
                "data_source": "MOCK_SIMULATED" if use_mock else "LIVE_GEMINI_API",
                "is_simulated": use_mock,
                "error_type": "INTERNAL_PIPELINE_ERROR",
                "error_message": redact_secret(f"Lỗi nội bộ khi thực thi pipeline: {exc}"),
            }), 500

        return jsonify(res), int(res.get("http_status") or 200)

    @app.post("/api/generate")
    def api_generate():
        payload = request.get_json(silent=True) or {}
        try:
            profile = _resolve_profile(payload)
        except ValueError as exc:
            return jsonify({"success": False, "http_status": 400, "error_message": str(exc)}), 400

        course_code = str(payload.get("course_code") or "IT1108").strip()
        course_name = str(payload.get("course_name") or "Nhập môn lập trình").strip()
        knowledge_block = (payload.get("knowledge_block") or "").strip() or None
        review_topics = payload.get("review_topics")
        try:
            num_questions = int(payload.get("num_questions", 5))
        except (TypeError, ValueError):
            return jsonify({"success": False, "http_status": 400, "error_message": "num_questions phải là số nguyên."}), 400

        difficulty = str(payload.get("difficulty") or "auto").strip().lower()
        scenario_type = str(payload.get("scenario_type") or "detailed_profile").strip()
        req_mode = str(payload.get("mode") or "auto").strip().lower()
        simulate_error: Optional[str] = payload.get("simulate_error")
        if simulate_error == "":
            simulate_error = None

        try:
            effective_mode = resolve_mode(req_mode)
        except ValueError as exc:
            return jsonify({"success": False, "http_status": 400, "error_message": str(exc)}), 400

        client = GeminiQuizClient(mode=effective_mode)
        try:
            res = client.generate_questions(
                course_code=course_code,
                course_name=course_name,
                knowledge_block=knowledge_block,
                review_topics=review_topics,
                num_questions=num_questions,
                difficulty=difficulty,
                scenario_type=scenario_type,
                profile=profile,
                mode=effective_mode,
                simulate_error=simulate_error,
            )
        except ValueError as exc:
            return jsonify({"success": False, "http_status": 400, "error_message": str(exc)}), 400

        metrics = compute_quiz_metrics(profile, res)
        res["metrics"] = metrics
        return jsonify(res), int(res.get("http_status") or 200)

    @app.post("/api/compare")
    def api_compare():
        payload = request.get_json(silent=True) or {}
        try:
            profile = _resolve_profile(payload)
        except ValueError as exc:
            return jsonify({"success": False, "http_status": 400, "error_message": str(exc)}), 400

        course_code = str(payload.get("course_code") or "IT1108").strip()
        course_name = str(payload.get("course_name") or "Nhập môn lập trình").strip()
        knowledge_block = (payload.get("knowledge_block") or "").strip() or None
        review_topics = payload.get("review_topics")
        try:
            num_questions = int(payload.get("num_questions", 5))
        except (TypeError, ValueError):
            return jsonify({"success": False, "http_status": 400, "error_message": "num_questions phải là số nguyên."}), 400
        difficulty = str(payload.get("difficulty") or "auto").strip().lower()

        req_mode = str(payload.get("mode") or "auto").strip().lower()
        try:
            effective_mode = resolve_mode(req_mode)
        except ValueError as exc:
            return jsonify({"success": False, "http_status": 400, "error_message": str(exc)}), 400

        client = GeminiQuizClient(mode=effective_mode)
        results: Dict[str, Any] = {}
        summary_rows = []
        for sc in ALLOWED_SCENARIOS:
            res = client.generate_questions(
                course_code=course_code,
                course_name=course_name,
                knowledge_block=knowledge_block,
                review_topics=review_topics,
                num_questions=num_questions,
                difficulty=difficulty,
                scenario_type=sc,
                profile=profile,
                mode=effective_mode,
            )
            metrics = compute_quiz_metrics(profile, res)
            res["metrics"] = metrics
            results[sc] = res
            summary_rows.append(metrics)

        return jsonify({
            "success": True,
            "effective_mode": effective_mode,
            "summary": summary_rows,
            "results": results,
        }), 200

    return app


app = create_app()


if __name__ == "__main__":
    print("=== Smart LMS & Generative AI - Tuần 7+8 GenAI Playground ===")
    print("Truy cập: http://127.0.0.1:5057")
    app.run(host="127.0.0.1", port=5057, debug=False)
