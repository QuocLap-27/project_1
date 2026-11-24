# -*- coding: utf-8 -*-
"""
Bộ Unit Tests kiểm thử toàn diện phân hệ GenAI (Tuần 7+8) theo Prompt 2:
- Đọc Mapping CSV read-only từ Tuần 5+6.
- Chuẩn hóa 6 trường JSON bắt buộc: question, options (4 lựa chọn), correct_answer, explanation, topic, difficulty.
- Sinh câu hỏi với đủ 7 tham số cấu hình + 3 kịch bản Prompt (tái sử dụng build_prompt_scenarios).
- Bảo mật ẩn danh PII Guard (NFR-02), sample_profiles.json không chứa PII/dữ liệu thật.
- Tự động nhận diện Vertex AI Express ('AQ.') vs AI Studio ('AIza'), ẩn API key khỏi summary & error log.
- Xử lý lỗi Timeout <= 20s (NFR-01), Rate Limit 429 (503), Invalid JSON / Schema (500) (NFR-03).
"""

from __future__ import annotations

import json
import os
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

WEEK78_DIR = Path(__file__).resolve().parent.parent
if str(WEEK78_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK78_DIR))

from src.config import get_api_provider, get_runtime_summary, redact_secret  # noqa: E402
from src.curriculum_loader import (  # noqa: E402
    get_block_context,
    load_assessment_mappings,
    load_curriculum_blocks,
)
from src.experiment_runner import load_sample_profiles, run_experiments  # noqa: E402
from src.gemini_client import GeminiQuizClient  # noqa: E402
from src.pii_guard import FORBIDDEN_PII_KEYS, PIILeakageError, sanitize_profile_for_prompt, scan_prompt_for_pii  # noqa: E402
from src.prompt_builder import SYSTEM_INSTRUCTION_IT1108, build_all_prompt_scenarios  # noqa: E402
from src.schemas import CORE_QUESTION_KEYS, parse_and_validate_quiz_json  # noqa: E402


class TestCurriculumLoader(unittest.TestCase):
    def test_load_6_it1108_blocks_readonly(self):
        blocks = load_curriculum_blocks(include_general=False)
        self.assertEqual(len(blocks), 6)
        for i in range(1, 7):
            code = f"IT1108_KB0{i}"
            self.assertIn(code, blocks)
            self.assertTrue(blocks[code]["it1108_block_name"])
            self.assertTrue(blocks[code]["it1108_lessons"])

    def test_exclude_exam_assessment_mapping(self):
        mappings = load_assessment_mappings(exclude_exam=True)
        self.assertEqual(len(mappings), 12)
        self.assertTrue(all(m["mapping_type"] != "Excluded_Exam" for m in mappings))

    def test_get_block_context_kb03(self):
        ctx = get_block_context("IT1108_KB03")
        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertEqual(ctx["it1108_block_code"], "IT1108_KB03")
        self.assertGreaterEqual(len(ctx["simulated_assessments"]), 2)


class TestSecurityConfigAndPIIGuard(unittest.TestCase):
    def test_sample_profiles_contain_no_pii_or_sensitive_fields(self):
        profiles = load_sample_profiles()
        self.assertGreaterEqual(len(profiles), 4)
        for p in profiles:
            self.assertNotIn("_raw_test_metadata", p)
            for forbidden_k in FORBIDDEN_PII_KEYS:
                self.assertNotIn(forbidden_k, p)

    def test_sanitize_strips_synthetic_pii_keys(self):
        synthetic = {
            "risk_level": "HIGH",
            "risk_probability": 0.72,
            "_raw_test_metadata": {
                "id_student": 99999,
                "full_name": "Test Synthetic",
                "email": "synthetic@example.com",
            },
            "weak_blocks": [{"it1108_block_code": "IT1108_KB03", "submission_vs_score_status": "MISSING_SUBMISSION"}],
        }
        clean, stripped, forbidden = sanitize_profile_for_prompt(synthetic)
        self.assertNotIn("_raw_test_metadata", clean)
        self.assertIn("99999", forbidden)
        self.assertIn("Test Synthetic", forbidden)
        self.assertTrue(any("_raw_test_metadata" in k for k in stripped))

    def test_scan_prompt_rejects_leaked_pii(self):
        with self.assertRaises(PIILeakageError):
            scan_prompt_for_pii(
                "Sinh viên Test Synthetic (email: synthetic@example.com, id_student: 99999)",
                forbidden_values=["Test Synthetic", "99999"],
                strict=True,
            )

    def test_provider_auto_detection_and_secret_redaction(self):
        with patch.dict(os.environ, {"GEMINI_API_PROVIDER": "auto"}, clear=False):
            self.assertEqual(get_api_provider("AQ.SampleStudioKey12345"), "ai_studio")
            self.assertEqual(get_api_provider("AIzaSampleStudioKey12345"), "ai_studio")

        with patch.dict(os.environ, {"GEMINI_API_PROVIDER": "vertex_express"}, clear=False):
            self.assertEqual(get_api_provider("AQ.SampleVertexKey12345"), "vertex_express")

        fake_secret = "AQ.FakeSecretValue987654321"
        redacted = redact_secret(f"Request failed with key {fake_secret}", fake_secret)
        self.assertNotIn(fake_secret, redacted)
        self.assertIn("[REDACTED_API_KEY]", redacted)

        summary = get_runtime_summary()
        self.assertNotIn("api_key_preview", summary)
        self.assertIn("has_api_key", summary)
        self.assertIn("api_provider", summary)


class TestPromptBuilderAndSevenParameters(unittest.TestCase):
    def setUp(self):
        self.profile = load_sample_profiles()[0]

    def test_system_instruction_forbids_absolute_self_diagnosis(self):
        self.assertIn("KHÔNG tự chẩn đoán", SYSTEM_INSTRUCTION_IT1108)
        self.assertIn("kết luận tuyệt đối", SYSTEM_INSTRUCTION_IT1108)

    def test_three_scenarios_differentiated_and_pii_safe(self):
        all_prompts = build_all_prompt_scenarios(self.profile)
        self.assertEqual(set(all_prompts.keys()), {"baseline", "risk_level", "detailed_profile"})

        for sc, bundle in all_prompts.items():
            self.assertTrue(bundle["pii_report"]["pii_safe"], f"Kịch bản {sc} bị rò rỉ PII!")

        baseline_prompt = all_prompts["baseline"]["user_prompt"]
        self.assertNotIn("0.72", baseline_prompt)
        self.assertNotIn("85.0%", baseline_prompt)

        risk_prompt = all_prompts["risk_level"]["user_prompt"]
        self.assertIn("0.72", risk_prompt)
        self.assertIn("CAO", risk_prompt)
        self.assertNotIn("42.0/100", risk_prompt)

        detailed_prompt = all_prompts["detailed_profile"]["user_prompt"]
        self.assertIn("85.0%", detailed_prompt)
        self.assertIn("42.0/100", detailed_prompt)
        self.assertIn("guided_basic", detailed_prompt)
        self.assertIn("code_trace", detailed_prompt)

    def test_generate_questions_with_all_7_parameters_in_mock_mode(self):
        client = GeminiQuizClient(mode="mock")
        res = client.generate_questions(
            course_code="IT1108",
            course_name="Nhập môn lập trình",
            knowledge_block="IT1108_KB03",
            review_topics=["Vòng lặp for", "Cấu trúc switch-case"],
            num_questions=3,
            difficulty="easy",
            scenario_type="baseline",
            mode="mock",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["http_status"], 200)
        quiz = res["quiz_data"]
        self.assertEqual(quiz["course_code"], "IT1108")
        self.assertEqual(quiz["course_name"], "Nhập môn lập trình")
        self.assertEqual(quiz["target_blocks"], ["IT1108_KB03"])
        self.assertEqual(len(quiz["questions"]), 3)
        for q in quiz["questions"]:
            for core_key in CORE_QUESTION_KEYS:
                self.assertIn(core_key, q)
            self.assertEqual(q["difficulty"], "easy")
            self.assertEqual(set(q["options"].keys()), {"A", "B", "C", "D"})
            self.assertIn(q["correct_answer"], ("A", "B", "C", "D"))


class TestSchemaValidator(unittest.TestCase):
    def setUp(self):
        self.client = GeminiQuizClient(mode="mock")
        self.profile = load_sample_profiles()[0]

    def test_valid_quiz_passes_6_core_keys_and_isjson(self):
        res = self.client.generate_quiz(self.profile, scenario_type="detailed_profile", mode="mock")
        self.assertTrue(res["success"])
        self.assertEqual(res["http_status"], 200)
        self.assertTrue(res["validation"]["is_valid"])
        self.assertTrue(res["validation"]["sql_isjson_compatible"])
        self.assertEqual(len(res["quiz_data"]["questions"]), 5)
        for q in res["quiz_data"]["questions"]:
            for key in CORE_QUESTION_KEYS:
                self.assertIn(key, q)
            self.assertEqual(len(q["options"]), 4)

    def test_normalizes_list_options_and_markdown_fence(self):
        res = self.client.generate_quiz(self.profile, scenario_type="baseline", num_questions=1, mode="mock")
        obj = dict(res["quiz_data"])
        q0 = dict(obj["questions"][0])
        q0["options"] = [q0["options"]["A"], q0["options"]["B"], q0["options"]["C"], q0["options"]["D"]]
        obj["questions"] = [q0]

        fenced = f"```json\n{json.dumps(obj, ensure_ascii=False)}\n```"
        val = parse_and_validate_quiz_json(fenced, expected_num_questions=1)
        self.assertTrue(val.is_valid, f"Unexpected errors: {val.errors}")
        self.assertEqual(set(val.normalized_data["questions"][0]["options"].keys()), {"A", "B", "C", "D"})

    def test_rejects_invalid_json_and_schema_defects(self):
        bad_json = parse_and_validate_quiz_json("{invalid json")
        self.assertFalse(bad_json.json_valid)
        self.assertFalse(bad_json.is_valid)

        incomplete_obj = {
            "course_code": "IT1108",
            "course_name": "Nhập môn lập trình",
            "scenario_type": "baseline",
            "target_blocks": ["INVALID_BLK"],
            "difficulty_strategy": "test",
            "study_advice": "test advice",
            "questions": [
                {
                    "question_number": 1,
                    "it1108_block_code": "INVALID_BLK",
                    "topic": "Sai",
                    "difficulty": "easy",
                    "question_type": "concept",
                    "question": "Câu hỏi test hợp lệ dài hơn 10 ký tự?",
                    "code_snippet": "",
                    "options": {"A": "Giống nhau", "B": "Giống nhau", "C": "Khác", "D": "Khác 2"},
                    "correct_answer": "Z",
                    "hint": "gợi ý",
                    "explanation": "ngắn",
                }
            ],
        }
        bad_schema = parse_and_validate_quiz_json(json.dumps(incomplete_obj, ensure_ascii=False))
        self.assertTrue(bad_schema.json_valid)
        self.assertFalse(bad_schema.schema_valid)
        self.assertGreaterEqual(len(bad_schema.errors), 5)


class TestErrorHandlingNFR01AndNFR03(unittest.TestCase):
    def setUp(self):
        self.client = GeminiQuizClient(mode="mock", timeout_seconds=20.0, max_retries=1)
        self.profile = load_sample_profiles()[0]

    def test_timeout_returns_503(self):
        res = self.client.generate_quiz(self.profile, mode="mock", simulate_error="timeout")
        self.assertFalse(res["success"])
        self.assertEqual(res["http_status"], 503)
        self.assertEqual(res["error_type"], "TIMEOUT")

    def test_rate_limit_returns_503(self):
        res = self.client.generate_quiz(self.profile, mode="mock", simulate_error="rate_limit")
        self.assertFalse(res["success"])
        self.assertEqual(res["http_status"], 503)
        self.assertEqual(res["error_type"], "RATE_LIMIT")

    def test_malformed_json_returns_500(self):
        res = self.client.generate_quiz(self.profile, mode="mock", simulate_error="malformed_json")
        self.assertFalse(res["success"])
        self.assertEqual(res["http_status"], 500)
        self.assertEqual(res["error_type"], "INVALID_JSON")

    def test_schema_error_returns_500(self):
        res = self.client.generate_quiz(self.profile, mode="mock", simulate_error="schema_error")
        self.assertFalse(res["success"])
        self.assertEqual(res["http_status"], 500)
        self.assertEqual(res["error_type"], "SCHEMA_VALIDATION_ERROR")

    def test_live_mode_without_api_key_returns_401(self):
        live_client = GeminiQuizClient(mode="live", api_key="")
        res = live_client.generate_quiz(self.profile, mode="live")
        self.assertFalse(res["success"])
        self.assertEqual(res["http_status"], 401)
        self.assertEqual(res["error_type"], "MISSING_API_KEY")

    def test_mocked_live_transport_success_timeout_and_secret_redaction(self):
        fake_key = "AQ.FakeVertexTestKey123456"
        live_client = GeminiQuizClient(mode="live", api_key=fake_key, max_retries=0)
        mock_sample = GeminiQuizClient(mode="mock").generate_quiz(
            self.profile, scenario_type="detailed_profile", mode="mock"
        )
        valid_json_text = json.dumps(mock_sample["quiz_data"], ensure_ascii=False)

        with patch.dict(os.environ, {"GEMINI_API_PROVIDER": "auto"}, clear=False), patch.object(
            live_client,
            "_call_live_via_sdk",
            side_effect=ImportError("google-genai not installed"),
        ), patch.object(
            live_client,
            "_call_live_via_rest",
            return_value=(valid_json_text, {"prompt_tokens": 300, "output_tokens": 600}, "gemini-rest-api"),
        ):
            res_ok = live_client.generate_quiz(self.profile, scenario_type="detailed_profile", mode="live")
            self.assertTrue(res_ok["success"])
            self.assertEqual(res_ok["http_status"], 200)
            self.assertEqual(res_ok["mode_used"], "live")
            self.assertEqual(res_ok["api_provider"], "ai_studio")

        with patch.object(
            live_client,
            "_call_live_via_sdk",
            side_effect=ImportError("google-genai not installed"),
        ), patch.object(
            live_client,
            "_call_live_via_rest",
            side_effect=RuntimeError(f"Connection error with {fake_key}"),
        ):
            res_err = live_client.generate_quiz(self.profile, scenario_type="detailed_profile", mode="live")
            self.assertFalse(res_err["success"])
            self.assertEqual(res_err["http_status"], 503)
            self.assertNotIn(fake_key, res_err["error_message"])

        with patch.object(
            live_client,
            "_call_live_via_sdk",
            side_effect=ImportError("google-genai not installed"),
        ), patch.object(
            live_client,
            "_call_live_via_rest",
            side_effect=socket.timeout("timed out"),
        ):
            res_to = live_client.generate_quiz(self.profile, scenario_type="detailed_profile", mode="live")
            self.assertFalse(res_to["success"])
            self.assertEqual(res_to["http_status"], 503)
            self.assertEqual(res_to["error_type"], "TIMEOUT")


class TestExperimentPedagogicalComparison(unittest.TestCase):
    def test_detailed_profile_outperforms_baseline(self):
        payload = run_experiments(
            mode="mock",
            repeats=1,
            profile_filter=["profile_01_high_mixed"],
            save_outputs=False,
        )
        summary = {row["scenario_type"]: row for row in payload["scenario_summary"]}
        self.assertEqual(summary["baseline"]["schema_valid_pct"], 100.0)
        self.assertEqual(summary["detailed_profile"]["schema_valid_pct"], 100.0)
        self.assertGreater(
            summary["detailed_profile"]["avg_pedagogical_alignment_score"],
            summary["baseline"]["avg_pedagogical_alignment_score"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
