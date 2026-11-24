# -*- coding: utf-8 -*-
"""
Bộ kiểm thử Tuần 8: Kết nối Machine Learning -> Curriculum/Topic Mapping -> Generative AI.
Kiểm chứng toàn bộ:
1. Suy luận ML thật từ predict.py + best_model.joblib (24 đặc trưng, ngưỡng 0.35 / 0.65,
   3 nhóm trạng thái NO_SUBMISSION / PARTIAL_SUBMISSION_LOW_SCORE / SUBMITTED_LOW_SCORE, và xử lý đầu vào lỗi).
2. Ánh xạ khối kiến thức từ OULAD FFF-2013J (có phân vị thật) và ca thiếu chi tiết khối (không bịa số).
3. Lựa chọn Prompt Scenario (detailed_profile -> fallback risk_level; baseline chỉ khi chủ động chọn).
4. Gia cố API: retry tối đa 1 lần cho INVALID_JSON, xử lý riêng 503 UNAVAILABLE & TIMEOUT,
   cách ly lỗi từng câu hỏi, không tự chuyển sang Mock, và che API key khỏi log/error.
"""

from __future__ import annotations

import json
import logging
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

WEEK78_DIR = Path(__file__).resolve().parent.parent
if str(WEEK78_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK78_DIR))

from src.gemini_client import GeminiQuizClient, _ServiceUnavailableAPIError  # noqa: E402
from src.ml_genai_pipeline import SmartLMSPipeline, select_prompt_scenario  # noqa: E402
from src.ml_predictor_bridge import MLPredictorBridge, find_representative_oulad_students  # noqa: E402
from src.schemas import CORE_QUESTION_KEYS, parse_and_validate_quiz_json  # noqa: E402
from src.topic_mapper import OULADBlockAnalyzer, build_personalized_profile  # noqa: E402


class TestWeek8MLAndMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = MLPredictorBridge()
        cls.reps = find_representative_oulad_students()
        cls.analyzer = OULADBlockAnalyzer()

    def test_ml_prediction_real_best_model_thresholds_and_behaviors(self):
        self.assertEqual(len(self.bridge.expected_features), 24)

        for status_key in ("NO_SUBMISSION", "PARTIAL_SUBMISSION_LOW_SCORE", "SUBMITTED_LOW_SCORE", "NORMAL"):
            self.assertIn(status_key, self.reps)
            res = self.bridge.predict_student_risk(self.reps[status_key]["features"])
            self.assertTrue(res["success"], f"ML failed for {status_key}: {res.get('error')}")
            self.assertEqual(res["decision_threshold"], 0.35)
            self.assertEqual(res["submission_behavior"]["status_code"], status_key)
            prob = res["risk_probability"]
            if prob < 0.35:
                self.assertEqual(res["risk_level"], "LOW")
            elif prob < 0.65:
                self.assertEqual(res["risk_level"], "MEDIUM")
            else:
                self.assertEqual(res["risk_level"], "HIGH")

    def test_ml_invalid_input_returns_clear_error(self):
        bad_features = dict(self.reps["NO_SUBMISSION"]["features"])
        bad_features.pop("avg_score")
        res_missing = self.bridge.predict_student_risk(bad_features)
        self.assertFalse(res_missing["success"])
        self.assertIn("Thiếu", res_missing["error"])

        bad_range = dict(self.reps["NO_SUBMISSION"]["features"])
        bad_range["assessment_completion_rate"] = 1.5
        res_range = self.bridge.predict_student_risk(bad_range)
        self.assertFalse(res_range["success"])
        self.assertIn("[0, 1]", res_range["error"])

    def test_topic_mapping_real_oulad_percentile_vs_missing_block_detail(self):
        stu = self.reps["PARTIAL_SUBMISSION_LOW_SCORE"]
        ml_res = self.bridge.predict_student_risk(stu["features"])

        full_profile = build_personalized_profile(
            simulated_student_code="SIM_TEST_FULL",
            ml_result=ml_res,
            oulad_student_id=stu["id_student"],
            analyzer=self.analyzer,
        )
        self.assertTrue(full_profile["has_detailed_block_data"])
        self.assertTrue(full_profile["has_class_percentile"])
        self.assertGreaterEqual(len(full_profile["weak_blocks"]), 1)
        for blk in full_profile["weak_blocks"]:
            self.assertIsNotNone(blk["lower_than_class_pct"])
            self.assertGreaterEqual(blk["lower_than_class_pct"], 0.0)
            self.assertLessEqual(blk["lower_than_class_pct"], 100.0)

        sc_full, _ = select_prompt_scenario(full_profile)
        self.assertEqual(sc_full, "detailed_profile")

        # Ca thiếu dữ liệu chi tiết từng khối -> KHÔNG bịa điểm hay phân vị, tự fallback sang risk_level
        partial_profile = build_personalized_profile(
            simulated_student_code="SIM_TEST_PARTIAL",
            ml_result=ml_res,
            oulad_student_id=None,
            block_records=None,
            target_block_codes=["IT1108_KB03", "IT1108_KB04"],
        )
        self.assertFalse(partial_profile["has_detailed_block_data"])
        self.assertFalse(partial_profile["has_class_percentile"])
        for blk in partial_profile["weak_blocks"]:
            self.assertIsNone(blk["avg_submitted_score"])
            self.assertIsNone(blk["lower_than_class_pct"])
            self.assertIsNone(blk["completion_rate"])

        sc_fallback, reason = select_prompt_scenario(partial_profile)
        self.assertEqual(sc_fallback, "risk_level")
        self.assertIn("fallback", reason.lower())

        sc_base, _ = select_prompt_scenario(full_profile, requested_scenario="baseline")
        self.assertEqual(sc_base, "baseline")


class TestWeek8APIHardeningAndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = SmartLMSPipeline()
        cls.reps = find_representative_oulad_students()
        cls.stu = cls.reps["PARTIAL_SUBMISSION_LOW_SCORE"]

    def _sample_valid_quiz_json(self, scenario: str = "detailed_profile") -> str:
        res = GeminiQuizClient(mode="mock").generate_questions(
            knowledge_block="IT1108_KB03",
            num_questions=5,
            scenario_type=scenario,
            mode="mock",
        )
        return json.dumps(res["quiz_data"], ensure_ascii=False)

    def test_invalid_json_retries_once_and_recovers(self):
        fake_key = "AQ.FakeWeek8TestKey999999"
        client = GeminiQuizClient(mode="live", api_key=fake_key, max_retries=0, max_json_retries=1, timeout_seconds=30.0)
        valid_json = self._sample_valid_quiz_json("detailed_profile")

        with patch.object(
            client,
            "_call_live_via_sdk",
            side_effect=ImportError("no sdk"),
        ), patch.object(
            client,
            "_call_live_via_rest",
            side_effect=[
                ("{broken json without closing", {"prompt_tokens": 100, "output_tokens": 20}, "gemini-rest-api"),
                (valid_json, {"prompt_tokens": 110, "output_tokens": 500}, "gemini-rest-api"),
            ],
        ) as rest_mock:
            res = self.pipeline.run(
                student_features=self.stu["features"],
                simulated_student_code="SIM_JSON_RETRY_OK",
                oulad_student_id=self.stu["id_student"],
                use_mock=False,
                genai_client=client,
            )
            self.assertEqual(rest_mock.call_count, 2)
            self.assertTrue(res["success"])
            self.assertEqual(res["http_status"], 200)
            self.assertEqual(res["genai_result"]["json_retries_used"], 1)
            self.assertEqual(res["data_source"], "LIVE_GEMINI_API")
            self.assertFalse(res["is_simulated"])

    def test_invalid_json_stops_after_one_retry_and_never_switches_to_mock(self):
        fake_key = "AQ.FakeWeek8TestKey999999"
        client = GeminiQuizClient(mode="live", api_key=fake_key, max_retries=0, max_json_retries=1, timeout_seconds=30.0)

        with patch.object(
            client,
            "_call_live_via_sdk",
            side_effect=ImportError("no sdk"),
        ), patch.object(
            client,
            "_call_live_via_rest",
            return_value=("{still broken json", {"prompt_tokens": 100, "output_tokens": 20}, "gemini-rest-api"),
        ) as rest_mock:
            res = self.pipeline.run(
                student_features=self.stu["features"],
                simulated_student_code="SIM_JSON_RETRY_FAIL",
                oulad_student_id=self.stu["id_student"],
                use_mock=False,
                genai_client=client,
            )
            self.assertEqual(rest_mock.call_count, 2)
            self.assertFalse(res["success"])
            self.assertEqual(res["http_status"], 500)
            self.assertEqual(res["error_type"], "INVALID_JSON")
            self.assertEqual(res["genai_result"]["json_retries_used"], 1)
            self.assertTrue(res["can_retry"])
            self.assertFalse(res["is_simulated"])
            self.assertEqual(res["data_source"], "LIVE_GEMINI_API")

    def test_503_unavailable_and_timeout_handled_with_key_redaction(self):
        fake_key = "AQ.SecretShouldNeverBeLogged12345"
        client = GeminiQuizClient(mode="live", api_key=fake_key, max_retries=1, max_json_retries=1, timeout_seconds=30.0)

        with self.assertLogs("smart_lms.genai", level=logging.WARNING) as log_cm:
            with patch.object(
                client,
                "_call_live_via_sdk",
                side_effect=ImportError("no sdk"),
            ), patch.object(
                client,
                "_call_live_via_rest",
                side_effect=_ServiceUnavailableAPIError(
                    f"HTTP 503 UNAVAILABLE from Gemini API with secret {fake_key}"
                ),
            ) as rest_mock:
                res_503 = self.pipeline.run(
                    student_features=self.stu["features"],
                    simulated_student_code="SIM_503_TEST",
                    oulad_student_id=self.stu["id_student"],
                    use_mock=False,
                    genai_client=client,
                )
        self.assertEqual(rest_mock.call_count, 2)
        self.assertFalse(res_503["success"])
        self.assertEqual(res_503["http_status"], 503)
        self.assertEqual(res_503["error_type"], "SERVICE_UNAVAILABLE")
        self.assertTrue(res_503["can_retry"])
        self.assertFalse(res_503["is_simulated"])
        self.assertNotIn(fake_key, res_503["error_message"])
        joined_logs = "\n".join(log_cm.output)
        self.assertNotIn(fake_key, joined_logs)

        with patch.object(
            client,
            "_call_live_via_sdk",
            side_effect=ImportError("no sdk"),
        ), patch.object(
            client,
            "_call_live_via_rest",
            side_effect=socket.timeout("timed out"),
        ):
            res_to = self.pipeline.run(
                student_features=self.stu["features"],
                simulated_student_code="SIM_TIMEOUT_TEST",
                oulad_student_id=self.stu["id_student"],
                use_mock=False,
                genai_client=client,
            )
            self.assertFalse(res_to["success"])
            self.assertEqual(res_to["http_status"], 503)
            self.assertEqual(res_to["error_type"], "TIMEOUT")
            self.assertTrue(res_to["can_retry"])
            self.assertFalse(res_to["is_simulated"])

    def test_single_bad_question_does_not_crash_validator(self):
        raw_obj = json.loads(self._sample_valid_quiz_json("detailed_profile"))
        raw_obj["questions"][2]["correct_answer"] = "INVALID"
        val = parse_and_validate_quiz_json(json.dumps(raw_obj, ensure_ascii=False), expected_num_questions=5)
        self.assertTrue(val.json_valid)
        self.assertFalse(val.is_valid)
        self.assertEqual(val.invalid_question_indices, [2])
        self.assertEqual(len(val.valid_questions), 4)

    def test_end_to_end_pipeline_explicit_mock_mode_full_and_fallback(self):
        # Đủ dữ liệu chi tiết -> detailed_profile
        res_full = self.pipeline.run(
            student_features=self.stu["features"],
            simulated_student_code="SIM_MOCK_FULL",
            oulad_student_id=self.stu["id_student"],
            requested_scenario="auto",
            num_questions=5,
            use_mock=True,
        )
        self.assertTrue(res_full["success"])
        self.assertEqual(res_full["data_source"], "MOCK_SIMULATED")
        self.assertTrue(res_full["is_simulated"])
        self.assertIn("MOCK", res_full["mode_notice"])
        self.assertEqual(res_full["prompt_selection"]["selected_scenario"], "detailed_profile")
        self.assertEqual(len(res_full["questions"]), 5)
        for q in res_full["questions"]:
            for k in CORE_QUESTION_KEYS:
                self.assertIn(k, q)
            self.assertEqual(set(q["options"].keys()), {"A", "B", "C", "D"})
            self.assertIn(q["correct_answer"], ("A", "B", "C", "D"))

        # Thiếu chi tiết từng khối -> fallback risk_level
        res_fallback = self.pipeline.run(
            student_features=self.stu["features"],
            simulated_student_code="SIM_MOCK_FALLBACK",
            oulad_student_id=None,
            target_block_codes=["IT1108_KB03", "IT1108_KB04"],
            requested_scenario="auto",
            num_questions=5,
            use_mock=True,
        )
        self.assertTrue(res_fallback["success"])
        self.assertEqual(res_fallback["prompt_selection"]["selected_scenario"], "risk_level")
        self.assertEqual(len(res_fallback["questions"]), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
