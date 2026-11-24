# -*- coding: utf-8 -*-
"""
Module: tests/test_webui_and_acceptance.py
Dự án: SMART LMS & GENERATIVE AI (Tuần 7–8 – Kiểm thử nghiệm thu Web UI & Xử lý lỗi Live Gemini)
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

WEEK78_DIR = Path(__file__).resolve().parent.parent
if str(WEEK78_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK78_DIR))

import playground.app as playground_app_mod  # noqa: E402
from src.config import get_gemini_api_key  # noqa: E402
from src.gemini_client import GeminiQuizClient, _ServiceUnavailableAPIError  # noqa: E402
from src.ml_genai_pipeline import SmartLMSPipeline  # noqa: E402


class TestWeek8WebUIAndAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = SmartLMSPipeline()
        cls.app = playground_app_mod.create_app(pipeline=cls.pipeline)
        cls.client = cls.app.test_client()

    def _valid_live_quiz_json(self, scenario: str = "detailed_profile") -> str:
        res = GeminiQuizClient(mode="mock").generate_questions(
            knowledge_block="IT1108_KB03",
            num_questions=5,
            scenario_type=scenario,
            mode="mock",
        )
        return json.dumps(res["quiz_data"], ensure_ascii=False)

    def test_get_endpoints_and_no_secret_leak(self):
        real_key = get_gemini_api_key()
        for path in ("/", "/api/config", "/api/profiles", "/api/blocks", "/api/pipeline/students"):
            resp = self.client.get(path)
            self.assertEqual(resp.status_code, 200, f"GET {path} failed with {resp.status_code}")
            body_text = resp.get_data(as_text=True)
            if real_key:
                self.assertNotIn(real_key, body_text, f"API Key bị lộ trong phản hồi GET {path}")

        catalog = self.client.get("/api/pipeline/students").get_json()
        levels = {s["risk_level"] for s in catalog["students"]}
        self.assertTrue({"HIGH", "MEDIUM", "LOW"}.issubset(levels))
        self.assertEqual(len(catalog["demo_presets"]), 3)
        self.assertEqual(len(catalog["blocks"]), 6)

    def test_pipeline_run_invalid_and_missing_inputs(self):
        # Payload không phải JSON
        r_bad_json = self.client.post("/api/pipeline/run", data="not-json", content_type="text/plain")
        self.assertEqual(r_bad_json.status_code, 400)
        self.assertEqual(r_bad_json.get_json()["error_type"], "INVALID_REQUEST")

        # Thiếu cả oulad_student_id lẫn student_features
        r_missing = self.client.post("/api/pipeline/run", json={"mode": "mock"})
        self.assertEqual(r_missing.status_code, 400)
        self.assertEqual(r_missing.get_json()["error_type"], "MISSING_STUDENT_INPUT")

        # oulad_student_id không phải số nguyên
        r_bad_id = self.client.post("/api/pipeline/run", json={"oulad_student_id": "abc", "mode": "mock"})
        self.assertEqual(r_bad_id.status_code, 400)
        self.assertEqual(r_bad_id.get_json()["error_type"], "INVALID_STUDENT_ID")

        # oulad_student_id không tồn tại
        r_not_found = self.client.post("/api/pipeline/run", json={"oulad_student_id": 999999999, "mode": "mock"})
        self.assertEqual(r_not_found.status_code, 404)
        self.assertEqual(r_not_found.get_json()["error_type"], "STUDENT_NOT_FOUND")

        # student_features thiếu đặc trưng ML
        r_bad_ml = self.client.post(
            "/api/pipeline/run",
            json={"student_features": {"num_assessments": 3}, "mode": "mock"},
        )
        self.assertEqual(r_bad_ml.status_code, 400)
        data_bad_ml = r_bad_ml.get_json()
        self.assertEqual(data_bad_ml["step_failed"], "ml_prediction")
        self.assertEqual(data_bad_ml["error_type"], "INVALID_ML_INPUT")

        # Không cho simulate_error ở chế độ Live
        r_bad_sim = self.client.post(
            "/api/pipeline/run",
            json={"oulad_student_id": 54882, "mode": "live", "simulate_error": "service_unavailable"},
        )
        self.assertEqual(r_bad_sim.status_code, 400)
        self.assertEqual(r_bad_sim.get_json()["error_type"], "INVALID_SIMULATION_OPTION")

    def test_pipeline_run_full_oulad_profile_mock(self):
        resp = self.client.post(
            "/api/pipeline/run",
            json={
                "oulad_student_id": 54882,
                "simulated_student_code": "SIM_WEBUI_FULL_01",
                "include_block_details": True,
                "requested_scenario": "auto",
                "num_questions": 5,
                "mode": "mock",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data_source"], "MOCK_SIMULATED")
        self.assertTrue(data["is_simulated"])
        self.assertEqual(data["ml_prediction"]["risk_level"], "HIGH")
        self.assertEqual(data["submission_behavior"]["status_code"], "PARTIAL_SUBMISSION_LOW_SCORE")
        self.assertTrue(data["personalized_profile"]["has_detailed_block_data"])
        self.assertTrue(data["personalized_profile"]["has_class_percentile"])
        self.assertEqual(data["prompt_selection"]["selected_scenario"], "detailed_profile")
        self.assertEqual(len(data["questions"]), 5)

    def test_pipeline_run_missing_block_detail_fallback_risk_level(self):
        resp = self.client.post(
            "/api/pipeline/run",
            json={
                "oulad_student_id": 54882,
                "simulated_student_code": "SIM_WEBUI_PARTIAL_02",
                "include_block_details": False,
                "target_block_codes": ["IT1108_KB03", "IT1108_KB04"],
                "requested_scenario": "auto",
                "num_questions": 5,
                "mode": "mock",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data_source"], "MOCK_SIMULATED")
        self.assertTrue(data["is_simulated"])
        self.assertFalse(data["personalized_profile"]["has_detailed_block_data"])
        self.assertFalse(data["personalized_profile"]["has_class_percentile"])
        for blk in data["personalized_profile"]["weak_blocks"]:
            self.assertIsNone(blk["avg_submitted_score"])
            self.assertIsNone(blk["lower_than_class_pct"])
            self.assertIsNone(blk["completion_rate"])
        self.assertEqual(data["prompt_selection"]["selected_scenario"], "risk_level")
        self.assertEqual(len(data["questions"]), 5)

    def test_pipeline_run_mocked_live_gemini_success(self):
        fake_key = "AQ.WebUILiveSuccessKey12345"
        valid_json = self._valid_live_quiz_json("detailed_profile")
        with patch.dict(os.environ, {"GEMINI_API_KEY": fake_key}), patch.object(
            GeminiQuizClient, "_call_live_via_sdk", side_effect=ImportError("no sdk")
        ), patch.object(
            GeminiQuizClient,
            "_call_live_via_rest",
            return_value=(valid_json, {"prompt_tokens": 200, "output_tokens": 450}, "gemini-rest-api"),
        ):
            resp = self.client.post(
                "/api/pipeline/run",
                json={
                    "oulad_student_id": 54882,
                    "simulated_student_code": "SIM_WEBUI_LIVE_01",
                    "include_block_details": True,
                    "mode": "live",
                },
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data_source"], "LIVE_GEMINI_API")
        self.assertFalse(data["is_simulated"])
        self.assertEqual(data["genai_result"]["attempts_used"], 1)
        self.assertEqual(len(data["questions"]), 5)
        self.assertNotIn(fake_key, resp.get_data(as_text=True))

    def test_pipeline_run_mocked_live_503_retries_bounded_no_mock_fallback(self):
        fake_key = "AQ.WebUISecret503Key999"
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": fake_key,
                "GEMINI_MAX_RETRIES": "2",
                "GEMINI_TIMEOUT_SECONDS": "30",
            },
        ), patch("src.gemini_client.time.sleep", return_value=None), patch.object(
            GeminiQuizClient, "_call_live_via_sdk", side_effect=ImportError("no sdk")
        ), patch.object(
            GeminiQuizClient,
            "_call_live_via_rest",
            side_effect=_ServiceUnavailableAPIError(f"HTTP 503 UNAVAILABLE high demand secret={fake_key}"),
        ) as rest_mock:
            resp = self.client.post(
                "/api/pipeline/run",
                json={
                    "oulad_student_id": 54882,
                    "simulated_student_code": "SIM_WEBUI_503",
                    "include_block_details": True,
                    "mode": "live",
                },
            )
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(rest_mock.call_count, 3)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error_type"], "SERVICE_UNAVAILABLE")
        self.assertEqual(data["error_category"], "TRANSIENT_SERVICE_ERROR")
        self.assertEqual(data["data_source"], "LIVE_GEMINI_API")
        self.assertFalse(data["is_simulated"])
        self.assertTrue(data["can_retry"])
        self.assertEqual(data["genai_result"]["attempts_used"], 3)
        self.assertEqual(
            data["genai_result"]["models_tried"],
            ["gemini-3.5-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"],
        )
        self.assertNotIn(fake_key, resp.get_data(as_text=True))

    def test_pipeline_run_live_without_api_key_returns_401(self):
        with patch("src.config.load_env_settings", return_value=None), patch.dict(
            os.environ, {"GEMINI_API_KEY": ""}, clear=False
        ):
            resp = self.client.post(
                "/api/pipeline/run",
                json={
                    "oulad_student_id": 54882,
                    "include_block_details": True,
                    "mode": "live",
                },
            )
        self.assertEqual(resp.status_code, 401)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertEqual(data["error_type"], "MISSING_API_KEY")
        self.assertEqual(data["error_category"], "CONFIGURATION_ERROR")
        self.assertEqual(data["data_source"], "LIVE_GEMINI_API")
        self.assertFalse(data["is_simulated"])

    def test_week7_playground_routes_regression(self):
        r_gen = self.client.post(
            "/api/generate",
            json={
                "profile_id": "profile_01_high_mixed",
                "num_questions": 3,
                "scenario_type": "detailed_profile",
                "mode": "mock",
            },
        )
        self.assertEqual(r_gen.status_code, 200)
        d_gen = r_gen.get_json()
        self.assertTrue(d_gen["success"])
        self.assertEqual(len(d_gen["quiz_data"]["questions"]), 3)

        r_cmp = self.client.post(
            "/api/compare",
            json={"profile_id": "profile_01_high_mixed", "num_questions": 2, "mode": "mock"},
        )
        self.assertEqual(r_cmp.status_code, 200)
        d_cmp = r_cmp.get_json()
        self.assertTrue(d_cmp["success"])
        self.assertEqual(len(d_cmp["summary"]), 3)


if __name__ == "__main__":
    unittest.main()
