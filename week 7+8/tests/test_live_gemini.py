# -*- coding: utf-8 -*-
"""
Integration Test kết nối trực tiếp Google Gemini API / Vertex Express Mode
(chạy khi đã cấu hình GEMINI_API_KEY trong week 7+8/.env).
Kiểm tra cả 3 kịch bản Prompt (baseline, risk_level, detailed_profile) và chức năng 7 tham số,
xác minh đủ 6 trường chuẩn JSON: question, options (4 lựa chọn), correct_answer, explanation, topic, difficulty.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

WEEK78_DIR = Path(__file__).resolve().parent.parent
if str(WEEK78_DIR) not in sys.path:
    sys.path.insert(0, str(WEEK78_DIR))

from src.config import has_valid_api_key  # noqa: E402
from src.experiment_runner import load_sample_profiles  # noqa: E402
from src.gemini_client import GeminiQuizClient  # noqa: E402
from src.schemas import ALLOWED_SCENARIOS, CORE_QUESTION_KEYS  # noqa: E402


class TestLiveGeminiIntegration(unittest.TestCase):
    def _assert_valid_live_result(self, res: dict, expected_num_questions: int) -> None:
        self.assertTrue(res["success"], f"Live Gemini call failed ({res.get('error_type')}): {res.get('error_message')}")
        self.assertEqual(res["http_status"], 200)
        self.assertEqual(res["mode_used"], "live")
        self.assertTrue(res["within_timeout_nfr01"], f"Vượt quá 20s NFR-01: {res['latency_ms']}ms")
        self.assertTrue(res["pii_report"]["pii_safe"])
        self.assertTrue(res["validation"]["is_valid"], f"Schema errors: {res['validation'].get('errors')}")
        self.assertTrue(res["validation"]["sql_isjson_compatible"])

        questions = res["quiz_data"]["questions"]
        self.assertEqual(len(questions), expected_num_questions)
        for q in questions:
            for key in CORE_QUESTION_KEYS:
                self.assertIn(key, q)
            self.assertEqual(set(q["options"].keys()), {"A", "B", "C", "D"})
            self.assertIn(q["correct_answer"], ("A", "B", "C", "D"))
            self.assertIn(q["difficulty"], ("easy", "medium", "hard"))

    def test_live_gemini_three_prompt_scenarios(self):
        if not has_valid_api_key():
            self.skipTest("Chưa cấu hình GEMINI_API_KEY trong week 7+8/.env — bỏ qua Live Gemini integration test.")

        client = GeminiQuizClient(mode="live")
        profile = load_sample_profiles()[0]

        for scenario in ALLOWED_SCENARIOS:
            with self.subTest(scenario=scenario):
                res = client.generate_quiz(profile=profile, scenario_type=scenario, num_questions=5, mode="live")
                self._assert_valid_live_result(res, expected_num_questions=5)

    def test_live_gemini_generate_questions_7_params(self):
        if not has_valid_api_key():
            self.skipTest("Chưa cấu hình GEMINI_API_KEY trong week 7+8/.env — bỏ qua Live Gemini integration test.")

        client = GeminiQuizClient(mode="live")
        res = client.generate_questions(
            course_code="IT1108",
            course_name="Nhập môn lập trình",
            knowledge_block="IT1108_KB03",
            review_topics=["Cấu trúc rẽ nhánh if-else", "Vòng lặp for và while"],
            num_questions=3,
            difficulty="easy",
            scenario_type="baseline",
            mode="live",
        )
        self._assert_valid_live_result(res, expected_num_questions=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
