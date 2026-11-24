# -*- coding: utf-8 -*-
"""
Module: gemini_client.py
Dự án: SMART LMS & GENERATIVE AI (Tuần 7+8 - Kết nối ML & GenAI)
Chức năng:
- Kết nối Google Gemini API (AI Studio / Vertex Express) với responseMimeType='application/json'
  và responseSchema rút gọn (GEMINI_USE_RESPONSE_SCHEMA=true).
- Gia cố độ ổn định theo quyết định Tuần 8:
  + Tách riêng xử lý HTTP 503/UNAVAILABLE, Rate Limit 429 và TIMEOUT (retry có giới hạn).
  + Retry tối đa 1 lần riêng cho lỗi INVALID_JSON trong chế độ Live.
  + Ghi log lỗi qua logging nhưng tuyệt đối che API Key qua redact_secret().
  + Không tự động chuyển từ Live sang Mock Mode khi API lỗi; chỉ dùng Mock khi bật rõ ràng.
"""

from __future__ import annotations

import json
import logging
import random
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import (
    get_api_provider,
    get_fallback_model,
    get_gemini_api_key,
    get_gemini_model,
    get_max_json_retries,
    get_max_retries,
    get_thinking_level,
    get_timeout_seconds,
    get_use_response_schema,
    redact_secret,
    resolve_mode,
)
from .mock_provider import MockRateLimitError, MockServiceUnavailableError, generate_mock_quiz
from .pii_guard import PIILeakageError
from .prompt_builder import build_prompt_for_scenario
from .schemas import get_gemini_response_schema, parse_and_validate_quiz_json

_logger = logging.getLogger("smart_lms.genai")


def _classify_error_guidance(error_type: str) -> Tuple[str, str]:
    """Phân loại lỗi để Web UI hiển thị rõ bản chất sự cố và tuyệt đối không coi 503 là lỗi chất lượng AI."""
    et = (error_type or "").upper()
    if et in ("SERVICE_UNAVAILABLE", "RATE_LIMIT", "TIMEOUT", "API_ERROR"):
        return (
            "TRANSIENT_SERVICE_ERROR",
            "Đây là lỗi dịch vụ/kết nối tạm thời phía Google Gemini API (KHÔNG phải lỗi chất lượng nội dung AI). "
            "Hệ thống không tự động chuyển sang Mock Mode. Vui lòng bấm 'Thử lại' sau vài giây hoặc chủ động chọn chế độ Mock Offline để demo.",
        )
    if et in ("INVALID_JSON", "SCHEMA_VALIDATION_ERROR"):
        return (
            "CONTENT_FORMAT_ERROR",
            "Phản hồi từ mô hình chưa đạt cấu trúc JSON/Schema kỳ vọng sau lượt retry sửa lỗi. "
            "Bạn có thể bấm 'Thử lại' hoặc điều chỉnh giảm số lượng câu hỏi.",
        )
    if et in ("MISSING_API_KEY", "INVALID_API_KEY"):
        return (
            "CONFIGURATION_ERROR",
            "Chưa có API Key hợp lệ hoặc API Key bị từ chối. Vui lòng kiểm tra file week 7+8/.env hoặc chọn chế độ Mock Offline rõ ràng.",
        )
    return (
        "VALIDATION_OR_INPUT_ERROR",
        "Dữ liệu đầu vào hoặc quy trình kiểm tra an toàn không hợp lệ. Vui lòng kiểm tra lại tham số đầu vào.",
    )


class _RateLimitAPIError(RuntimeError):
    """Lỗi Rate Limit (HTTP 429) từ Gemini API."""


class _ServiceUnavailableAPIError(RuntimeError):
    """Lỗi quá tải / tạm ngưng dịch vụ (HTTP 503 / UNAVAILABLE)."""


class _AuthAPIError(RuntimeError):
    """Lỗi xác thực API Key (HTTP 401/403)."""


class _UpstreamAPIError(RuntimeError):
    """Lỗi dịch vụ Gemini API khác hoặc lỗi mạng."""


class GeminiQuizClient:
    """Client thống nhất điều phối gọi Google Gemini API hoặc Mock Provider với retry & telemetry."""

    def __init__(
        self,
        mode: str = "auto",
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        max_json_retries: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.configured_mode = mode
        self.model = model or get_gemini_model()
        self.fallback_model = fallback_model or get_fallback_model()
        self.timeout_seconds = float(
            timeout_seconds if timeout_seconds is not None else get_timeout_seconds()
        )
        self.max_retries = int(max_retries if max_retries is not None else get_max_retries())
        raw_json_retries = (
            max_json_retries if max_json_retries is not None else get_max_json_retries()
        )
        self.max_json_retries = min(max(0, int(raw_json_retries)), 1)
        self._explicit_api_key = api_key

    def _get_api_key(self) -> str:
        if self._explicit_api_key is not None:
            return self._explicit_api_key.strip()
        return get_gemini_api_key()

    def _call_live_via_sdk(
        self,
        api_key: str,
        model_name: str,
        system_instruction: str,
        user_prompt: str,
        timeout_sec: float,
    ) -> Tuple[str, Dict[str, int], str]:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        provider = get_api_provider(api_key)
        is_vertex = provider == "vertex_express"
        timeout_ms = max(1000, int(timeout_sec * 1000))

        try:
            http_opts = types.HttpOptions(timeout=timeout_ms)
            client = (
                genai.Client(vertexai=True, api_key=api_key, http_options=http_opts)
                if is_vertex
                else genai.Client(api_key=api_key, http_options=http_opts)
            )
        except Exception:
            client = (
                genai.Client(vertexai=True, api_key=api_key)
                if is_vertex
                else genai.Client(api_key=api_key)
            )

        sdk_kwargs: Dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": 0.3,
            "max_output_tokens": 3072,
            "response_mime_type": "application/json",
        }
        if get_use_response_schema():
            sdk_kwargs["response_schema"] = get_gemini_response_schema()

        thinking_level = get_thinking_level()
        if thinking_level:
            try:
                sdk_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
            except Exception:
                pass

        try:
            config = types.GenerateContentConfig(**sdk_kwargs)
        except Exception:
            sdk_kwargs.pop("thinking_config", None)
            config = types.GenerateContentConfig(**sdk_kwargs)

        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=config,
        )
        raw_text = getattr(response, "text", None) or ""
        usage_meta = getattr(response, "usage_metadata", None)
        usage = {
            "prompt_tokens": int(getattr(usage_meta, "prompt_token_count", 0) or 0),
            "output_tokens": int(getattr(usage_meta, "candidates_token_count", 0) or 0),
        }
        transport = "vertex-express-sdk" if is_vertex else "google-genai-sdk"
        return raw_text, usage, transport

    def _call_live_via_rest(
        self,
        api_key: str,
        model_name: str,
        system_instruction: str,
        user_prompt: str,
        timeout_sec: float,
    ) -> Tuple[str, Dict[str, int], str]:
        provider = get_api_provider(api_key)
        if provider == "vertex_express":
            url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model_name}:generateContent"
            transport = "vertex-express-rest"
        else:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
            transport = "gemini-rest-api"

        gen_config: Dict[str, Any] = {
            "temperature": 0.3,
            "maxOutputTokens": 3072,
            "responseMimeType": "application/json",
        }
        if get_use_response_schema():
            gen_config["responseSchema"] = get_gemini_response_schema()
        thinking_level = get_thinking_level()
        if thinking_level:
            gen_config["thinkingConfig"] = {"thinkingLevel": thinking_level}

        payload = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": gen_config,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                err_body = redact_secret(exc.read().decode("utf-8", errors="replace"), api_key)
                if exc.code == 429:
                    raise _RateLimitAPIError(f"HTTP 429 Rate Limit từ Gemini API ({provider}): {err_body}") from exc
                if exc.code == 503 or "UNAVAILABLE" in err_body.upper():
                    raise _ServiceUnavailableAPIError(
                        f"HTTP {exc.code} UNAVAILABLE từ Gemini API ({provider}): {err_body}"
                    ) from exc
                if exc.code in (401, 403):
                    raise _AuthAPIError(
                        f"HTTP {exc.code} API Key không hợp lệ hoặc chưa được cấp quyền ({provider}): {err_body}"
                    ) from exc
                raise _UpstreamAPIError(f"HTTP {exc.code} từ Gemini API ({provider}): {err_body}") from exc
            finally:
                exc.close()
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise TimeoutError(f"Hết thời gian chờ kết nối Gemini API ({timeout_sec:.1f}s).") from exc
            raise _UpstreamAPIError(
                redact_secret(f"Lỗi kết nối mạng tới Gemini API ({provider}): {exc.reason}", api_key)
            ) from exc

        candidates = resp_data.get("candidates") or []
        raw_text = ""
        if candidates:
            parts = (candidates[0].get("content") or {}).get("parts") or []
            raw_text = "".join(str(p.get("text", "")) for p in parts)

        usage_meta = resp_data.get("usageMetadata") or {}
        usage = {
            "prompt_tokens": int(usage_meta.get("promptTokenCount", 0) or 0),
            "output_tokens": int(usage_meta.get("candidatesTokenCount", 0) or 0),
        }
        return raw_text, usage, transport

    def _invoke_live_with_retries(
        self,
        api_key: str,
        prompt_bundle: Dict[str, Any],
        start_perf: float,
        user_prompt_override: Optional[str] = None,
        retry_trace: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, int], str, str]:
        system_instruction = prompt_bundle["system_instruction"]
        user_prompt = user_prompt_override or prompt_bundle["user_prompt"]
        last_exc: Optional[Exception] = None
        next_model = self.model
        trace = retry_trace if retry_trace is not None else {}

        for attempt in range(self.max_retries + 1):
            elapsed = time.perf_counter() - start_perf
            remaining = self.timeout_seconds - elapsed
            if remaining <= 0.2:
                raise TimeoutError(f"Đã vượt ngưỡng timeout {self.timeout_seconds:.0f}s (NFR-01).")

            active_model = next_model
            trace["attempts_used"] = int(trace.get("attempts_used", 0)) + 1
            trace.setdefault("models_tried", []).append(active_model)

            try:
                try:
                    raw_text, usage, transport = self._call_live_via_sdk(
                        api_key, active_model, system_instruction, user_prompt, remaining
                    )
                    return raw_text, usage, active_model, transport
                except ImportError:
                    raw_text, usage, transport = self._call_live_via_rest(
                        api_key, active_model, system_instruction, user_prompt, remaining
                    )
                    return raw_text, usage, active_model, transport
            except _AuthAPIError:
                raise
            except (TimeoutError, socket.timeout):
                raise
            except Exception as exc:
                err_msg = redact_secret(str(exc), api_key)
                is_timeout = "timeout" in err_msg.lower() or "timed out" in err_msg.lower()
                if is_timeout:
                    raise TimeoutError(err_msg) from exc

                is_404 = ("404" in err_msg) or ("NOT_FOUND" in err_msg.upper())
                is_rate_limit = (
                    isinstance(exc, _RateLimitAPIError)
                    or "429" in err_msg
                    or "RESOURCE_EXHAUSTED" in err_msg.upper()
                )
                is_503_unavail = (
                    isinstance(exc, _ServiceUnavailableAPIError)
                    or "503" in err_msg
                    or "UNAVAILABLE" in err_msg.upper()
                    or "HIGH DEMAND" in err_msg.upper()
                )

                if is_rate_limit:
                    last_exc = _RateLimitAPIError(err_msg)
                elif is_503_unavail:
                    last_exc = _ServiceUnavailableAPIError(err_msg)
                else:
                    last_exc = _UpstreamAPIError(err_msg)

                _logger.warning(
                    "Gemini attempt %d/%d failed (%s, model=%s): %s",
                    attempt + 1,
                    self.max_retries + 1,
                    type(last_exc).__name__,
                    active_model,
                    err_msg[:220],
                )

                if attempt < self.max_retries:
                    elapsed_now = time.perf_counter() - start_perf
                    rem_after = self.timeout_seconds - elapsed_now
                    if rem_after > 2.0:
                        if is_404 and self.fallback_model and active_model != self.fallback_model:
                            next_model = self.fallback_model
                            time.sleep(0.05)
                            continue
                        if attempt >= 1 and self.fallback_model and self.fallback_model != self.model:
                            next_model = self.fallback_model
                        else:
                            next_model = self.model
                        jitter = random.uniform(0.05, 0.25)
                        backoff = min(1.0 * (2 ** attempt) + jitter, max(0.1, rem_after - 1.5))
                        time.sleep(backoff)
                        continue
                break

        assert last_exc is not None
        raise last_exc

    def generate_questions(
        self,
        course_code: str = "IT1108",
        course_name: str = "Nhập môn lập trình",
        knowledge_block: Optional[str] = None,
        review_topics: Optional[Union[str, List[str]]] = None,
        num_questions: int = 5,
        difficulty: str = "auto",
        scenario_type: str = "detailed_profile",
        profile: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
        simulate_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sinh bộ câu hỏi ôn tập với đầy đủ 7 tham số cấu hình học thuật + hồ sơ mô phỏng."""
        base_profile: Dict[str, Any] = dict(profile) if isinstance(profile, dict) else {
            "risk_level": "MEDIUM",
            "risk_probability": 0.50,
            "weak_blocks": [],
        }

        if knowledge_block:
            existing_blocks = list(base_profile.get("weak_blocks") or [])
            matched = [b for b in existing_blocks if b.get("it1108_block_code") == knowledge_block]
            if matched:
                base_profile["weak_blocks"] = matched
            else:
                base_profile["weak_blocks"] = [
                    {
                        "it1108_block_code": knowledge_block,
                        "submission_vs_score_status": (
                            "NORMAL" if difficulty == "hard" else "LOW_SCORE_DESPITE_SUBMISSION"
                        ),
                    }
                ]

        return self._execute_generation(
            profile=base_profile,
            scenario_type=scenario_type,
            num_questions=num_questions,
            course_code=course_code,
            course_name=course_name,
            knowledge_block=knowledge_block,
            review_topics=review_topics,
            difficulty=difficulty,
            mode=mode,
            simulate_error=simulate_error,
        )

    def generate_quiz(
        self,
        profile: Dict[str, Any],
        scenario_type: str = "detailed_profile",
        num_questions: int = 5,
        mode: Optional[str] = None,
        simulate_error: Optional[str] = None,
        course_code: str = "IT1108",
        course_name: str = "Nhập môn lập trình",
        knowledge_block: Optional[str] = None,
        review_topics: Optional[Union[str, List[str]]] = None,
        difficulty: str = "auto",
    ) -> Dict[str, Any]:
        """Hàm tương thích với luồng kiểm thử theo hồ sơ sinh viên mô phỏng."""
        return self._execute_generation(
            profile=profile,
            scenario_type=scenario_type,
            num_questions=num_questions,
            course_code=course_code,
            course_name=course_name,
            knowledge_block=knowledge_block,
            review_topics=review_topics,
            difficulty=difficulty,
            mode=mode,
            simulate_error=simulate_error,
        )

    def _execute_generation(
        self,
        profile: Dict[str, Any],
        scenario_type: str,
        num_questions: int,
        course_code: str,
        course_name: str,
        knowledge_block: Optional[str],
        review_topics: Optional[Union[str, List[str]]],
        difficulty: str,
        mode: Optional[str],
        simulate_error: Optional[str],
    ) -> Dict[str, Any]:
        effective_mode = resolve_mode(mode or self.configured_mode)
        is_simulated = effective_mode == "mock"
        data_source = "MOCK_SIMULATED" if is_simulated else "LIVE_GEMINI_API"
        mode_notice = (
            "Đang chạy ở chế độ MOCK MÔ PHỎNG (không phải kết quả Gemini API thực tế)."
            if is_simulated
            else "Đang gọi Google Gemini API thực tế."
        )

        start_perf = time.perf_counter()
        api_key = self._get_api_key()

        try:
            prompt_bundle = build_prompt_for_scenario(
                profile=profile,
                scenario_type=scenario_type,
                num_questions=num_questions,
                course_code=course_code,
                course_name=course_name,
                knowledge_block=knowledge_block,
                review_topics=review_topics,
                difficulty=difficulty,
            )
        except PIILeakageError as exc:
            err_msg = redact_secret(str(exc), api_key)
            _logger.error("PII Guard rejected prompt: %s", err_msg)
            return self._build_error_response(
                http_status=400,
                error_type="PII_LEAKAGE",
                error_message=err_msg,
                mode_used=effective_mode,
                model_used=self.model,
                scenario_type=scenario_type,
                latency_ms=round((time.perf_counter() - start_perf) * 1000, 2),
                pii_safe=False,
                data_source=data_source,
                is_simulated=is_simulated,
                mode_notice=mode_notice,
                can_retry=False,
            )

        if effective_mode == "live" and not api_key:
            return self._build_error_response(
                http_status=401,
                error_type="MISSING_API_KEY",
                error_message=(
                    "Chưa cấu hình GEMINI_API_KEY cho chế độ Live. "
                    "Hệ thống KHÔNG tự chuyển sang Mock Mode. Hãy thêm key vào week 7+8/.env hoặc bật chế độ Mock rõ ràng."
                ),
                mode_used="live",
                model_used=self.model,
                scenario_type=scenario_type,
                latency_ms=round((time.perf_counter() - start_perf) * 1000, 2),
                prompt_bundle=prompt_bundle,
                data_source=data_source,
                is_simulated=False,
                mode_notice=mode_notice,
                can_retry=True,
            )

        raw_text = ""
        usage = {"prompt_tokens": 0, "output_tokens": 0}
        model_used = self.model
        transport = "mock-offline"
        json_retries_used = 0
        retry_trace: Dict[str, Any] = {"attempts_used": 0, "models_tried": []}

        try:
            if effective_mode == "mock":
                model_used = f"{self.model} (mock-offline)"
                for attempt in range(self.max_retries + 1):
                    retry_trace["attempts_used"] += 1
                    retry_trace["models_tried"].append(model_used)
                    try:
                        raw_text, usage = generate_mock_quiz(
                            profile=profile,
                            scenario_type=scenario_type,
                            prompt_bundle=prompt_bundle,
                            num_questions=num_questions,
                            difficulty=difficulty,
                            simulate_error=simulate_error,
                        )
                        break
                    except MockRateLimitError as exc:
                        if attempt < self.max_retries:
                            time.sleep(0.01)
                            continue
                        raise _RateLimitAPIError(str(exc)) from exc
                    except MockServiceUnavailableError as exc:
                        if attempt < self.max_retries:
                            time.sleep(0.01)
                            continue
                        raise _ServiceUnavailableAPIError(str(exc)) from exc
            else:
                raw_text, usage, model_used, transport = self._invoke_live_with_retries(
                    api_key=api_key,
                    prompt_bundle=prompt_bundle,
                    start_perf=start_perf,
                    retry_trace=retry_trace,
                )

            validation = parse_and_validate_quiz_json(
                raw_text=raw_text,
                expected_num_questions=num_questions,
                expected_scenario=scenario_type,
                expected_course_code=course_code,
            )

            # Quyết định 6 Tuần 8: Retry tối đa 1 lần khi gặp lỗi INVALID_JSON ở chế độ Live
            if effective_mode == "live" and not validation.json_valid and self.max_json_retries > 0:
                remaining_after = self.timeout_seconds - (time.perf_counter() - start_perf)
                if remaining_after > 3.0:
                    json_retries_used = 1
                    _logger.warning(
                        "INVALID_JSON on first Live attempt (model=%s); executing 1 JSON repair retry.",
                        model_used,
                    )
                    repair_prompt = (
                        prompt_bundle["user_prompt"]
                        + "\n\nLƯU Ý BẮT BUỘC: Lần trả lời trước bị lỗi cú pháp JSON. "
                        "Hãy chỉ xuất DUY NHẤT 1 JSON object hợp lệ, thoát đúng ký tự \\n trong code_snippet."
                    )
                    raw_text, usage, model_used, transport = self._invoke_live_with_retries(
                        api_key=api_key,
                        prompt_bundle=prompt_bundle,
                        start_perf=start_perf,
                        user_prompt_override=repair_prompt,
                        retry_trace=retry_trace,
                    )
                    validation = parse_and_validate_quiz_json(
                        raw_text=raw_text,
                        expected_num_questions=num_questions,
                        expected_scenario=scenario_type,
                        expected_course_code=course_code,
                    )

        except (TimeoutError, socket.timeout) as exc:
            latency_ms = round((time.perf_counter() - start_perf) * 1000, 2)
            err_msg = redact_secret(
                f"Hệ thống AI quá thời gian phản hồi (> {self.timeout_seconds:.0f}s): {exc}",
                api_key,
            )
            _logger.error("Gemini TIMEOUT: %s", err_msg)
            return self._build_error_response(
                http_status=503,
                error_type="TIMEOUT",
                error_message=err_msg,
                mode_used=effective_mode,
                model_used=model_used,
                scenario_type=scenario_type,
                latency_ms=latency_ms,
                prompt_bundle=prompt_bundle,
                data_source=data_source,
                is_simulated=is_simulated,
                mode_notice=mode_notice,
                json_retries_used=json_retries_used,
                attempts_used=retry_trace["attempts_used"],
                models_tried=retry_trace["models_tried"],
                can_retry=True,
            )
        except _ServiceUnavailableAPIError as exc:
            latency_ms = round((time.perf_counter() - start_perf) * 1000, 2)
            err_msg = redact_secret(
                f"Dịch vụ Gemini API đang quá tải (503 UNAVAILABLE). Vui lòng thử lại sau: {exc}",
                api_key,
            )
            _logger.error("Gemini 503 UNAVAILABLE: %s", err_msg[:240])
            return self._build_error_response(
                http_status=503,
                error_type="SERVICE_UNAVAILABLE",
                error_message=err_msg,
                mode_used=effective_mode,
                model_used=model_used,
                scenario_type=scenario_type,
                latency_ms=latency_ms,
                prompt_bundle=prompt_bundle,
                data_source=data_source,
                is_simulated=is_simulated,
                mode_notice=mode_notice,
                json_retries_used=json_retries_used,
                attempts_used=retry_trace["attempts_used"],
                models_tried=retry_trace["models_tried"],
                can_retry=True,
            )
        except _RateLimitAPIError as exc:
            latency_ms = round((time.perf_counter() - start_perf) * 1000, 2)
            err_msg = redact_secret(
                f"Hệ thống AI đang quá tải hoặc vượt giới hạn lượt gọi (Rate Limit 429). Vui lòng thử lại sau: {exc}",
                api_key,
            )
            _logger.error("Gemini 429 RATE_LIMIT: %s", err_msg[:240])
            return self._build_error_response(
                http_status=503,
                error_type="RATE_LIMIT",
                error_message=err_msg,
                mode_used=effective_mode,
                model_used=model_used,
                scenario_type=scenario_type,
                latency_ms=latency_ms,
                prompt_bundle=prompt_bundle,
                data_source=data_source,
                is_simulated=is_simulated,
                mode_notice=mode_notice,
                json_retries_used=json_retries_used,
                attempts_used=retry_trace["attempts_used"],
                models_tried=retry_trace["models_tried"],
                can_retry=True,
            )
        except _AuthAPIError as exc:
            latency_ms = round((time.perf_counter() - start_perf) * 1000, 2)
            err_msg = redact_secret(str(exc), api_key)
            _logger.error("Gemini Auth Error: %s", err_msg[:240])
            return self._build_error_response(
                http_status=401,
                error_type="INVALID_API_KEY",
                error_message=err_msg,
                mode_used=effective_mode,
                model_used=model_used,
                scenario_type=scenario_type,
                latency_ms=latency_ms,
                prompt_bundle=prompt_bundle,
                data_source=data_source,
                is_simulated=is_simulated,
                mode_notice=mode_notice,
                json_retries_used=json_retries_used,
                attempts_used=retry_trace["attempts_used"],
                models_tried=retry_trace["models_tried"],
                can_retry=False,
            )
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_perf) * 1000, 2)
            err_msg = redact_secret(f"Hệ thống AI đang bận hoặc gặp sự cố kết nối: {exc}", api_key)
            _logger.error("Gemini API Error: %s", err_msg[:240])
            return self._build_error_response(
                http_status=503,
                error_type="API_ERROR",
                error_message=err_msg,
                mode_used=effective_mode,
                model_used=model_used,
                scenario_type=scenario_type,
                latency_ms=latency_ms,
                prompt_bundle=prompt_bundle,
                data_source=data_source,
                is_simulated=is_simulated,
                mode_notice=mode_notice,
                json_retries_used=json_retries_used,
                attempts_used=retry_trace["attempts_used"],
                models_tried=retry_trace["models_tried"],
                can_retry=True,
            )

        latency_ms = round((time.perf_counter() - start_perf) * 1000, 2)
        within_timeout = latency_ms <= 20000.0

        if not validation.json_valid:
            err_msg = "Dữ liệu AI trả về không đúng định dạng JSON: " + "; ".join(validation.errors)
            _logger.error("Gemini INVALID_JSON (retries=%d): %s", json_retries_used, err_msg[:240])
            return self._build_error_response(
                http_status=500,
                error_type="INVALID_JSON",
                error_message=err_msg,
                mode_used=effective_mode,
                model_used=model_used,
                scenario_type=scenario_type,
                latency_ms=latency_ms,
                prompt_bundle=prompt_bundle,
                validation=validation.to_dict(),
                raw_response=redact_secret(raw_text, api_key),
                usage=usage,
                transport=transport,
                data_source=data_source,
                is_simulated=is_simulated,
                mode_notice=mode_notice,
                json_retries_used=json_retries_used,
                attempts_used=retry_trace["attempts_used"],
                models_tried=retry_trace["models_tried"],
                can_retry=True,
            )

        if not validation.is_valid:
            err_msg = "JSON từ AI thiếu trường hoặc sai cấu trúc Schema: " + "; ".join(validation.errors)
            _logger.error("Gemini SCHEMA_VALIDATION_ERROR: %s", err_msg[:240])
            return self._build_error_response(
                http_status=500,
                error_type="SCHEMA_VALIDATION_ERROR",
                error_message=err_msg,
                mode_used=effective_mode,
                model_used=model_used,
                scenario_type=scenario_type,
                latency_ms=latency_ms,
                prompt_bundle=prompt_bundle,
                validation=validation.to_dict(),
                raw_response=redact_secret(raw_text, api_key),
                usage=usage,
                transport=transport,
                data_source=data_source,
                is_simulated=is_simulated,
                mode_notice=mode_notice,
                json_retries_used=json_retries_used,
                attempts_used=retry_trace["attempts_used"],
                models_tried=retry_trace["models_tried"],
                can_retry=True,
            )

        return {
            "success": True,
            "http_status": 200,
            "mode_used": effective_mode,
            "data_source": data_source,
            "is_simulated": is_simulated,
            "mode_notice": mode_notice,
            "api_provider": get_api_provider(api_key) if effective_mode == "live" else "mock",
            "model_used": model_used,
            "models_tried": retry_trace["models_tried"],
            "attempts_used": retry_trace["attempts_used"],
            "transport": transport,
            "scenario_type": scenario_type,
            "latency_ms": latency_ms,
            "within_timeout_nfr01": within_timeout,
            "json_retries_used": json_retries_used,
            "can_retry": True,
            "usage": usage,
            "pii_report": prompt_bundle["pii_report"],
            "prompt_bundle": prompt_bundle,
            "validation": validation.to_dict(),
            "quiz_data": validation.normalized_data,
            "raw_response_preview": redact_secret(raw_text[:600], api_key),
            "error_type": None,
            "error_category": None,
            "error_message": None,
            "user_guidance": None,
        }

    def _build_error_response(
        self,
        http_status: int,
        error_type: str,
        error_message: str,
        mode_used: str,
        model_used: str,
        scenario_type: str,
        latency_ms: float,
        prompt_bundle: Optional[Dict[str, Any]] = None,
        validation: Optional[Dict[str, Any]] = None,
        raw_response: str = "",
        usage: Optional[Dict[str, int]] = None,
        transport: str = "",
        pii_safe: bool = True,
        data_source: Optional[str] = None,
        is_simulated: Optional[bool] = None,
        mode_notice: str = "",
        json_retries_used: int = 0,
        attempts_used: int = 0,
        models_tried: Optional[List[str]] = None,
        can_retry: bool = True,
    ) -> Dict[str, Any]:
        is_sim = (mode_used == "mock") if is_simulated is None else is_simulated
        ds = data_source or ("MOCK_SIMULATED" if is_sim else "LIVE_GEMINI_API")
        error_category, user_guidance = _classify_error_guidance(error_type)
        default_validation = {
            "is_valid": False,
            "json_valid": False,
            "schema_valid": False,
            "sql_isjson_compatible": False,
            "errors": [error_message],
            "warnings": [],
            "normalized_data": None,
            "valid_questions": [],
            "invalid_question_indices": [],
        }
        return {
            "success": False,
            "http_status": http_status,
            "mode_used": mode_used,
            "data_source": ds,
            "is_simulated": is_sim,
            "mode_notice": mode_notice,
            "api_provider": get_api_provider() if mode_used == "live" else "mock",
            "model_used": model_used,
            "models_tried": list(models_tried or ([model_used] if model_used else [])),
            "attempts_used": attempts_used,
            "transport": transport or ("mock-offline" if mode_used == "mock" else "gemini-api"),
            "scenario_type": scenario_type,
            "latency_ms": latency_ms,
            "within_timeout_nfr01": latency_ms <= 20000.0,
            "json_retries_used": json_retries_used,
            "can_retry": can_retry,
            "usage": usage or {"prompt_tokens": 0, "output_tokens": 0},
            "pii_report": (prompt_bundle or {}).get(
                "pii_report", {"pii_safe": pii_safe, "stripped_pii_keys": [], "findings": []}
            ),
            "prompt_bundle": prompt_bundle,
            "validation": validation or default_validation,
            "quiz_data": None,
            "raw_response_preview": raw_response[:600],
            "error_type": error_type,
            "error_category": error_category,
            "error_message": error_message,
            "user_guidance": user_guidance,
        }
