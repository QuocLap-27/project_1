# -*- coding: utf-8 -*-
"""
Module: config.py
Dự án: SMART LMS & GENERATIVE AI (Tuần 7+8 - Kiểm thử GenAI)
Chức năng:
- Quản lý đường dẫn đọc dữ liệu Mapping & module predict.py từ Tuần 5+6 (read-only).
- Nạp biến môi trường từ file week 7+8/.env (hỗ trợ python-dotenv hoặc parser chuẩn Python).
- Tự động nhận diện API provider ('ai_studio' cho key AIza... hoặc 'vertex_express' cho key AQ....).
- Bảo vệ bí mật API Key: không trả key về frontend/summary, cung cấp redact_secret() lọc log/lỗi.
- Cấu hình ngưỡng Timeout NFR-01 (20 giây) và chế độ chạy (auto / live / mock).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union

OptionalPath = Optional[Union[Path, str]]

PLACEHOLDER_API_KEYS = {
    "",
    "your_google_ai_studio_api_key_here",
    "your_gemini_or_vertex_express_api_key_here",
    "your_api_key_here",
    "changeme",
    "none",
    "null",
}

WEEK78_DIR: Path = Path(__file__).resolve().parent.parent
PROJECT_ROOT: Path = WEEK78_DIR.parent
WEEK56_DIR: Path = PROJECT_ROOT / "week 5+6"
WEEK34_DIR: Path = PROJECT_ROOT / "week 3+4"

CURRICULUM_TOPICS_CSV: Path = WEEK56_DIR / "D_curriculum_mapping" / "curriculum_topics.csv"
ASSESSMENT_MAPPING_CSV: Path = WEEK56_DIR / "D_curriculum_mapping" / "assessment_topic_mapping.csv"
VLE_MAPPING_CSV: Path = WEEK56_DIR / "D_curriculum_mapping" / "vle_topic_mapping.csv"
WEEK56_PREDICT_PY: Path = WEEK56_DIR / "A_source_code" / "predict.py"
WEEK56_MODEL_PATH: Path = WEEK56_DIR / "B_models_and_results" / "best_model.joblib"
WEEK56_METADATA_PATH: Path = WEEK56_DIR / "B_models_and_results" / "model_metadata.json"
WEEK34_STUDENT_FEATURES_CSV: Path = WEEK34_DIR / "data" / "output" / "student_features.csv"
OULAD_ZIP_PATH: Path = PROJECT_ROOT / "open+university+learning+analytics+dataset.zip"

SAMPLE_PROFILES_JSON: Path = WEEK78_DIR / "tests" / "sample_profiles.json"
OUTPUTS_DIR: Path = WEEK78_DIR / "outputs"
ENV_FILE_PATH: Path = WEEK78_DIR / ".env"

DEFAULT_API_PROVIDER: str = "auto"
DEFAULT_GEMINI_MODEL: str = "gemini-3.5-flash"
DEFAULT_FALLBACK_MODEL: str = "gemini-3.5-flash-lite"
DEFAULT_THINKING_LEVEL: str = "MINIMAL"
DEFAULT_USE_RESPONSE_SCHEMA: bool = True
DEFAULT_TIMEOUT_SECONDS: float = 20.0
DEFAULT_MAX_RETRIES: int = 2
DEFAULT_MAX_JSON_RETRIES: int = 1
DEFAULT_GENAI_MODE: str = "auto"


def _parse_dotenv_file(env_path: Path) -> Dict[str, str]:
    """Đọc file .env bằng parser thuần Python (không phụ thuộc thư viện ngoài)."""
    parsed: Dict[str, str] = {}
    if not env_path.exists() or not env_path.is_file():
        return parsed

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        if key:
            parsed[key] = value
    return parsed


def load_env_settings(env_path: OptionalPath = None) -> None:
    """Nạp file .env vào os.environ mà không ghi đè biến môi trường đã đặt sẵn."""
    target = Path(env_path) if env_path is not None else ENV_FILE_PATH
    if not target.exists():
        return

    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=target, override=False)
        return
    except ImportError:
        pass

    for key, value in _parse_dotenv_file(target).items():
        if key not in os.environ:
            os.environ[key] = value


# Tự động nạp .env ngay khi import module
load_env_settings()


def _read_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
        return val if val > 0 else default
    except ValueError:
        return default


def _read_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
        return val if val >= 0 else default
    except ValueError:
        return default


def get_gemini_api_key() -> str:
    """Trả về GEMINI_API_KEY (rỗng nếu chưa cấu hình hoặc vẫn là placeholder)."""
    load_env_settings()
    raw = os.environ.get("GEMINI_API_KEY", "").strip()
    if raw.lower() in PLACEHOLDER_API_KEYS:
        return ""
    return raw


def has_valid_api_key() -> bool:
    """Kiểm tra đã có GEMINI_API_KEY thực tế hay chưa."""
    return bool(get_gemini_api_key())


def get_api_provider(api_key: Optional[str] = None) -> str:
    """
    Xác định API provider thực tế ('ai_studio' hoặc 'vertex_express'):
    - Nếu GEMINI_API_PROVIDER='vertex_express' -> trả về 'vertex_express'.
    - Mọi trường hợp khác (bao gồm 'auto' và 'ai_studio') -> mặc định trả về 'ai_studio'
      (tương thích cả key AIza... lẫn key AQ.... trên Google AI Studio).
    """
    load_env_settings()
    configured = os.environ.get("GEMINI_API_PROVIDER", DEFAULT_API_PROVIDER).strip().lower()
    if configured == "vertex_express":
        return "vertex_express"
    return "ai_studio"


def redact_secret(text: str, secret: Optional[str] = None) -> str:
    """Thay thế mọi xuất hiện của API key trong thông báo lỗi/log bằng '[REDACTED_API_KEY]'."""
    if not text:
        return ""
    result = str(text)
    target_secret = (secret if secret is not None else get_gemini_api_key()).strip()
    if len(target_secret) >= 6 and target_secret in result:
        result = result.replace(target_secret, "[REDACTED_API_KEY]")
    return result


def get_gemini_model() -> str:
    """Model chính mặc định: gemini-2.5-flash (tối ưu tốc độ & chi phí)."""
    val = os.environ.get("GEMINI_MODEL", "").strip()
    return val or DEFAULT_GEMINI_MODEL


def get_fallback_model() -> str:
    """Model dự phòng tiết kiệm quota nhất: gemini-3.5-flash-lite."""
    val = os.environ.get("GEMINI_FALLBACK_MODEL", "").strip()
    return val or DEFAULT_FALLBACK_MODEL


def get_thinking_level() -> str:
    """Mức suy nghĩ cho Gemini 3.5+ ('MINIMAL', 'LOW', 'MEDIUM', 'HIGH' hoặc '' nếu 'none'/'default')."""
    load_env_settings()
    raw = os.environ.get("GEMINI_THINKING_LEVEL", DEFAULT_THINKING_LEVEL).strip()
    if raw.lower() in ("none", "default", "off"):
        return ""
    return raw.upper() or DEFAULT_THINKING_LEVEL


def get_use_response_schema() -> bool:
    """Có gắn kèm responseSchema vào generationConfig hay dùng json-only (nhanh hơn dưới ngưỡng 20s)."""
    load_env_settings()
    raw = os.environ.get("GEMINI_USE_RESPONSE_SCHEMA", "").strip().lower()
    if not raw:
        return DEFAULT_USE_RESPONSE_SCHEMA
    return raw in ("1", "true", "yes", "on")


def get_timeout_seconds() -> float:
    """Timeout gọi socket API (mặc định 20 giây theo NFR-01, có thể tăng qua GEMINI_TIMEOUT_SECONDS)."""
    load_env_settings()
    return _read_float_env("GEMINI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)


def get_max_retries() -> int:
    """Số lần thử lại tối đa khi gặp lỗi tạm thời (ví dụ Rate Limit 429 hoặc 503)."""
    load_env_settings()
    return _read_int_env("GEMINI_MAX_RETRIES", DEFAULT_MAX_RETRIES)


def get_max_json_retries() -> int:
    """Số lần thử lại tối đa cho lỗi INVALID_JSON (giới hạn trong [0, 1] theo quyết định Tuần 8)."""
    load_env_settings()
    val = _read_int_env("GEMINI_MAX_JSON_RETRIES", DEFAULT_MAX_JSON_RETRIES)
    return min(max(0, val), 1)


def get_configured_mode() -> str:
    """Đọc chế độ GENAI_MODE trong môi trường ('auto', 'live', hoặc 'mock')."""
    mode = os.environ.get("GENAI_MODE", DEFAULT_GENAI_MODE).strip().lower()
    if mode in ("auto", "live", "mock"):
        return mode
    return DEFAULT_GENAI_MODE


def resolve_mode(requested_mode: str = "auto") -> str:
    """
    Quyết định chế độ thực thi thực tế ('live' hoặc 'mock'):
    - 'mock': luôn trả về 'mock'.
    - 'live': luôn trả về 'live' (kể cả khi thiếu key để client báo lỗi thiếu key rõ ràng).
    - 'auto' (mặc định): trả về 'live' nếu có API key hợp lệ, ngược lại 'mock'.
    """
    mode = (requested_mode or get_configured_mode()).strip().lower()
    if mode not in ("auto", "live", "mock"):
        raise ValueError(f"Chế độ GENAI_MODE không hợp lệ: '{requested_mode}'. Chỉ chấp nhận 'auto', 'live', 'mock'.")
    if mode == "mock":
        return "mock"
    if mode == "live":
        return "live"
    return "live" if has_valid_api_key() else "mock"


def get_runtime_summary() -> Dict[str, object]:
    """Trả về thông tin cấu hình an toàn (tuyệt đối không chứa bất kỳ phần nào của API key)."""
    has_key = has_valid_api_key()
    return {
        "configured_mode": get_configured_mode(),
        "effective_mode": resolve_mode(get_configured_mode()),
        "has_api_key": has_key,
        "api_provider": get_api_provider(),
        "model": get_gemini_model(),
        "fallback_model": get_fallback_model(),
        "thinking_level": get_thinking_level(),
        "use_response_schema": get_use_response_schema(),
        "timeout_seconds": get_timeout_seconds(),
        "max_retries": get_max_retries(),
        "max_json_retries": get_max_json_retries(),
        "curriculum_csv_exists": CURRICULUM_TOPICS_CSV.exists(),
        "assessment_mapping_csv_exists": ASSESSMENT_MAPPING_CSV.exists(),
        "week56_predict_exists": WEEK56_PREDICT_PY.exists(),
        "model_exists": WEEK56_MODEL_PATH.exists(),
        "oulad_zip_exists": OULAD_ZIP_PATH.exists(),
        "student_features_csv_exists": WEEK34_STUDENT_FEATURES_CSV.exists(),
    }


# Hằng số tiện dùng cho import trực tiếp (không export API key ở module-level để tránh lộ khi inspect)
GEMINI_MODEL: str = get_gemini_model()
GEMINI_FALLBACK_MODEL: str = get_fallback_model()
GEMINI_TIMEOUT_SECONDS: float = get_timeout_seconds()
GEMINI_MAX_RETRIES: int = get_max_retries()
GENAI_MODE: str = get_configured_mode()
