# -*- coding: utf-8 -*-
"""
Module: pii_guard.py
Dự án: SMART LMS & GENERATIVE AI (Tuần 7+8 - Kiểm thử GenAI)
Chức năng:
- Thực thi yêu cầu phi chức năng NFR-02 (Bảo mật & Ẩn danh):
  Tuyệt đối KHÔNG truyền id_student, MSSV, họ tên, email hay bất kỳ thông tin định danh cá nhân (PII)
  nào vào Prompt gửi cho Google Gemini API.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Set, Tuple


class PIILeakageError(ValueError):
    """Ngoại lệ kích hoạt khi phát hiện dữ liệu định danh (PII) chuẩn bị lọt vào Prompt."""


FORBIDDEN_PII_KEYS: Set[str] = {
    "id_student",
    "student_id",
    "user_id",
    "oulad_student_id",
    "full_name",
    "name",
    "student_name",
    "email",
    "username",
    "mssv",
    "phone",
    "phone_number",
    "address",
    "citizen_id",
    "cccd",
    "cmnd",
}

METADATA_WRAPPER_KEYS: Set[str] = {
    "_raw_test_metadata",
    "raw_pii",
    "pii_metadata",
}

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_VN_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+84|0)(?:3|5|7|8|9)(?:[\s.\-]?\d){8}(?!\d)")
_LABELED_ID_PATTERN = re.compile(
    r"\b(?:id_student|student_id|user_id|oulad_student_id|mssv|mã\s*sinh\s*viên|họ\s*(?:và\s*)?tên)\s*[:=]\s*[^\s,;]+",
    flags=re.IGNORECASE,
)


def _collect_forbidden_values(node: Any, collector: List[str]) -> None:
    if isinstance(node, dict):
        for v in node.values():
            _collect_forbidden_values(v, collector)
    elif isinstance(node, list):
        for item in node:
            _collect_forbidden_values(item, collector)
    elif node is not None:
        text = str(node).strip()
        if len(text) >= 3 and text not in collector:
            collector.append(text)


def _strip_pii_recursive(
    node: Any,
    stripped_keys: List[str],
    forbidden_values: List[str],
    parent_path: str = "",
) -> Any:
    if isinstance(node, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in node.items():
            key_lower = str(k).strip().lower()
            full_key = f"{parent_path}.{k}" if parent_path else str(k)
            if key_lower in FORBIDDEN_PII_KEYS or key_lower in METADATA_WRAPPER_KEYS:
                if full_key not in stripped_keys:
                    stripped_keys.append(full_key)
                _collect_forbidden_values(v, forbidden_values)
                continue
            cleaned[k] = _strip_pii_recursive(v, stripped_keys, forbidden_values, full_key)
        return cleaned
    if isinstance(node, list):
        return [
            _strip_pii_recursive(item, stripped_keys, forbidden_values, f"{parent_path}[]")
            for item in node
        ]
    return copy.deepcopy(node)


def sanitize_profile_for_prompt(
    profile: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Loại bỏ toàn bộ các trường chứa thông tin định danh cá nhân (PII) khỏi hồ sơ sinh viên.
    Trả về tuple: (clean_profile, stripped_keys, forbidden_values).
    """
    if not isinstance(profile, dict):
        raise TypeError("Hồ sơ đầu vào phải là dict.")

    stripped_keys: List[str] = []
    forbidden_values: List[str] = []
    clean_profile = _strip_pii_recursive(profile, stripped_keys, forbidden_values)
    return clean_profile, stripped_keys, forbidden_values


def scan_prompt_for_pii(
    prompt_text: str,
    forbidden_values: Optional[List[str]] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Quét văn bản Prompt trước khi gửi cho Gemini API nhằm đảm bảo NFR-02.
    Kiểm tra:
    - Định dạng Email, số điện thoại Việt Nam, nhãn id_student/MSSV/họ tên.
    - Bất kỳ giá trị nhạy cảm nào đã bị tách khỏi hồ sơ (forbidden_values).
    """
    text = prompt_text or ""
    findings: List[str] = []

    for match in _EMAIL_PATTERN.findall(text):
        findings.append(f"Phát hiện địa chỉ email trong Prompt: '{match}'")

    for match in _VN_PHONE_PATTERN.findall(text):
        findings.append(f"Phát hiện số điện thoại trong Prompt: '{match}'")

    for match in _LABELED_ID_PATTERN.findall(text):
        findings.append(f"Phát hiện nhãn định danh sinh viên trong Prompt: '{match}'")

    if forbidden_values:
        for val in forbidden_values:
            candidate = str(val).strip()
            if len(candidate) < 3:
                continue
            if candidate.isdigit():
                pattern = re.compile(rf"(?<!\d){re.escape(candidate)}(?!\d)")
                if pattern.search(text):
                    findings.append(f"Phát hiện mã số nhạy cảm '{candidate}' bị lọt vào Prompt.")
            else:
                if re.search(re.escape(candidate), text, flags=re.IGNORECASE):
                    findings.append(f"Phát hiện giá trị định danh '{candidate}' bị lọt vào Prompt.")

    report = {
        "pii_safe": len(findings) == 0,
        "findings": findings,
    }
    if strict and findings:
        raise PIILeakageError("Vi phạm bảo mật NFR-02: " + "; ".join(findings))
    return report
