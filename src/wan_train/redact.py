"""Redact URLs and cloud credentials from operator-facing error strings."""
from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+")
_AWS_KEY_RE = re.compile(r"(?i)(AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16})")
_SECRETish_RE = re.compile(
    r"(?i)(secret[_-]?access[_-]?key|aws_secret_access_key)\s*[=:]\s*\S+"
)


def redact_error_message(message: object, *, limit: int = 500) -> str:
    """Strip URLs and obvious AWS credential material from an error string."""
    text = str(message)[:limit]
    text = _URL_RE.sub("[url-redacted]", text)
    text = _AWS_KEY_RE.sub("[aws-key-redacted]", text)
    text = _SECRETish_RE.sub("[secret-redacted]", text)
    return text
