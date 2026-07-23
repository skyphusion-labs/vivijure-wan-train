"""Tests for error redaction helpers."""
from wan_train.redact import redact_error_message


def test_redact_error_message_strips_urls_and_aws_keys():
    raw = (
        "download failed https://abc.r2.cloudflarestorage.com/bucket/key "
        "AKIAIOSFODNN7EXAMPLE secret_access_key=supersecret"
    )
    out = redact_error_message(raw)
    assert "https://" not in out
    assert "AKIA" not in out
    assert "supersecret" not in out
    assert "[url-redacted]" in out
    assert "[aws-key-redacted]" in out
