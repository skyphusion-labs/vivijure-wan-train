"""R2 (S3-compatible) object I/O for the Wan train worker.

Tenant job I/O is one credential per job: the payload `r2` block on a pooled
endpoint, or the four `R2_*` env vars on a dedicated endpoint. Never log field
values; this object is a credential.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .redact import redact_error_message

# Optional per-job tenant R2 block and its required fields. Named here so the
# handler, the tests, and the control plane all read one definition.
PAYLOAD_KEY = "r2"
PAYLOAD_REQUIRED = ("endpoint", "access_key_id", "secret_access_key", "bucket")


@dataclass(frozen=True)
class R2Config:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    session_token: str | None = None

    @classmethod
    def from_env(cls, env: dict | None = None) -> "R2Config":
        """Dedicated-endpoint path: the four `R2_*` env vars. Raise if any are missing."""
        e = env if env is not None else os.environ
        missing = [k for k in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
                   if not e.get(k)]
        if missing:
            raise RuntimeError("R2 config incomplete; missing env: " + ", ".join(missing))
        return cls(e["R2_ENDPOINT"], e["R2_ACCESS_KEY_ID"], e["R2_SECRET_ACCESS_KEY"], e["R2_BUCKET"])

    @classmethod
    def from_payload_block(cls, block: object) -> "R2Config":
        """Build from the payload `r2` block. Raises on any malformation.

        Every message names FIELDS only, never values.
        """
        if not isinstance(block, dict):
            raise RuntimeError(
                f"job R2 config: {PAYLOAD_KEY!r} must be an object, got {type(block).__name__}")
        missing = [k for k in PAYLOAD_REQUIRED
                   if not (isinstance(block.get(k), str) and block[k].strip())]
        if missing:
            raise RuntimeError(
                "job R2 config incomplete; missing or blank fields: " + ", ".join(missing))
        token = block.get("session_token")
        if token is not None and not (isinstance(token, str) and token.strip()):
            raise RuntimeError(
                "job R2 config: session_token, when present, must be a non-empty string")
        return cls(
            endpoint=block["endpoint"].strip(),
            access_key_id=block["access_key_id"].strip(),
            secret_access_key=block["secret_access_key"].strip(),
            bucket=block["bucket"].strip(),
            session_token=token.strip() if isinstance(token, str) else None,
        )

    @classmethod
    def from_payload_or_env(cls, payload: dict | None, env: dict | None = None) -> "R2Config":
        """Payload block when the job carries one; endpoint env when it does not.

        PRESENT + malformed refuses. It never falls back to the environment: a
        silent fallback would train a tenant against the wrong bucket under the
        wrong credential. An explicit `"r2": null` is a producer defect and is
        refused the same way. ABSENT (key omitted) is `from_env`, so an operator
        CF studio / dedicated endpoint keeps working unchanged.
        """
        if isinstance(payload, dict) and PAYLOAD_KEY in payload:
            return cls.from_payload_block(payload[PAYLOAD_KEY])
        return cls.from_env(env)

    @staticmethod
    def strip_from_payload(payload: dict) -> dict:
        """A COPY of `payload` with the credential block removed.

        Nothing below the handler needs it (the store is injected), so removing
        it makes a leak structurally impossible rather than merely absent today.
        """
        return {k: v for k, v in payload.items() if k != PAYLOAD_KEY}


class R2:
    def __init__(self, config: R2Config):
        self.config = config
        self._cli = None

    def _client(self):
        if self._cli is None:
            import boto3
            from botocore.config import Config
            self._cli = boto3.client(
                "s3",
                endpoint_url=self.config.endpoint,
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                aws_session_token=self.config.session_token,
                config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
                region_name="auto",
            )
        return self._cli

    def get_file(self, key: str, dest: Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        from botocore.exceptions import ClientError
        try:
            head = self._client().head_object(Bucket=self.config.bucket, Key=key)
            expected = head.get("ContentLength")
            self._client().download_file(self.config.bucket, key, str(dest))
        except ClientError as e:
            raise RuntimeError(redact_error_message(e)) from None
        actual = dest.stat().st_size
        if expected is not None and actual != expected:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                redact_error_message(
                    f"R2 download truncated: {key!r} expected {expected} bytes, got {actual}"
                )
            )
        return dest

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self._client().head_object(Bucket=self.config.bucket, Key=key)
            return True
        except ClientError:
            return False

    def put_file(self, path: Path, key: str, *, content_type: str | None = None) -> str:
        extra: dict[str, object] = {}
        if content_type:
            extra["ContentType"] = content_type
        self._client().upload_file(str(path), self.config.bucket, key, ExtraArgs=extra or None)
        return key

    def put_bytes(self, data: bytes, key: str, *, content_type: str | None = None) -> str:
        kwargs: dict[str, object] = {"Bucket": self.config.bucket, "Key": key, "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        self._client().put_object(**kwargs)
        return key
