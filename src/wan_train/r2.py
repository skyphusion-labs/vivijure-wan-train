"""R2 (S3-compatible) object I/O for the Wan train worker."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .redact import redact_error_message


@dataclass(frozen=True)
class R2Config:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @classmethod
    def from_env(cls, env: dict | None = None) -> "R2Config":
        e = env if env is not None else os.environ
        missing = [k for k in ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
                   if not e.get(k)]
        if missing:
            raise RuntimeError("R2 config incomplete; missing env: " + ", ".join(missing))
        return cls(e["R2_ENDPOINT"], e["R2_ACCESS_KEY_ID"], e["R2_SECRET_ACCESS_KEY"], e["R2_BUCKET"])


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
