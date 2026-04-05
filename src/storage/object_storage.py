"""S3-compatible object storage adapter for local S3 and Cloudflare R2."""

from __future__ import annotations

import gzip
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = ROOT_DIR / ".env"

_OBJECT_STORAGE_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "bucket": (
        "CF_R2_BUCKET",
        "R2_BUCKET",
        "MINIO_BUCKET",
        "OBJECT_STORAGE_BUCKET",
        "S3_BUCKET",
    ),
    "endpoint": (
        "CF_R2_ENDPOINT",
        "CF_R2_ENDPOINT_URL",
        "MINIO_ENDPOINT",
        "OBJECT_STORAGE_ENDPOINT",
        "S3_ENDPOINT",
    ),
    "access_key": (
        "CF_R2_ACCESS_KEY",
        "CF_R2_ACCESS_KEY_ID",
        "MINIO_ACCESS_KEY",
        "OBJECT_STORAGE_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
    ),
    "secret_key": (
        "CF_R2_SECRET_KEY",
        "CF_R2_SECRET_ACCESS_KEY",
        "MINIO_SECRET_KEY",
        "OBJECT_STORAGE_SECRET_KEY",
        "AWS_SECRET_ACCESS_KEY",
    ),
}


@dataclass(frozen=True)
class ObjectStorageConfig:
    """Runtime configuration for an S3-compatible object storage backend."""

    bucket: str
    endpoint: str
    access_key: str
    secret_key: str
    region: str = "auto"
    public_url_base: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        env: dict[str, str] | os._Environ[str] | None = None,
        env_path: Path = DEFAULT_ENV_PATH,
    ) -> ObjectStorageConfig:
        """Build config from env vars used by local S3 and Cloudflare R2."""

        environment = os.environ if env is None else env
        bucket = _resolve_env_var_any(
            _OBJECT_STORAGE_ENV_ALIASES["bucket"],
            env=environment,
            env_path=env_path,
        )
        endpoint = _resolve_env_var_any(
            _OBJECT_STORAGE_ENV_ALIASES["endpoint"],
            env=environment,
            env_path=env_path,
        )
        access_key = _resolve_env_var_any(
            _OBJECT_STORAGE_ENV_ALIASES["access_key"],
            env=environment,
            env_path=env_path,
        )
        secret_key = _resolve_env_var_any(
            _OBJECT_STORAGE_ENV_ALIASES["secret_key"],
            env=environment,
            env_path=env_path,
        )

        required_values = {
            "bucket": bucket,
            "endpoint": endpoint,
            "access_key": access_key,
            "secret_key": secret_key,
        }
        missing = [name for name, value in required_values.items() if not value]
        if missing:
            joined = ", ".join(
                f"{name} ({' | '.join(_OBJECT_STORAGE_ENV_ALIASES[name])})"
                for name in missing
            )
            raise ValueError(f"Missing required object storage env vars: {joined}")

        bucket_str = cast(str, bucket)
        endpoint_str = cast(str, endpoint)
        access_key_str = cast(str, access_key)
        secret_key_str = cast(str, secret_key)

        region = _resolve_env_var_any(
            (
                "CF_R2_REGION",
                "MINIO_REGION",
                "AWS_REGION",
                "OBJECT_STORAGE_REGION",
            ),
            env=environment,
            env_path=env_path,
        )

        public_url_base = _resolve_env_var_any(
            (
                "CF_R2_PUBLIC_URL_BASE",
                "OBJECT_STORAGE_PUBLIC_URL_BASE",
            ),
            env=environment,
            env_path=env_path,
        )

        return cls(
            bucket=bucket_str,
            endpoint=endpoint_str,
            access_key=access_key_str,
            secret_key=secret_key_str,
            region=region or "auto",
            public_url_base=public_url_base,
        )


def _resolve_env_var_any(
    names: tuple[str, ...],
    *,
    env: dict[str, str] | os._Environ[str],
    env_path: Path,
) -> str | None:
    for name in names:
        value = _resolve_env_var(name, env=env, env_path=env_path)
        if value:
            return value
    return None


def _resolve_env_var(
    name: str,
    *,
    env: dict[str, str] | os._Environ[str],
    env_path: Path,
) -> str | None:
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            if key.strip() == name:
                resolved = raw_value.strip()
                if resolved:
                    return resolved

    value = env.get(name)
    if value:
        return value
    return None


class S3CompatibleObjectStore:
    """Writes compressed JSON payloads to S3-compatible object storage."""

    def __init__(self, *, s3_client: Any, bucket: str) -> None:
        self._s3_client = s3_client
        self._bucket = bucket

    @classmethod
    def from_config(cls, config: ObjectStorageConfig) -> S3CompatibleObjectStore:
        """Create a boto3 S3 client configured for local S3 or Cloudflare R2."""

        boto3_module = importlib.import_module("boto3")
        client_kwargs: dict[str, Any] = {
            "endpoint_url": config.endpoint,
            "aws_access_key_id": config.access_key,
            "aws_secret_access_key": config.secret_key,
        }
        if config.region and config.region != "auto":
            client_kwargs["region_name"] = config.region

        s3_client = boto3_module.client("s3", **client_kwargs)
        return cls(s3_client=s3_client, bucket=config.bucket)

    def put_json_gzip(
        self,
        *,
        key: str,
        payload: dict[str, Any],
        metadata: dict[str, str],
    ) -> str:
        """Store the full payload as gzip-compressed canonical JSON."""

        body_bytes = gzip.compress(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        return self.upload_document(
            key=key,
            content=body_bytes,
            content_type="application/json",
            metadata=metadata,
            content_encoding="gzip",
        )

    def upload_document(
        self,
        *,
        key: str,
        content: bytes,
        content_type: str,
        metadata: dict[str, str],
        content_encoding: str | None = None,
    ) -> str:
        """Upload raw bytes with caller-provided content type and metadata."""

        request: dict[str, Any] = {
            "Bucket": self._bucket,
            "Key": key,
            "Body": content,
            "ContentType": content_type,
            "Metadata": metadata,
        }
        if content_encoding:
            request["ContentEncoding"] = content_encoding

        self._s3_client.put_object(
            **request,
        )
        return key

    def fetch_document(self, *, key: str) -> bytes:
        """Fetch raw document bytes for a key from object storage."""

        response = self._s3_client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()

    def delete_prefix(self, *, prefix: str) -> int:
        """Delete all objects under a prefix and return the number of deleted keys."""

        deleted = 0
        paginator = self._s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            contents = page.get("Contents") or []
            if not contents:
                continue

            for start in range(0, len(contents), 1000):
                batch = contents[start : start + 1000]
                self._s3_client.delete_objects(
                    Bucket=self._bucket,
                    Delete={
                        "Objects": [{"Key": item["Key"]} for item in batch],
                        "Quiet": True,
                    },
                )
                deleted += len(batch)
        return deleted

    def validate_bucket_access(self, *, smoke_test: bool = False) -> None:
        """Validate list access and optionally perform a write/delete smoke test."""

        self._s3_client.head_bucket(Bucket=self._bucket)
        if not smoke_test:
            return

        test_key = "healthcheck/raw-storage-smoke-test.txt"
        self._s3_client.put_object(Bucket=self._bucket, Key=test_key, Body=b"ok")
        self._s3_client.get_object(Bucket=self._bucket, Key=test_key)
        self._s3_client.delete_object(Bucket=self._bucket, Key=test_key)
