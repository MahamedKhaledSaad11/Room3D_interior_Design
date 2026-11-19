from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

from fastapi import Depends, Header, HTTPException, Query, Request, status

from .config import get_settings


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt$16384$8$1${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt, expected = encoded.split("$")
        if algorithm != "scrypt":
            return False
        actual = hashlib.scrypt(password.encode(), salt=_unb64(salt), n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(actual, _unb64(expected))
    except (ValueError, TypeError):
        return False


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _unb64(value: str) -> bytes:
    decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    if _b64(decoded) != value:
        raise ValueError("non-canonical base64url")
    return decoded


def create_access_token(subject: str, expires_in: int = 3600) -> str:
    payload = {"sub": subject, "iat": int(time.time()), "exp": int(time.time()) + expires_in}
    return _sign_token(payload)


def create_asset_token(subject: str, project_id: str) -> str:
    return _sign_token({"sub": subject, "scope": "asset_read", "project_id": project_id,
                        "iat": int(time.time()), "exp": int(time.time()) + 300})


def _sign_token(payload: dict[str, object]) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = f"{_b64(json.dumps(header, separators=(',', ':')).encode())}.{_b64(json.dumps(payload, separators=(',', ':')).encode())}"
    signature = hmac.new(get_settings().jwt_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64(signature)}"


def verify_access_token(token: str) -> dict[str, Any]:
    try:
        encoded_header, encoded_payload, encoded_signature = token.split(".")
        signing_input = f"{encoded_header}.{encoded_payload}"
        expected = hmac.new(get_settings().jwt_secret.encode(), signing_input.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _unb64(encoded_signature)):
            raise ValueError("invalid signature")
        header = json.loads(_unb64(encoded_header))
        payload = json.loads(_unb64(encoded_payload))
        if header.get("alg") != "HS256" or not isinstance(payload.get("sub"), str):
            raise ValueError("invalid claims")
        if int(payload.get("exp", 0)) <= int(time.time()):
            raise ValueError("expired token")
        return payload
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


def require_worker_token(x_worker_token: str | None = Header(default=None)) -> None:
    if not hmac.compare_digest(x_worker_token or "", get_settings().worker_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid worker token")


def get_current_user(
    authorization: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> str | None:
    raw_token: str | None = None
    if authorization:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")
        raw_token = authorization[7:]
    elif token:
        raise HTTPException(401, "Session tokens must be sent in the Authorization header")
    if not raw_token:
        if not get_settings().auth_required:
            return None
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")
    claims = verify_access_token(raw_token)
    if claims.get("scope") is not None:
        raise HTTPException(401, "Asset credentials cannot authorize API requests")
    return str(claims["sub"])


def get_storage_actor(
    request: Request,
    authorization: str | None = Header(default=None),
    x_worker_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> tuple[bool, str | None]:
    if x_worker_token and hmac.compare_digest(x_worker_token, get_settings().worker_token):
        return (True, None)
    if token:
        claims = verify_access_token(token)
        if (request.method != "GET" or claims.get("scope") != "asset_read"
                or claims.get("project_id") != request.path_params.get("project_id")):
            raise HTTPException(401, "Invalid asset scope")
        return (False, str(claims["sub"]))
    return (False, get_current_user(authorization=authorization, token=None))
