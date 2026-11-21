import base64
import io
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web_app"))
os.environ["CANVAS_SKIP_APP"] = "1"  # don't build the module-level app from a real .env

import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# 1x1 transparent WebP
TINY_WEBP = base64.b64encode(
    bytes.fromhex("524946461a000000574542505650384c0d0000002f00000010071011118888fe0700")
).decode()

HEADERS = {"X-Requested-With": "canvas"}


class FakeRenderer:
    def __init__(self):
        self.calls = []

    async def __call__(self, payload):
        self.calls.append(payload)
        base = payload["seed"] if payload["seed"] is not None else 1000
        return {
            "images": [{"seed": base + i, "image_b64": TINY_WEBP} for i in range(payload["count"])],
            "generation_seconds": 4.0,
            "cold_start": True,
            "load_seconds": 20.0,
        }


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSION_SECRET", "x" * 40)
    monkeypatch.setenv("ALLOWED_USERS", "alice:pw-a,bob:pw-b")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("COOKIE_SECURE", "0")
    monkeypatch.setenv("MONTHLY_GPU_MINUTES_CAP", "1800")
    for k in ("LLM_BASE_URL", "LLM_MODEL", "MODAL_ENDPOINT_URL"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


@pytest.fixture
def renderer():
    return FakeRenderer()


@pytest.fixture
def make_client(env, renderer):
    def _make(user=None, password=None):
        app = server.create_app(renderer=renderer)
        c = TestClient(app)
        if user:
            r = c.post("/api/login", json={"username": user, "password": password}, headers=HEADERS)
            assert r.status_code == 200, r.text
        return c
    return _make
