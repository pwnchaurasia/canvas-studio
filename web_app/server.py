"""Canvas Studio web app: auth, image proxy to the Modal worker, prompt enhancer, gallery, chat.

Run locally:  uvicorn server:app --host 127.0.0.1 --port 8090
"""

import base64
import hmac
import os
import re
import sqlite3
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, URLSafeTimedSerializer
from pydantic import BaseModel, Field

import prompt_tools
import writer

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
load_dotenv(BASE_DIR / ".env", override=False)

SESSION_COOKIE = "cs_session"
SESSION_MAX_AGE = 30 * 24 * 3600
LOGIN_LIMIT, LOGIN_WINDOW_S = 5, 60
IDLE_BILLING_S = 25  # matches scaledown_window on the worker
ASPECTS = ("9:16", "1:1", "16:9", "draft")
USERNAME_RE = re.compile(r"^[a-z0-9-]{1,32}$")
FILENAME_RE = re.compile(r"^([a-z0-9-]{1,32})_(\d{10,16})_(\d{1,10})\.webp$")

Renderer = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


# ---------------------------------------------------------------- settings


class Settings:
    def __init__(self) -> None:
        env = os.environ.get
        self.session_secret = env("SESSION_SECRET", "")
        self.users = parse_users(env("ALLOWED_USERS", ""))
        self.modal_url = env("MODAL_ENDPOINT_URL", "").rstrip("/")
        self.studio_api_key = env("STUDIO_API_KEY", "")
        self.modal_proxy_key = env("MODAL_PROXY_KEY", "")
        self.modal_proxy_secret = env("MODAL_PROXY_SECRET", "")
        self.llm_base_url = env("LLM_BASE_URL", "").rstrip("/")
        self.llm_api_key = env("LLM_API_KEY", "")
        self.llm_models = parse_models(env("LLM_MODELS", ""))
        legacy = env("LLM_MODEL", "")
        self.llm_default_model = (
            env("LLM_DEFAULT_MODEL", "") or legacy or (self.llm_models[0]["id"] if self.llm_models else "")
        )
        if self.llm_default_model and self.llm_default_model not in [m["id"] for m in self.llm_models]:
            self.llm_models.insert(0, {"label": "Default", "id": self.llm_default_model})
        self.enhancer_model = env("ENHANCER_MODEL", "") or self.llm_default_model
        self.local_prompts_path = Path(env("WRITER_PROMPTS_FILE", str(BASE_DIR / "prompts.local.json")))
        self.cookie_secure = env("COOKIE_SECURE", "1") != "0"
        self.monthly_cap_min = float(env("MONTHLY_GPU_MINUTES_CAP", "1800"))
        self.data_dir = Path(env("DATA_DIR", str(BASE_DIR)))

        if len(self.session_secret) < 32:
            raise RuntimeError("SESSION_SECRET must be set (>= 32 chars)")
        if not self.users:
            raise RuntimeError("ALLOWED_USERS must list at least one user:password pair")

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_base_url and self.llm_default_model)


def parse_models(raw: str) -> list[dict[str, str]]:
    """'Label|model_id,Label 2|model_id_2' -> [{'label': ..., 'id': ...}]"""
    out: list[dict[str, str]] = []
    for item in filter(None, (p.strip() for p in raw.split(","))):
        label, sep, model_id = item.partition("|")
        model_id = (model_id if sep else label).strip()
        if model_id and model_id not in [m["id"] for m in out]:
            out.append({"label": (label if sep else model_id).strip(), "id": model_id})
    return out


def parse_users(raw: str) -> dict[str, str]:
    users: dict[str, str] = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        name, sep, pw = pair.partition(":")
        name = name.strip().lower()
        if not sep or not pw or not USERNAME_RE.fullmatch(name):
            raise RuntimeError(f"Invalid ALLOWED_USERS entry for user {name!r} (use lowercase a-z, 0-9, -)")
        users[name] = pw
    return users


# ---------------------------------------------------------------- storage


class Store:
    def __init__(self, data_dir: Path) -> None:
        self.gallery = (data_dir / "gallery").resolve()
        self.gallery.mkdir(parents=True, exist_ok=True)
        self.db_path = data_dir / "studio.db"
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY,
                    user TEXT NOT NULL,
                    filename TEXT NOT NULL UNIQUE,
                    prompt TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    aspect TEXT NOT NULL,
                    steps INTEGER NOT NULL,
                    adapter_scale REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    estimated_gpu_sec REAL NOT NULL
                )"""
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_images_user ON images(user, created_at)")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, row: dict[str, Any]) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO images (user, filename, prompt, seed, aspect, steps, adapter_scale, created_at, "
                "estimated_gpu_sec) VALUES (:user, :filename, :prompt, :seed, :aspect, :steps, :adapter_scale, "
                ":created_at, :estimated_gpu_sec)",
                row,
            )

    def list_for(self, user: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT filename, prompt, seed, aspect, steps, adapter_scale, created_at FROM images "
                "WHERE user = ? ORDER BY id DESC LIMIT 500",
                (user,),
            ).fetchall()
        return [dict(r) for r in rows]

    def owns(self, user: str, filename: str) -> bool:
        with self._conn() as c:
            return c.execute(
                "SELECT 1 FROM images WHERE user = ? AND filename = ?", (user, filename)
            ).fetchone() is not None

    def delete(self, user: str, filename: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM images WHERE user = ? AND filename = ?", (user, filename))

    def gpu_seconds_this_month(self, user: str | None = None) -> float:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        q = "SELECT COALESCE(SUM(estimated_gpu_sec), 0) FROM images WHERE substr(created_at, 1, 7) = ?"
        args: tuple = (month,)
        if user:
            q += " AND user = ?"
            args += (user,)
        with self._conn() as c:
            return float(c.execute(q, args).fetchone()[0])

    def safe_path(self, user: str, filename: str) -> Path:
        """Resolve a gallery filename for `user`, rejecting traversal and other users' files."""
        m = FILENAME_RE.fullmatch(filename)
        if not m or m.group(1) != user:
            raise HTTPException(status_code=404, detail="not found")
        path = (self.gallery / filename).resolve()
        if path.parent != self.gallery or not self.owns(user, filename):
            raise HTTPException(status_code=404, detail="not found")
        return path


# ---------------------------------------------------------------- request models


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=32)
    password: str = Field(min_length=1, max_length=256)


class GenerateBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    aspect: str = Field(default="9:16", pattern=r"^(9:16|1:1|16:9|draft)$")
    count: int = Field(default=1, ge=1, le=4)
    steps: int = Field(default=9, ge=4, le=12)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    adapter_scale: float = Field(default=0.0, ge=0.0, le=1.0)


class EnhanceBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)


# ---------------------------------------------------------------- app factory


def create_app(
    settings: Settings | None = None,
    renderer: Renderer | None = None,
    llm_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or Settings()
    store = Store(settings.data_dir)
    wstore = writer.WriterStore(store.db_path)
    signer = URLSafeTimedSerializer(settings.session_secret, salt="canvas-session")
    login_attempts: dict[str, deque[float]] = defaultdict(deque)
    http = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0), follow_redirects=True)

    async def modal_render(payload: dict[str, Any]) -> dict[str, Any]:
        if not settings.modal_url:
            raise HTTPException(status_code=503, detail="GPU worker not configured")
        headers = {"X-API-Key": settings.studio_api_key}
        if settings.modal_proxy_key:
            headers |= {"Modal-Key": settings.modal_proxy_key, "Modal-Secret": settings.modal_proxy_secret}
        try:
            r = await http.post(f"{settings.modal_url}/generate", json=payload, headers=headers)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"GPU worker unreachable: {type(e).__name__}") from e
        if r.status_code != 200:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = None
            msg = detail if isinstance(detail, str) else f"GPU worker error {r.status_code}"
            raise HTTPException(status_code=502, detail=msg)
        return r.json()

    render = renderer or modal_render
    llm_http = httpx.AsyncClient(transport=llm_transport) if llm_transport else http
    llm = writer.LLMClient(llm_http, settings.llm_base_url, settings.llm_api_key)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await http.aclose()
        if llm_http is not http:
            await llm_http.aclose()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.settings, app.state.store = settings, store

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["X-Robots-Tag"] = "noindex, nofollow"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'"
        )
        return resp

    # ------------------------------------------------ auth helpers

    def client_ip(request: Request) -> str:
        # uvicorn is bound to 127.0.0.1 behind ngrok, so the forwarded header is the only real signal.
        fwd = request.headers.get("x-forwarded-for", "")
        return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")

    def require_user(request: Request) -> str:
        token = request.cookies.get(SESSION_COOKIE)
        if not token:
            raise HTTPException(status_code=401, detail="login required")
        try:
            user = signer.loads(token, max_age=SESSION_MAX_AGE)
        except BadSignature:  # includes SignatureExpired
            raise HTTPException(status_code=401, detail="session expired")
        if user not in settings.users:
            raise HTTPException(status_code=401, detail="login required")
        return user

    def same_origin(request: Request) -> None:
        # Custom header forces a CORS preflight on cross-site requests, which we never approve (CSRF guard).
        if request.headers.get("x-requested-with") != "canvas":
            raise HTTPException(status_code=403, detail="forbidden")

    # ------------------------------------------------ public routes

    @app.get("/robots.txt", include_in_schema=False)
    def robots() -> PlainTextResponse:
        return PlainTextResponse("User-agent: *\nDisallow: /\n")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    for name, media in (("manifest.json", "application/manifest+json"), ("sw.js", "text/javascript"),
                        ("icon.svg", "image/svg+xml")):
        def _serve(name: str = name, media: str = media) -> FileResponse:
            return FileResponse(STATIC_DIR / name, media_type=media, headers={"Cache-Control": "no-cache"})
        app.add_api_route(f"/{name}", _serve, methods=["GET"], include_in_schema=False)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.post("/api/login", dependencies=[Depends(same_origin)])
    def login(body: LoginBody, request: Request, response: Response) -> dict:
        ip, now = client_ip(request), time.monotonic()
        attempts = login_attempts[ip]
        while attempts and now - attempts[0] > LOGIN_WINDOW_S:
            attempts.popleft()
        if len(attempts) >= LOGIN_LIMIT:
            raise HTTPException(status_code=429, detail="too many attempts, wait a minute")
        attempts.append(now)

        user = body.username.strip().lower()
        expected = settings.users.get(user)
        # compare against a dummy when the user is unknown so timing does not reveal valid names
        if not hmac.compare_digest(body.password.encode(), (expected or "\x00invalid").encode()) or not expected:
            raise HTTPException(status_code=401, detail="invalid credentials")

        attempts.clear()
        response.set_cookie(
            SESSION_COOKIE, signer.dumps(user), max_age=SESSION_MAX_AGE, httponly=True,
            secure=settings.cookie_secure, samesite="lax", path="/",
        )
        return {"user": user}

    @app.post("/api/logout", dependencies=[Depends(same_origin)])
    def logout(response: Response) -> dict:
        response.delete_cookie(SESSION_COOKIE, path="/")
        return {"ok": True}

    # ------------------------------------------------ authed routes

    def usage(user: str) -> dict:
        llm_user, llm_total = wstore.usage_this_month(user), wstore.usage_this_month()
        return {
            "gpu_minutes_user": round(store.gpu_seconds_this_month(user) / 60, 1),
            "gpu_minutes_total": round(store.gpu_seconds_this_month() / 60, 1),
            "gpu_minutes_cap": settings.monthly_cap_min,
            "llm_tokens_user": llm_user["tokens"],
            "llm_cost_user": llm_user["cost_usd"],
            "llm_tokens_total": llm_total["tokens"],
            "llm_cost_total": llm_total["cost_usd"],
        }

    @app.get("/api/me")
    def me(user: str = Depends(require_user)) -> dict:
        return {"user": user, "writer_enabled": settings.llm_enabled, "llm_enhancer": settings.llm_enabled,
                "usage": usage(user)}

    async def render_and_save(user: str, req: dict[str, Any]) -> dict:
        """Run the GPU worker and persist results to the user's gallery. Shared by Studio and Writer."""
        if reason := prompt_tools.blocked_reason(req["prompt"]):
            raise HTTPException(status_code=400, detail=reason)
        if store.gpu_seconds_this_month() / 60 >= settings.monthly_cap_min:
            raise HTTPException(status_code=429, detail="Monthly GPU budget reached")

        result = await render(req)
        images = result.get("images") or []
        if not images:
            raise HTTPException(status_code=502, detail="GPU worker returned no images")

        billed = float(result.get("generation_seconds", 0)) + IDLE_BILLING_S + float(result.get("load_seconds", 0))
        per_image = round(billed / len(images), 2)
        now = datetime.now(timezone.utc)
        ts = int(now.timestamp() * 1000)
        saved = []
        for img in images:
            seed = int(img["seed"])
            filename = f"{user}_{ts}_{seed}.webp"
            (store.gallery / filename).write_bytes(base64.b64decode(img["image_b64"]))
            store.add({
                "user": user, "filename": filename, "prompt": req["prompt"], "seed": seed,
                "aspect": req["aspect"], "steps": req["steps"], "adapter_scale": req["adapter_scale"],
                "created_at": now.isoformat(), "estimated_gpu_sec": per_image,
            })
            saved.append({"filename": filename, "url": f"/api/image/{filename}", "seed": seed})

        return {
            "images": saved,
            "cold_start": bool(result.get("cold_start")),
            "generation_seconds": result.get("generation_seconds"),
            "estimated_gpu_sec": round(billed, 1),
            "usage": usage(user),
        }

    @app.post("/api/generate", dependencies=[Depends(same_origin)])
    async def generate(body: GenerateBody, user: str = Depends(require_user)) -> dict:
        return await render_and_save(user, body.model_dump())

    @app.post("/api/enhance-prompt", dependencies=[Depends(same_origin)])
    async def enhance(body: EnhanceBody, user: str = Depends(require_user)) -> dict:
        if reason := prompt_tools.blocked_reason(body.prompt):
            raise HTTPException(status_code=400, detail=reason)
        rule_text, changes = prompt_tools.rule_based_enhance(body.prompt)
        source = "rules"
        text = rule_text
        if settings.llm_enabled:
            try:
                llm_text, usage_info = await llm.complete(
                    [{"role": "system", "content": prompt_tools.ENHANCER_SYSTEM_PROMPT},
                     {"role": "user", "content": body.prompt}],
                    model=settings.enhancer_model, temperature=0.4, max_tokens=400,
                )
                wstore.log_usage(user, "enhance", settings.enhancer_model, usage_info, body.prompt, llm_text)
                cleaned, more = prompt_tools.strip_negations(prompt_tools.scrub_terms(llm_text.strip().strip('"')))
                if len(cleaned.split()) >= 30 and not prompt_tools.blocked_reason(cleaned):
                    text, source = cleaned, "llm"
                    changes += more
            except HTTPException:
                pass  # fall back to rule-based result
        return {"prompt": text, "source": source, "changes": changes, "words": len(text.split())}

    @app.get("/api/gallery")
    def gallery(user: str = Depends(require_user)) -> dict:
        items = store.list_for(user)
        for it in items:
            it["url"] = f"/api/image/{it['filename']}"
        return {"items": items, "usage": usage(user)}

    @app.get("/api/image/{filename}")
    def get_image(filename: str, download: bool = False, user: str = Depends(require_user)) -> FileResponse:
        path = store.safe_path(user, filename)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(
            path, media_type="image/webp",
            filename=filename if download else None,
            headers={"Cache-Control": "private, max-age=86400"},
        )

    @app.delete("/api/image/{filename}", dependencies=[Depends(same_origin)])
    def delete_image(filename: str, user: str = Depends(require_user)) -> dict:
        path = store.safe_path(user, filename)
        path.unlink(missing_ok=True)
        store.delete(user, filename)
        return {"ok": True}

    app.include_router(writer.build_router(
        settings=settings, wstore=wstore, llm=llm, require_user=require_user, same_origin=same_origin,
        render_and_save=render_and_save, usage=usage, local_prompts_path=settings.local_prompts_path,
    ))

    return app


if os.environ.get("CANVAS_SKIP_APP") != "1":
    app = create_app()
