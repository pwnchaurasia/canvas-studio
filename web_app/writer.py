"""Writer: threaded script/story co-writing with streaming, a per-thread story bible, and a bridge
from any scene to the image worker. All rows are owned by one user and every query filters on it."""

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

import prompt_tools

FormatMode = Literal["chat", "screenplay", "reel", "story", "outline"]
Language = Literal["en", "hi", "hinglish", "bilingual"]
ToneMode = Literal["standard", "uncut"]
Length = Literal["normal", "long"]

HISTORY_LIMIT = 30
CONTINUE_TEXT = "Continue seamlessly from where you stopped with full detail."

# ---------------------------------------------------------------- system prompt

BASE_RULES = (
    "You are a professional screenwriter, novelist and short-form video writer working as the user's "
    "co-writer. Follow the user's direction precisely and keep continuity with everything written so far. "
    "Format output in clean Markdown: headings for scenes or sections, **bold** for character names before "
    "dialogue, and short readable paragraphs. Never summarise what you are about to write; just write it."
)

FORMAT_RULES: dict[str, str] = {
    "screenplay": (
        "FORMAT: Screenplay. Use scene headings (INT./EXT. LOCATION - TIME), vivid present-tense action "
        "lines with sensory detail, tight character dialogue with occasional parentheticals, and clear beats."
    ),
    "reel": (
        "FORMAT: Vertical Reel script, 45-60 seconds. Sections: **HOOK** (first 2 seconds), numbered "
        "**SCENE BEATS** each with **Voiceover** (max 25 words), **On-screen text**, and **Visual cue**, "
        "then a closing line or call to action. Punchy and visual."
    ),
    "story": (
        "FORMAT: Long-form prose. Rich, immersive storytelling with sensory detail, natural pacing, "
        "dialogue, and interiority. Never rush or summarise events; dramatise them."
    ),
    "outline": (
        "FORMAT: Structured outline. Break the work into episodes or chapters, each with scene goals, "
        "key beats, conflicts, and the emotional arc of the main characters."
    ),
    "chat": "FORMAT: Open conversation. Brainstorm, answer questions, and draft as the user asks.",
}

LANGUAGE_RULES: dict[str, str] = {
    "en": "LANGUAGE: Natural, cinematic English.",
    "hi": (
        "LANGUAGE: Write entirely in natural Hindi using Devanagari script, with the rhythm of real spoken "
        "Hindi. Avoid stiff textbook or machine-translated phrasing."
    ),
    "hinglish": (
        "LANGUAGE: Hinglish in Roman script: conversational Hindi mixed naturally with English, the way "
        "modern urban Indians talk and text. Do not use Devanagari."
    ),
    "bilingual": (
        "LANGUAGE: Bilingual. Write scene headings, action and directions in English; write all spoken "
        "dialogue in natural Hindi or Hinglish, as in contemporary Indian screenplays."
    ),
}

TONE_RULES: dict[str, str] = {
    "standard": "TONE: Polished writing suitable for a general audience.",
    "uncut": (
        "TONE: Mature fiction for an adult audience. Follow the user's creative direction faithfully, "
        "without moralising, disclaimers, or fade-to-black shortcuts, and keep descriptions detailed and "
        "realistic. All characters are adults (18+)."
    ),
}

LONG_RULE = (
    "LENGTH: Write at full length, roughly 1,500-3,000 words for this reply, with complete scenes. "
    "Stop only at a natural scene break."
)

SCENE_SYSTEM_PROMPT = f"""You convert a scene from a script or story into ONE prompt for a photorealistic text-to-image model.

Rules:
1. Output ONLY the prompt: a single English paragraph of 80-160 words. No preamble, no quotes, no lists.
2. The scene may be in Hindi, Hinglish or English. The prompt is always in English.
3. Describe only what IS visible. Never use negations such as "no", "not", "without", "barely" or "avoid".
4. Never write the word "camera". Describe the lens look instead, e.g. "85mm lens look, shallow depth of field".
5. Copy each character's physical description from the STORY BIBLE exactly (age, skin tone, hair, build, clothing).
6. Choose the single most visual moment of the scene. Include framing, setting, one cohesive lighting setup,
   realistic skin texture and materials, and mood.
7. All people depicted are adults."""

_LOCAL_PROMPTS: dict[str, Any] = {"mtime": None, "data": {}}


def load_local_prompts(path: Path) -> dict[str, str]:
    """Optional gitignored overrides: {"base": "...", "standard": "...", "uncut": "..."}."""
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return {}
    if _LOCAL_PROMPTS["mtime"] != mtime:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _LOCAL_PROMPTS["data"] = {k: v for k, v in data.items() if isinstance(v, str)} if isinstance(data, dict) else {}
        except (OSError, ValueError):
            _LOCAL_PROMPTS["data"] = {}
        _LOCAL_PROMPTS["mtime"] = mtime
    return _LOCAL_PROMPTS["data"]


def build_writer_system_prompt(thread: dict, overrides: dict[str, str] | None = None, length: str = "normal") -> str:
    overrides = overrides or {}
    tone = thread.get("tone_mode", "standard")
    parts = [
        overrides.get("base") or BASE_RULES,
        FORMAT_RULES.get(thread.get("format_mode", "chat"), FORMAT_RULES["chat"]),
        LANGUAGE_RULES.get(thread.get("language", "en"), LANGUAGE_RULES["en"]),
        overrides.get(tone) or TONE_RULES.get(tone, TONE_RULES["standard"]),
    ]
    if length == "long":
        parts.append(LONG_RULE)
    bible = (thread.get("story_bible") or "").strip()
    if bible:
        parts.append(
            "STORY BIBLE (canon; never contradict it: keep every character's age, appearance, relationships "
            f"and established facts consistent):\n{bible}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------- LLM client


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMClient:
    def __init__(self, http: httpx.AsyncClient, base_url: str, api_key: str):
        self.http, self.base_url, self.api_key = http, base_url, api_key

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    @staticmethod
    def _error(status: int, body: bytes) -> HTTPException:
        try:
            j = json.loads(body)
            msg = (j.get("error") or {}).get("message") if isinstance(j.get("error"), dict) else j.get("error")
            msg = msg or j.get("detail") or j.get("message")
        except ValueError:
            msg = None
        msg = str(msg or body[:200].decode("utf-8", "replace")).strip()[:200]
        return HTTPException(status_code=502, detail=f"Writer model error {status}: {msg}")

    async def complete(self, messages: list[dict], model: str, temperature: float, max_tokens: int) -> tuple[str, dict]:
        try:
            r = await self.http.post(
                f"{self.base_url}/chat/completions",
                json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                headers=self._headers, timeout=120.0,
            )
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Writer model unreachable: {type(e).__name__}") from e
        if r.status_code != 200:
            raise self._error(r.status_code, r.content)
        try:
            j = r.json()
            text = j["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise HTTPException(status_code=502, detail="Writer model returned an unexpected response") from e
        return _THINK.sub("", text).strip(), j.get("usage") or {}

    async def stream(
        self, messages: list[dict], model: str, temperature: float, max_tokens: int
    ) -> AsyncIterator[tuple[str, Any]]:
        payload = {
            "model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True},
        }
        try:
            async with self.http.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers,
                timeout=httpx.Timeout(180.0, connect=10.0),
            ) as r:
                if r.status_code != 200:
                    raise self._error(r.status_code, await r.aread())
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if chunk.get("usage"):
                        yield "usage", chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            yield "delta", delta
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Writer model connection lost: {type(e).__name__}") from e


# ---------------------------------------------------------------- storage


class WriterStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    user TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    model_id TEXT NOT NULL DEFAULT '',
                    format_mode TEXT NOT NULL DEFAULT 'chat',
                    language TEXT NOT NULL DEFAULT 'en',
                    tone_mode TEXT NOT NULL DEFAULT 'standard',
                    story_bible TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_threads_user ON threads(user, updated_at);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
                    user TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    model_id TEXT NOT NULL DEFAULT '',
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, id);
                CREATE TABLE IF NOT EXISTS llm_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_llm_usage_user ON llm_usage(user, created_at);
                """
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # threads
    def create_thread(self, user: str, fields: dict) -> dict:
        now = int(time.time())
        row = {"id": uuid.uuid4().hex, "user": user, "created_at": now, "updated_at": now, **fields}
        with self._conn() as c:
            c.execute(
                "INSERT INTO threads (id, user, title, model_id, format_mode, language, tone_mode, story_bible, "
                "created_at, updated_at) VALUES (:id, :user, :title, :model_id, :format_mode, :language, "
                ":tone_mode, :story_bible, :created_at, :updated_at)",
                row,
            )
        return self.get_thread(user, row["id"])

    def get_thread(self, user: str, thread_id: str) -> dict:
        with self._conn() as c:
            r = c.execute("SELECT * FROM threads WHERE id = ? AND user = ?", (thread_id, user)).fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="thread not found")
        return dict(r)

    def list_threads(self, user: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, title, model_id, format_mode, language, tone_mode, updated_at FROM threads "
                "WHERE user = ? ORDER BY updated_at DESC LIMIT 300",
                (user,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_thread(self, user: str, thread_id: str, fields: dict) -> dict:
        self.get_thread(user, thread_id)
        if fields:
            cols = ", ".join(f"{k} = :{k}" for k in fields)
            with self._conn() as c:
                c.execute(
                    f"UPDATE threads SET {cols}, updated_at = :now WHERE id = :id AND user = :user",
                    {**fields, "now": int(time.time()), "id": thread_id, "user": user},
                )
        return self.get_thread(user, thread_id)

    def touch(self, user: str, thread_id: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE threads SET updated_at = ? WHERE id = ? AND user = ?", (int(time.time()), thread_id, user))

    def delete_thread(self, user: str, thread_id: str) -> None:
        self.get_thread(user, thread_id)
        with self._conn() as c:
            c.execute("DELETE FROM messages WHERE thread_id = ? AND user = ?", (thread_id, user))
            c.execute("DELETE FROM threads WHERE id = ? AND user = ?", (thread_id, user))

    # messages
    def add_message(self, user: str, thread_id: str, role: str, content: str, model_id: str = "",
                    prompt_tokens: int = 0, completion_tokens: int = 0) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO messages (thread_id, user, role, content, model_id, prompt_tokens, completion_tokens, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (thread_id, user, role, content, model_id, prompt_tokens, completion_tokens, int(time.time())),
            )
            return int(cur.lastrowid)

    def messages(self, user: str, thread_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, role, content, model_id, prompt_tokens, completion_tokens, created_at FROM messages "
                "WHERE thread_id = ? AND user = ? ORDER BY id",
                (thread_id, user),
            ).fetchall()
        out = []
        for r in rows:
            m = dict(r)
            if m["role"] == "image":
                try:
                    m["content"] = json.loads(m["content"])
                except ValueError:
                    m["content"] = {}
            out.append(m)
        return out

    def history(self, user: str, thread_id: str, limit: int = HISTORY_LIMIT) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT role, content FROM messages WHERE thread_id = ? AND user = ? AND role IN ('user', 'assistant') "
                "ORDER BY id DESC LIMIT ?",
                (thread_id, user, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def drop_trailing_assistant(self, user: str, thread_id: str) -> int:
        """Delete assistant (and inline image) messages after the last user message; returns how many."""
        with self._conn() as c:
            last_user = c.execute(
                "SELECT MAX(id) FROM messages WHERE thread_id = ? AND user = ? AND role = 'user'", (thread_id, user)
            ).fetchone()[0]
            if last_user is None:
                return 0
            cur = c.execute(
                "DELETE FROM messages WHERE thread_id = ? AND user = ? AND id > ? AND role = 'assistant'",
                (thread_id, user, last_user),
            )
            return cur.rowcount

    # usage
    def log_usage(self, user: str, kind: str, model_id: str, usage: dict, fallback_prompt: str = "",
                  fallback_completion: str = "") -> tuple[int, int]:
        pt = int(usage.get("prompt_tokens") or 0) or len(fallback_prompt) // 4
        ct = int(usage.get("completion_tokens") or 0) or len(fallback_completion) // 4
        cost = float(usage.get("cost") or 0.0)
        with self._conn() as c:
            c.execute(
                "INSERT INTO llm_usage (user, kind, model_id, prompt_tokens, completion_tokens, cost_usd, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user, kind, model_id, pt, ct, cost, int(time.time())),
            )
        return pt, ct

    def usage_this_month(self, user: str | None = None) -> dict:
        start = time.gmtime()
        month_start = int(time.mktime((start.tm_year, start.tm_mon, 1, 0, 0, 0, 0, 0, 0)) - time.timezone)
        q = "SELECT COALESCE(SUM(prompt_tokens + completion_tokens), 0), COALESCE(SUM(cost_usd), 0) FROM llm_usage WHERE created_at >= ?"
        args: tuple = (month_start,)
        if user:
            q += " AND user = ?"
            args += (user,)
        with self._conn() as c:
            tokens, cost = c.execute(q, args).fetchone()
        return {"tokens": int(tokens), "cost_usd": round(float(cost), 4)}


# ---------------------------------------------------------------- request models


class ThreadCreate(BaseModel):
    title: str = Field(default="", max_length=120)
    format_mode: FormatMode = "story"
    language: Language = "en"
    tone_mode: ToneMode = "standard"
    model_id: str = Field(default="", max_length=200)
    story_bible: str = Field(default="", max_length=20000)


class ThreadPatch(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    format_mode: FormatMode | None = None
    language: Language | None = None
    tone_mode: ToneMode | None = None
    model_id: str | None = Field(default=None, max_length=200)
    story_bible: str | None = Field(default=None, max_length=20000)


class StreamBody(BaseModel):
    content: str = Field(default="", max_length=20000)
    regenerate: bool = False
    length: Length = "normal"


class ScenePromptBody(BaseModel):
    scene_text: str = Field(min_length=1, max_length=20000)
    style_preset: Literal["standard", "custom"] = "standard"


class RenderInlineBody(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    aspect: str = Field(default="9:16", pattern=r"^(9:16|1:1|16:9|draft)$")
    count: int = Field(default=1, ge=1, le=2)
    steps: int = Field(default=9, ge=4, le=12)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    adapter_scale: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------- routes


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def build_router(
    *,
    settings: Any,
    wstore: WriterStore,
    llm: LLMClient,
    require_user: Callable[..., str],
    same_origin: Callable[..., None],
    render_and_save: Callable[[str, dict], Awaitable[dict]],
    usage: Callable[[str], dict],
    local_prompts_path: Path,
) -> APIRouter:
    router = APIRouter(prefix="/api")
    mutate = [Depends(same_origin)]

    def require_llm() -> None:
        if not settings.llm_enabled:
            raise HTTPException(status_code=503, detail="Writer is not configured (set LLM_BASE_URL and models)")

    def resolve_model(model_id: str | None) -> str:
        allowed = [m["id"] for m in settings.llm_models]
        if model_id and model_id in allowed:
            return model_id
        if model_id and model_id not in allowed:
            raise HTTPException(status_code=422, detail="unknown model")
        return settings.llm_default_model

    def guard_uncut(tone: str, *texts: str) -> None:
        if tone == "uncut":
            for t in texts:
                if t and (reason := prompt_tools.blocked_reason(t)):
                    raise HTTPException(status_code=400, detail="Uncut mode: all characters must be adults. " + reason)

    @router.get("/models")
    def models(user: str = Depends(require_user)) -> dict:
        return {"models": settings.llm_models, "default": settings.llm_default_model, "enabled": settings.llm_enabled}

    @router.get("/threads")
    def list_threads(user: str = Depends(require_user)) -> dict:
        return {"threads": wstore.list_threads(user)}

    @router.post("/threads", dependencies=mutate)
    def create_thread(body: ThreadCreate, user: str = Depends(require_user)) -> dict:
        guard_uncut(body.tone_mode, body.story_bible)
        fields = body.model_dump()
        fields["model_id"] = resolve_model(body.model_id)
        return wstore.create_thread(user, fields)

    @router.get("/threads/{thread_id}")
    def get_thread(thread_id: str, user: str = Depends(require_user)) -> dict:
        return wstore.get_thread(user, thread_id)

    @router.patch("/threads/{thread_id}", dependencies=mutate)
    def patch_thread(thread_id: str, body: ThreadPatch, user: str = Depends(require_user)) -> dict:
        current = wstore.get_thread(user, thread_id)
        fields = {k: v for k, v in body.model_dump().items() if v is not None}
        if "model_id" in fields:
            fields["model_id"] = resolve_model(fields["model_id"])
        guard_uncut(fields.get("tone_mode", current["tone_mode"]), fields.get("story_bible", current["story_bible"]))
        return wstore.update_thread(user, thread_id, fields)

    @router.delete("/threads/{thread_id}", dependencies=mutate)
    def delete_thread(thread_id: str, user: str = Depends(require_user)) -> dict:
        wstore.delete_thread(user, thread_id)
        return {"ok": True}

    @router.get("/threads/{thread_id}/messages")
    def thread_messages(thread_id: str, user: str = Depends(require_user)) -> dict:
        thread = wstore.get_thread(user, thread_id)
        return {"thread": thread, "messages": wstore.messages(user, thread_id)}

    @router.get("/threads/{thread_id}/export")
    def export_thread(thread_id: str, user: str = Depends(require_user)) -> PlainTextResponse:
        t = wstore.get_thread(user, thread_id)
        lines = [f"# {t['title'] or 'Untitled'}", "",
                 f"_Format: {t['format_mode']} · Language: {t['language']} · Mode: {t['tone_mode']}_", ""]
        if t["story_bible"].strip():
            lines += ["## Story bible", "", t["story_bible"].strip(), ""]
        lines += ["---", ""]
        for m in wstore.messages(user, thread_id):
            if m["role"] == "user":
                lines += [f"> **Prompt:** {m['content']}", ""]
            elif m["role"] == "assistant":
                lines += [m["content"], ""]
            elif m["role"] == "image":
                c = m["content"]
                lines += [f"_[Image: {c.get('filename', '')} · seed {c.get('seed', '')}]_", ""]
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", t["title"] or "script").strip("-")[:60] or "script"
        return PlainTextResponse(
            "\n".join(lines), media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{safe}.md"'},
        )

    @router.post("/threads/{thread_id}/stream", dependencies=mutate)
    async def stream(thread_id: str, body: StreamBody, user: str = Depends(require_user)) -> StreamingResponse:
        require_llm()
        thread = wstore.get_thread(user, thread_id)
        content = body.content.strip()
        if body.regenerate:
            wstore.drop_trailing_assistant(user, thread_id)
            if not wstore.history(user, thread_id, limit=1):
                raise HTTPException(status_code=400, detail="nothing to regenerate")
        else:
            if not content:
                raise HTTPException(status_code=422, detail="message is empty")
            guard_uncut(thread["tone_mode"], content)
            wstore.add_message(user, thread_id, "user", content)
            if not thread["title"]:
                title = re.sub(r"\s+", " ", content)[:60].strip()
                thread = wstore.update_thread(user, thread_id, {"title": title})

        model = thread["model_id"] or settings.llm_default_model
        system = build_writer_system_prompt(thread, load_local_prompts(local_prompts_path), body.length)
        messages = [{"role": "system", "content": system}] + wstore.history(user, thread_id)
        temperature = 0.9 if thread["format_mode"] in ("story", "screenplay", "reel") else 0.8
        max_tokens = 8000 if body.length == "long" else 2500
        prompt_chars = "".join(m["content"] for m in messages)

        async def events():
            parts: list[str] = []
            usage_info: dict = {}
            saved_id: int | None = None
            error: str | None = None
            yield _sse({"type": "start", "title": thread["title"], "model": model})
            upstream = llm.stream(messages, model, temperature, max_tokens)
            try:
                async for kind, value in upstream:
                    if kind == "delta":
                        parts.append(value)
                        yield _sse({"type": "delta", "t": value})
                    else:
                        usage_info = value
            except HTTPException as e:
                error = str(e.detail)
            except Exception as e:  # never leak a half-open stream
                error = f"Writer stream failed: {type(e).__name__}"
            finally:
                try:
                    await upstream.aclose()  # stop the provider (and its billing) when the user presses Stop
                except BaseException:
                    pass
                text = _THINK.sub("", "".join(parts)).strip()
                if text:
                    pt, ct = wstore.log_usage(user, "write", model, usage_info, prompt_chars, text)
                    saved_id = wstore.add_message(user, thread_id, "assistant", text, model, pt, ct)
                    wstore.touch(user, thread_id)
            if error:
                yield _sse({"type": "error", "detail": error, "message_id": saved_id})
            else:
                yield _sse({"type": "done", "message_id": saved_id, "usage": usage(user)})

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.post("/threads/{thread_id}/scene-prompt", dependencies=mutate)
    async def scene_prompt(thread_id: str, body: ScenePromptBody, user: str = Depends(require_user)) -> dict:
        require_llm()
        thread = wstore.get_thread(user, thread_id)
        bible = thread["story_bible"].strip() or "(none provided)"
        model = thread["model_id"] or settings.llm_default_model
        text, usage_info = await llm.complete(
            [{"role": "system", "content": SCENE_SYSTEM_PROMPT},
             {"role": "user", "content": f"STORY BIBLE:\n{bible}\n\nSCENE:\n{body.scene_text[:12000]}"}],
            model=model, temperature=0.4, max_tokens=600,
        )
        wstore.log_usage(user, "scene", model, usage_info, body.scene_text, text)
        cleaned = re.sub(r"\bbarely\b\s*", "", text.strip().strip('"'), flags=re.IGNORECASE)
        cleaned, _ = prompt_tools.strip_negations(prompt_tools.scrub_terms(cleaned))
        words = cleaned.split()
        if len(words) > 180:
            cleaned = " ".join(words[:180]).rstrip(",;") + "."
        if len(cleaned.split()) < 15:
            raise HTTPException(status_code=502, detail="Writer model returned an unusable visual prompt; try again")
        if reason := prompt_tools.blocked_reason(cleaned):
            raise HTTPException(status_code=400, detail=reason)
        return {
            "visual_prompt": cleaned,
            "recommended_adapter_scale": 0.8 if body.style_preset == "custom" else 0.0,
            "words": len(cleaned.split()),
        }

    @router.post("/threads/{thread_id}/render-inline", dependencies=mutate)
    async def render_inline(thread_id: str, body: RenderInlineBody, user: str = Depends(require_user)) -> dict:
        wstore.get_thread(user, thread_id)
        result = await render_and_save(user, body.model_dump())
        messages = []
        for img in result["images"]:
            payload = {"url": img["url"], "filename": img["filename"], "prompt": body.prompt, "seed": img["seed"],
                       "adapter_scale": body.adapter_scale, "aspect": body.aspect}
            mid = wstore.add_message(user, thread_id, "image", json.dumps(payload))
            messages.append({"id": mid, "role": "image", "content": payload})
        wstore.touch(user, thread_id)
        return {**result, "messages": messages}

    return router
