"""Voice: shared voice bank (consented reference clips), single-voice narration, and multi-speaker
scene dialogue rendered on the Modal TTS worker. Audio files are private to the user who made them."""

import base64
import json
import re
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Literal

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

PRESETS: dict[str, tuple[float, float]] = {  # name -> (exaggeration, cfg_weight)
    "natural": (0.5, 0.5),
    "slow_deep": (0.65, 0.3),
    "expressive": (0.85, 0.25),
}
TTS_IDLE_S = 20                       # scaledown_window on the TTS worker
T4_TO_L4_COST = 0.000164 / 0.000222   # TTS seconds expressed in image-GPU budget seconds
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_EXT = {".wav", ".mp3", ".m4a"}
MAX_SPEAK_CHARS = 6000
AUDIO_RE = re.compile(r"^([a-z0-9-]{1,32})_(\d{10,16})_([a-f0-9]{6})(_dlg)?\.mp3$")
VOICE_ID_RE = re.compile(r"^[a-z0-9_-]{1,48}$")

TRANSLIT_PROMPT = (
    "Transliterate this Roman-script Hinglish into natural spoken Hindi written in Devanagari, verbatim. "
    "Keep English loanwords as they are naturally spoken (written in Devanagari). Do not translate meaning, "
    "do not add or remove anything, no commentary. Output only the transliterated text."
)

PARSE_PROMPT = """Convert the text below into an ordered list of spoken turns for a multi-voice audio scene.

Return ONLY JSON: {{"characters": [names in order of first appearance], "turns": [{{"slot": "A"|"B"|"narrator", "speaker_name": "...", "text": "...", "preset": "natural"|"slow_deep"|"expressive"}}]}}

Rules:
- Slot "A" is the first character who speaks, "B" the second. If a third character speaks, assign them to whichever of A or B fits better.
- {narrator_rule}
- Keep every spoken line faithful to the source. Do not censor, summarise, soften or reword.
- Remove markdown, asterisks, parenthetical stage directions and character-name labels from "text".
- Choose "preset" from the line's emotion: calm -> natural, low/intense -> slow_deep, heightened -> expressive.
- {language_rule}
- At most 30 turns. Merge consecutive lines by the same speaker."""


class VoiceStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS voices (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    gender TEXT NOT NULL,
                    voice_lang TEXT NOT NULL,
                    default_exaggeration REAL NOT NULL,
                    default_cfg REAL NOT NULL,
                    consent_confirmed INTEGER NOT NULL DEFAULT 0,
                    created_by TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audio_clips (
                    id TEXT PRIMARY KEY,
                    user TEXT NOT NULL,
                    thread_id TEXT,
                    message_id INTEGER,
                    voice_id TEXT NOT NULL,
                    language_id TEXT NOT NULL,
                    filename TEXT NOT NULL UNIQUE,
                    text_preview TEXT NOT NULL DEFAULT '',
                    duration_sec REAL NOT NULL DEFAULT 0,
                    estimated_gpu_sec REAL NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audio_user ON audio_clips(user, created_at);
                """
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def voices(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM voices ORDER BY gender, display_name")]

    def voice(self, voice_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM voices WHERE id = ?", (voice_id,)).fetchone()
        return dict(r) if r else None

    def add_voice(self, row: dict) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO voices (id, display_name, gender, voice_lang, default_exaggeration, default_cfg, "
                "consent_confirmed, created_by, created_at) VALUES (:id, :display_name, :gender, :voice_lang, "
                ":default_exaggeration, :default_cfg, :consent_confirmed, :created_by, :created_at)",
                row,
            )

    def delete_voice(self, voice_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM voices WHERE id = ?", (voice_id,))

    def add_clip(self, row: dict) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO audio_clips (id, user, thread_id, message_id, voice_id, language_id, filename, "
                "text_preview, duration_sec, estimated_gpu_sec, created_at) VALUES (:id, :user, :thread_id, "
                ":message_id, :voice_id, :language_id, :filename, :text_preview, :duration_sec, "
                ":estimated_gpu_sec, :created_at)",
                row,
            )

    def clips(self, user: str, thread_id: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM audio_clips WHERE user = ?", [user]
        if thread_id:
            q += " AND thread_id = ?"
            args.append(thread_id)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY created_at DESC LIMIT 300", args)]

    def clip(self, user: str, clip_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM audio_clips WHERE id = ? AND user = ?", (clip_id, user)).fetchone()
        return dict(r) if r else None

    def owns_file(self, user: str, filename: str) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM audio_clips WHERE user = ? AND filename = ?", (user, filename)).fetchone() is not None

    def delete_clip(self, user: str, clip_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM audio_clips WHERE id = ? AND user = ?", (clip_id, user))

    def tts_gpu_seconds_this_month(self, user: str | None = None) -> float:
        t = time.gmtime()
        start = int(time.mktime((t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 0, 0)) - time.timezone)
        q, args = "SELECT COALESCE(SUM(estimated_gpu_sec), 0) FROM audio_clips WHERE created_at >= ?", [start]
        if user:
            q += " AND user = ?"
            args.append(user)
        with self._conn() as c:
            return float(c.execute(q, args).fetchone()[0])


# ---------------------------------------------------------------- helpers


def latin_majority(text: str) -> bool:
    latin = len(re.findall(r"[A-Za-z]", text))
    deva = len(re.findall(r"[ऀ-ॿ]", text))
    return latin > deva


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:24] or "voice"


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------- request models


class SpeakBody(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_SPEAK_CHARS)
    voice_id: str = Field(pattern=VOICE_ID_RE.pattern)
    language_id: Literal["en", "hi"] = "en"
    exaggeration: float | None = Field(default=None, ge=0.0, le=2.0)
    cfg_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    convert_hinglish: bool = False
    thread_id: str | None = Field(default=None, max_length=64)
    message_id: int | None = None


class ParseBody(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_SPEAK_CHARS * 2)
    language_id: Literal["en", "hi"] = "en"
    convert_hinglish: bool = True
    include_narrator: bool = True


class TurnIn(BaseModel):
    slot: Literal["A", "B", "narrator"]
    speaker_name: str = Field(default="", max_length=60)
    text: str = Field(min_length=1, max_length=2000)
    preset: Literal["natural", "slow_deep", "expressive"] | None = None


class DialogueBody(BaseModel):
    language_id: Literal["en", "hi"] = "en"
    voice_map: dict[Literal["A", "B", "narrator"], str]
    turns: list[TurnIn] = Field(min_length=1, max_length=30)
    thread_id: str | None = Field(default=None, max_length=64)
    message_id: int | None = None


# ---------------------------------------------------------------- routes


def build_router(
    *,
    settings: Any,
    vstore: VoiceStore,
    wstore: Any,
    llm: Any,
    http: httpx.AsyncClient,
    require_user: Callable[..., str],
    same_origin: Callable[..., None],
    usage: Callable[[str], dict],
    budget_left: Callable[[], bool],
    data_dir: Path,
) -> APIRouter:
    router = APIRouter(prefix="/api")
    mutate = [Depends(same_origin)]
    audio_dir = (data_dir / "audio").resolve()
    voice_dir = (data_dir / "voices").resolve()
    audio_dir.mkdir(parents=True, exist_ok=True)
    voice_dir.mkdir(parents=True, exist_ok=True)

    def require_tts() -> None:
        if not settings.modal_tts_url:
            raise HTTPException(status_code=503, detail="Voice worker not configured (set MODAL_TTS_URL)")

    async def tts_call(path: str, payload: dict) -> httpx.Response:
        require_tts()
        headers = {"X-API-Key": settings.studio_api_key}
        if settings.modal_proxy_key:
            headers |= {"Modal-Key": settings.modal_proxy_key, "Modal-Secret": settings.modal_proxy_secret}
        try:
            r = await http.post(f"{settings.modal_tts_url}{path}", json=payload, headers=headers, timeout=900.0)
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Voice worker unreachable: {type(e).__name__}") from e
        if r.status_code != 200:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = None
            raise HTTPException(status_code=502, detail=detail if isinstance(detail, str) else f"Voice worker error {r.status_code}")
        return r

    def check_thread(user: str, thread_id: str | None) -> None:
        if thread_id:
            wstore.get_thread(user, thread_id)  # 404 if not owned

    def save_audio(user: str, r: httpx.Response, *, voice_id: str, language_id: str, text: str,
                   thread_id: str | None, message_id: int | None, dialogue: bool) -> dict:
        clip_id = secrets.token_hex(3)
        filename = f"{user}_{int(time.time() * 1000)}_{clip_id}{'_dlg' if dialogue else ''}.mp3"
        (audio_dir / filename).write_bytes(r.content)
        duration = float(r.headers.get("X-Audio-Duration") or 0)
        gen_s = float(r.headers.get("X-Gen-Seconds") or 0)
        load_s = float(r.headers.get("X-Load-Seconds") or 0)
        gpu_sec = round((gen_s + load_s + TTS_IDLE_S) * T4_TO_L4_COST, 2)
        vstore.add_clip({
            "id": clip_id, "user": user, "thread_id": thread_id, "message_id": message_id, "voice_id": voice_id,
            "language_id": language_id, "filename": filename, "text_preview": text[:200],
            "duration_sec": duration, "estimated_gpu_sec": gpu_sec, "created_at": int(time.time()),
        })
        return {"id": clip_id, "url": f"/api/audio/{filename}", "filename": filename, "duration_sec": duration,
                "message_id": message_id, "thread_id": thread_id, "voice_id": voice_id,
                "cold_start": r.headers.get("X-Cold-Start") == "1", "usage": usage(user)}

    async def to_devanagari(user: str, text: str) -> str:
        if not settings.llm_enabled:
            return text
        out, usage_info = await llm.complete(
            [{"role": "system", "content": TRANSLIT_PROMPT}, {"role": "user", "content": text}],
            model=settings.llm_default_model, temperature=0.2, max_tokens=min(8000, len(text) * 2 + 200),
        )
        wstore.log_usage(user, "transliterate", settings.llm_default_model, usage_info, text, out)
        return out.strip() or text

    # ---- voice bank (shared between the family accounts)

    @router.get("/voices")
    def list_voices(user: str = Depends(require_user)) -> dict:
        out: dict[str, list] = {"male": [], "female": []}
        for v in vstore.voices():
            v["mine"] = v["created_by"] == user
            out.setdefault(v["gender"], []).append(v)
        return {**out, "presets": PRESETS, "enabled": bool(settings.modal_tts_url)}

    @router.post("/voices", dependencies=mutate)
    async def upload_voice(
        file: UploadFile = File(...),
        display_name: str = Form(..., min_length=1, max_length=40),
        gender: Literal["male", "female"] = Form(...),
        voice_lang: Literal["en", "hi"] = Form("en"),
        preset: Literal["natural", "slow_deep", "expressive"] = Form("natural"),
        consent: bool = Form(False),
        user: str = Depends(require_user),
    ) -> dict:
        if not consent:
            raise HTTPException(status_code=400, detail="Confirm you have permission to use this voice")
        ext = Path(file.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            raise HTTPException(status_code=422, detail="Upload a .wav, .mp3 or .m4a file")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Clip must be 5 MB or smaller")
        if len(data) < 1000:
            raise HTTPException(status_code=422, detail="Clip is empty or too short")
        voice_id = f"{gender[0]}_{slugify(display_name)}_{secrets.token_hex(2)}"
        backup = voice_dir / f"{voice_id}{ext}"
        backup.write_bytes(data)
        try:
            r = await tts_call("/voices/upload", {
                "voice_id": voice_id, "audio_b64": base64.b64encode(data).decode(), "filename": f"clip{ext}",
            })
        except HTTPException:
            backup.unlink(missing_ok=True)
            raise
        exg, cfg = PRESETS[preset]
        vstore.add_voice({
            "id": voice_id, "display_name": display_name.strip(), "gender": gender, "voice_lang": voice_lang,
            "default_exaggeration": exg, "default_cfg": cfg, "consent_confirmed": 1,
            "created_by": user, "created_at": int(time.time()),
        })
        return {**vstore.voice(voice_id), "reference_seconds": r.json().get("seconds")}

    @router.delete("/voices/{voice_id}", dependencies=mutate)
    async def delete_voice(voice_id: str, user: str = Depends(require_user)) -> dict:
        v = vstore.voice(voice_id)
        if not v:
            raise HTTPException(status_code=404, detail="voice not found")
        if v["created_by"] != user:
            raise HTTPException(status_code=403, detail="Only the person who added this voice can delete it")
        await tts_call("/voices/delete", {"voice_id": voice_id})
        for p in voice_dir.glob(f"{voice_id}.*"):
            p.unlink(missing_ok=True)
        vstore.delete_voice(voice_id)
        return {"ok": True}

    # ---- single voice

    @router.post("/tts/speak", dependencies=mutate)
    async def speak(body: SpeakBody, user: str = Depends(require_user)) -> dict:
        require_tts()
        check_thread(user, body.thread_id)
        v = vstore.voice(body.voice_id)
        if not v:
            raise HTTPException(status_code=400, detail="unknown voice")
        if not budget_left():
            raise HTTPException(status_code=429, detail="Monthly GPU budget reached")
        text = body.text
        if body.language_id == "hi" and (body.convert_hinglish or latin_majority(text)):
            text = await to_devanagari(user, text)
        r = await tts_call("/synthesize", {
            "text": text, "voice_id": body.voice_id, "language_id": body.language_id, "voice_lang": v["voice_lang"],
            "exaggeration": body.exaggeration if body.exaggeration is not None else v["default_exaggeration"],
            "cfg_weight": body.cfg_weight if body.cfg_weight is not None else v["default_cfg"],
        })
        out = save_audio(user, r, voice_id=body.voice_id, language_id=body.language_id, text=text,
                         thread_id=body.thread_id, message_id=body.message_id, dialogue=False)
        return {**out, "spoken_text": text}

    # ---- scene dialogue

    @router.post("/tts/parse-dialogue", dependencies=mutate)
    async def parse_dialogue(body: ParseBody, user: str = Depends(require_user)) -> dict:
        if not settings.llm_enabled:
            raise HTTPException(status_code=503, detail="Scene splitting needs the Writer model configured")
        narrator_rule = (
            'Descriptive action and narration become "narrator" turns, kept in order.' if body.include_narrator
            else "Omit all narration and action description; include only spoken dialogue."
        )
        language_rule = (
            "Write every turn's text in natural Hindi using Devanagari script; transliterate Roman Hinglish "
            "verbatim, keeping tone and English loanwords." if body.language_id == "hi" and body.convert_hinglish
            else "Keep each line in its original language and script."
        )
        raw, usage_info = await llm.complete(
            [{"role": "system", "content": PARSE_PROMPT.format(narrator_rule=narrator_rule, language_rule=language_rule)},
             {"role": "user", "content": body.text}],
            model=settings.llm_default_model, temperature=0.2, max_tokens=6000,
        )
        wstore.log_usage(user, "parse", settings.llm_default_model, usage_info, body.text, raw)
        try:
            data = extract_json(raw)
        except ValueError:
            raise HTTPException(status_code=502, detail="Could not split this scene; try again or trim the text")
        turns = []
        for t in data.get("turns") or []:
            if not isinstance(t, dict):
                continue
            slot = str(t.get("slot", "")).strip()
            slot = {"a": "A", "b": "B", "narrator": "narrator", "n": "narrator"}.get(slot.lower(), None)
            text = re.sub(r"[*_`#]+", "", str(t.get("text", ""))).strip()
            if not slot or not text or (slot == "narrator" and not body.include_narrator):
                continue
            preset = t.get("preset") if t.get("preset") in PRESETS else "natural"
            turns.append({"slot": slot, "speaker_name": str(t.get("speaker_name", ""))[:60], "text": text[:2000],
                          "preset": preset})
        if not turns:
            raise HTTPException(status_code=422, detail="No spoken lines found in this text")
        chars = [str(c)[:60] for c in (data.get("characters") or []) if isinstance(c, (str, int))][:10]
        return {"characters": chars, "turns": turns[:30]}

    @router.post("/tts/dialogue", dependencies=mutate)
    async def dialogue(body: DialogueBody, user: str = Depends(require_user)) -> dict:
        require_tts()
        check_thread(user, body.thread_id)
        voices: dict[str, dict] = {}
        for slot in {t.slot for t in body.turns}:
            vid = body.voice_map.get(slot)
            v = vstore.voice(vid) if vid else None
            if not v:
                raise HTTPException(status_code=400, detail=f"Pick a voice for {'Narrator' if slot == 'narrator' else 'Voice ' + slot}")
            voices[slot] = v
        if not budget_left():
            raise HTTPException(status_code=429, detail="Monthly GPU budget reached")
        turns = []
        for t in body.turns:
            v = voices[t.slot]
            exg, cfg = PRESETS[t.preset] if t.preset else (v["default_exaggeration"], v["default_cfg"])
            turns.append({"speaker": t.slot, "voice_id": v["id"], "voice_lang": v["voice_lang"], "text": t.text,
                          "exaggeration": exg, "cfg_weight": cfg})
        r = await tts_call("/synthesize-dialogue", {"language_id": body.language_id, "turns": turns})
        label = "dialogue:" + "+".join(sorted(voices))
        preview = " / ".join(t.text for t in body.turns)
        return save_audio(user, r, voice_id=label, language_id=body.language_id, text=preview,
                          thread_id=body.thread_id, message_id=body.message_id, dialogue=True)

    # ---- audio library

    @router.get("/audio-history")
    def audio_history(thread_id: str | None = None, user: str = Depends(require_user)) -> dict:
        items = vstore.clips(user, thread_id)
        for it in items:
            it["url"] = f"/api/audio/{it['filename']}"
        return {"items": items}

    @router.get("/audio/{filename}")
    def get_audio(filename: str, download: bool = False, user: str = Depends(require_user)) -> FileResponse:
        m = AUDIO_RE.fullmatch(filename)
        if not m:
            raise HTTPException(status_code=404, detail="not found")
        if m.group(1) != user:
            raise HTTPException(status_code=403, detail="forbidden")
        path = (audio_dir / filename).resolve()
        if path.parent != audio_dir or not path.is_file() or not vstore.owns_file(user, filename):
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path, media_type="audio/mpeg", filename=filename if download else None,
                            headers={"Cache-Control": "private, max-age=86400"})

    @router.delete("/audio/{clip_id}", dependencies=mutate)
    def delete_audio(clip_id: str, user: str = Depends(require_user)) -> dict:
        clip = vstore.clip(user, clip_id)
        if not clip:
            raise HTTPException(status_code=404, detail="not found")
        (audio_dir / clip["filename"]).unlink(missing_ok=True)
        vstore.delete_clip(user, clip_id)
        return {"ok": True}

    return router
